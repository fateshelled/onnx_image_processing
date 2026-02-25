"""
Essential Matrix Estimator from Sinkhorn assignment matrix.

Implements the weighted 8-point algorithm with Hartley normalization,
optionally refined via IRLS (Iteratively Reweighted Least Squares)
and Sampson error-based refinement.
Designed for ONNX export at opset 14 with dynamic N and M axes.

Key ONNX compatibility decisions
---------------------------------
* ``torch.linalg.svd`` / ``torch.linalg.eigh`` / ``torch.linalg.solve`` are
  **not** supported by the TorchScript-based ONNX exporter at opset 14.
  → Replaced by fixed-iteration *power iteration* (unrolled loops).

* ``torch.trace``  is not exported at opset 14.
  → Replaced by ``torch.einsum("ii", M)``.

* ``torch.diag(v)`` (1-D → 2-D diagonal matrix) is not exported at opset 14.
  → Replaced by ``torch.eye(n) * v.unsqueeze(1)`` (element-wise scale of I),
    which is semantically identical and fully ONNX-compatible.
    The module uses ``torch.sign`` for sign computation (as required) and the
    eye-scale idiom wherever a diagonal matrix would otherwise be created.

Dependencies: torch, onnx, onnxruntime
"""

import torch
from torch import nn


class EssentialMatrixEstimator(nn.Module):
    """
    Estimates the Essential Matrix from a Sinkhorn probability matrix.

    Consumes the (N+1, M+1) output of a SinkhornMatcher (dustbin included)
    and returns the 3x3 Essential Matrix via the weighted 8-point algorithm.

    Algorithm overview
    ------------------
    1.  Strip dustbin row/column → P_core (N, M).
    2.  Bidirectional top-K mask AND absolute-value threshold → weight matrix.
    3.  Map row/column indices to pixel positions on a pre-defined grid;
        convert to normalised image coords via K⁻¹.
    4.  Hartley-normalise both point sets (weighted centroid + scale).
    5.  Build the (N·M, 9) epipolar design matrix A; form M = Aᵀ diag(w) A.
    6.  Minimum eigenvector of M via shifted power iteration → E_raw.
    7.  Hartley denormalise: E ← T₂ᵀ E_raw T₁.
    8.  Project onto the Essential Matrix manifold (singular values → [σ,σ,0])
        via power-iteration SVD with branch-free sign correction.
    9.  IRLS refinement: reweight correspondences using the Cauchy robust
        kernel on algebraic residuals and re-run steps 4–8.
    10. Sampson error refinement: further reweight correspondences using the
        Cauchy kernel on Sampson errors (a first-order approximation to the
        geometric reprojection error) and re-run steps 4–8.

    All operations use PyTorch tensor ops only; no Python branching on tensor
    *values* — ensuring ONNX exportability at opset 14.

    Args:
        K:                Camera intrinsic matrix, shape (3, 3).
        image_shape:      (H, W) of the feature-point grid. Feature point index
                          ``i`` maps to pixel coordinate ``(i % W, i // W)``.
                          Must satisfy ``H * W >= max(N, M)`` at inference time.
                          Default: (32, 32).
        top_k:            Number of top-probability entries kept per row *and* per
                          column in the bidirectional filter. Default: 3.
        n_iter:           Power-iteration steps for the 9x9 eigenvector solve
                          (min eigenvector of the weighted normal equations).
                          More iterations → better accuracy, larger ONNX graph.
                          Default: 30.
        n_iter_manifold:  Power-iteration steps for each 3x3 eigenvector solve
                          inside the Essential Matrix manifold projection.
                          3x3 matrices converge much faster than the 9x9 case;
                          10 iterations is sufficient in almost all practical
                          scenarios. Default: 10.
        n_irls:           Number of IRLS (Iteratively Reweighted Least Squares)
                          refinement iterations.  Each iteration recomputes
                          correspondence weights via the Cauchy robust kernel
                          on algebraic residuals and re-solves the weighted
                          8-point algorithm.  More iterations increase the ONNX
                          graph size proportionally.  0 disables IRLS.
                          Default: 5.
        irls_sigma:       Scale parameter σ for the Cauchy kernel used in IRLS:
                          ``w(r) = 1 / (1 + (r / σ)²)``.  Correspondences
                          whose algebraic residual ``|x₂ᵀ E x₁|`` is much
                          larger than σ are strongly down-weighted.
                          Default: 0.01.
        n_sampson:        Number of Sampson-error refinement iterations that
                          run after IRLS.  Each iteration reweights using the
                          Cauchy kernel on Sampson errors — a first-order
                          approximation to the geometric reprojection error —
                          and re-solves the weighted 8-point algorithm.
                          0 disables Sampson refinement.  Default: 3.
        sampson_sigma:    Scale parameter σ for the Cauchy kernel used in
                          Sampson refinement:
                          ``w(d) = 1 / (1 + d / σ²)``  where *d* is the
                          Sampson error (squared Sampson distance).
                          Default: 0.01.
    """

    def __init__(
        self,
        K: torch.Tensor,
        image_shape: tuple[int, int] = (32, 32),
        top_k: int = 3,
        n_iter: int = 30,
        n_iter_manifold: int = 10,
        n_irls: int = 5,
        irls_sigma: float = 0.01,
        n_sampson: int = 3,
        sampson_sigma: float = 0.01,
    ) -> None:
        super().__init__()

        # Register K and K⁻¹ as persistent buffers (device-portable).
        # K_inv is computed once at construction; it becomes a constant
        # in the ONNX graph (not computed at inference time).
        K_f = K.float()
        K_inv = torch.linalg.inv(K_f)
        self.register_buffer("K", K_f)
        self.register_buffer("K_inv", K_inv)

        self.top_k = top_k
        self.n_iter = n_iter
        self.n_iter_manifold = n_iter_manifold
        self.n_irls = n_irls
        self.irls_sigma = irls_sigma
        self.n_sampson = n_sampson
        self.sampson_sigma = sampson_sigma
        H, W = image_shape
        self.H = H
        self.W = W

        # Precompute pixel grid coordinates as a buffer.
        # Feature point i → pixel (x = i % W, y = i // W).
        idx = torch.arange(H * W, dtype=torch.float32)
        px = idx % W   # x-coordinate (column index)
        py = idx // W  # y-coordinate (row index)
        pixel_coords = torch.stack([px, py], dim=-1)  # (H*W, 2)
        self.register_buffer("pixel_coords", pixel_coords)

        # Precompute normalised image coordinates for every grid point.
        # pixel_coords_n[i] = K⁻¹ · [px, py, 1]ᵀ (x, y only).
        # Moving this from forward() eliminates two homogeneous-coordinate
        # cat ops and two (H*W, 3)@(3, 3) GEMMs from the ONNX graph.
        ones = torch.ones(H * W, 1)
        pixel_coords_h = torch.cat([pixel_coords, ones], dim=-1)   # (H*W, 3)
        pixel_coords_n = (pixel_coords_h @ K_inv.T)[:, :2]         # (H*W, 2)
        self.register_buffer("pixel_coords_n", pixel_coords_n)

    # ------------------------------------------------------------------
    # Private helpers (all ONNX-safe)
    # ------------------------------------------------------------------

    @staticmethod
    def _det3(M: torch.Tensor) -> torch.Tensor:
        """Analytical determinant of a 3x3 matrix via cofactor expansion.

        Args:
            M: Shape (3, 3).

        Returns:
            Scalar determinant (0-dim tensor).
        """
        return (
            M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
            - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
            + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
        )

    @staticmethod
    def _make_diag3(v: torch.Tensor) -> torch.Tensor:
        """Create a 3x3 diagonal matrix from a length-3 vector (ONNX-safe).

        ONNX opset 14 does not support ``torch.diag`` for 1-D → 2-D creation.
        The equivalent ``torch.eye(3) * v.unsqueeze(1)`` is fully exportable:
        element (i,j) = eye[i,j] · v[i], which is v[i] when i==j and 0 otherwise.

        Args:
            v: Shape (3,).

        Returns:
            Diagonal matrix, shape (3, 3).
        """
        # torch.sign is applied externally; here we only build diag(v).
        return torch.eye(3, dtype=v.dtype, device=v.device) * v.unsqueeze(1)

    def _min_eigvec9(self, M: torch.Tensor) -> torch.Tensor:
        """Minimum eigenvector of a 9x9 symmetric matrix via shifted power iteration.

        Strategy: shifted iteration on ``M_shifted = trace(M)·I − M``.
        The maximum eigenvector of ``M_shifted`` equals the minimum eigenvector
        of ``M`` (signs are preserved by the shift).

        Args:
            M: Symmetric 9x9 matrix.

        Returns:
            Unit-length minimum eigenvector, shape (9,).
        """
        # trace(M) ≥ lambda_max for PSD M; ensures M_shifted is PSD.
        lam = torch.einsum("ii", M)   # ONNX-safe trace
        M_s = lam * torch.eye(9, dtype=M.dtype, device=M.device) - M

        # Initialise with uniform vector then iterate.
        v = M.new_ones(9) / 3.0
        for _ in range(self.n_iter):
            v = M_s @ v
            v = v / (v.norm() + 1e-8)
        return v

    def _project_onto_E_manifold(self, E: torch.Tensor) -> torch.Tensor:
        """Project a 3x3 matrix onto the Essential Matrix manifold.

        The Essential Matrix manifold requires singular values [σ, σ, 0].
        This method uses power iteration on ``Eᵀ E`` to find the right
        singular vectors, then reconstructs E with enforced singular values
        and branch-free determinant sign correction (uses ``torch.sign``).

        Args:
            E: Input 3x3 matrix.

        Returns:
            Projected Essential Matrix, shape (3, 3).
        """
        # ---- Right singular vectors via power iteration on B = EᵀE ----
        B = E.T @ E   # symmetric PSD 3x3; eigenvalues = squared singular values

        # trace(B) ≥ lambda_max; used as shift for min-eigvec iteration.
        lam = torch.einsum("ii", B)

        # v1 = right singular vector for the largest singular value.
        # 3x3 matrices converge quickly; n_iter_manifold (default 10) suffices.
        v1 = B.new_ones(3) / torch.sqrt(B.new_tensor(3.0))   # 1/√3
        for _ in range(self.n_iter_manifold):
            v1 = B @ v1
            v1 = v1 / (v1.norm() + 1e-8)

        # v3 = right singular vector for the smallest singular value (≈ 0 for E).
        B_s = lam * torch.eye(3, dtype=B.dtype, device=B.device) - B
        v3 = B.new_ones(3) / torch.sqrt(B.new_tensor(3.0))   # 1/√3
        for _ in range(self.n_iter_manifold):
            v3 = B_s @ v3
            v3 = v3 / (v3.norm() + 1e-8)

        # v2 completes the orthonormal right basis via cross product.
        v2 = torch.linalg.cross(v3, v1)
        v2 = v2 / (v2.norm() + 1e-8)

        # V = [v1 | v2 | v3]: columns are right singular vectors.
        V = torch.stack([v1, v2, v3], dim=-1)   # (3, 3)

        # Sign correction for V: ensure det(V) = +1 (proper rotation).
        # Branch-free: multiply last column of V by sign(det(V)).
        one = V.new_ones(1).squeeze(0)
        sign_V = torch.sign(self._det3(V))   # ±1 without branching
        V = V @ self._make_diag3(torch.stack([one, one, sign_V]))

        # ---- Left singular vectors from U = E·V / σ ----
        sigma1 = (E @ V[:, 0]).norm()
        sigma2 = (E @ V[:, 1]).norm()
        s_avg = (sigma1 + sigma2) / 2.0   # average first two SVs

        u1 = E @ V[:, 0] / (sigma1 + 1e-8)
        u2 = E @ V[:, 1] / (sigma2 + 1e-8)
        u3 = torch.linalg.cross(u1, u2)   # third left singular vector

        # U = [u1 | u2 | u3]: columns are left singular vectors.
        U = torch.stack([u1, u2, u3], dim=-1)   # (3, 3)

        # Sign correction for U: ensure det(U) = +1.
        sign_U = torch.sign(self._det3(U))
        U = U @ self._make_diag3(torch.stack([one, one, sign_U]))

        # ---- Reconstruct E with singular values [s_avg, s_avg, 0] ----
        z = E.new_zeros(1).squeeze(0)
        S_proj = self._make_diag3(torch.stack([s_avg, s_avg, z]))
        return U @ S_proj @ V.T

    def _hartley_normalization(
        self,
        pts: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the Hartley normalization matrix for a weighted point set.

        Translates the weighted centroid to the origin and scales so that
        the weighted root-mean-squared distance from the centroid is √2.

        Args:
            pts:     Point set, shape (N, 2), in any coordinate system.
            weights: Per-point scalar weights, shape (N,).

        Returns:
            T:        3x3 normalization matrix.
            scale:    Scalar scale factor (0-dim tensor).
            centroid: Weighted centroid, shape (2,).
        """
        # Weighted centroid: c = Σ(w_i · p_i) / Σ w_i
        w_sum = weights.sum() + 1e-8
        centroid = (weights.unsqueeze(-1) * pts).sum(dim=0) / w_sum   # (2,)

        # Center the point set.
        pts_c = pts - centroid   # (N, 2)

        # Weighted RMS distance from the centroid.
        dist_sq = (pts_c ** 2).sum(dim=-1)   # (N,)
        mean_dist = torch.sqrt((weights * dist_sq).sum() / w_sum + 1e-8)

        # Scale so that mean distance becomes √2.
        scale = torch.sqrt(pts.new_tensor(2.0)) / (mean_dist + 1e-8)

        # Build T = [[s, 0, -s·cx], [0, s, -s·cy], [0, 0, 1]]
        # without index-assignment (avoids ScatterND in ONNX).
        z = pts.new_zeros(1).squeeze(0)    # scalar zero
        o = pts.new_ones(1).squeeze(0)     # scalar one
        cx, cy = centroid[0], centroid[1]

        row0 = torch.stack([scale,  z,      -scale * cx])
        row1 = torch.stack([z,      scale,  -scale * cy])
        row2 = torch.stack([z,      z,       o          ])
        T = torch.stack([row0, row1, row2])   # (3, 3)

        return T, scale, centroid

    def _weighted_8point_core(
        self,
        weights: torch.Tensor,
        pts1_n: torch.Tensor,
        pts2_n: torch.Tensor,
    ) -> torch.Tensor:
        """Run the weighted 8-point algorithm and return the Essential Matrix.

        Performs Hartley normalisation, builds the normal equations via
        Kronecker factorisation, solves for the minimum eigenvector,
        Hartley denormalises, and projects onto the E manifold.

        Args:
            weights: Pair weights, shape (N, M).
            pts1_n:  Normalised image coordinates for image 1, shape (N, 2).
            pts2_n:  Normalised image coordinates for image 2, shape (M, 2).

        Returns:
            Essential Matrix, shape (3, 3).
        """
        N = weights.shape[0]
        M = weights.shape[1]

        # ── Hartley normalisation ──────────────────────────────────────
        w1 = weights.sum(dim=1)   # (N,)
        w2 = weights.sum(dim=0)   # (M,)

        T1, s1, c1 = self._hartley_normalization(pts1_n, w1)
        T2, s2, c2 = self._hartley_normalization(pts2_n, w2)

        pts1_hn = (pts1_n - c1) * s1   # (N, 2)
        pts2_hn = (pts2_n - c2) * s2   # (M, 2)

        # ── Weighted normal equations via Kronecker factorisation ──────
        f1 = torch.cat([pts1_hn, pts1_hn.new_ones(N, 1)], dim=-1)   # (N, 3)
        f2 = torch.cat([pts2_hn, pts2_hn.new_ones(M, 1)], dim=-1)   # (M, 3)

        F1_flat = (f1.unsqueeze(-1) * f1.unsqueeze(-2)).reshape(N, 9)  # (N, 9)
        F2_flat = (f2.unsqueeze(-1) * f2.unsqueeze(-2)).reshape(M, 9)  # (M, 9)

        WF2 = weights @ F2_flat       # (N, 9)
        M_flat = F1_flat.T @ WF2      # (9, 9)
        M_mat = M_flat.reshape(3, 3, 3, 3).permute(0, 2, 1, 3).reshape(9, 9)

        # ── Minimum eigenvector → raw E ────────────────────────────────
        e = self._min_eigvec9(M_mat)     # (9,)
        E_raw = e.reshape(3, 3)          # (3, 3)

        # ── Hartley denormalisation ────────────────────────────────────
        E_denorm = T2.T @ E_raw @ T1    # (3, 3)

        # ── Project onto Essential Matrix manifold ─────────────────────
        return self._project_onto_E_manifold(E_denorm)

    @staticmethod
    def _compute_algebraic_residuals(
        E: torch.Tensor,
        pts1_n: torch.Tensor,
        pts2_n: torch.Tensor,
    ) -> torch.Tensor:
        """Compute algebraic residuals x₂ᵀ E x₁ for all (i, j) pairs.

        Args:
            E:      Essential Matrix, shape (3, 3).
            pts1_n: Normalised image coordinates for image 1, shape (N, 2).
            pts2_n: Normalised image coordinates for image 2, shape (M, 2).

        Returns:
            Residual matrix, shape (N, M).  Element (i, j) is x₂[j]ᵀ E x₁[i].
        """
        N = pts1_n.shape[0]
        M = pts2_n.shape[0]
        f1 = torch.cat([pts1_n, pts1_n.new_ones(N, 1)], dim=-1)   # (N, 3)
        f2 = torch.cat([pts2_n, pts2_n.new_ones(M, 1)], dim=-1)   # (M, 3)
        return f1 @ E.T @ f2.T   # (N, M)

    @staticmethod
    def _compute_sampson_errors(
        E: torch.Tensor,
        pts1_n: torch.Tensor,
        pts2_n: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Sampson errors for all (i, j) correspondence pairs.

        The Sampson error is a first-order approximation to the geometric
        (reprojection) error.  For pair (i, j):

            r   = x₂[j]ᵀ E x₁[i]            (algebraic residual)
            l₂  = E  x₁[i]                   (epipolar line in image 2)
            l₁  = Eᵀ x₂[j]                   (epipolar line in image 1)
            err = r² / (l₂[0]² + l₂[1]² + l₁[0]² + l₁[1]²)

        All operations are element-wise or standard matmuls → ONNX-safe.

        Args:
            E:      Essential Matrix, shape (3, 3).
            pts1_n: Normalised image coordinates for image 1, shape (N, 2).
            pts2_n: Normalised image coordinates for image 2, shape (M, 2).

        Returns:
            Sampson error matrix, shape (N, M).  Each entry ≥ 0.
        """
        N = pts1_n.shape[0]
        M = pts2_n.shape[0]
        f1 = torch.cat([pts1_n, pts1_n.new_ones(N, 1)], dim=-1)   # (N, 3)
        f2 = torch.cat([pts2_n, pts2_n.new_ones(M, 1)], dim=-1)   # (M, 3)

        # Epipolar lines: l₂[i] = E x₁[i],  l₁[j] = Eᵀ x₂[j].
        l2 = f1 @ E.T   # (N, 3)
        l1 = f2 @ E     # (M, 3)

        # Algebraic residuals: r[i, j] = x₂[j]ᵀ E x₁[i].
        r = l2 @ f2.T   # (N, M)

        # Sampson denominator: sum of squared normal components of both
        # epipolar lines (first two elements only — the third is the offset).
        l2_sq = (l2[:, :2] ** 2).sum(dim=-1)   # (N,)
        l1_sq = (l1[:, :2] ** 2).sum(dim=-1)   # (M,)
        denom = l2_sq.unsqueeze(1) + l1_sq.unsqueeze(0)   # (N, M)

        return r ** 2 / (denom + 1e-8)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        """Estimate the Essential Matrix from a Sinkhorn probability matrix.

        Args:
            P: Sinkhorn output, shape (N+1, M+1).
               Last row and last column are the dustbin entries.

        Returns:
            E: Essential Matrix, shape (3, 3).
        """
        # ── Step 1: Remove dustbin row and column ──────────────────────
        N = P.shape[0] - 1   # number of feature points in image 1
        M = P.shape[1] - 1   # number of feature points in image 2
        P_core = P[:N, :M]   # (N, M) core probability matrix

        # ── Step 2: Bidirectional top-K mask ──────────────────────────
        k = self.top_k

        # Top-K per row: True iff P_core[i,j] is among the k largest in row i.
        topk_row_vals = torch.topk(
            P_core, k=k, dim=1, largest=True, sorted=True
        ).values                                   # (N, k)
        thresh_row = topk_row_vals[:, k - 1 : k]  # (N, 1): k-th largest per row
        mask_row = P_core >= thresh_row            # (N, M)

        # Top-K per column: True iff P_core[i,j] is among the k largest in col j.
        topk_col_vals = torch.topk(
            P_core, k=k, dim=0, largest=True, sorted=True
        ).values                                   # (k, M)
        thresh_col = topk_col_vals[k - 1 : k, :]  # (1, M): k-th largest per col
        mask_col = P_core >= thresh_col            # (N, M)

        # AND both directional masks with an absolute value threshold.
        # No Python branching on tensor values → fully ONNX-safe.
        mask_thresh = P_core > 0.01                         # (N, M)
        mask = mask_row & mask_col & mask_thresh            # (N, M)

        # Masked probability values serve as pair weights.
        weights = P_core * mask.to(P_core.dtype)            # (N, M)

        # ── Steps 3 & 4: Normalised image coordinates (precomputed in __init__) ─
        # pixel_coords_n[i] = K⁻¹ · [px_i, py_i, 1]ᵀ (x, y components).
        # This avoids two homogeneous-cat ops and two (N/M, 3)@(3,3) GEMMs.
        pixel_coords_n = self.pixel_coords_n.to(P)   # (H*W, 2)
        pts1_n = pixel_coords_n[:N]                  # (N, 2) normalised
        pts2_n = pixel_coords_n[:M]                  # (M, 2) normalised

        # ── Step 5: Weighted 8-point algorithm → initial E ───────────
        E = self._weighted_8point_core(weights, pts1_n, pts2_n)

        # ── Step 6: IRLS refinement (Cauchy-weighted algebraic residuals)
        # Re-estimate E by down-weighting correspondences whose epipolar
        # residual |x₂ᵀ E x₁| is large, using the Cauchy robust kernel.
        # Weights are recomputed from the base (Sinkhorn) weights each
        # iteration: w_irls = w_base / (1 + (r / σ)²).
        for _ in range(self.n_irls):
            residuals = self._compute_algebraic_residuals(E, pts1_n, pts2_n)
            cauchy_w = 1.0 / (1.0 + (residuals / self.irls_sigma) ** 2)
            E = self._weighted_8point_core(weights * cauchy_w, pts1_n, pts2_n)

        # ── Step 7: Sampson error refinement ───────────────────────────
        # Further refine E using the Sampson error — a first-order
        # approximation to the geometric reprojection error — which is
        # more geometrically meaningful than the algebraic residual used
        # in IRLS.  Reweighting uses the Cauchy kernel:
        #   w_sampson = w_base / (1 + err / σ²)
        # where err = (x₂ᵀ E x₁)² / (‖Ex₁‖₁₂² + ‖Eᵀx₂‖₁₂²).
        for _ in range(self.n_sampson):
            sampson_err = self._compute_sampson_errors(E, pts1_n, pts2_n)
            cauchy_w = 1.0 / (1.0 + sampson_err / (self.sampson_sigma ** 2))
            E = self._weighted_8point_core(weights * cauchy_w, pts1_n, pts2_n)

        return E


# ──────────────────────────────────────────────────────────────────────────────
# Standalone demo / ONNX export verification
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import onnxruntime as ort

    # ── Camera intrinsic for a 32x32 synthetic image ─────────────────────
    # fx = fy = 16 px, principal point at centre (16, 16).
    K = torch.tensor(
        [[16.0,  0.0, 16.0],
         [ 0.0, 16.0, 16.0],
         [ 0.0,  0.0,  1.0]],
        dtype=torch.float32,
    )

    model = EssentialMatrixEstimator(
        K=K,
        image_shape=(32, 32),
        top_k=3,
        n_iter=30,
        n_iter_manifold=10,
    )
    model.eval()

    # ── Random Sinkhorn-shaped probability matrix (N=M=1024, dustbin→1025) ─
    torch.manual_seed(42)
    P = torch.rand(1025, 1025)

    # ── PyTorch forward pass ──────────────────────────────────────────────
    with torch.no_grad():
        E_pt = model(P)
    print("PyTorch output E:")
    print(E_pt)
    print(f"  shape: {E_pt.shape}, dtype: {E_pt.dtype}")

    # ── ONNX export ───────────────────────────────────────────────────────
    onnx_path = "essential_matrix_estimator.onnx"
    torch.onnx.export(
        model,
        (P,),
        onnx_path,
        opset_version=14,
        input_names=["P"],
        output_names=["E"],
        dynamic_axes={
            "P": {0: "N_plus_1", 1: "M_plus_1"},
        },
    )
    print(f"\nONNX model exported → {os.path.abspath(onnx_path)}")

    # ── Verify with onnxruntime ───────────────────────────────────────────
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = sess.run(["E"], {"P": P.numpy()})
    E_ort = torch.from_numpy(outputs[0])

    max_diff = (E_ort - E_pt).abs().max().item()
    print(f"\nONNX Runtime output E: {E_ort}")
    print(f"  Max absolute difference (PyTorch vs ORT): {max_diff:.2e}")

    if max_diff < 1e-4:
        print("\n✓ Outputs match within tolerance.")
    else:
        print("\n✗ WARNING: outputs differ by more than 1e-4.")

    # ── Dynamic-shape smoke test ──────────────────────────────────────────
    # Verify the exported graph is not tied to the (1025, 1025) shape.
    P2 = torch.rand(513, 257)   # N=512, M=256
    with torch.no_grad():
        E_pt2 = model(P2)
    E_ort2 = torch.from_numpy(
        sess.run(["E"], {"P": P2.numpy()})[0]
    )
    diff2 = (E_ort2 - E_pt2).abs().max().item()
    print(f"\nDynamic-shape test (513x257): max diff = {diff2:.2e}")
    if diff2 < 1e-4:
        print("✓ Dynamic shape verified.")
    else:
        print("✗ Dynamic shape test failed.")
