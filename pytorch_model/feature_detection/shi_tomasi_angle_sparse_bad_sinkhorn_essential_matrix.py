"""
Shi-Tomasi + Angle + Sparse BAD + Sinkhorn + Essential Matrix Estimator.

This module provides a combined model that connects the feature matching
pipeline (Shi-Tomasi corner detection, angle estimation, sparse BAD descriptors,
and Sinkhorn matching) with Essential Matrix estimation.

Unlike the standalone EssentialMatrixEstimator — which assumes feature points
lie on a regular pixel grid — this combined model uses the **actual detected
keypoint positions** for geometrically accurate Essential Matrix estimation.

Pipeline:
    1. Shi-Tomasi corner detection + angle estimation → score + orientation maps
    2. NMS + top-k → keypoint selection (actual pixel positions, not a grid)
    3. Rotation-aware BAD descriptor computation at keypoints (sparse)
    4. Sinkhorn matching → probability matrix
    5. Essential Matrix estimation via weighted 8-point algorithm
       - Uses actual keypoint coordinates (converted to normalised image coords)
       - Invalid/padded keypoints are suppressed via a validity mask
       - Optionally refined via IRLS with Cauchy-weighted algebraic residuals
       - Further refined via Sampson error-based reweighting

Designed for ONNX export (opset 14+) with batch_size=1.
"""

import torch
from torch import nn

from pytorch_model.feature_detection.shi_tomasi_angle import ShiTomasiWithAngle
from pytorch_model.descriptor.bad import SparseBAD
from pytorch_model.matching.sinkhorn import SinkhornMatcher
from pytorch_model.geometry.essential_matrix_estimator import EssentialMatrixEstimator
from pytorch_model.utils import apply_nms_maxpool, select_topk_keypoints


class ShiTomasiAngleSparseBADSinkhornWithEssentialMatrix(nn.Module):
    """
    Combined feature matching and Essential Matrix estimation model.

    Chains Shi-Tomasi + Angle + Sparse BAD + Sinkhorn with Essential Matrix
    estimation, using the actual detected keypoint positions (in pixel space,
    normalised by the camera intrinsic matrix K) for geometrically accurate
    computation.

    Invalid/padded keypoints (marked as ``(-1, -1)`` by the keypoint selector)
    are excluded from the Essential Matrix computation via a validity mask that
    zeroes the corresponding rows and columns of the Sinkhorn weight matrix.

    Args:
        K: Camera intrinsic matrix, shape (3, 3).
        max_keypoints: Maximum number of keypoints to detect per image.
                       Outputs are padded to this size.
        block_size: Block size for Shi-Tomasi corner detection (must be odd).
                   Default is 5.
        patch_size: Patch size for angle estimation (must be odd). Default is 15.
        sigma: Gaussian sigma for angle estimation. Default is 2.5.
        num_pairs: Number of BAD descriptor comparison pairs (256 or 512).
                   Default is 256.
        binarize: If True, output binarized BAD descriptors. Default is False.
        soft_binarize: If True and binarize=True, use sigmoid for soft
                       binarization. Default is True.
        temperature: Temperature for soft sigmoid binarization. Default is 10.0.
        sinkhorn_iterations: Number of Sinkhorn iterations. Default is 20.
        epsilon: Entropy regularization parameter for Sinkhorn. Default is 1.0.
        unused_score: Score for dustbin entries in Sinkhorn. Default is 1.0.
        distance_type: Distance metric for Sinkhorn cost matrix ('l1' or 'l2').
                       Default is 'l2'.
        nms_radius: Radius for non-maximum suppression. Default is 3.
        score_threshold: Minimum score threshold for keypoint selection.
                        Default is 0.0.
        normalize_descriptors: If True, L2-normalize descriptors before
                              matching. Default is True.
        sampling_mode: Sampling mode for descriptor grid sampling
                       ('nearest' or 'bilinear'). Default is 'nearest'.
        border_margin: Margin from image border (in pixels) to exclude keypoints.
                      If None, uses descriptor's max_radius. Default is None.
        top_k: Number of top-probability entries kept per row and per column
               in the bidirectional filter for Essential Matrix estimation.
               Default is 3.
        n_iter: Power-iteration steps for the 9×9 eigenvector solve inside
                Essential Matrix estimation. Default is 30.
        n_iter_manifold: Power-iteration steps for each 3×3 eigenvector solve
                         inside the Essential Matrix manifold projection.
                         Default is 10.
        n_irls: Number of IRLS refinement iterations. 0 disables. Default is 5.
        irls_kernel: Robust kernel for IRLS reweighting. One of 'cauchy',
                     'tukey', 'geman_mcclure', or 'huber'. Default is 'cauchy'.
        irls_sigma: Scale parameter σ for the IRLS robust kernel.
                    Default is 0.01.
        n_sampson: Number of Sampson error refinement iterations (runs after
                   IRLS). Uses the Sampson error — a first-order approximation
                   to the geometric reprojection error. 0 disables. Default is 3.
        sampson_sigma: Scale parameter σ for the Cauchy kernel used in Sampson
                       refinement. Default is 0.01.

    Example:
        >>> K = torch.eye(3)
        >>> K[0, 0] = K[1, 1] = 525.0
        >>> K[0, 2] = 320.0; K[1, 2] = 240.0
        >>> model = ShiTomasiAngleSparseBADSinkhornWithEssentialMatrix(
        ...     K=K, max_keypoints=512)
        >>> img1 = torch.randn(1, 1, 480, 640)
        >>> img2 = torch.randn(1, 1, 480, 640)
        >>> kpts1, kpts2, probs, E = model(img1, img2)
        >>> print(kpts1.shape)   # [1, 512, 2]
        >>> print(probs.shape)   # [1, 513, 513]
        >>> print(E.shape)       # [3, 3]
    """

    def __init__(
        self,
        K: torch.Tensor,
        max_keypoints: int,
        block_size: int = 5,
        patch_size: int = 15,
        sigma: float = 2.5,
        num_pairs: int = 256,
        binarize: bool = False,
        soft_binarize: bool = True,
        temperature: float = 10.0,
        sinkhorn_iterations: int = 20,
        epsilon: float = 1.0,
        unused_score: float = 1.0,
        distance_type: str = "l2",
        nms_radius: int = 3,
        score_threshold: float = 0.0,
        normalize_descriptors: bool = True,
        sampling_mode: str = "nearest",
        border_margin: int | None = None,
        top_k: int = 3,
        n_iter: int = 30,
        n_iter_manifold: int = 10,
        n_irls: int = 5,
        irls_kernel: str = "cauchy",
        irls_sigma: float = 0.01,
        n_sampson: int = 3,
        sampson_sigma: float = 0.01,
    ) -> None:
        super().__init__()

        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.score_threshold = score_threshold
        self.top_k = top_k

        # Feature detector + orientation estimator
        self.detector = ShiTomasiWithAngle(
            block_size=block_size,
            patch_size=patch_size,
            sigma=sigma,
        )

        # Sparse BAD descriptor with rotation-awareness
        self.descriptor = SparseBAD(
            num_pairs=num_pairs,
            binarize=binarize,
            soft_binarize=soft_binarize,
            temperature=temperature,
            normalize_descriptors=normalize_descriptors,
            sampling_mode=sampling_mode,
        )

        # Border margin: default to descriptor max_radius for safety
        if border_margin is None:
            self.border_margin = self.descriptor.max_radius
        else:
            self.border_margin = border_margin

        # Sinkhorn matcher
        self.matcher = SinkhornMatcher(
            iterations=sinkhorn_iterations,
            epsilon=epsilon,
            unused_score=unused_score,
            distance_type=distance_type,
        )

        # EssentialMatrixEstimator is used as a helper for:
        #   - _hartley_normalization()
        #   - _min_eigvec9()
        #   - _project_onto_E_manifold()
        # image_shape=(1, 1) is a minimal placeholder; the precomputed pixel
        # grid in EssentialMatrixEstimator is NOT used — actual keypoint
        # coordinates are computed in forward() and passed directly.
        self.estimator = EssentialMatrixEstimator(
            K=K,
            image_shape=(1, 1),
            top_k=top_k,
            n_iter=n_iter,
            n_iter_manifold=n_iter_manifold,
            n_irls=n_irls,
            irls_kernel=irls_kernel,
            irls_sigma=irls_sigma,
            n_sampson=n_sampson,
            sampson_sigma=sampson_sigma,
        )

        # K_inv buffer for normalising pixel keypoint coordinates
        K_f = K.float()
        K_inv = torch.linalg.inv(K_f)
        self.register_buffer("K_inv", K_inv)

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _estimate_essential_matrix(
        self,
        P: torch.Tensor,
        pts1_n: torch.Tensor,
        pts2_n: torch.Tensor,
        valid1: torch.Tensor,
        valid2: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the Essential Matrix from a Sinkhorn matrix and actual keypoints.

        Implements the weighted 8-point algorithm with Hartley normalisation,
        using actual normalised keypoint coordinates instead of a pixel grid.

        Args:
            P: Sinkhorn probability matrix, shape (K+1, K+1).
               Last row/column are dustbin entries.
            pts1_n: Normalised keypoints in image 1, shape (K, 2) as (x, y).
            pts2_n: Normalised keypoints in image 2, shape (K, 2) as (x, y).
            valid1: Validity mask for keypoints1, shape (K,). True = valid.
            valid2: Validity mask for keypoints2, shape (K,). True = valid.

        Returns:
            E: Essential Matrix, shape (3, 3).
        """
        N = P.shape[0] - 1   # number of keypoints in image 1
        M = P.shape[1] - 1   # number of keypoints in image 2
        P_core = P[:N, :M]   # (N, M) core probability matrix

        # ── Validity masking ────────────────────────────────────────────
        # Zero out rows/columns for invalid (padded) keypoints so that
        # their coordinates — stored as (-1, -1) — have zero weight and
        # therefore do not corrupt the Essential Matrix estimate.
        v1 = valid1.to(P_core)          # (N,) float, 1.0 = valid
        v2 = valid2.to(P_core)          # (M,) float, 1.0 = valid
        P_core = P_core * v1.unsqueeze(1) * v2.unsqueeze(0)   # (N, M)

        # ── Bidirectional top-K mask ────────────────────────────────────
        k = self.top_k

        topk_row_vals = torch.topk(
            P_core, k=k, dim=1, largest=True, sorted=True
        ).values                                    # (N, k)
        thresh_row = topk_row_vals[:, k - 1 : k]   # (N, 1)
        mask_row = P_core >= thresh_row             # (N, M)

        topk_col_vals = torch.topk(
            P_core, k=k, dim=0, largest=True, sorted=True
        ).values                                    # (k, M)
        thresh_col = topk_col_vals[k - 1 : k, :]   # (1, M)
        mask_col = P_core >= thresh_col             # (N, M)

        mask_thresh = P_core > 0.01
        mask = mask_row & mask_col & mask_thresh
        weights = P_core * mask.to(P_core.dtype)    # (N, M)

        # ── Weighted 8-point algorithm → initial E ─────────────────────
        E = self.estimator._weighted_8point_core(weights, pts1_n, pts2_n)

        # ── IRLS refinement ──────────────────────────────────────────────
        for _ in range(self.estimator.n_irls):
            residuals = EssentialMatrixEstimator._compute_algebraic_residuals(
                E, pts1_n, pts2_n,
            )
            irls_w = self.estimator._compute_irls_weights(residuals)
            E = self.estimator._weighted_8point_core(
                weights * irls_w, pts1_n, pts2_n,
            )

        # ── Sampson error refinement ─────────────────────────────────────
        for _ in range(self.estimator.n_sampson):
            sampson_err = EssentialMatrixEstimator._compute_sampson_errors(
                E, pts1_n, pts2_n,
            )
            cauchy_w = 1.0 / (
                1.0 + sampson_err / (self.estimator.sampson_sigma ** 2)
            )
            E = self.estimator._weighted_8point_core(
                weights * cauchy_w, pts1_n, pts2_n,
            )

        return E

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect keypoints, compute matches, and estimate the Essential Matrix.

        Args:
            image1: First grayscale image, shape (1, 1, H, W).
                    Batch size must be 1 for Essential Matrix estimation.
            image2: Second grayscale image, shape (1, 1, H, W).

        Returns:
            Tuple of:
                - keypoints1: Detected keypoints in image 1, shape (1, K, 2)
                  in (y, x) pixel format. Invalid/padded entries are (-1, -1).
                - keypoints2: Detected keypoints in image 2, shape (1, K, 2)
                  in (y, x) pixel format. Invalid/padded entries are (-1, -1).
                - matching_probs: Sinkhorn probability matrix, shape (1, K+1, K+1).
                  Last row/column are dustbin entries.
                - E: Estimated Essential Matrix, shape (3, 3).
        """
        # ── Step 1: Detect features and compute orientations ───────────
        scores1, angles1 = self.detector(image1)   # (1, 1, H, W) each
        scores2, angles2 = self.detector(image2)
        scores1 = scores1.squeeze(1)   # (1, H, W)
        scores2 = scores2.squeeze(1)

        # ── Step 2: Non-maximum suppression ────────────────────────────
        nms_mask1 = apply_nms_maxpool(scores1, self.nms_radius)
        nms_mask2 = apply_nms_maxpool(scores2, self.nms_radius)

        # ── Step 3: Select top-k keypoints ─────────────────────────────
        # kpt_scores > 0 ↔ valid keypoint (not padded)
        keypoints1, kpt_scores1 = select_topk_keypoints(
            scores1, nms_mask1, self.max_keypoints,
            self.score_threshold, self.border_margin,
        )
        keypoints2, kpt_scores2 = select_topk_keypoints(
            scores2, nms_mask2, self.max_keypoints,
            self.score_threshold, self.border_margin,
        )

        # ── Step 4: Rotation-aware BAD descriptors ─────────────────────
        desc1 = self.descriptor(image1, keypoints1, angles1)
        desc2 = self.descriptor(image2, keypoints2, angles2)

        # ── Step 5: Sinkhorn matching ───────────────────────────────────
        matching_probs = self.matcher(desc1, desc2)   # (1, K+1, K+1)

        # ── Step 6: Essential Matrix estimation ─────────────────────────
        # Squeeze batch dimension (batch_size=1 required here).
        kpts1 = keypoints1[0].float()    # (K, 2) in (y, x)
        kpts2 = keypoints2[0].float()    # (K, 2) in (y, x)
        P     = matching_probs[0]        # (K+1, K+1)

        # Validity masks: True for valid keypoints (score > 0).
        valid1 = kpt_scores1[0] > 0      # (K,)
        valid2 = kpt_scores2[0] > 0      # (K,)

        # Convert keypoint coordinates from (y, x) pixel format to
        # normalised image coordinates via K⁻¹.
        #
        # EssentialMatrixEstimator uses the (x, y) convention, so we
        # swap the axis order: kpts[:, 0] = y → second slot;
        #                      kpts[:, 1] = x → first slot.
        #
        # For an invalid keypoint at (-1, -1), the normalised coordinate
        # is K⁻¹·[-1, -1, 1]ᵀ — a garbage value — but its weight is
        # zeroed by valid1/valid2 in _estimate_essential_matrix, so it
        # does not affect the result.
        pts1_xy = torch.stack([kpts1[:, 1], kpts1[:, 0]], dim=-1)   # (K, 2) x,y
        pts2_xy = torch.stack([kpts2[:, 1], kpts2[:, 0]], dim=-1)   # (K, 2) x,y

        K_inv  = self.K_inv.to(pts1_xy)
        ones_k = pts1_xy.new_ones(self.max_keypoints, 1)

        # [x, y, 1] @ K_inv.T → [(x−cx)/fx, (y−cy)/fy, 1], keep first 2
        pts1_n = (torch.cat([pts1_xy, ones_k], dim=-1) @ K_inv.T)[:, :2]   # (K,2)
        pts2_n = (torch.cat([pts2_xy, ones_k], dim=-1) @ K_inv.T)[:, :2]   # (K,2)

        E = self._estimate_essential_matrix(P, pts1_n, pts2_n, valid1, valid2)

        return keypoints1, keypoints2, matching_probs, E
