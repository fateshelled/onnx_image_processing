"""Sliding-window pose-graph optimizer for visual odometry.

v1 scope:
- Nodes: camera-to-world SE(3) poses (4x4). The oldest live node is the
  anchor (fixed); the window slides by dropping nodes beyond window_size.
- Edges: relative SE(3) measurements M_ij with p_j = R_ij p_i + t_ij,
  satisfying M_ij = T_j^{-1} T_i when consistent. This matches the
  recoverPose [R|t] point-transform convention used by the VO pipeline
  (see PR #33).
- Residual per edge: e = Log(M_ij^{-1} T_j^{-1} T_i)  (rotation part first).
- Solver: Gauss-Newton with LM damping; numerical Jacobian via central
  differences on 6-dof right-increments T' = T Exp(delta). Windows are
  small (<= ~12 nodes) so dense solving is adequate and numpy-only.
- Robustness: per-edge Huber reweighting.
- Marginalization: v1 drops removed edges without propagating information
  (documented approximation; full Schur/multi-node prior is future work).

Optional per-edge scale (monocular scale drift):
- recoverPose translations are unit-norm, so the translation magnitude of a
  relative-pose measurement is unknown. With ``optimize_scale=True`` an edge
  can be declared ``scale_free``: its translation becomes s_ij * t_ij with
  s_ij an optimized variable (stored as a log-scale). The first free edge is
  the gauge and is held at its initial scale (relative scales are observed
  through loops); a soft prior ``scale_prior_sigma`` on log-scale keeps
  unobserved edges near 1.

Large graphs switch to a sparse (scipy) normal-equation solve so the memory
scales with the number of nonzeros instead of the square of the variables;
the dense LM path is kept for small windows. Runs on CPU. Not ONNX-related.
"""

import numpy as np
import scipy.linalg as sla
from scipy import sparse as sp
from scipy.sparse.linalg import splu

from .se3 import (
    se3_ad,
    se3_exp,
    se3_log,
    se3_left_jacobian_inv,
    se3_right_jacobian_inv,
)


def _edge_residual(T_i, T_j, M):
    """Residual e = Log(M^{-1} T_j^{-1} T_i); zero when M == T_j^{-1} T_i."""
    P = np.linalg.inv(T_j) @ T_i
    Q = np.linalg.inv(M) @ P
    return se3_log(Q)


class SlidingWindowOptimizer:
    """Pose-only sliding window over SE(3) nodes.

    Node ids must be strictly increasing (typically step/frame counters).
    """

    def __init__(
        self,
        window_size: int | None = 10,
        max_iterations: int = 30,
        lambda_init: float = 1e-3,
        step_scale_t: float = 0.05,
        step_scale_r: float = 0.05,
        huber: float = 1.0,
        optimize_scale: bool = False,
        scale_prior_sigma: float = 0.0,
        tsvd_ratio: float = 0.0,
        dense_max_cols: int = 1500,
        loop_robust: str = "none",
        loop_robust_phi: float = 1.0,
        loop_robust_min_weight: float = 0.0,
    ) -> None:
        if window_size is not None and window_size < 2:
            raise ValueError(f"window_size must be >= 2 or None, got {window_size}")
        self.window_size = window_size
        self.max_iterations = max_iterations
        self.lambda_init = lambda_init
        self.step_scale_t = step_scale_t
        self.step_scale_r = step_scale_r
        self.huber = huber
        self.optimize_scale = optimize_scale
        self.scale_prior_sigma = scale_prior_sigma
        # TSVD (Truncated SVD): drop optimization directions whose Hessian
        # eigenvalue is below tsvd_ratio * max_eigenvalue. 0 disables.
        self.tsvd_ratio = float(tsvd_ratio)
        # Normal equations are dense below this many variables and sparse
        # (scipy) above it, keeping the exact legacy solver for small windows.
        self.dense_max_cols = int(dense_max_cols)
        if loop_robust not in ("none", "dcs"):
            raise ValueError(f"unsupported loop_robust mode: {loop_robust}")
        self.loop_robust = loop_robust
        self.loop_robust_phi = float(loop_robust_phi)
        self.loop_robust_min_weight = float(loop_robust_min_weight)
        if self.loop_robust_phi <= 0.0:
            raise ValueError("loop_robust_phi must be positive")
        if not 0.0 <= self.loop_robust_min_weight <= 1.0:
            raise ValueError("loop_robust_min_weight must be in [0, 1]")
        # Above this many variables the TSVD eigendecomposition is skipped
        # (a dense eigendecomposition would need ~n^2 memory and OOM).
        self.tsvd_dense_max_cols = 12000
        self.tsvd_kept = 0
        self.tsvd_total = 0
        self.pose_ids: list[int] = []
        # Node ids held fixed (no variables); they still participate in
        # residuals and pin the gauge for fixed-lag optimization.
        self.fixed_ids: set[int] = set()
        self.T: dict[int, np.ndarray] = {}
        self.edges: list[tuple] = []  # (i, j, M, sigma_t, sigma_r)
        # Optional dense 6x6 whitening matrix W (cost = ||W e||^2, i.e.
        # information Omega = W^T W). None -> diagonal 1/sigma per axis.
        # Used for marginalization (relative) priors.
        self.edge_whiten: list[np.ndarray | None] = []
        # Only accepted loop-closure edges opt into the additional robust
        # kernel. Odometry, spokes and marginalized priors keep legacy weights.
        self.edge_robust: list[bool] = []
        self.last_robust_weights: list[float] = []
        self.n_robust_downweighted: int = 0
        # Dense marginalization priors: each entry is (ids, L, mu, x0) with
        # cost ||L (delta - mu)||^2 over right-increments delta_i =
        # Log(x0_i^-1 T_i). Used to project an eliminated subgraph onto its
        # (multi-node) Markov blanket.
        self.prior_factors: list[tuple] = []
        # Parallel bookkeeping for per-edge scale.
        self.edge_scale: list[float] = []
        self.edge_scale_sigma: list[float] = []  # per-edge scale-prior width
        self.scale_col: list[int | None] = []  # None = fixed / gauge
        self.scale_prior_mean: list[float] = []  # prior mean of log(scale)
        self.n_scale: int = 0
        self.gauge_edge: int | None = None
        self.last_cost = float("inf")
        self.last_steps: int = 0

    # -- graph ---------------------------------------------------------------

    def add_node(self, node_id: int, T: np.ndarray, fixed: bool = False) -> None:
        if node_id in self.T:
            raise ValueError(f"duplicate node id {node_id}")
        if self.pose_ids and node_id <= self.pose_ids[-1]:
            raise ValueError("node ids must be strictly increasing")
        self.T[node_id] = np.asarray(T, float).copy()
        self.pose_ids.append(node_id)
        if fixed:
            self.fixed_ids.add(node_id)
        self._drop_oldest()

    def add_edge(
        self,
        i: int,
        j: int,
        M: np.ndarray,
        sigma_t: float | None = None,
        sigma_r: float | None = None,
        scale_free: bool = False,
        scale: float = 1.0,
        scale_prior_mean: float = 0.0,
        omega: np.ndarray | None = None,
        scale_prior_sigma: float | None = None,
        robust: bool = False,
    ) -> None:
        if i not in self.T or j not in self.T:
            raise KeyError("edge endpoints must be live nodes in the window")
        if omega is not None:
            # W with W^T W = Omega (cost e^T Omega e). Eigen-based so that a
            # rank-deficient / round-off-degraded Omega still yields a valid
            # whitener (negative eigenvalues clipped to zero).
            omega = np.asarray(omega, float)
            omega = 0.5 * (omega + omega.T)
            w_e, V = np.linalg.eigh(omega)
            self.edge_whiten.append(
                np.sqrt(np.clip(w_e, 0.0, None))[:, None] * V.T)
        else:
            self.edge_whiten.append(None)
        self.edges.append((
            i, j, np.asarray(M, float).copy(),
            self.step_scale_t if sigma_t is None else sigma_t,
            self.step_scale_r if sigma_r is None else sigma_r,
        ))
        self.edge_robust.append(bool(robust))
        s = float(scale)
        if s <= 0.0:
            s = 1.0
        self.edge_scale.append(s)
        self.scale_prior_mean.append(float(scale_prior_mean))
        # Per-edge scale-prior width (None -> the window default).
        self.edge_scale_sigma.append(
            self.scale_prior_sigma if scale_prior_sigma is None
            else float(scale_prior_sigma))
        col = None
        if self.optimize_scale and scale_free:
            if self.gauge_edge is None:
                # First free edge fixes the (unobservable) global scale.
                self.gauge_edge = len(self.edges) - 1
            else:
                col = self.n_scale
                self.n_scale += 1
        self.scale_col.append(col)

    def add_prior_factor(self, ids, H, b, x0) -> None:
        """Add a dense Gaussian prior over several nodes.

        ``cost = 0.5 * (delta - mu)^T Lambda (delta - mu)`` with
        ``Lambda = H`` and ``mu = -H^+ b``. ``x0`` is the linearization point
        (poses aligned with ``ids``). Stored whitened for row assembly.
        """
        ids = tuple(int(i) for i in ids)
        H = np.asarray(H, float)
        H = 0.5 * (H + H.T)
        b = np.asarray(b, float).ravel()
        mu = -np.linalg.pinv(H) @ b
        w_e, V = np.linalg.eigh(H)
        L = np.sqrt(np.clip(w_e, 0.0, None))[:, None] * V.T
        x0 = [np.asarray(p, float).copy() for p in x0]
        self.prior_factors.append((ids, L, mu, x0))

    def pop_last_prior_factor(self) -> bool:
        """Remove the most recently added prior factor (e.g. NL-Reg).

        Used so a temporary regularizer is not baked into a marginalization
        prior. Returns True when a factor was removed.
        """
        if not self.prior_factors:
            return False
        self.prior_factors.pop()
        return True

    def add_nl_regularization(self, strength=1.0, tau=1.0, length=1.0):
        """Direction-dependent Tikhonov (NL-Reg) anchored to the current poses.

        Builds the unit-balanced node Hessian ``H_bal = S^-1 H S^-1``
        (rotation dofs scaled by ``length`` so translation/rotation are
        comparable), weights its spectrum with the Wiener filter
        ``g(mu) = strength * tau / (mu + tau)`` (continuous, no threshold),
        and adds a dense prior factor anchoring every node to its current
        pose with information ``Lambda = S Lambda_bal S``. Weak directions are
        pulled back to the prior; well-conditioned directions are untouched.
        Scales are not regularized (node poses only).
        """
        free = [nd for nd in self.pose_ids if nd not in self.fixed_ids]
        if not free:
            return
        H, _g, _ncol = self._full_normal()
        nnode = 6 * len(free)
        Hn = H[:nnode, :nnode]
        Hn = Hn.toarray() if sp.issparse(Hn) else np.asarray(Hn)
        Hn = 0.5 * (Hn + Hn.T)
        s = np.ones(nnode)
        s[0::6] = s[1::6] = s[2::6] = float(length)  # rotation dofs
        dinv = 1.0 / s
        Hb = Hn * dinv[:, None] * dinv[None, :]
        w_e, V = sla.eigh(Hb)
        w_e = np.clip(w_e, 0.0, None)
        gw = float(strength) * float(tau) / (w_e + float(tau))
        Lam_bal = (V * gw) @ V.T
        Lam = Lam_bal * s[:, None] * s[None, :]
        Lam = 0.5 * (Lam + Lam.T)
        self.add_prior_factor(free, Lam, np.zeros(nnode),
                              [self.T[nd] for nd in free])

    def get_scale(self, edge_index: int) -> float:
        return self.edge_scale[edge_index]

    def get_pose(self, node_id: int) -> np.ndarray:
        if node_id not in self.T:
            raise KeyError(node_id)
        return self.T[node_id].copy()

    def _drop_oldest(self) -> None:
        if self.window_size is None:
            return
        while len(self.pose_ids) > self.window_size:
            drop = self.pose_ids.pop(0)
            self.T.pop(drop, None)
            keep = [k for k, e in enumerate(self.edges)
                    if e[0] != drop and e[1] != drop]
            self.edges = [self.edges[k] for k in keep]
            self.edge_whiten = [self.edge_whiten[k] for k in keep]
            self.edge_robust = [self.edge_robust[k] for k in keep]
            self.edge_scale = [self.edge_scale[k] for k in keep]
            self.edge_scale_sigma = [self.edge_scale_sigma[k] for k in keep]
            self.scale_prior_mean = [self.scale_prior_mean[k] for k in keep]
            cols = [self.scale_col[k] for k in keep]
            self.scale_col = []
            self.n_scale = 0
            for c in cols:
                if c is None:
                    self.scale_col.append(None)
                else:
                    self.scale_col.append(self.n_scale)
                    self.n_scale += 1

    # -- internals ------------------------------------------------------------

    def _scaled_M(self, k: int) -> np.ndarray:
        M = self.edges[k][2]
        s = self.edge_scale[k]
        if s == 1.0:
            return M
        Ms = M.copy()
        Ms[:3, 3] = Ms[:3, 3] * s
        return Ms

    def _n_prior(self) -> int:
        # Only edges whose per-edge scale-prior width is positive contribute a
        # row (a 0 width means "no scale prior for this edge").
        if not self.optimize_scale:
            return 0
        return sum(1 for c, s in zip(self.scale_col, self.edge_scale_sigma)
                   if c is not None and s > 0.0)

    def _residuals(self) -> np.ndarray:
        """Scaled residual vector; edge rows then scale-prior rows."""
        if not self.edges:
            return np.zeros(0)
        out = [
            self._apply_weight(
                k, _edge_residual(self.T[i], self.T[j], self._scaled_M(k)))
            for k, (i, j, M, st, sr) in enumerate(self.edges)
        ]
        r = np.concatenate(out)
        if self._n_prior():
            pri = [
                np.array([(np.log(max(self.edge_scale[e], 1e-12))
                           - self.scale_prior_mean[e])
                          / self.edge_scale_sigma[e]])
                for e, c in enumerate(self.scale_col)
                if c is not None and self.edge_scale_sigma[e] > 0.0
            ]
            r = np.concatenate([r] + pri)
        for fac in self.prior_factors:
            r = np.concatenate([r, self._prior_factor_residual(fac)])
        return r

    def _prior_factor_delta(self, fac) -> np.ndarray:
        ids, _L, _mu, x0 = fac
        return np.concatenate([
            se3_log(np.linalg.inv(x0[k]) @ self.T[nd])
            for k, nd in enumerate(ids)])

    def _prior_factor_residual(self, fac) -> np.ndarray:
        ids, L, mu, _x0 = fac
        return L @ (self._prior_factor_delta(fac) - mu)

    def _edge_scale(self, st, sr):
        return np.concatenate([np.full(3, sr), np.full(3, st)])

    def _apply_weight(self, k, v):
        """Whiten a 6-vector: dense W @ v, or per-axis v / sigma."""
        W = self.edge_whiten[k]
        if W is not None:
            return W @ v
        _, _, _, st, sr = self.edges[k]
        return v / self._edge_scale(st, sr)

    def _apply_weight_jac(self, k, block):
        """Whiten a 6x6 Jacobian block (rows = residual axes)."""
        W = self.edge_whiten[k]
        if W is not None:
            return W @ block
        _, _, _, st, sr = self.edges[k]
        return block / self._edge_scale(st, sr)[:, None]

    def _huber_weights(self) -> np.ndarray:
        if not self.edges:
            return np.zeros(0)
        w = np.ones(6 * len(self.edges))
        robust_weights = []
        for k, (i, j, M, st, sr) in enumerate(self.edges):
            e = _edge_residual(self.T[i], self.T[j], self._scaled_M(k))
            n = float(np.linalg.norm(self._apply_weight(k, e)))
            if n > self.huber:
                w[6 * k: 6 * k + 6] = self.huber / max(n, 1e-12)
            rw = self._loop_robust_weight(k, n)
            w[6 * k: 6 * k + 6] *= rw
            if self.edge_robust[k]:
                robust_weights.append(rw)
        self.last_robust_weights = robust_weights
        self.n_robust_downweighted = sum(x < 0.5 for x in robust_weights)
        if self._n_prior():
            w = np.concatenate([w, np.ones(self._n_prior())])
        for fac in self.prior_factors:
            w = np.concatenate([w, np.ones(6 * len(fac[0]))])
        return w

    def _loop_robust_weight(self, k: int, norm: float) -> float:
        """DCS information weight for an opted-in loop edge.

        DCS leaves small residuals unchanged and smoothly reduces the
        information of inconsistent constraints. The returned scalar is
        multiplied with the existing Huber information weight.
        """
        if not self.edge_robust[k] or self.loop_robust == "none":
            return 1.0
        n2 = float(norm) * float(norm)
        phi = self.loop_robust_phi
        weight = min(1.0, 2.0 * phi / (phi + n2))
        return max(self.loop_robust_min_weight, weight)

    def _cost(self) -> float:
        r = self._residuals()
        if r.size == 0:
            return 0.0
        w = self._huber_weights()
        return float(0.5 * np.sum((np.sqrt(w) * r) ** 2))

    def _scale_jacobian(self, k: int) -> np.ndarray:
        """d(scaled residual_k)/d(log scale_k), a 6-vector.

        With A = M_s^{-1} T_j^{-1} T_i and dM_s/du = [0 | M_s t],
        d e / du = J_r(e) * ( A^{-1} dA/du )^vee, and
        A^{-1} dA/du = -P^{-1} (dM_s/du) A  (P = T_j^{-1} T_i).
        """
        i, j, M, st, sr = self.edges[k]
        Ms = self._scaled_M(k)
        sc = self._edge_scale(st, sr)
        e = _edge_residual(self.T[i], self.T[j], Ms)
        P = np.linalg.inv(self.T[j]) @ self.T[i]
        A = np.linalg.inv(Ms) @ P
        dM = np.zeros((4, 4))
        dM[:3, 3] = Ms[:3, 3]
        G = -np.linalg.inv(P) @ dM @ A
        w = 0.5 * np.array([G[2, 1] - G[1, 2], G[0, 2] - G[2, 0], G[1, 0] - G[0, 1]])
        v = G[:3, 3]
        twist = np.concatenate([w, v])
        return (se3_right_jacobian_inv(e) @ twist) / sc

    def _jacobian_coo(self, eps=1e-6):
        """Sparse Jacobian as COO triplets ``(rows, cols, vals, shape)``.

        Each edge contributes 6 residual rows that touch only its two
        endpoint node blocks (12 columns) plus its own log-scale column,
        so the matrix is extremely sparse for large pose graphs.

        Node columns are analytic (verified against finite differences);
        edges whose residual rotation norm exceeds the Bernoulli series
        comfort range (||omega|| > 1.8) fall back to central differences on
        that row only. Scale columns use :meth:`_scale_jacobian`.
        """
        free = [nd for nd in self.pose_ids if nd not in self.fixed_ids]
        n = len(free)
        ncol = 6 * n + self.n_scale
        col = {node: k for k, node in enumerate(free)}
        nrow = self._residuals().size
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for k, (i, j, M, st, sr) in enumerate(self.edges):
            r0 = 6 * k
            sc = self._edge_scale(st, sr)
            Ms = self._scaled_M(k)
            e = _edge_residual(self.T[i], self.T[j], Ms)
            if float(np.linalg.norm(e[:3])) > 1.8:
                self._jacobian_row_numeric(k, col, rows, cols, vals, eps)
            else:
                Ji = se3_right_jacobian_inv(e)
                Jj = -se3_left_jacobian_inv(e) @ se3_ad(np.linalg.inv(Ms))
                Bi = self._apply_weight_jac(k, Ji)
                Bj = self._apply_weight_jac(k, Jj)
                base_i = 6 * col[i] if i in col else None
                base_j = 6 * col[j] if j in col else None
                for a in range(6):
                    for d in range(6):
                        v = Bi[a, d]
                        if v != 0.0 and base_i is not None:
                            rows.append(r0 + a)
                            cols.append(base_i + d)
                            vals.append(v)
                        v = Bj[a, d]
                        if v != 0.0 and base_j is not None:
                            rows.append(r0 + a)
                            cols.append(base_j + d)
                            vals.append(v)
            c = self.scale_col[k]
            if c is not None:
                sj = self._scale_jacobian(k)
                for a in range(6):
                    if sj[a] != 0.0:
                        rows.append(r0 + a)
                        cols.append(6 * n + c)
                        vals.append(sj[a])
        if self._n_prior():
            base = 6 * len(self.edges)
            for e, c in enumerate(self.scale_col):
                if c is not None and self.edge_scale_sigma[e] > 0.0:
                    rows.append(base)
                    cols.append(6 * n + c)
                    vals.append(1.0 / self.edge_scale_sigma[e])
                    base += 1
        base = 6 * len(self.edges) + self._n_prior()
        for (ids, L, mu, _x0) in self.prior_factors:
            m6 = 6 * len(ids)
            for kk, nd in enumerate(ids):
                if nd not in col:
                    continue  # fixed in the factor: delta is pinned
                cb = 6 * col[nd]
                blk = L[:, 6 * kk:6 * kk + 6]
                for a in range(m6):
                    for d in range(6):
                        v = blk[a, d]
                        if v != 0.0:
                            rows.append(base + a)
                            cols.append(cb + d)
                            vals.append(v)
            base += m6
        return (np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
                np.asarray(vals, dtype=float),
                (nrow, ncol))

    def _jacobian(self, eps=1e-6):
        """Dense Jacobian (kept for tests and small problems)."""
        rows, cols, vals, shape = self._jacobian_coo(eps)
        J = np.zeros(shape)
        if rows.size:
            J[rows, cols] = vals
        return J

    def _jacobian_row_numeric(self, k, col, rows, cols, vals, eps=1e-6):
        """Central differences for one edge row, appended to COO lists."""
        i, j, M, st, sr = self.edges[k]
        for node in (i, j):
            if node not in col:
                continue  # fixed endpoint: no columns
            base = 6 * col[node]
            T0 = self.T[node].copy()
            for d in range(6):
                delta = np.zeros(6)
                delta[d] = eps
                self.T[node] = T0 @ se3_exp(delta)
                rp = _edge_residual(self.T[i], self.T[j], self._scaled_M(k))
                self.T[node] = T0 @ se3_exp(-delta)
                rm = _edge_residual(self.T[i], self.T[j], self._scaled_M(k))
                self.T[node] = T0
                g = self._apply_weight(k, (rp - rm) / (2.0 * eps))
                for a in range(6):
                    if g[a] != 0.0:
                        rows.append(6 * k + a)
                        cols.append(base + d)
                        vals.append(g[a])

    def _sparse_normal(self, r, w):
        """Sparse reduced normal equations with node 0 (gauge) eliminated.

        Returns ``(H, g)`` over variables 6..ncol-1. No dense matrix of the
        full graph is ever materialised.
        """
        rows, cols, vals, shape = self._jacobian_coo()
        J = sp.csr_matrix((vals, (rows, cols)), shape=shape)
        Jw = J.multiply(w[:, None]).tocsr()
        H = (J.T @ Jw).tocsr()
        g = np.asarray(J.T @ (w * r)).ravel()
        return H[6:, 6:].tocsr(), g[6:]

    def _solve_sparse(self, H, rhs, lam):
        """Solve ``(H + lam I) dx = rhs`` with sparse LU (reduced space)."""
        n = H.shape[0]
        A = (H + sp.eye(n, format="csr") * lam).tocsc()
        return splu(A).solve(rhs)

    def _full_normal(self):
        """Gauss-Newton normal equations over all variables (sparse)."""
        r = self._residuals()
        w = self._huber_weights()
        rows, cols, vals, shape = self._jacobian_coo()
        J = sp.csr_matrix((vals, (rows, cols)), shape=shape)
        Jw = J.multiply(w[:, None]).tocsr()
        return (J.T @ Jw).tocsr(), np.asarray(J.T @ (w * r)).ravel(), shape[1]

    def marginalize_relative(self, eliminate, keep, fix_scales=False):
        """Schur-marginalize ``eliminate`` and return a relative prior.

        ``keep`` must be exactly two nodes (the Markov blanket). Returns
        ``(a, b, G, Omega)`` so that adding ``add_edge(a, b, G, omega=Omega)``
        reproduces the marginal cost of the eliminated subgraph, up to the
        linearization point used here. The relative-mean transform G and the
        6x6 information Omega are both exact for the linearized system.

        With ``fix_scales=True`` every log-scale is held at its current value
        (no scale variable is optimized or eliminated), which keeps the prior
        well conditioned when the segment's scales were already determined by
        the surrounding graph.
        """
        if len(keep) != 2:
            raise ValueError("marginalize_relative expects exactly 2 kept nodes")
        free = [nd for nd in self.pose_ids if nd not in self.fixed_ids]
        col = {nd: k for k, nd in enumerate(free)}
        elim = set(eliminate)
        if elim & self.fixed_ids or any(nd in self.fixed_ids for nd in keep):
            raise ValueError("cannot marginalize fixed nodes")
        elim_idx: list[int] = []
        for nd in eliminate:
            b = 6 * col[nd]
            elim_idx.extend(range(b, b + 6))
        if not fix_scales:
            # Scale variables whose edge touches an eliminated node are
            # eliminated with it; retained free-scale edges are unsupported.
            for k, (i, j, *_rest) in enumerate(self.edges):
                c = self.scale_col[k]
                if c is None:
                    continue
                if i in elim or j in elim:
                    elim_idx.append(6 * len(free) + c)
                else:
                    raise NotImplementedError(
                        "retained free-scale edge in marginalize_relative")
        keep_idx = [6 * col[nd] + d for nd in keep for d in range(6)]
        H, g, _ = self._full_normal()
        if fix_scales:
            nnode = 6 * len(free)
            H = H[:nnode, :nnode]
            g = g[:nnode]
        idx = elim_idx + keep_idx
        Hs = H[idx][:, idx].toarray()
        gs = g[idx]
        ne = len(elim_idx)
        Hee_inv = np.linalg.pinv(Hs[:ne, :ne])
        H_r = Hs[ne:, ne:] - Hs[ne:, :ne] @ Hee_inv @ Hs[:ne, ne:]
        H_r = 0.5 * (H_r + H_r.T)  # enforce symmetry against round-off
        g_r = gs[ne:] - Hs[ne:, :ne] @ Hee_inv @ gs[:ne]
        d_r = -np.linalg.pinv(H_r) @ g_r
        meanT = {nd: self.T[nd] @ se3_exp(d_r[6 * m:6 * m + 6])
                 for m, nd in enumerate(keep)}
        a, b = keep
        G = np.linalg.inv(meanT[b]) @ meanT[a]
        # Jacobian of r = Log(G^-1 T_b^-1 T_a) at the mean (e = 0).
        Jrel = np.zeros((6, 12))
        Jrel[:, :6] = se3_right_jacobian_inv(np.zeros(6))
        Jrel[:, 6:] = -se3_left_jacobian_inv(np.zeros(6)) @ se3_ad(np.linalg.inv(G))
        M = Jrel @ np.linalg.pinv(H_r) @ Jrel.T
        M = 0.5 * (M + M.T)
        Omega = np.linalg.pinv(M)
        Omega = 0.5 * (Omega + Omega.T)
        return a, b, G, np.asarray(Omega, float)

    def marginalize_general(self, eliminate, keep):
        """Schur-marginalize ``eliminate`` onto an arbitrary ``keep`` set.

        Returns ``(keep, H_r, b_r)`` so that
        ``add_prior_factor(keep, H_r, b_r, [pose(k) for k in keep])`` reproduces
        the marginal cost over the retained nodes. Scales are fixed at their
        current values (same convention as the fixed-lag transient
        marginalization), so the prior is over the 6-dof poses only.
        """
        free = [nd for nd in self.pose_ids if nd not in self.fixed_ids]
        col = {nd: k for k, nd in enumerate(free)}
        elim = set(eliminate)
        if elim & self.fixed_ids:
            raise ValueError("cannot marginalize fixed nodes")
        elim_idx: list[int] = []
        for nd in eliminate:
            b = 6 * col[nd]
            elim_idx.extend(range(b, b + 6))
        keep_idx = [6 * col[nd] + d for nd in keep for d in range(6)]
        H, g, _ = self._full_normal()
        nnode = 6 * len(free)
        H = H[:nnode, :nnode]
        g = g[:nnode]
        idx = elim_idx + keep_idx
        Hs = H[idx][:, idx].toarray()
        gs = g[idx]
        ne = len(elim_idx)
        Hee_inv = np.linalg.pinv(Hs[:ne, :ne])
        H_r = Hs[ne:, ne:] - Hs[ne:, :ne] @ Hee_inv @ Hs[:ne, ne:]
        H_r = 0.5 * (H_r + H_r.T)
        g_r = gs[ne:] - Hs[ne:, :ne] @ Hee_inv @ gs[:ne]
        return list(keep), H_r, g_r

    # -- solver ---------------------------------------------------------------

    def optimize(self, verbose: bool = False) -> float:
        live = list(self.pose_ids)
        free = [nd for nd in live if nd not in self.fixed_ids]
        if len(live) < 3 or not self.edges or not free:
            self.last_cost = self._cost()
            return self.last_cost
        n_node = len(free)
        ncol = 6 * n_node + self.n_scale
        use_sparse = ncol > self.dense_max_cols
        lam = self.lambda_init
        prev_cost = self._cost()
        steps = 0
        eig_cache = None  # (w, V, keep) in the anchored/reduced space
        for _ in range(self.max_iterations):
            r = self._residuals()
            w = self._huber_weights()
            if use_sparse:
                # Reduced normal equations over variables 6..ncol-1: node 0 is
                # the gauge and never moves. H stays sparse end to end.
                Hred, gred = self._sparse_normal(r, w)
            else:
                J = self._jacobian()
                H = J.T @ (w[:, None] * J)
                g = J.T @ (w * r)
                # Anchor the first node (gauge for the pose graph).
                Hred = H.copy()
                Hred[:6, :] = 0.0
                Hred[:, :6] = 0.0
                Hred[:6, :6] = np.eye(6)
                gred = g.copy()
                gred[:6] = 0.0
            if self.tsvd_ratio > 0.0 and eig_cache is None:
                # TSVD (Eckart-Young): directions with eigenvalue <= ratio*max
                # are degenerate and receive no update, so unobservable scales
                # / translation directions cannot drift. The sparse path keeps
                # everything sparse via eigsh (no dense Hessian), the dense
                # path uses scipy.linalg.eigh.
                if use_sparse and Hred.shape[0] > self.tsvd_dense_max_cols:
                    # Too large for a dense eigendecomposition: skip TSVD and
                    # rely on the sparse solve (avoids OOM on huge graphs).
                    self.tsvd_total = int(Hred.shape[0])
                    self.tsvd_kept = int(Hred.shape[0])
                    eig_cache = ("skip",)
                else:
                    Hd = Hred.toarray() if use_sparse else Hred
                    w_e, V = sla.eigh(Hd)
                    keep = w_e > self.tsvd_ratio * float(w_e.max())
                    self.tsvd_total = int(w_e.size)
                    self.tsvd_kept = int(np.count_nonzero(keep))
                    eig_cache = ("eig", w_e, V, keep)
            accepted = False
            # Levenberg-Marquardt: re-solve with stronger damping until the
            # step is accepted (the previous code re-applied the same step).
            for _attempt in range(8):
                try:
                    if eig_cache is not None and eig_cache[0] == "eig":
                        w_e, V, keep = eig_cache[1:]
                        inv = np.zeros_like(w_e)
                        inv[keep] = 1.0 / (w_e[keep] + lam)
                        dxred = -(V * inv) @ (V.T @ gred)
                    elif use_sparse:
                        dxred = self._solve_sparse(Hred, -gred, lam)
                    else:
                        # Hf = Hred + lam*I is symmetric positive definite, so
                        # use a Cholesky-based solve (much faster than the
                        # default general LU for the dense window).
                        Hf = Hred + np.eye(ncol) * lam
                        dxred = sla.solve(Hf, -gred, assume_a="pos")
                except (np.linalg.LinAlgError, RuntimeError):
                    lam *= 10.0
                    if lam > 1e7:
                        break
                    continue
                dx = (np.concatenate([np.zeros(6), dxred])
                      if use_sparse else dxred)
                if float(np.max(np.abs(dx))) < 1e-10:
                    break
                backup = {n_: self.T[n_].copy() for n_ in live}
                backup_scale = list(self.edge_scale)
                for k, node in enumerate(free):
                    self.T[node] = backup[node] @ se3_exp(dx[6 * k: 6 * k + 6])
                for e, c in enumerate(self.scale_col):
                    if c is not None:
                        self.edge_scale[e] = max(
                            backup_scale[e]
                            * float(np.exp(dx[6 * n_node + c])), 1e-9)
                cost = self._cost()
                if cost <= prev_cost - 1e-12:
                    lam = max(lam / 3.0, 1e-9)
                    prev_cost = cost
                    steps += 1
                    accepted = True
                    break
                self.T.update(backup)
                self.edge_scale = backup_scale
                lam *= 10.0
                if lam > 1e7:
                    break
            if not accepted:
                break
        self.last_cost = prev_cost
        self.last_steps = steps
        return prev_cost
