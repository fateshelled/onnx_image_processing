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

Runs on CPU with numpy only. Not ONNX-related.
"""

import numpy as np

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
        self.pose_ids: list[int] = []
        self.T: dict[int, np.ndarray] = {}
        self.edges: list[tuple] = []  # (i, j, M, sigma_t, sigma_r)
        # Parallel bookkeeping for per-edge scale.
        self.edge_scale: list[float] = []
        self.scale_col: list[int | None] = []  # None = fixed / gauge
        self.n_scale: int = 0
        self.gauge_edge: int | None = None
        self.last_cost = float("inf")
        self.last_steps: int = 0

    # -- graph ---------------------------------------------------------------

    def add_node(self, node_id: int, T: np.ndarray) -> None:
        if node_id in self.T:
            raise ValueError(f"duplicate node id {node_id}")
        if self.pose_ids and node_id <= self.pose_ids[-1]:
            raise ValueError("node ids must be strictly increasing")
        self.T[node_id] = np.asarray(T, float).copy()
        self.pose_ids.append(node_id)
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
    ) -> None:
        if i not in self.T or j not in self.T:
            raise KeyError("edge endpoints must be live nodes in the window")
        self.edges.append((
            i, j, np.asarray(M, float).copy(),
            self.step_scale_t if sigma_t is None else sigma_t,
            self.step_scale_r if sigma_r is None else sigma_r,
        ))
        s = float(scale)
        if s <= 0.0:
            s = 1.0
        self.edge_scale.append(s)
        col = None
        if self.optimize_scale and scale_free:
            if self.gauge_edge is None:
                # First free edge fixes the (unobservable) global scale.
                self.gauge_edge = len(self.edges) - 1
            else:
                col = self.n_scale
                self.n_scale += 1
        self.scale_col.append(col)

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
            self.edge_scale = [self.edge_scale[k] for k in keep]
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
        if self.optimize_scale and self.scale_prior_sigma > 0.0:
            return self.n_scale
        return 0

    def _residuals(self) -> np.ndarray:
        """Scaled residual vector; edge rows then scale-prior rows."""
        if not self.edges:
            return np.zeros(0)
        out = [
            _edge_residual(self.T[i], self.T[j], self._scaled_M(k))
            / self._edge_scale(st, sr)
            for k, (i, j, M, st, sr) in enumerate(self.edges)
        ]
        r = np.concatenate(out)
        if self._n_prior():
            pri = [
                np.array([np.log(self.edge_scale[e]) / self.scale_prior_sigma])
                for e, c in enumerate(self.scale_col) if c is not None
            ]
            r = np.concatenate([r] + pri)
        return r

    def _edge_scale(self, st, sr):
        return np.concatenate([np.full(3, sr), np.full(3, st)])

    def _huber_weights(self) -> np.ndarray:
        if not self.edges:
            return np.zeros(0)
        w = np.ones(6 * len(self.edges))
        for k, (i, j, M, st, sr) in enumerate(self.edges):
            e = _edge_residual(self.T[i], self.T[j], self._scaled_M(k))
            sc = self._edge_scale(st, sr)
            n = float(np.linalg.norm(e / sc))
            if n > self.huber:
                w[6 * k: 6 * k + 6] = self.huber / max(n, 1e-12)
        if self._n_prior():
            w = np.concatenate([w, np.ones(self._n_prior())])
        return w

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

    def _jacobian(self, eps=1e-6):
        """Jacobian of residuals wrt node right-increments and log-scales.

        Node columns analytic (verified against finite differences); edges
        whose residual rotation norm exceeds the Bernoulli series comfort
        range (||omega|| > 1.8) fall back to central differences on that
        row only. Scale columns use the analytic expression in
        :meth:`_scale_jacobian`.
        """
        live = list(self.pose_ids)
        n = len(live)
        ncol = 6 * n + self.n_scale
        col = {node: k for k, node in enumerate(live)}
        r0 = self._residuals()
        J = np.zeros((r0.size, ncol))
        for k, (i, j, M, st, sr) in enumerate(self.edges):
            rows = slice(6 * k, 6 * k + 6)
            sc = self._edge_scale(st, sr)
            Ms = self._scaled_M(k)
            e = _edge_residual(self.T[i], self.T[j], Ms)
            if float(np.linalg.norm(e[:3])) > 1.8:
                self._jacobian_row_numeric(k, J, col, eps)
            else:
                Ji = se3_right_jacobian_inv(e)
                Jj = -se3_left_jacobian_inv(e) @ se3_ad(np.linalg.inv(Ms))
                base_i = 6 * col[i]
                base_j = 6 * col[j]
                inv_sc = 1.0 / sc
                J[rows, base_i:base_i + 6] = Ji * inv_sc[:, None]
                J[rows, base_j:base_j + 6] = Jj * inv_sc[:, None]
            c = self.scale_col[k]
            if c is not None:
                J[rows, 6 * n + c] = self._scale_jacobian(k)
        if self._n_prior():
            base = 6 * len(self.edges)
            for e, c in enumerate(self.scale_col):
                if c is not None:
                    J[base, 6 * n + c] = 1.0 / self.scale_prior_sigma
                    base += 1
        return J

    def _jacobian_row_numeric(self, k, J, col, eps=1e-6):
        """Central differences for a single edge row (6+6 columns)."""
        i, j, M, st, sr = self.edges[k]
        sc = self._edge_scale(st, sr)
        inv_sc = 1.0 / sc
        for node in (i, j):
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
                J[6 * k: 6 * k + 6, base + d] = ((rp - rm) * inv_sc) / (2.0 * eps)

    # -- solver ---------------------------------------------------------------

    def optimize(self, verbose: bool = False) -> float:
        live = list(self.pose_ids)
        if len(live) < 3 or not self.edges:
            self.last_cost = self._cost()
            return self.last_cost
        n_node = len(live)
        lam = self.lambda_init
        prev_cost = self._cost()
        steps = 0
        for _ in range(self.max_iterations):
            J = self._jacobian()
            r = self._residuals()
            w = self._huber_weights()
            H = J.T @ (w[:, None] * J)
            g = J.T @ (w * r)
            accepted = False
            # Levenberg-Marquardt: re-solve with stronger damping until the
            # step is accepted (the previous code re-applied the same step).
            for _attempt in range(8):
                Hf = H.copy()
                # Anchor the first node (gauge for the pose graph).
                Hf[:6, :] = 0.0
                Hf[:, :6] = 0.0
                Hf[:6, :6] = np.eye(6)
                rf = g.copy()
                rf[:6] = 0.0
                Hf += np.eye(H.shape[0]) * lam
                try:
                    dx = np.linalg.solve(Hf, -rf)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    if lam > 1e7:
                        break
                    continue
                if float(np.max(np.abs(dx))) < 1e-10:
                    break
                backup = {n_: self.T[n_].copy() for n_ in live}
                backup_scale = list(self.edge_scale)
                for k, node in enumerate(live):
                    self.T[node] = backup[node] @ se3_exp(dx[6 * k: 6 * k + 6])
                for e, c in enumerate(self.scale_col):
                    if c is not None:
                        self.edge_scale[e] = (
                            backup_scale[e] * float(np.exp(dx[6 * n_node + c])))
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
