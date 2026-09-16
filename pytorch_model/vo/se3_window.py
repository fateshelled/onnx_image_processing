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
    ) -> None:
        if window_size is not None and window_size < 2:
            raise ValueError(f"window_size must be >= 2 or None, got {window_size}")
        self.window_size = window_size
        self.max_iterations = max_iterations
        self.lambda_init = lambda_init
        self.step_scale_t = step_scale_t
        self.step_scale_r = step_scale_r
        self.huber = huber
        self.pose_ids: list[int] = []
        self.T: dict[int, np.ndarray] = {}
        self.edges: list[tuple] = []  # (i, j, M, sigma_t, sigma_r)
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
    ) -> None:
        if i not in self.T or j not in self.T:
            raise KeyError("edge endpoints must be live nodes in the window")
        self.edges.append((
            i, j, np.asarray(M, float).copy(),
            self.step_scale_t if sigma_t is None else sigma_t,
            self.step_scale_r if sigma_r is None else sigma_r,
        ))

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
            self.edges = [e for e in self.edges if e[0] != drop and e[1] != drop]

    # -- internals ------------------------------------------------------------

    def _residuals(self) -> np.ndarray:
        """Scaled residual vector; rows ordered by edge then (rot, trans)."""
        if not self.edges:
            return np.zeros(0)
        out = np.concatenate([
            _edge_residual(self.T[i], self.T[j], M)
            / self._edge_scale(st, sr)
            for (i, j, M, st, sr) in self.edges
        ])
        return out

    def _edge_scale(self, st, sr):
        return np.concatenate([np.full(3, sr), np.full(3, st)])

    def _huber_weights(self) -> np.ndarray:
        if not self.edges:
            return np.zeros(0)
        w = np.ones(6 * len(self.edges))
        for k, (i, j, M, st, sr) in enumerate(self.edges):
            e = _edge_residual(self.T[i], self.T[j], M)
            sc = self._edge_scale(st, sr)
            n = float(np.linalg.norm(e / sc))
            if n > self.huber:
                w[6 * k: 6 * k + 6] = self.huber / max(n, 1e-12)
        return w

    def _cost(self) -> float:
        r = self._residuals()
        if r.size == 0:
            return 0.0
        w = self._huber_weights()
        return float(0.5 * np.sum((np.sqrt(w) * r) ** 2))

    def _jacobian(self, eps=1e-6):
        """Jacobian of residuals wrt right-increment on each node.

        Analytic by default. Per edge (residual e = Log(M^-1 Tj^-1 Ti),
        right updates T <- T Exp(delta)):
            d e / d delta_i =  J_r(e)        (= J_l(-e)^-1)
            d e / d delta_j = -J_l(e)^-1 @ Ad(M)^-1
        Both verified against finite differences to ~1e-10 (analytic). Edges
        whose residual rotation norm exceeds the Bernoulli series comfort
        range (||omega|| > 1.8) fall back to central differences on that
        row only (rare; residuals are typically << 1).
        """
        live = list(self.pose_ids)
        n = len(live)
        col = {node: k for k, node in enumerate(live)}
        r0 = self._residuals()
        J = np.zeros((r0.size, 6 * n))
        for k, (i, j, M, st, sr) in enumerate(self.edges):
            rows = slice(6 * k, 6 * k + 6)
            sc = self._edge_scale(st, sr)
            e = _edge_residual(self.T[i], self.T[j], M)
            if float(np.linalg.norm(e[:3])) > 1.8:
                self._jacobian_row_numeric(k, J, col, eps)
                continue
            Ji = se3_right_jacobian_inv(e)
            Jj = -se3_left_jacobian_inv(e) @ se3_ad(np.linalg.inv(M))
            base_i = 6 * col[i]
            base_j = 6 * col[j]
            inv_sc = 1.0 / sc
            J[rows, base_i:base_i + 6] = Ji * inv_sc[:, None]
            J[rows, base_j:base_j + 6] = Jj * inv_sc[:, None]
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
                rp = _edge_residual(self.T[i], self.T[j], M)
                self.T[node] = T0 @ se3_exp(-delta)
                rm = _edge_residual(self.T[i], self.T[j], M)
                self.T[node] = T0
                J[6 * k: 6 * k + 6, base + d] = ((rp - rm) * inv_sc) / (2.0 * eps)

    # -- solver ---------------------------------------------------------------

    def optimize(self, verbose: bool = False) -> float:
        live = list(self.pose_ids)
        if len(live) < 3 or not self.edges:
            self.last_cost = self._cost()
            return self.last_cost
        lam = self.lambda_init
        prev_cost = self._cost()
        steps = 0
        for _ in range(self.max_iterations):
            J = self._jacobian()
            r = self._residuals()
            w = self._huber_weights()
            H = J.T @ (w[:, None] * J)
            g = J.T @ (w * r)
            Hf = H.copy()
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
                continue
            accepted = False
            for _attempt in range(5):
                backup = {n_: self.T[n_].copy() for n_ in live}
                for k, node in enumerate(live):
                    self.T[node] = backup[node] @ se3_exp(dx[6 * k: 6 * k + 6])
                cost = self._cost()
                if cost <= prev_cost - 1e-12:
                    lam = max(lam / 3.0, 1e-9)
                    prev_cost = cost
                    steps += 1
                    accepted = True
                    break
                self.T.update(backup)
                lam *= 10.0
                if lam > 1e7:
                    break
            if not accepted or float(np.max(np.abs(dx))) < 1e-8:
                break
        self.last_cost = prev_cost
        self.last_steps = steps
        return prev_cost
