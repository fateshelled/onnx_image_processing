"""Tests for per-edge scale optimization in the sliding-window pose graph.

Covers:
- analytic scale Jacobian vs finite differences (node and scale columns),
- recovery of per-edge translation scales on a closed triangle where the
  measured translations are unit-norm (monocular convention) but the true
  translations have different lengths.
"""

import numpy as np

from vo.se3 import se3_exp
from vo.se3_window import SlidingWindowOptimizer


def _Rz(theta):
    return se3_exp(np.array([0.0, 0.0, theta, 0.0, 0.0, 0.0]))[:3, :3]


def _M(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t, float).reshape(3)
    return M


def test_jacobian_matches_finite_difference():
    rng = np.random.default_rng(0)
    T = [np.eye(4)]
    for _ in range(3):
        T.append(T[-1] @ se3_exp(rng.normal(size=6) * 0.3))

    opt = SlidingWindowOptimizer(max_iterations=1,
                                 optimize_scale=True, scale_prior_sigma=0.3)
    for i in range(4):
        opt.add_node(i, T[i])
    for i in range(3):
        opt.add_edge(i, i + 1, np.linalg.inv(T[i + 1]) @ T[i],
                     scale_free=(i > 0), scale=1.0 + 0.2 * i)

    J = opt._jacobian()
    n = len(opt.pose_ids)
    ncol = J.shape[1]
    assert ncol == 6 * n + opt.n_scale
    assert opt.n_scale == 1  # 2 free edges, first is the gauge

    eps = 1e-6
    Jn = np.zeros_like(J)
    for k, node in enumerate(opt.pose_ids):
        T0 = opt.T[node].copy()
        for d in range(6):
            delta = np.zeros(6)
            delta[d] = eps
            opt.T[node] = T0 @ se3_exp(delta)
            rp = opt._residuals()
            opt.T[node] = T0 @ se3_exp(-delta)
            rm = opt._residuals()
            opt.T[node] = T0
            Jn[:, 6 * k + d] = (rp - rm) / (2 * eps)
    for e, c in enumerate(opt.scale_col):
        if c is None:
            continue
        s0 = opt.edge_scale[e]
        opt.edge_scale[e] = s0 * np.exp(eps)
        rp = opt._residuals()
        opt.edge_scale[e] = s0 * np.exp(-eps)
        rm = opt._residuals()
        opt.edge_scale[e] = s0
        Jn[:, 6 * n + c] = (rp - rm) / (2 * eps)

    # Scale columns must match tightly; node columns carry central-difference
    # truncation noise (relative ~1e-6) so allow a looser tolerance there.
    scale_cols = [6 * n + c for c in opt.scale_col if c is not None]
    node_cols = [k for k in range(6 * n)]
    assert np.allclose(J[:, scale_cols], Jn[:, scale_cols], atol=1e-6)
    assert np.allclose(J[:, node_cols], Jn[:, node_cols], atol=1e-3), \
        f"max node diff {np.max(np.abs(J[:, node_cols] - Jn[:, node_cols]))}"


def _triangle_setup():
    """True edges: lengths 1 and 2, turns 120 and 150 degrees."""
    M1 = _M(_Rz(0.0), [1.0, 0.0, 0.0])
    M2 = _M(_Rz(np.radians(120.0)), [2.0, 0.0, 0.0])
    T1 = np.linalg.inv(M1)
    T2 = T1 @ np.linalg.inv(M2)
    M3_true = np.linalg.inv(T2)  # loop edge node2 -> node0
    L3 = float(np.linalg.norm(M3_true[:3, 3]))
    M3_meas = M3_true.copy()
    M3_meas[:3, 3] = M3_true[:3, 3] / L3
    M1_meas = M1.copy()  # L1 = 1 already
    M2_meas = M2.copy()
    M2_meas[:3, 3] = M2[:3, 3] / 2.0
    return (M1, M1_meas), (M2, M2_meas), (M3_true, M3_meas), L3


def test_recovers_per_edge_scales_on_triangle():
    (M1, M1_meas), (M2, M2_meas), (M3_true, M3_meas), L3 = _triangle_setup()

    T0 = np.eye(4)
    T1 = T0 @ np.linalg.inv(M1_meas)
    T2 = T1 @ np.linalg.inv(M2_meas)

    opt = SlidingWindowOptimizer(max_iterations=200,
                                 optimize_scale=True, scale_prior_sigma=0.0)
    for i, Ti in enumerate([T0, T1, T2]):
        opt.add_node(i, Ti)
    opt.add_edge(0, 1, M1_meas, scale_free=True)   # gauge, true scale 1
    opt.add_edge(1, 2, M2_meas, scale_free=True)   # true scale 2
    opt.add_edge(0, 2, M3_meas, scale_free=True)   # true scale L3 (loop)

    cost = opt.optimize()
    assert cost < 1e-6, f"triangle should close, residual {cost}"
    s2 = opt.get_scale(1)
    s3 = opt.get_scale(2)
    assert abs(s2 - 2.0) < 1e-2, f"edge1 scale {s2} != 2"
    assert abs(s3 - L3) < 1e-2, f"edge2 scale {s3} != {L3}"


def test_fixed_scale_cannot_close_triangle():
    (M1, M1_meas), (M2, M2_meas), (M3_true, M3_meas), L3 = _triangle_setup()

    T1 = np.linalg.inv(M1_meas)
    T2 = T1 @ np.linalg.inv(M2_meas)

    opt = SlidingWindowOptimizer(max_iterations=200,
                                 optimize_scale=False)
    for i, Ti in enumerate([np.eye(4), T1, T2]):
        opt.add_node(i, Ti)
    opt.add_edge(0, 1, M1_meas)
    opt.add_edge(1, 2, M2_meas)
    opt.add_edge(0, 2, M3_meas)

    cost = opt.optimize()
    assert cost > 1e-3, f"unit translations should not close exactly, got {cost}"


if __name__ == "__main__":
    test_jacobian_matches_finite_difference()
    test_recovers_per_edge_scales_on_triangle()
    test_fixed_scale_cannot_close_triangle()
    print("scale window tests passed")
