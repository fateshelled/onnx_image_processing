"""Synthetic tests for full-graph (window_size=None) pose optimization + loop edges."""

import numpy as np

from pytorch_model.vo.se3 import se3_exp, se3_log
from pytorch_model.vo.se3_window import SlidingWindowOptimizer, _edge_residual


def _chain(n, seed=0, step=0.1):
    rng = np.random.default_rng(seed)
    T = [np.eye(4)]
    for _ in range(n - 1):
        d = rng.normal(size=6) * step
        T.append(T[-1] @ se3_exp(d))
    return T


def test_full_graph_keeps_all_nodes():
    opt = SlidingWindowOptimizer(window_size=None, max_iterations=5)
    T = _chain(20, seed=2)
    for i, Ti in enumerate(T):
        opt.add_node(i, Ti)
    assert len(opt.pose_ids) == 20, "window_size=None must keep all nodes"
    for i in range(19):
        opt.add_edge(i, i + 1, np.linalg.inv(T[i + 1]) @ T[i])
    cost = opt.optimize()
    assert cost < 1e-6, f"clean graph should converge to ~0 residual, got {cost}"


def test_loop_closure_corrects_drift():
    rng = np.random.default_rng(3)
    n = 12
    Tgt = _chain(n, seed=3, step=0.15)

    # Odometry edges carry a constant translation bias -> drift on integration.
    bias = np.zeros(6)
    bias[3] = 0.08  # +0.08 m per step in x
    opt = SlidingWindowOptimizer(window_size=None, max_iterations=200,
                                 huber=1.0, step_scale_t=0.02, step_scale_r=0.02)

    # Initial (drifted) node poses: integrate biased odometry only.
    drifted = [np.eye(4)]
    for i in range(n - 1):
        M = np.linalg.inv(Tgt[i + 1]) @ Tgt[i] @ se3_exp(bias)
        drifted.append(drifted[-1] @ M)
        opt.add_node(i, drifted[i].copy())
    opt.add_node(n - 1, drifted[-1].copy())

    for i in range(n - 1):
        M = np.linalg.inv(Tgt[i + 1]) @ Tgt[i] @ se3_exp(bias)
        opt.add_edge(i, i + 1, M)

    drift_err = float(np.linalg.norm(se3_log(np.linalg.inv(Tgt[n - 1]) @ drifted[n - 1])))
    assert drift_err > 0.2, "setup: drift must be non-trivial"

    # Add the true loop edge (no bias) between node 0 and node n-1.
    opt.add_edge(0, n - 1, np.linalg.inv(Tgt[n - 1]) @ Tgt[0])
    cost = opt.optimize()

    # After loop closure, node n-1 should be pulled back close to GT.
    corr_err = float(np.linalg.norm(se3_log(np.linalg.inv(Tgt[n - 1]) @ opt.get_pose(n - 1))))
    assert corr_err < drift_err * 0.5, (
        f"loop closure should reduce endpoint error {drift_err:.3f} -> {corr_err:.3f}"
    )
    # Loop edge residual itself must be small.
    loop_res = float(np.linalg.norm(_edge_residual(opt.get_pose(0), opt.get_pose(n - 1),
                                                   np.linalg.inv(Tgt[n - 1]) @ Tgt[0])))
    assert loop_res < 0.1, f"loop edge residual should be small, got {loop_res}"


if __name__ == "__main__":
    test_full_graph_keeps_all_nodes()
    test_loop_closure_corrects_drift()
    print("loop closure tests passed")
