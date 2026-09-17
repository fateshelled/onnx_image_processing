"""Synthetic tests for full-graph (window_size=None) pose optimization + loop edges.

Also covers the numpy-only integrated keyframe-matching / loop-closure edge
selection logic in ``vo.loop_closure``.
"""

import numpy as np

from vo.se3 import se3_exp, se3_log
from vo.se3_window import SlidingWindowOptimizer, _edge_residual
from vo.loop_closure import (
    confirmed_loop_hits,
    edge_key,
    local_candidate,
    temporal_confirmed,
)


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


# --------------------------------------------------------------------------
# Integrated keyframe matching / loop-closure edge selection
# --------------------------------------------------------------------------
def _hit(a, inl=0.5, n=100):
    return (int(a), np.eye(3), np.zeros(3), float(inl), int(n))


def test_edge_key_is_undirected():
    assert edge_key(3, 7) == edge_key(7, 3) == (3, 7)


def test_temporal_gate_requires_consecutive_history():
    # need=1 -> gate off
    assert temporal_confirmed(0, 2, [[], [_hit(0)], [_hit(2)]], 1, margin=4)
    # need=3 -> keyframe 0 has no hit, so the chain is broken
    assert not temporal_confirmed(0, 2, [[], [_hit(0)], [_hit(2)]], 3, margin=4)
    # full chain confirms
    assert temporal_confirmed(0, 2, [[_hit(1)], [_hit(0)], [_hit(2)]], 3, margin=4)


def test_temporal_gate_margin():
    hits_per_kf = [[], [_hit(0)], []]
    assert temporal_confirmed(3, 2, hits_per_kf, 2, margin=4)
    assert not temporal_confirmed(3, 2, hits_per_kf, 2, margin=2)


def test_confirmed_loop_hits_skips_duplicates():
    hits = [_hit(0), _hit(8)]
    out = confirmed_loop_hits(16, hits, 1, [[], []], need=1, margin=4,
                              added_edges={edge_key(0, 16)})
    assert [int(h[0]) for h in out] == [8]


def test_local_candidate_branches():
    kf = [0, 16, 32]
    assert local_candidate(kf, 0, set()) is None
    assert local_candidate(kf, 2, set()) == 16
    assert local_candidate(kf, 2, {edge_key(16, 32)}) is None


def test_integrated_loop_preferred_then_local_fallback():
    kf = [0, 16, 32, 48]
    hits_per_kf = [[], [_hit(0, inl=0.9)], [_hit(0, inl=0.9)], []]
    # Keyframe 32: the previous keyframe saw the same loop spot -> loop path.
    confirmed = confirmed_loop_hits(32, [_hit(0, inl=0.9)], 2, hits_per_kf,
                                    need=2, margin=16, added_edges=set())
    assert len(confirmed) == 1 and int(confirmed[0][0]) == 0
    # Keyframe 48: no loop hit -> local fallback to kf[2] == 32.
    assert confirmed_loop_hits(48, [], 3, hits_per_kf, 2, 16, set()) == []
    assert local_candidate(kf, 3, set()) == 32


if __name__ == "__main__":
    test_full_graph_keeps_all_nodes()
    test_loop_closure_corrects_drift()
    test_edge_key_is_undirected()
    test_temporal_gate_requires_consecutive_history()
    test_temporal_gate_margin()
    test_confirmed_loop_hits_skips_duplicates()
    test_local_candidate_branches()
    test_integrated_loop_preferred_then_local_fallback()
    print("loop closure tests passed")
