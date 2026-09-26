#!/usr/bin/env python3
"""Tests for the online keyframe pose graph (vo.online_graph)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vo.online_graph import OnlinePoseGraph
from vo.se3 import se3_exp


def _relative(T_prev, T_curr):
    """GT measurement (R, t) with x_curr = R x_prev + t convention."""
    R = T_curr[:3, :3].T @ T_prev[:3, :3]
    t = T_curr[:3, :3].T @ (T_prev[:3, 3] - T_curr[:3, 3])
    return R, t


def test_online_graph_tracks_ground_truth():
    rng = np.random.default_rng(0)
    N = 60
    gt = [np.eye(4)]
    for _ in range(N - 1):
        gt.append(gt[-1] @ se3_exp(rng.normal(size=6) * 0.08))

    def match_fn(a, b):  # "images" are frame ids
        R, t = _relative(gt[a], gt[b])
        return {"ok": True, "R": R, "t": t, "inlier_ratio": 1.0, "n_matches": 200}

    params = {"kf_max_gap": 16, "max_keyframes": 3, "kf_trans_thresh": 8.0,
              "kf_rot_thresh": 10.0, "loop_iterations": 10, "step_scale_t": 0.1,
              "scale_prior_sigma": 2.0, "loop_window": 80, "loop_min_gap": 30,
              "loop_min_inlier": 0.4, "nl_reg": True, "nl_reg_c": 10.0,
              "nl_reg_tau": 10.0, "nl_reg_length": 1.0}
    graph = OnlinePoseGraph(params, cam=None, match_fn=match_fn)
    est = [graph.add_frame(k) for k in range(N)]

    err = [np.linalg.norm(est[k][:3, 3] - gt[k][:3, 3]) for k in range(N)]
    assert np.isfinite(err).all()
    # Exact measurements: the optimized trajectory should stay near GT.
    assert max(err) < 0.2, f"max position error {max(err):.3f}"


def test_online_graph_node_set_is_bounded():
    """The active node set must not grow with the sequence length."""
    rng = np.random.default_rng(1)
    N = 240
    gt = [np.eye(4)]
    for _ in range(N - 1):
        gt.append(gt[-1] @ se3_exp(rng.normal(size=6) * 0.08))

    def match_fn(a, b):
        R, t = _relative(gt[a], gt[b])
        return {"ok": True, "R": R, "t": t, "inlier_ratio": 1.0,
                "n_matches": 300}

    params = {"kf_max_gap": 5, "kf_min_gap": 4, "max_keyframes": 3,
              "kf_trans_thresh": 0.0, "kf_rot_thresh": 0.0,
              "loop_iterations": 10, "step_scale_t": 0.1,
              "scale_prior_sigma": 2.0, "loop_window": 80,
              # Candidate keyframes must still be active for a loop to be
              # usable under bounding, so keep the gap small.
              "loop_min_gap": 8, "loop_min_inlier": 0.4, "nl_reg": True,
              "nl_reg_c": 10.0, "nl_reg_tau": 10.0, "nl_reg_length": 1.0}
    graph = OnlinePoseGraph(params, cam=None, match_fn=match_fn)
    max_nodes = 0
    for k in range(N):
        graph.add_frame(k)
        max_nodes = max(max_nodes, graph.last_n_nodes)
    assert graph.n_kf > 20
    assert graph.n_loop > 0
    # Bounded: with max_keyframes=3 / held_cap=6 the node set stays small and
    # does not grow with N.
    assert max_nodes < 40, f"node set not bounded: {max_nodes}"


def test_loop_enable_false_disables_loop_closure():
    """loop_enable=False must add no loop edges and skip the verifier."""
    rng = np.random.default_rng(2)
    N = 60
    gt = [np.eye(4)]
    for _ in range(N - 1):
        gt.append(gt[-1] @ se3_exp(rng.normal(size=6) * 0.08))

    def match_fn(a, b):
        R, t = _relative(gt[a], gt[b])
        return {"ok": True, "R": R, "t": t, "inlier_ratio": 1.0,
                "n_matches": 300}

    base = {"kf_max_gap": 5, "kf_min_gap": 4, "max_keyframes": 3,
            "kf_trans_thresh": 0.0, "kf_rot_thresh": 0.0,
            "loop_iterations": 10, "step_scale_t": 0.1,
            "scale_prior_sigma": 2.0, "loop_window": 80,
            "loop_min_gap": 8, "loop_min_inlier": 0.4, "nl_reg": True,
            "nl_reg_c": 10.0, "nl_reg_tau": 10.0, "nl_reg_length": 1.0}

    enabled = OnlinePoseGraph(dict(base), cam=None, match_fn=match_fn)
    for k in range(N):
        enabled.add_frame(k)
    assert enabled.n_loop > 0  # control: this setup does close loops

    called = []
    disabled = OnlinePoseGraph({**base, "loop_enable": False}, cam=None,
                               match_fn=match_fn)
    disabled.loop_verifier = lambda a, b: called.append((a, b)) or True
    for k in range(N):
        disabled.add_frame(k)
    assert disabled.n_loop == 0
    assert called == []


def test_scale_kf_path_runs_and_zero_sigma_is_safe():
    """The Kalman-filter scale path must run and not divide by zero."""
    from vo.online_graph import DEFAULT_PARAMS
    for sigma in (0.5, 0.0):
        rng = np.random.default_rng(3)
        N = 80
        gt = [np.eye(4)]
        for _ in range(N - 1):
            gt.append(gt[-1] @ se3_exp(rng.normal(size=6) * 0.08))

        def match_fn(a, b):
            R, t = _relative(gt[a], gt[b])
            return {"ok": True, "R": R, "t": t, "inlier_ratio": 1.0,
                    "n_matches": 300}

        params = dict(DEFAULT_PARAMS)
        params.update(scale_kf=True, scale_kf_sigma=sigma, kf_max_gap=5,
                      kf_min_gap=4, nl_reg=True, max_keyframes=3)
        graph = OnlinePoseGraph(params, cam=None, match_fn=match_fn)
        for k in range(N):
            T = graph.add_frame(k)
            assert np.isfinite(T).all()
        assert graph._scale_kf is not None
        assert graph._scale_kf.k is not None and np.isfinite(graph._scale_kf.k)
        assert graph._scale_kf.k > 0.0


def test_rotation_cycle_gate_rejects_inconsistent_loop():
    """The online path must honor cycle_threshold_deg before accepting loops."""
    gt = [np.eye(4)]
    step = se3_exp(np.array([0.0, 0.0, 0.01, 0.05, 0.0, 0.0]))
    for _ in range(39):
        gt.append(gt[-1] @ step)

    def match_fn(a, b):
        R, t = _relative(gt[a], gt[b])
        if b - a >= 8:
            bad = se3_exp(np.array(
                [0.0, 0.0, np.radians(25.0), 0.0, 0.0, 0.0]))[:3, :3]
            R = R @ bad
        return {"ok": True, "R": R, "t": t, "inlier_ratio": 1.0,
                "n_matches": 300}

    params = {"kf_max_gap": 4, "max_keyframes": 3,
              "kf_trans_thresh": 0.0, "kf_rot_thresh": 0.0,
              "loop_iterations": 3, "step_scale_t": 0.1,
              "scale_prior_sigma": 2.0, "loop_window": 20,
              "loop_min_gap": 8, "loop_min_inlier": 0.4,
              "loop_temporal_k": 1, "cycle_threshold_deg": 5.0,
              "global_opt_on_loop": False, "nl_reg": False}
    graph = OnlinePoseGraph(params, cam=None, match_fn=match_fn)
    for k in range(len(gt)):
        graph.add_frame(k)

    assert graph.n_cycle_rejected > 0
    assert graph.n_loop == 0


def test_rotation_cycle_gate_can_be_disabled():
    """A zero threshold preserves the pre-existing loop acceptance behavior."""
    gt = [np.eye(4)]
    step = se3_exp(np.array([0.0, 0.0, 0.01, 0.05, 0.0, 0.0]))
    for _ in range(23):
        gt.append(gt[-1] @ step)

    def match_fn(a, b):
        R, t = _relative(gt[a], gt[b])
        if b - a >= 8:
            bad = se3_exp(np.array(
                [0.0, 0.0, np.radians(25.0), 0.0, 0.0, 0.0]))[:3, :3]
            R = R @ bad
        return {"ok": True, "R": R, "t": t, "inlier_ratio": 1.0,
                "n_matches": 300}

    params = {"kf_max_gap": 4, "max_keyframes": 3,
              "kf_trans_thresh": 0.0, "kf_rot_thresh": 0.0,
              "loop_iterations": 3, "step_scale_t": 0.1,
              "scale_prior_sigma": 2.0, "loop_window": 20,
              "loop_min_gap": 8, "loop_min_inlier": 0.4,
              "loop_temporal_k": 1, "cycle_threshold_deg": 0.0,
              "global_opt_on_loop": False, "nl_reg": False}
    graph = OnlinePoseGraph(params, cam=None, match_fn=match_fn)
    for k in range(len(gt)):
        graph.add_frame(k)

    assert graph.n_cycle_rejected == 0
    assert graph.n_loop > 0


def test_loop_mad_history_is_bounded_and_unique():
    params = {"loop_mad_window": 3, "loop_rot_sigma": 0.1,
              "loop_dir_sigma": 0.4}
    graph = OnlinePoseGraph(params, cam=None, match_fn=lambda _a, _b: None)
    graph._commit_loop_history([
        (0, 2, 0.1, 0.2), (1, 3, 0.2, 0.3), (2, 4, 0.3, 0.4),
        (3, 5, 0.4, 0.5), (3, 5, 9.0, 9.0),
    ])
    assert list(graph._hist_rot) == [(1, 3), (2, 4), (3, 5)]
    assert graph._hist_rot[(3, 5)] == pytest.approx(0.4)
    assert len(graph._hist_rot) == len(graph._hist_dir) == 3


def test_loop_verifier_hook_counts_rejects_and_abstains():
    graph = OnlinePoseGraph({}, cam=None, match_fn=lambda _a, _b: None)
    verdicts = {0: False, 10: None, 20: True}
    graph.loop_verifier = lambda a, b: verdicts[a]
    hits = [(a, np.eye(3), np.zeros(3), 1.0, 10) for a in (0, 10, 20)]
    verified = graph._verified_hits(30, hits)
    assert [hit[0] for hit in verified] == [10, 20]
    assert graph.n_verifier_rejected == 1
    assert graph.n_verifier_abstained == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
