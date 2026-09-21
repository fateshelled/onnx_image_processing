#!/usr/bin/env python3
"""Tests for the online keyframe pose graph (vo.online_graph)."""

import sys
from pathlib import Path

import numpy as np

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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
