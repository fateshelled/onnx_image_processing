"""Coarse OFAT sweep of the covariance IRLS gate knobs (discretized levels).

Follows notes/20260920-tuning-policy.md: few discrete levels, coarse first
pass, per-series ATE reported (no mean+worst scalar).  Selection is meant to
use desk/desk2/room as the dev set; other series are diagnostics/holdout and
must not drive selection.

Usage: .venv/bin/python scripts/tune_covariant_gate.py --output notes/x.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from eval.rustuna_tune_loop import (  # noqa: E402
    SEQ_OPT1_DEFAULTS, eval_seq, load_cache, load_frames,
)
from eval.eval_tum_vo import intrinsics_for  # noqa: E402
from eval.torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

CACHE = REPO / "eval/results/tune_cache_loop"
FIELDS = ("ATE_median", "ATE_rmse", "ATE_p90", "ATE_max", "RPE_trans_1m_median", "RPE_trans_1m_rmse", "RPE_rot_1m_median_deg", "RPE_trans_1m_drift_pct", "RPE_trans_2m_median", "RPE_trans_2m_rmse", "RPE_rot_2m_median_deg", "RPE_trans_2m_drift_pct", "RPE_trans_5m_median", "RPE_trans_5m_rmse", "RPE_rot_5m_median_deg", "RPE_trans_5m_drift_pct", "RPE_trans_10m_median", "RPE_trans_10m_rmse", "RPE_rot_10m_median_deg", "RPE_trans_10m_drift_pct", "n_loop", "verifier_accept", "verifier_reject",
          "verifier_inconsistent_rejected", "verifier_direction_z_rejected")

CENTER = {
    "loop_verifier": "sim3_cov", "loop_verifier_kernel": "gm",
    "loop_verifier_kernel_delta": 5.0, "loop_verifier_whitened_iterations": 3,
    "loop_verifier_consistency_chi": 2.80,
    "loop_verifier_min_consistent": 0.3,
    "loop_verifier_max_median_residual": 6.0,
}

# OFAT: vary one axis at a time from CENTER.
GRID = {
    "loop_verifier_consistency_chi": [2.0, 4.0, 6.0],
    "loop_verifier_whitened_iterations": [0, 1, 5],
    "loop_verifier_min_consistent": [0.7],
    "loop_verifier_max_median_residual": [1e9, 3.0],
    "loop_verifier_kernel_delta": [1.0, 10.0],
}

# The consistency gate is disabled with a tiny positive fraction and a huge
# residual bound; 0.0 is rejected by the verifier's own validation.
DISABLE = 1e-9


def _configs():
    configs = [("center", dict(CENTER))]
    tag = lambda key, value: (f"{key.replace('loop_verifier_', '')[:12]}"
                              f"={value:g}")
    for key, values in GRID.items():
        for value in values:
            if value == CENTER.get(key):
                continue
            overrides = dict(CENTER)
            overrides[key] = value
            configs.append((tag(key, value), overrides))
    # pass-through control: consistency gate disabled, IRLS on
    control = dict(CENTER)
    control["loop_verifier_min_consistent"] = DISABLE
    control["loop_verifier_max_median_residual"] = 1e9
    configs.append(("irls_only_nogate", control))
    return configs


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="desk,desk2,room")
    parser.add_argument("--dataset-root", default="/home/ubuntu/datasets/tum_rgbd")
    parser.add_argument("--configs", default="",
                        help="comma-separated config tags to run (default all)")
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    only = {tag.strip() for tag in opts.configs.split(",") if tag.strip()}
    configs = [c for c in _configs() if not only or c[0] in only]

    matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")
    reports = {}
    for seq in (name.strip() for name in opts.seq.split(",") if name):
        c = load_cache(CACHE, seq)
        frames = load_frames(opts.dataset_root, seq)
        if len(frames) != int(c.get("n_frames", len(frames))):
            raise ValueError(
                f"{seq}: frame count {len(frames)} != cache n_frames "
                f"{c.get('n_frames')}")
        gt_pose = np.array(
            [frames[i][1] for i in range(0, len(frames), int(c["stride"]))],
            dtype=np.float64)
        if len(gt_pose) != len(c["gt_pos"]):
            raise ValueError(
                f"{seq}: gt_pose {len(gt_pose)} != gt_pos {len(c['gt_pos'])}")
        c["gt_pose"] = gt_pose
        fx, fy, cx, cy = intrinsics_for(opts.dataset_root, seq,
                                        (525., 525., 320., 240.))
        cam = CameraIntrinsics(fx, fy, cx, cy, 640, 480)
        match_path = CACHE / f"match_cache_{seq}_torch.pkl"
        if not match_path.exists():
            match_path = CACHE / f"match_cache_{seq}_numpy.pkl"
        with match_path.open("rb") as handle:
            match_cache = pickle.load(handle)
        for tag, overrides in configs:
            params = {**SEQ_OPT1_DEFAULTS, **overrides}
            try:
                result = eval_seq(c, params, cam, matcher, match_cache)
                report = {name: _finite(result.get(name)) for name in FIELDS}
            except Exception as exc:  # noqa: BLE001
                report = {"error": f"{type(exc).__name__}: {exc}"}
            reports[f"{seq}:{tag}"] = report
            print(f"{seq}:{tag} {report}", flush=True)
            opts.output.parent.mkdir(parents=True, exist_ok=True)
            opts.output.write_text(
                json.dumps(reports, indent=2, ensure_ascii=False,
                           allow_nan=False) + "\n")
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
