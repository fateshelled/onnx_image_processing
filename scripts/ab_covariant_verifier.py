"""A/B the covariance-weighted Sim(3) loop gate against baseline/current sim3.

Runs the same fixed-cache kf_prior evaluator for several verifier operating
points and reports the trajectory ATE plus the verifier counters, so the gate
is judged on ATE rather than on offline direction medians.

Usage: .venv/bin/python scripts/ab_covariant_verifier.py --output notes/ab.json
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
FIELDS = ("ATE_median", "ATE_rmse", "ATE_p90", "ATE_max", "RPE_trans_1m_median", "RPE_trans_1m_rmse", "RPE_rot_1m_median_deg", "RPE_trans_1m_drift_pct", "RPE_trans_2m_median", "RPE_trans_2m_rmse", "RPE_rot_2m_median_deg", "RPE_trans_2m_drift_pct", "RPE_trans_5m_median", "RPE_trans_5m_rmse", "RPE_rot_5m_median_deg", "RPE_trans_5m_drift_pct", "RPE_trans_10m_median", "RPE_trans_10m_rmse", "RPE_rot_10m_median_deg", "RPE_trans_10m_drift_pct", "n_loop", "n_kf", "verifier_accept", "verifier_reject",
          "verifier_abstain", "verifier_direction_reject",
          "verifier_inconsistent_rejected", "verifier_direction_z_rejected",
          "verifier_uncertain_abstained")

CONFIGS = [
    ("baseline", {"loop_verifier": "none"}),
    ("sim3_g6", {}),
    ("sim3_cov_gm_d5", {
        "loop_verifier": "sim3_cov", "loop_verifier_kernel": "gm",
        "loop_verifier_kernel_delta": 5.0,
        "loop_verifier_min_consistent": 0.7,
        "loop_verifier_max_median_residual": 3.0}),
    ("sim3_cov_gm_d5_loose", {
        "loop_verifier": "sim3_cov", "loop_verifier_kernel": "gm",
        "loop_verifier_kernel_delta": 5.0,
        "loop_verifier_min_consistent": 0.4,
        "loop_verifier_max_median_residual": 6.0}),
    ("sim3_cov_gm_d5_veryloose", {
        "loop_verifier": "sim3_cov", "loop_verifier_kernel": "gm",
        "loop_verifier_kernel_delta": 5.0,
        "loop_verifier_min_consistent": 0.2,
        "loop_verifier_max_median_residual": 12.0}),
    ("sim3_cov_gm_d5_zdir", {
        "loop_verifier": "sim3_cov", "loop_verifier_kernel": "gm",
        "loop_verifier_kernel_delta": 5.0,
        "loop_verifier_min_consistent": 0.4,
        "loop_verifier_max_median_residual": 6.0,
        "loop_verifier_max_direction_z": 3.0}),
    ("sim3_cov_irls_d5_loose", {
        "loop_verifier": "sim3_cov", "loop_verifier_kernel": "gm",
        "loop_verifier_kernel_delta": 5.0,
        "loop_verifier_min_consistent": 0.4,
        "loop_verifier_max_median_residual": 6.0,
        "loop_verifier_whitened_iterations": 5}),
    ("sim3_cov_irls_d5_veryloose", {
        "loop_verifier": "sim3_cov", "loop_verifier_kernel": "gm",
        "loop_verifier_kernel_delta": 5.0,
        "loop_verifier_min_consistent": 0.2,
        "loop_verifier_max_median_residual": 12.0,
        "loop_verifier_whitened_iterations": 5}),
    ("sim3_cov_irls_d1_veryloose", {
        "loop_verifier": "sim3_cov", "loop_verifier_kernel": "gm",
        "loop_verifier_kernel_delta": 1.0,
        "loop_verifier_min_consistent": 0.2,
        "loop_verifier_max_median_residual": 12.0,
        "loop_verifier_whitened_iterations": 5}),
]


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
    configs = [c for c in CONFIGS if not only or c[0] in only]

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
            except Exception as exc:  # noqa: BLE001 - keep other configs going
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
