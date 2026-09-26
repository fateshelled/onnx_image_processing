"""A/B inlier-based gates for additive keyframe edges.

Reuses one match cache per sequence (preloaded from disk when available), so
only unseen pairs cost anything.

Usage: TORCH_THREADS=4 .venv/bin/python scripts/ab_inlier_gate.py
"""

import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import torch

if os.environ.get("TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

spec = importlib.util.spec_from_file_location(
    "rtl", REPO / "eval/rustuna_tune_loop.py")
rtl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtl)

from vo.pose_estimation import CameraIntrinsics  # noqa: E402

# (min_inlier_ratio, min_inlier_count)
CONFIGS = [(0.0, 0), (0.2, 0), (0.3, 0), (0.0, 75), (0.0, 150), (0.3, 100)]
SEQ = ["desk", "desk2", "room"]

base = json.load(open(REPO / "eval/results/rustuna_tune_loop_room_kf.json"))
params = dict(base["best_params"])  # odom_ref=kf
params["loop_iterations"] = 10
params["trans_gate_deg"] = 0.0

cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0,
                       width=640, height=480)
matcher = rtl.NumpySinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")

print("kf params: trans=%.2f rot=%.1f loop_iterations=%d"
      % (params["kf_trans_thresh"], params["kf_rot_thresh"],
         params["loop_iterations"]))
print(f"{'seq':6} " + " ".join(f"r{r}_n{n}" for r, n in CONFIGS))

result = {}
for seq in SEQ:
    cache = rtl.load_cache(REPO / "eval/results/tune_cache_loop", seq)
    pkl = REPO / f"eval/results/tune_cache_loop/match_cache_{seq}_torch.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            match_cache = pickle.load(f)
        print(f"[{seq}] preloaded {len(match_cache)} cached pairs", flush=True)
    else:
        match_cache = {}
    row = {}
    for ratio, count in CONFIGS:
        p = dict(params)
        p["kf_edge_min_inlier"] = ratio
        p["kf_edge_min_inliers"] = count
        t = time.perf_counter()
        r = rtl.eval_seq(cache, p, cam, matcher, match_cache)
        row[f"r{ratio}_n{count}"] = {
            "ratio": ratio, "count": count, "ATE_median": r["ATE_median"],
            "n_loop": r["n_loop"], "n_kf": r["n_kf"]}
        print(f"{seq:6} r={ratio:<4} n={count:<4}: ATE={r['ATE_median']:.4f} "
              f"n_loop={r['n_loop']} n_kf={r['n_kf']} "
              f"({time.perf_counter()-t:.0f}s)", flush=True)
    result[seq] = row
    print()

json.dump({"params": params, "configs": CONFIGS, "result": result},
          open(REPO / "eval/results/ab_inlier_gate.json", "w"), indent=2)
print("wrote eval/results/ab_inlier_gate.json")
