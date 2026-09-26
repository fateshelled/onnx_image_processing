"""A/B the local-map keyframe connectivity (--kf-local-map-k).

K=1 is the previous single-hub additive behaviour; K>=2 constrains each frame
to the last K keyframes. One match cache per sequence is reused.

Usage: TORCH_THREADS=4 .venv/bin/python scripts/ab_local_map.py
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

KS = [1, 2, 3]
SEQ = ["desk", "desk2", "room"]

base = json.load(open(REPO / "eval/results/rustuna_tune_loop_room_kf.json"))
params = dict(base["best_params"])  # odom_ref=kf
params["loop_iterations"] = 10
params["trans_gate_deg"] = 0.0
params["kf_edge_min_inlier"] = 0.0

cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0,
                       width=640, height=480)
matcher = rtl.NumpySinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")

print("kf params: trans=%.2f rot=%.1f loop_iterations=%d"
      % (params["kf_trans_thresh"], params["kf_rot_thresh"],
         params["loop_iterations"]))

result = {}
for seq in SEQ:
    cache = rtl.load_cache(REPO / "eval/results/tune_cache_loop", seq)
    pkl = REPO / f"eval/results/tune_cache_loop/match_cache_{seq}_torch.pkl"
    match_cache = {}
    if pkl.exists():
        with open(pkl, "rb") as f:
            match_cache = pickle.load(f)
        print(f"[{seq}] preloaded {len(match_cache)} cached pairs", flush=True)
    row = {}
    for k in KS:
        p = dict(params)
        p["kf_local_map_k"] = k
        t = time.perf_counter()
        r = rtl.eval_seq(cache, p, cam, matcher, match_cache)
        row[k] = {"ATE_median": r["ATE_median"], "n_loop": r["n_loop"],
                  "n_kf": r["n_kf"]}
        print(f"{seq:6} K={k}: ATE={r['ATE_median']:.4f} n_loop={r['n_loop']} "
              f"n_kf={r['n_kf']} pairs_cached={len(match_cache)} "
              f"({time.perf_counter()-t:.0f}s)", flush=True)
    result[seq] = row
    print()

json.dump({"params": params, "ks": KS, "result": result},
          open(REPO / "eval/results/ab_local_map.json", "w"), indent=2)
print("wrote eval/results/ab_local_map.json")
