"""A/B the translation-consistency gate for additive keyframe edges.

Reuses one in-memory match cache per sequence, so only the first gate value
pays the matching cost.

Usage: TORCH_THREADS=4 .venv/bin/python scripts/ab_trans_gate.py
"""

import importlib.util
import json
import os
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

GATES = [0.0, 15.0, 25.0, 40.0]
SEQ = ["desk", "desk2", "room"]

base = json.load(open(REPO / "eval/results/rustuna_tune_loop_room_kf.json"))
params = dict(base["best_params"])  # odom_ref=kf already
params["loop_iterations"] = 10  # search-time setting (fast); matches tuning

cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0,
                       width=640, height=480)
matcher = rtl.NumpySinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")

print(f"params: odom_ref={params['odom_ref']} kf_trans={params['kf_trans_thresh']:.2f} "
      f"kf_rot={params['kf_rot_thresh']:.1f} loop_iterations={params['loop_iterations']}")
print(f"{'seq':6} " + " ".join(f"gate={g:>4.0f}" for g in GATES))
result = {}
for seq in SEQ:
    cache = rtl.load_cache(REPO / "eval/results/tune_cache_loop", seq)
    match_cache = {}
    row = {}
    for g in GATES:
        p = dict(params)
        p["trans_gate_deg"] = g
        t = time.perf_counter()
        r = rtl.eval_seq(cache, p, cam, matcher, match_cache)
        row[g] = {"ATE_median": r["ATE_median"], "n_loop": r["n_loop"],
                  "n_kf": r["n_kf"]}
        print(f"{seq:6} gate={g:>4.0f}: ATE={r['ATE_median']:.4f} "
              f"n_loop={r['n_loop']} n_kf={r['n_kf']} ({time.perf_counter()-t:.0f}s)",
              flush=True)
    result[seq] = row
    print()

json.dump({"params": params, "result": result},
          open(REPO / "eval/results/ab_trans_gate.json", "w"), indent=2)
print("wrote eval/results/ab_trans_gate.json")
