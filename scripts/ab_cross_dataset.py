"""Cross-dataset check of the adaptive scale gate (fr1 / fr2 / fr3).

Evaluates configs `off` (no KF/TSVD) and `gate` (KF + adaptive-TSVD by loop
density) on every sequence with an existing feature cache, using per-camera
TUM intrinsics.

Usage: TORCH_THREADS=4 .venv/bin/python scripts/ab_cross_dataset.py
"""

import argparse
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
from eval_tum_vo import intrinsics_for  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

DATASET_ROOT = "/home/ubuntu/datasets/tum_rgbd"
SEQ = ["desk", "desk2", "room", "360", "xyz",
       "freiburg2_desk", "freiburg2_xyz", "freiburg2_rpy",
       "freiburg3_long_office_household", "freiburg3_sitting_xyz"]
CONFIGS = [("off", False, 0.0), ("gate", True, 0.15)]

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default=",".join(SEQ),
                help="comma-separated subset to evaluate")
ap.add_argument("--resume", action="store_true",
                help="skip sequences already present in the output json")
ap.add_argument("--force", action="store_true",
                help="re-evaluate sequences even if already present")
ap.add_argument("--matcher", default="torch", choices=["numpy", "torch"])
args = ap.parse_args()
SEQ = [s for s in args.seq.split(",") if s]

OUT = REPO / "eval/results/ab_cross_dataset.json"
result = {}
if OUT.exists():
    try:
        result = json.load(open(OUT)).get("result", {})
    except json.JSONDecodeError:
        result = {}

base = json.load(open(REPO / "eval/results/rustuna_tune_loop_room_kf.json"))
params = dict(base["best_params"])
params["loop_iterations"] = 10
params["kf_local_map_k"] = 1
params["kf_edge_min_inlier"] = 0.0
params["scale_kf_q"] = 1e-3
params["scale_kf_r"] = 0.1
params["scale_kf_sigma"] = 0.5

if args.matcher == "torch":
    from torch_sinkhorn import TorchSinkhornMatcher
    matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")
else:
    matcher = rtl.NumpySinkhornMatcher(iterations=20, epsilon=0.05,
                                       unused_score=1.0, distance_type="l2")
for seq in SEQ:
    npz = REPO / f"eval/results/tune_cache_loop/{seq}.npz"
    if not npz.exists():
        print(f"{seq:32} SKIP (no cache)")
        continue
    if seq in result and args.resume and not args.force:
        print(f"{seq:32} SKIP (already evaluated)")
        continue
    fx, fy, cx, cy = intrinsics_for(DATASET_ROOT, seq, (525., 525., 320., 240.))
    cam = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=640, height=480)
    cache = rtl.load_cache(REPO / "eval/results/tune_cache_loop", seq)
    pkl = REPO / (f"eval/results/tune_cache_loop/"
                  f"match_cache_{seq}_{args.matcher}.pkl")
    mc = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
    row = {}
    for label, kf, lr in CONFIGS:
        p = dict(params)
        p["scale_kf"] = kf
        p["scale_kf_adapt_loop_ratio"] = lr
        p["tsvd_ratio"] = 1e-3
        p["graph_mode"] = "stride"  # batch baseline (not the sequential default)
        p["nl_reg"] = False
        diag = {}
        t = time.perf_counter()
        r = rtl.eval_seq(cache, p, cam, matcher, mc, diag=diag)
        row[label] = {"ATE_median": r["ATE_median"], "n_loop": r["n_loop"],
                      "n_kf": r["n_kf"],
                      "loop_ratio": diag.get("loop_ratio")}
        print(f"{seq:32} {label:4}: ATE={r['ATE_median']:.4f} "
              f"n_loop={r['n_loop']:3d} n_kf={r['n_kf']:3d} "
              f"loop_ratio={diag.get('loop_ratio', float('nan')):.3f} "
              f"({time.perf_counter()-t:.0f}s)", flush=True)
    result[seq] = row
    with open(pkl, "wb") as f:
        pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)
    json.dump({"params": params, "result": result},
              open(REPO / "eval/results/ab_cross_dataset.json", "w"), indent=2)
    print()

json.dump({"params": params, "result": result},
          open(REPO / "eval/results/ab_cross_dataset.json", "w"), indent=2)
print("wrote eval/results/ab_cross_dataset.json")
