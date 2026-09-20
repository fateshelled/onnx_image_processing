"""A/B the adaptive TSVD gate.

Config: (label, kf, adaptive_innov, tau, loop_ratio, tsvd).

Usage: TORCH_THREADS=4 .venv/bin/python scripts/ab_adapt_gate.py
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

CONFIGS = [
    ("off",         False, False, 3.0,  0.0,  0.0),
    ("kf_r0.1",     True,  False, 3.0,  0.0,  0.0),
    ("tsvd1e-3",    False, False, 3.0,  0.0,  1e-3),
    ("gate_lr0.10", True,  False, 3.0,  0.10, 1e-3),
    ("gate_lr0.20", True,  False, 3.0,  0.20, 1e-3),
]
SEQ = ["desk", "desk2", "room"]

base = json.load(open(REPO / "eval/results/rustuna_tune_loop_room_kf.json"))
params = dict(base["best_params"])
params["loop_iterations"] = 10
params["kf_local_map_k"] = 1
params["trans_gate_deg"] = 0.0
params["kf_edge_min_inlier"] = 0.0
params["scale_kf_q"] = 1e-3
params["scale_kf_r"] = 0.1
params["scale_kf_sigma"] = 0.5

cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0,
                       width=640, height=480)
matcher = rtl.NumpySinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")

result = {}
for seq in SEQ:
    cache = rtl.load_cache(REPO / "eval/results/tune_cache_loop", seq)
    pkl = REPO / f"eval/results/tune_cache_loop/match_cache_{seq}_torch.pkl"
    mc = {}
    if pkl.exists():
        with open(pkl, "rb") as f:
            mc = pickle.load(f)
    row = {}
    for label, kf, adapt, tau, lr, tsvd in CONFIGS:
        p = dict(params)
        p["scale_kf"] = kf
        p["scale_kf_adaptive"] = adapt
        p["scale_kf_innov_tau"] = tau
        p["scale_kf_adapt_loop_ratio"] = lr
        p["tsvd_ratio"] = tsvd
        diag = {}
        t = time.perf_counter()
        res = rtl.eval_seq(cache, p, cam, matcher, mc, diag=diag)
        row[label] = {"ATE_median": res["ATE_median"],
                      "loop_ratio": diag.get("loop_ratio"),
                      "innov_med": diag.get("kf_innov_median")}
        print(f"{seq:6} {label:12}: ATE={res['ATE_median']:.4f} "
              f"loop_ratio={diag.get('loop_ratio', float('nan')):.3f} "
              f"innov={diag.get('kf_innov_median', float('nan')):.3f} "
              f"({time.perf_counter()-t:.0f}s)", flush=True)
    result[seq] = row
    with open(pkl, "wb") as f:
        pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)
    print()

json.dump({"params": params, "configs": [c[0] for c in CONFIGS],
           "result": result},
          open(REPO / "eval/results/ab_adapt_gate.json", "w"), indent=2)
print("wrote eval/results/ab_adapt_gate.json")
