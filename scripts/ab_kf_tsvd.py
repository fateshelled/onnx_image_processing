"""A/B the linear-KF scale pre-pass combined with TSVD.

Config: (label, kf_enable, q, r, sigma, tsvd_ratio).

Usage: TORCH_THREADS=4 .venv/bin/python scripts/ab_kf_tsvd.py
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
    ("off",           False, None, None, None, 0.0),
    ("tsvd1e-3",      False, None, None, None, 1e-3),
    ("kf_r0.1",       True,  1e-3, 0.1,  0.5,  0.0),
    ("kf_r0.1_t1e-3", True,  1e-3, 0.1,  0.5,  1e-3),
    ("kf_r0.1_t1e-2", True,  1e-3, 0.1,  0.5,  1e-2),
]
SEQ = ["desk", "desk2", "room"]

base = json.load(open(REPO / "eval/results/rustuna_tune_loop_room_kf.json"))
params = dict(base["best_params"])
params["loop_iterations"] = 10
params["kf_local_map_k"] = 1
params["trans_gate_deg"] = 0.0
params["kf_edge_min_inlier"] = 0.0

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
    for label, kf_flag, q, r, sig, tsvd in CONFIGS:
        p = dict(params)
        p["scale_kf"] = kf_flag
        p["tsvd_ratio"] = tsvd
        if kf_flag:
            p["scale_kf_q"], p["scale_kf_r"], p["scale_kf_sigma"] = q, r, sig
        t = time.perf_counter()
        res = rtl.eval_seq(cache, p, cam, matcher, mc)
        row[label] = {"ATE_median": res["ATE_median"], "n_loop": res["n_loop"],
                      "n_kf": res["n_kf"]}
        print(f"{seq:6} {label:14}: ATE={res['ATE_median']:.4f} "
              f"n_loop={res['n_loop']} n_kf={res['n_kf']} "
              f"({time.perf_counter()-t:.0f}s)", flush=True)
    result[seq] = row
    with open(pkl, "wb") as f:
        pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)
    print()

json.dump({"params": params, "configs": [c[0] for c in CONFIGS],
           "result": result},
          open(REPO / "eval/results/ab_kf_tsvd.json", "w"), indent=2)
print("wrote eval/results/ab_kf_tsvd.json")
