"""Compare stride batch vs sequential option-1 fixed-lag on fr2/fr3 series.

For every series it evaluates:
  - stride_off   : batch graph, no gate (sparse solver)
  - stride_gate  : batch graph with the adaptive scale-KF gate
  - seq_opt1     : sequential fixed-lag (max_keyframes=3, seq_tsvd=0.1) with
                   deferred multi-node marginalization (option 1)

Match caches are loaded/saved in eval/results/tune_cache_loop so repeated runs
are cheap. Results are written incrementally to
eval/results/fr23_seq_option1.json (resume-friendly).
"""

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

spec = importlib.util.spec_from_file_location(
    "rtl", REPO / "eval/rustuna_tune_loop.py")
rtl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtl)
from eval_tum_vo import intrinsics_for  # noqa: E402
from torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

DATASET_ROOT = "/home/ubuntu/datasets/tum_rgbd"
CACHE = REPO / "eval/results/tune_cache_loop"
OUT = REPO / "eval/results/fr23_seq_option1.json"
# Heaviest last so partial results are still useful.
SEQ = ["freiburg3_long_office_household", "freiburg3_sitting_xyz",
       "freiburg2_rpy", "freiburg2_xyz"]

base = json.load(open(REPO / "eval/results/rustuna_tune_loop_room_kf.json"))
params = dict(base["best_params"])
params["loop_iterations"] = 10
params["kf_local_map_k"] = 1
params["kf_edge_min_inlier"] = 0.0
params["scale_kf_q"] = 1e-3
params["scale_kf_r"] = 0.1
params["scale_kf_sigma"] = 0.5

matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                               unused_score=1.0, distance_type="l2")

result = {}
if OUT.exists():
    try:
        result = json.load(open(OUT))
    except json.JSONDecodeError:
        result = {}


def run(seq, tag, p):
    cache = rtl.load_cache(CACHE, seq)
    pkl = CACHE / f"match_cache_{seq}_torch.pkl"
    mc = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
    fx, fy, cx, cy = intrinsics_for(DATASET_ROOT, seq, (525., 525., 320., 240.))
    cam = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=640, height=480)
    t = time.perf_counter()
    r = rtl.eval_seq(cache, p, cam, matcher, mc)
    with open(pkl, "wb") as f:
        pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)
    result.setdefault(seq, {})[tag] = {
        "ATE_median": r["ATE_median"], "n_loop": r["n_loop"],
        "n_kf": r["n_kf"]}
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"{seq:32} {tag:12}: ATE={r['ATE_median']:.5f} "
          f"n_loop={r['n_loop']} n_kf={r['n_kf']} "
          f"({time.perf_counter()-t:.0f}s)", flush=True)


for seq in SEQ:
    if seq in result and "seq_opt1" in result[seq]:
        print(f"{seq:32} SKIP (done)", flush=True)
        continue
    if not (CACHE / f"{seq}.npz").exists():
        print(f"{seq:32} SKIP (no npz cache)", flush=True)
        continue
    p = dict(params)
    p["scale_kf"] = False
    p["scale_kf_adapt_loop_ratio"] = 0.0
    p["tsvd_ratio"] = 1e-3
    p["graph_mode"] = "stride"
    run(seq, "stride_off", p)

    p = dict(params)
    p["scale_kf"] = True
    p["scale_kf_adapt_loop_ratio"] = 0.15
    p["tsvd_ratio"] = 1e-3
    p["graph_mode"] = "stride"
    run(seq, "stride_gate", p)

    p = dict(params)
    p["graph_mode"] = "kf_prior"
    p["max_keyframes"] = 3
    p["seq_tsvd_ratio"] = 0.1
    p["nl_reg"] = False  # legacy TSVD variant (kept for reference)
    run(seq, "seq_opt1", p)

print("wrote", OUT, flush=True)
