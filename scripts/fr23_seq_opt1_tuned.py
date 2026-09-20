"""Holdout validation of the tuned seq_opt1 graph on fr2/fr3 series.

Runs the sequential fixed-lag graph (graph_mode=kf_prior, NL-Reg on) with the
parameters selected by the optuna tuning (eval/results/tune_seq_opt1.json) and
records ATE per sequence. fr2/fr3 are holdout and are NOT used for selection.
Match caches are loaded/saved; results are written incrementally (resume).
"""

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

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
OUT = REPO / "eval/results/fr23_seq_opt1_tuned.json"
SEQ = ["freiburg3_long_office_household", "freiburg3_sitting_xyz",
       "freiburg2_rpy", "freiburg2_xyz"]

params = json.load(open(REPO / "eval/results/tune_seq_opt1.json"))["best_params"]
matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                               unused_score=1.0, distance_type="l2")

result = json.load(open(OUT)) if OUT.exists() else {}

for seq in SEQ:
    if seq in result:
        print(f"{seq:32} SKIP (done)", flush=True)
        continue
    npz = CACHE / f"{seq}.npz"
    if not npz.exists():
        print(f"{seq:32} SKIP (no npz cache)", flush=True)
        continue
    cache = rtl.load_cache(CACHE, seq)
    pkl = CACHE / f"match_cache_{seq}_torch.pkl"
    mc = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
    fx, fy, cx, cy = intrinsics_for(DATASET_ROOT, seq, (525., 525., 320., 240.))
    cam = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=640, height=480)
    t = time.perf_counter()
    r = rtl.eval_seq(cache, params, cam, matcher, mc)
    with open(pkl, "wb") as f:
        pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)
    result[seq] = {"ATE_median": r["ATE_median"], "n_loop": r["n_loop"],
                   "n_kf": r["n_kf"]}
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"{seq:32} seq_opt1_tuned: ATE={r['ATE_median']:.5f} "
          f"n_loop={r['n_loop']} n_kf={r['n_kf']} "
          f"({time.perf_counter()-t:.0f}s)", flush=True)

print("wrote", OUT, flush=True)
