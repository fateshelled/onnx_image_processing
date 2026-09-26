"""Verify the room-best parameters on desk / desk2 with the production
NumPy matcher (loop_iterations=60).

Usage: .venv/bin/python scripts/verify_best_desk.py
"""

import importlib.util
import json
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

from vo.sinkhorn_numpy import NumpySinkhornMatcher  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

res = json.load(open(REPO / "eval/results/rustuna_tune_loop_room_kf.json"))
params = dict(res["best_params"])
params["loop_iterations"] = 60

cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0,
                       width=640, height=480)
matcher = NumpySinkhornMatcher(iterations=20, epsilon=0.05,
                               unused_score=1.0, distance_type="l2")

out = {}
for seq in ["desk", "desk2"]:
    cache = rtl.load_cache(REPO / "eval/results/tune_cache_loop", seq)
    t = time.perf_counter()
    r = rtl.eval_seq(cache, params, cam, matcher)
    dt = time.perf_counter() - t
    out[seq] = {"ATE_median": r["ATE_median"], "n_loop": r["n_loop"],
                "n_kf": r["n_kf"]}
    print(f"{seq}: ATE={r['ATE_median']:.4f} n_loop={r['n_loop']} "
          f"n_kf={r['n_kf']} ({dt:.0f}s)", flush=True)

payload = {"best_room_value": res["best_value"],
           "best_params": res["best_params"], "loop_iterations": 60,
           "baseline": {"desk": 0.247, "desk2": 0.246}, "verify": out}
json.dump(payload, open(REPO / "eval/results/verify_best_desk.json", "w"),
          indent=2)
print("wrote eval/results/verify_best_desk.json")
