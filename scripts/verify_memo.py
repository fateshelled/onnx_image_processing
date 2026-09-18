"""Verify match-cache memoization: second eval must match the first and be
much faster (all frame pairs already memoized).

Usage: .venv/bin/python scripts/verify_memo.py [seq]
"""

import importlib.util
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

spec = importlib.util.spec_from_file_location(
    "rtl", REPO_ROOT / "eval/rustuna_tune_loop.py")
rtl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtl)

from vo.sinkhorn_numpy import NumpySinkhornMatcher  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

seq = sys.argv[1] if len(sys.argv) > 1 else "desk2"
cache = rtl.load_cache("eval/results/tune_cache_loop", seq)
cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0,
                       width=640, height=480)
matcher = NumpySinkhornMatcher(iterations=20, epsilon=0.05,
                               unused_score=1.0, distance_type="l2")
params = {
    "odom_ref": "kf", "kf_mode": "motion", "keyframe_decim": 12,
    "kf_trans_thresh": 7.899, "kf_rot_thresh": 32.301, "loop_window": 65,
    "loop_min_gap": 22, "loop_min_inlier": 0.327, "loop_temporal_k": 3,
    "scale_prior_sigma": 1.806, "step_scale_t": 0.136, "loop_sigma_scale": 1.152,
    "loop_iterations": 10, "loop_rot_only": False, "cycle_threshold_deg": 0.0,
}
match_cache = {}
t0 = time.perf_counter()
r1 = rtl.eval_seq(cache, params, cam, matcher, match_cache)
t1 = time.perf_counter() - t0
n_pairs = len(match_cache)

t0 = time.perf_counter()
r2 = rtl.eval_seq(cache, params, cam, matcher, match_cache)
t2 = time.perf_counter() - t0

print(f"seq={seq} memoized_pairs={n_pairs}")
print(f"run1 ATE={r1['ATE_median']:.8f} n_loop={r1['n_loop']} "
      f"n_kf={r1['n_kf']} s={t1:.1f}")
print(f"run2 ATE={r2['ATE_median']:.8f} n_loop={r2['n_loop']} "
      f"n_kf={r2['n_kf']} s={t2:.1f}")
print(f"match_cache size after run2={len(match_cache)}")
same = (r1["ATE_median"] == r2["ATE_median"] and r1["n_loop"] == r2["n_loop"]
        and r1["n_kf"] == r2["n_kf"])
print("IDENTICAL:", same, f"speedup={t1 / t2:.1f}x")
