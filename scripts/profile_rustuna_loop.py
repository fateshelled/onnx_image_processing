"""One-shot profiling of eval_seq subcomponents to find the bottleneck.

Usage: .venv/bin/python scripts/profile_rustuna_loop.py [seq] [--full]
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

import importlib.util as _ilu  # noqa: E402

spec = _ilu.spec_from_file_location("rtl", REPO_ROOT / "eval/rustuna_tune_loop.py")
rtl = _ilu.module_from_spec(spec)
spec.loader.exec_module(rtl)

from vo.sinkhorn_numpy import NumpySinkhornMatcher  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

seq = sys.argv[1] if len(sys.argv) > 1 else "room"

timing = {"sinkhorn_calls": 0, "sinkhorn_s": 0.0, "pose_calls": 0, "pose_s": 0.0,
          "optimize_s": 0.0}

_orig_match = NumpySinkhornMatcher.match_probs
_orig_pose = rtl.estimate_pose_from_matches
_orig_opt = rtl.SlidingWindowOptimizer.optimize


def match_probs(self, *a, **k):
    t = time.perf_counter()
    r = _orig_match(self, *a, **k)
    timing["sinkhorn_calls"] += 1
    timing["sinkhorn_s"] += time.perf_counter() - t
    return r


def pose(*a, **k):
    t = time.perf_counter()
    r = _orig_pose(*a, **k)
    timing["pose_calls"] += 1
    timing["pose_s"] += time.perf_counter() - t
    return r


def optimize(self, *a, **k):
    t = time.perf_counter()
    r = _orig_opt(self, *a, **k)
    timing["optimize_s"] += time.perf_counter() - t
    return r


NumpySinkhornMatcher.match_probs = match_probs
rtl.estimate_pose_from_matches = pose
rtl.SlidingWindowOptimizer.optimize = optimize

t0 = time.perf_counter()
cache = rtl.load_cache("eval/results/tune_cache_loop", seq)
t_load = time.perf_counter() - t0

params = {
    "odom_ref": "kf", "kf_mode": "motion", "keyframe_decim": 12,
    "kf_trans_thresh": 7.899, "kf_rot_thresh": 32.301, "loop_window": 65,
    "loop_min_gap": 22, "loop_min_inlier": 0.327, "loop_temporal_k": 3,
    "scale_prior_sigma": 1.806, "step_scale_t": 0.136, "loop_sigma_scale": 1.152,
    "loop_iterations": 10, "loop_rot_only": False, "cycle_threshold_deg": 0.0,
}
cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0,
                       width=640, height=480)
matcher = NumpySinkhornMatcher(iterations=20, epsilon=0.05,
                               unused_score=1.0, distance_type="l2")

t0 = time.perf_counter()
res = rtl.eval_seq(cache, params, cam, matcher)
total = time.perf_counter() - t0

print(f"seq={seq} n_frames={cache['n_frames']} stride={cache['stride']}")
print(f"cache_load_s={t_load:.2f}")
print(f"eval_total_s={total:.1f}  ATE={res['ATE_median']:.4f} "
      f"n_loop={res['n_loop']} n_kf={res['n_kf']}")
print(f"sinkhorn: calls={timing['sinkhorn_calls']} s={timing['sinkhorn_s']:.1f}")
print(f"pose(est): calls={timing['pose_calls']} s={timing['pose_s']:.1f}")
print(f"opt.optimize: s={timing['optimize_s']:.1f}")
other = total - timing["sinkhorn_s"] - timing["pose_s"] - timing["optimize_s"]
print(f"other_s={other:.1f}")
