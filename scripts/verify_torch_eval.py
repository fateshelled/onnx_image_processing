"""End-to-end equivalence check: eval_seq with numpy vs torch matcher.

Usage: .venv/bin/python scripts/verify_torch_eval.py [seq]
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
from torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

seq = sys.argv[1] if len(sys.argv) > 1 else "desk2"
cache = rtl.load_cache("eval/results/tune_cache_loop", seq)
cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0,
                       width=640, height=480)
params = {
    "odom_ref": "kf", "kf_mode": "motion", "keyframe_decim": 12,
    "kf_trans_thresh": 7.899, "kf_rot_thresh": 32.301, "loop_window": 65,
    "loop_min_gap": 22, "loop_min_inlier": 0.327, "loop_temporal_k": 3,
    "scale_prior_sigma": 1.806, "step_scale_t": 0.136, "loop_sigma_scale": 1.152,
    "loop_iterations": 10, "loop_rot_only": False, "cycle_threshold_deg": 0.0,
}
res = {}
for name, matcher in (
        ("numpy", NumpySinkhornMatcher(iterations=20, epsilon=0.05,
                                       unused_score=1.0, distance_type="l2")),
        ("torch", TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                       unused_score=1.0, distance_type="l2"))):
    t = time.perf_counter()
    r = rtl.eval_seq(cache, params, cam, matcher)
    dt = time.perf_counter() - t
    res[name] = r
    print(f"{name}: ATE={r['ATE_median']:.8f} n_loop={r['n_loop']} "
          f"n_kf={r['n_kf']} s={dt:.1f}")

d = abs(res["numpy"]["ATE_median"] - res["torch"]["ATE_median"])
print(f"ATE abs diff = {d:.3e}")
print("EQUIVALENT (atol 1e-4):", d <= 1e-4
      and res["numpy"]["n_loop"] == res["torch"]["n_loop"])
