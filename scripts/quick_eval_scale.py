"""Single-sequence online eval with scale_kf (Kalman filter) toggled."""

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))
spec = importlib.util.spec_from_file_location("rtl", REPO / "eval/rustuna_tune_loop.py")
rtl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtl)
from eval_tum_vo import intrinsics_for  # noqa: E402
from torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="desk")
ap.add_argument("--scale-kf", type=int, default=1)
ap.add_argument("--overrides", default="{}", help="JSON dict of param overrides")
args = ap.parse_args()

CACHE = REPO / "eval/results/tune_cache_loop"
c = rtl.load_cache(CACHE, args.seq)
pkl = CACHE / f"match_cache_{args.seq}_torch.pkl"
mc = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
fx, fy, cx, cy = intrinsics_for("/home/ubuntu/datasets/tum_rgbd", args.seq,
                                (525., 525., 320., 240.))
cam = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=640, height=480)
matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05, unused_score=1.0,
                               distance_type="l2")
params = dict(rtl.SEQ_OPT1_DEFAULTS)
params["scale_kf"] = bool(args.scale_kf)
params.update(json.loads(args.overrides))
r = rtl.eval_seq(c, params, cam, matcher, mc)
tag = "scale_kf=ON" if args.scale_kf else "scale_kf=OFF"
print(f"{args.seq} {tag}: ATE={r['ATE_median']:.5f} n_loop={r['n_loop']} "
      f"n_kf={r['n_kf']} "
      f"robust_downweight_samples={r.get('n_robust_downweighted', 0)} "
      f"robust_rot={r.get('n_robust_rot_downweighted', 0)} "
      f"robust_dir={r.get('n_robust_dir_downweighted', 0)} "
      f"overrides={args.overrides}", flush=True)
