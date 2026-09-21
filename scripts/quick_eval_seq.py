"""Quick single-sequence online-ified eval (for A/B debugging)."""
import argparse
import importlib.util
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
ap.add_argument("--max-kf", default=None,
                help="override max_keyframes (int, or 'all'/None for unbounded)")
args = ap.parse_args()
params = dict(rtl.SEQ_OPT1_DEFAULTS)
if args.max_kf is not None:
    params["max_keyframes"] = (None if args.max_kf in ("all", "none")
                               else int(args.max_kf))

CACHE = REPO / "eval/results/tune_cache_loop"
c = rtl.load_cache(CACHE, args.seq)
pkl = CACHE / f"match_cache_{args.seq}_torch.pkl"
mc = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
fx, fy, cx, cy = intrinsics_for("/home/ubuntu/datasets/tum_rgbd", args.seq, (525., 525., 320., 240.))
cam = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=640, height=480)
m = TorchSinkhornMatcher(iterations=20, epsilon=0.05, unused_score=1.0, distance_type="l2")
r = rtl.eval_seq(c, params, cam, m, mc)
print(f"{args.seq}: ATE={r['ATE_median']:.5f} n_loop={r['n_loop']} n_kf={r['n_kf']}", flush=True)
