"""Compare the online-ified eval_seq against reference values (desk/desk2)."""
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

CACHE = REPO / "eval/results/tune_cache_loop"
for seq in ["desk", "desk2"]:
    c = rtl.load_cache(CACHE, seq)
    pkl = CACHE / f"match_cache_{seq}_torch.pkl"
    mc = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
    fx, fy, cx, cy = intrinsics_for("/home/ubuntu/datasets/tum_rgbd", seq, (525., 525., 320., 240.))
    cam = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=640, height=480)
    m = TorchSinkhornMatcher(iterations=20, epsilon=0.05, unused_score=1.0, distance_type="l2")
    r = rtl.eval_seq(c, dict(rtl.SEQ_OPT1_DEFAULTS), cam, m, mc)
    print(f"{seq}: ATE={r['ATE_median']:.5f} n_loop={r['n_loop']} n_kf={r['n_kf']}", flush=True)
