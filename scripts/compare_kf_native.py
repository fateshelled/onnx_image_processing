"""desk: compare KF-only ATE vs all-frame (native) ATE for legacy and online."""
import importlib.util, os, pickle, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
spec = importlib.util.spec_from_file_location("rtl", REPO / "eval/rustuna_tune_loop.py")
rtl = importlib.util.module_from_spec(spec); spec.loader.exec_module(rtl)
from eval_tum_vo import intrinsics_for
from torch_sinkhorn import TorchSinkhornMatcher
from vo.pose_estimation import CameraIntrinsics
from vo.online_graph import OnlinePoseGraph

CACHE = REPO / "eval/results/tune_cache_loop"
c = rtl.load_cache(CACHE, "desk")
stride = int(c["stride"])
mc = pickle.load(open(CACHE / "match_cache_desk_torch.pkl", "rb"))
fx, fy, cx, cy = intrinsics_for("/home/ubuntu/datasets/tum_rgbd", "desk", (525., 525., 320., 240.))
cam = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=640, height=480)
dm = TorchSinkhornMatcher(iterations=20, epsilon=0.05, unused_score=1.0, distance_type="l2")
gt_pos = c["gt_pos"]

def ate(est, gt):
    s, R, t = rtl.umeyama(est, gt, with_scale=True)
    return float(np.median(np.linalg.norm(s * (est @ R.T) + t - gt, axis=1)))

# legacy (diag provides ate_kf and native)
os.environ["LEGACY_KF_PRIOR"] = "1"; os.environ.pop("CAUSAL_KF_PRIOR", None)
diag = {}
res = rtl.eval_seq(c, {**rtl.SEQ_OPT1_DEFAULTS, "max_keyframes": None}, cam, dm, match_cache=mc, diag=diag)
print(f"LEGACY kf=all: native={res['ATE_median']:.5f} kf_only={diag.get('ate_kf')}")
os.environ.pop("LEGACY_KF_PRIOR", None)

# online
def match_fn(a, b):
    r = mc.get((a, b))
    return r if r is not None else {"ok": False, "n_matches": 0, "inlier_ratio": 0.0}
g = OnlinePoseGraph({**rtl.SEQ_OPT1_DEFAULTS, "max_keyframes": None}, cam, match_fn)
g.add_frame(0); keys = [0]
for n, o in enumerate(c["odom"]):
    idx = (n + 1) * stride
    od = (o.get("R"), o.get("t"), float(o.get("inlier", 1.0))) if o.get("ok") else (None, None, 0.0)
    g.add_frame(idx, odom=od); keys.append(idx)
est_all = np.array([g.pose(k)[:3, 3] for k in keys]); gt_all = np.array([gt_pos[k // stride] for k in keys])
est_k = np.array([g._est[k][:3, 3] for k in g._kf]); gt_k = np.array([gt_pos[k // stride] for k in g._kf])
print(f"ONLINE kf=all: native={ate(est_all, gt_all):.5f} kf_only={ate(est_k, gt_k):.5f}")
