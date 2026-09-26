"""Run the online pose graph on a TUM sequence and report ATE vs ground truth.

Used to check that the online graph (vo.online_graph + vo.onnx_matcher) gives
a trajectory comparable to the offline eval (eval/rustuna_tune_loop.py). The
image pipeline mirrors the eval: grayscale, resized to the model input, and
processed every ``--stride`` frames.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from eval_tum_vo import (  # noqa: E402
    nearest_timestamp, quat_to_se3, read_path_file, read_tum_file,
    resolve_dataset, umeyama,
)
from vo.online_graph import DEFAULT_PARAMS, OnlinePoseGraph  # noqa: E402
from vo.onnx_matcher import OnnxSessionMatcher  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="desk")
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--model", default=str(REPO / "eval/pyramid_k512_l2_wd.onnx"))
ap.add_argument("--dataset-root", default="/home/ubuntu/datasets/tum_rgbd")
ap.add_argument("--max-frames", type=int, default=None)
args = ap.parse_args()

base, cam_id = resolve_dataset(args.dataset_root, args.seq)
gt_poses = [(ts, quat_to_se3(v[:3], v[3:7]))
            for ts, v in read_tum_file(base / "groundtruth.txt")]
rgb_rows = read_path_file(base / "rgb.txt")

# Model input size + intrinsics for this camera.
sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
in_names = [i.name for i in sess.get_inputs()]
out_names = [o.name for o in sess.get_outputs()]
h, w = sess.get_inputs()[0].shape[2], sess.get_inputs()[0].shape[3]
if cam_id == "freiburg1":
    fx, fy, cx, cy = 525.0, 525.0, 320.0, 240.0
elif cam_id == "freiburg2":
    fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
else:
    fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
cam = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)

oi = {"k1": 0, "k2": 1, "probs": 2}
for i, n in enumerate(out_names):
    nl = n.lower()
    if "keypoints1" in nl:
        oi["k1"] = i
    elif "keypoints2" in nl:
        oi["k2"] = i
    elif "matching" in nl or "probs" in nl:
        oi["probs"] = i

matcher = OnnxSessionMatcher(sess, cam, in_names, oi)
images = {}


def match_fn(i, j):
    if i not in images or j not in images:
        return {"ok": False, "n_matches": 0, "inlier_ratio": 0.0}
    return matcher.match(images[i], images[j])


graph = OnlinePoseGraph(dict(DEFAULT_PARAMS), cam, match_fn)

est, gt = [], []
t0 = time.time()
idx = 0
for k in range(0, len(rgb_rows), args.stride):
    ts, rel = rgb_rows[k]
    gm = nearest_timestamp(gt_poses, ts, 0.05)
    if gm is None:
        continue
    img = cv2.imread(str(base / rel), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    images[idx] = cv2.resize(img, (w, h)).astype(np.float32)[None, None]
    T = graph.add_frame(idx)
    keep = set(graph.match_keyframes) | {idx - 1, idx}
    for m in [m for m in images if m not in keep]:
        images.pop(m, None)
    est.append(T[:3, 3])
    gt.append(gm[1][:3, 3])
    idx += 1
    if args.max_frames is not None and len(est) >= args.max_frames:
        break

est = np.array(est)
gt = np.array(gt)
s, R, t = umeyama(est, gt, with_scale=True)
aligned = s * (est @ R.T) + t
ate = float(np.median(np.linalg.norm(aligned - gt, axis=1)))
print(f"{args.seq}: frames={len(est)} ATE_median={ate:.5f} "
      f"({time.time()-t0:.0f}s, {len(est)/(time.time()-t0):.1f} fps)")
