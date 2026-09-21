"""Compare ONNX Sinkhorn vs PyTorch Sinkhorn on identical descriptors.

For each frame pair we take the pair model's descriptors and compute the
Sinkhorn probability matrix two ways: the model's own ``matching_probs``
(ONNX) and ``TorchSinkhornMatcher`` on the same descriptors. We then compare
the probability matrices, the extracted match sets, and the relative pose.
"""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from eval_tum_vo import read_path_file, resolve_dataset  # noqa: E402
from torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.onnx_matcher import extract_matches  # noqa: E402
from vo.pose_estimation import CameraIntrinsics, estimate_pose_ransac  # noqa: E402

SEQ = "desk"
STRIDE = 2
N_PAIRS = 40
base, _ = resolve_dataset("/home/ubuntu/datasets/tum_rgbd", SEQ)
rows = read_path_file(base / "rgb.txt")

sess = ort.InferenceSession(str(REPO / "eval/pyramid_k512_l2_wd.onnx"),
                            providers=["CPUExecutionProvider"])
in_names = [i.name for i in sess.get_inputs()]
out_names = [o.name for o in sess.get_outputs()]
oi = {"k1": 0, "k2": 1, "d1": 2, "d2": 3, "probs": 4}
for i, n in enumerate(out_names):
    nl = n.lower()
    for key, pat in (("k1", "keypoints1"), ("k2", "keypoints2"),
                     ("d1", "descriptors1"), ("d2", "descriptors2"),
                     ("probs", "matching")):
        if pat in nl:
            oi[key] = i
cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0,
                       width=640, height=480)
torch_m = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                               unused_score=1.0, distance_type="l2")


def pose(mk1, mk2):
    R, t, mask = estimate_pose_ransac(mk1, mk2, cam, ransac_threshold=1.4,
                                      method=cv2.USAC_MAGSAC)
    return R, t, (int(np.sum(mask)) if mask is not None else 0)


def angle_between(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))


p_diffs, n_diff, i_diff, t_diffs = [], [], [], []
for k in range(0, min(len(rows) - STRIDE, N_PAIRS * STRIDE), STRIDE):
    a = cv2.resize(cv2.imread(str(base / rows[k][1]), cv2.IMREAD_GRAYSCALE),
                   (640, 480)).astype(np.float32)[None, None]
    b = cv2.resize(cv2.imread(str(base / rows[k + STRIDE][1]),
                              cv2.IMREAD_GRAYSCALE),
                   (640, 480)).astype(np.float32)[None, None]
    outs = sess.run(None, {in_names[0]: a, in_names[1]: b})
    k1, k2, d1, d2, P_onnx = outs[oi["k1"]], outs[oi["k2"]], outs[oi["d1"]], \
        outs[oi["d2"]], outs[oi["probs"]]
    P_torch = torch_m.match_probs(d1[0], d2[0])
    Po = np.asarray(P_onnx)[0]
    Pt = np.asarray(P_torch)
    p_diffs.append(float(np.max(np.abs(Po - Pt))))

    mo = extract_matches(k1, k2, P_onnx, 0.1, 1024, 0.1)
    mt = extract_matches(k1, k2, Pt[None], 0.1, 1024, 0.1)
    n_diff.append(len(mo[0]) - len(mt[0]))
    Ro, to, io = pose(mo[0], mo[1])
    Rt, tt, it = pose(mt[0], mt[1])
    i_diff.append(io - it)
    if Ro is not None and Rt is not None:
        t_diffs.append(angle_between(to, tt))
    print(f"pair {k:4}: |dP|max={p_diffs[-1]:.2e} n_onnx={len(mo[0])} "
          f"n_torch={len(mt[0])} inl={io}/{it}", flush=True)

print(f"\nN={len(p_diffs)} pairs")
print(f"|P_onnx - P_torch| max: mean={np.mean(p_diffs):.2e} "
      f"max={np.max(p_diffs):.2e}")
print(f"match count diff (onnx-torch): mean={np.mean(n_diff):.2f} "
      f"max|.|={np.max(np.abs(n_diff))}")
print(f"inlier count diff (onnx-torch): mean={np.mean(i_diff):.2f} "
      f"max|.|={np.max(np.abs(i_diff))}")
print(f"relative translation direction diff [deg]: "
      f"mean={np.mean(t_diffs):.3f} max={np.max(t_diffs):.3f}")
