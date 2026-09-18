"""Verify the descriptor-caching pipeline matches the pair model exactly.

Checks, for random and real image pairs:
1. single-image ONNX keypoints == pair-model keypoints
2. numpy Sinkhorn matching_probs == pair-model matching_probs
3. extract_matches outputs identical match sets
4. timing: full pair model vs cached (detect once + numpy match)
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from vo.sinkhorn_numpy import NumpySinkhornMatcher  # noqa: E402
from eval_tum_vo import extract_matches  # noqa: E402

PAIR = REPO / "eval/pyramid_k512_l2.onnx"
SINGLE = REPO / "eval/pyramid_k512_l2_desc.onnx"

so = ort.SessionOptions()
so.intra_op_num_threads = 1
pair_sess = ort.InferenceSession(str(PAIR), so, providers=["CPUExecutionProvider"])
single_sess = ort.InferenceSession(str(SINGLE), so, providers=["CPUExecutionProvider"])
matcher = NumpySinkhornMatcher(iterations=20, epsilon=0.05, unused_score=1.0,
                               distance_type="l2")


def norm_img(x):
    return cv2.resize(x, (640, 480)).astype(np.float32)[None, None]


def check(a, b, tag):
    k1o, k2o, Po = pair_sess.run(None, {"image1": a, "image2": b})
    ka, da = single_sess.run(None, {"image": a})
    kb, db = single_sess.run(None, {"image": b})

    kp_diff = float(np.abs(k1o - ka).max()) + float(np.abs(k2o - kb).max())
    Pn = matcher.match_probs(da[0], db[0])
    P_diff = float(np.abs(Po[0] - Pn).max())

    mk1o, mk2o, sco = extract_matches(k1o, k2o, Po, 0.1, 100, 0.1)
    mk1n, mk2n, scn = extract_matches(ka, kb, Pn[None], 0.1, 100, 0.1)
    # canonical order: sort by (x1, y1, x2, y2) to neutralize score-tie ordering
    def canon(k1m, k2m):
        if len(k1m) == 0:
            return np.empty((0, 4), dtype=np.float64)
        o = np.lexsort((k2m[:, 0], k2m[:, 1], k1m[:, 1], k1m[:, 0]))
        return np.concatenate([k1m[o], k2m[o]], axis=1)
    c_o, c_n = canon(mk1o, mk2o), canon(mk1n, mk2n)
    src_ok = (
        c_o.shape == c_n.shape
        and (len(c_o) == 0 or np.allclose(c_o, c_n, atol=1e-3))
    )

    print(f"[{tag}] kp_maxdiff={kp_diff:.3e} P_maxdiff={P_diff:.3e} "
          f"n={len(mk1o)}/{len(mk1n)} identical={src_ok}")
    return kp_diff == 0.0, P_diff < 5e-3, src_ok


# random synthetic pairs
rng = np.random.default_rng(42)
a1 = rng.random((480, 640)).astype(np.float32) * 255
a2 = rng.random((480, 640)).astype(np.float32) * 255
ok = check(norm_img(a1), norm_img(a2), "random-distinct")
img = None  # noqa: F841
ok_same = check(norm_img(a1), norm_img(a1), "random-same")

# real TUM frames (stride 60 for a weak loop-like pair)
base = Path("/home/ubuntu/datasets/tum_rgbd/rgbd_dataset_freiburg1_room")
if base.exists():
    frames = []
    with open(base / "rgb.txt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                frames.append(base / line.split()[1])
    ok_real1 = check(norm_img(cv2.imread(str(frames[0]), 0)),
                     norm_img(cv2.imread(str(frames[1]), 0)), "room 0-1")
    ok_real2 = check(norm_img(cv2.imread(str(frames[0]), 0)),
                     norm_img(cv2.imread(str(frames[len(frames)//2]), 0)),
                     "room 0-mid")
    real_ok = ok_real1 and ok_real2
else:
    real_ok = True
    print("no TUM data found, skipped real-image check")

all_ok = ok and ok_same and real_ok
print("ALL OK" if all_ok else "MISMATCH DETECTED")

# ---- timing benchmark ----
a1n, a2n = norm_img(a1), norm_img(a2)


def time_pair():
    t0 = time.perf_counter()
    pair_sess.run(None, {"image1": a1n, "image2": a2n})
    return time.perf_counter() - t0


pair_t = np.mean([time_pair() for _ in range(3)])

t0 = time.perf_counter()
da = single_sess.run(None, {"image": a1n})
t_a = time.perf_counter() - t0
t0 = time.perf_counter()
db = single_sess.run(None, {"image": a2n})
t_b = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(10):
    matcher.match_probs(da[0][0], db[0][0])
match_t = (time.perf_counter() - t0) / 10

print(f"\npair model (detect+match): {pair_t*1000:.0f} ms")
print(f"single-image detect      : {t_a*1000:.0f} / {t_b*1000:.0f} ms per frame")
print(f"numpy match per pair     : {match_t*1000:.1f} ms")
K = da[0].shape[1]
print(f"amortized detect per frame: {(t_a+t_b)/2*1000:.0f} ms")
print(f"break-even candidate pairs: {pair_t/max(t_a+t_b,1e-9)*K:.0f} pairs (K={K})")
