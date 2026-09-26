"""Parameter sensitivity around the Rustuna-tuned best config.

Evaluates a coarse grid (method=magsac, guided off) on cached sequences and
reports ATE_med per config, so the exact optimum can be rounded to simple
values without losing accuracy.

Run (cache must exist):
    .venv/bin/python eval/param_sensitivity.py --model eval/pyramid_k512_l2.onnx
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from rustuna_tune import load_all_cache, eval_seq  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

BASE = {"method": "magsac", "threshold": 1.4294175439754488, "dbin": 0.11538520396791509,
        "guided_inlier_thresh": 0.0, "guided_sampson": 4.0}

# coarse grid around the optimum: threshold x dbin
GRID = [
    (1.0, 0.1), (1.2, 0.1), (1.4, 0.1), (2.0, 0.1),
    (1.0, 0.12), (1.2, 0.12), (1.4, 0.12), (1.5, 0.12), (2.0, 0.12),
    (1.0, 0.15), (1.2, 0.15), (1.4, 0.15), (2.0, 0.15),
    (1.4, 0.3),  # old dbin 0.3 for reference
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset-root", default="/home/ubuntu/datasets/tum_rgbd")
    ap.add_argument("--seq", default="all")
    ap.add_argument("--cache-dir", default="eval/results/tune_cache")
    ap.add_argument("--fx", type=float, default=525.0)
    ap.add_argument("--fy", type=float, default=525.0)
    ap.add_argument("--cx", type=float, default=320.0)
    ap.add_argument("--cy", type=float, default=240.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default="eval/results/param_sensitivity.json")
    args = ap.parse_args()

    seqs = ["desk", "desk2", "room"] if args.seq == "all" else [args.seq]
    cache = load_all_cache(args.cache_dir, seqs)
    cam = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                           width=args.width, height=args.height)

    configs = [("best(exact)", BASE)]
    configs += [
        (f"magsac thr{thr} dbin{dbin}",
         {"method": "magsac", "threshold": thr, "dbin": dbin,
          "guided_inlier_thresh": 0.0, "guided_sampson": 4.0})
        for (thr, dbin) in GRID
    ]
    rows = []
    for name, params in configs:
        t0 = time.time()
        vals = {seq: eval_seq(cache[seq], params, cam) for seq in seqs}
        med = [vals[seq]["ATE_median"] for seq in seqs]
        okr = [vals[seq]["n_ok"] / vals[seq]["n_pairs"] for seq in seqs]
        row = {
            "config": name, "params": params,
            "ATE_med": {seq: vals[seq]["ATE_median"] for seq in seqs},
            "mean": float(np.mean(med)), "max": float(np.max(med)),
            "ok_rate": {seq: okr[i] for i, seq in enumerate(seqs)},
        }
        rows.append(row)
        per = " ".join(f"{seq}={med[i]:.3f}/{okr[i]:.2f}" for i, seq in enumerate(seqs))
        print(f"{name:24s} mean={row['mean']:.3f} max={row['max']:.3f} {per}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
