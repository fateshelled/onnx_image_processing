"""Plot optimized edge scale against GT baseline (independent of the optimizer),
which is the intuitive view: does the estimated edge length track true motion?

Usage: TORCH_THREADS=4 .venv/bin/python scripts/plot_scale_gt.py
"""

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

if os.environ.get("TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

spec = importlib.util.spec_from_file_location(
    "rtl", REPO / "eval/rustuna_tune_loop.py")
rtl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtl)

from vo.pose_estimation import CameraIntrinsics  # noqa: E402

base = json.load(open(REPO / "eval/results/rustuna_tune_loop_room_kf.json"))
params = dict(base["best_params"])
params["loop_iterations"] = 10
params["kf_local_map_k"] = 1
params["tsvd_ratio"] = 0.0
params["trans_gate_deg"] = 0.0
params["kf_edge_min_inlier"] = 0.0

cam = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0,
                       width=640, height=480)
matcher = rtl.NumpySinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")

SEQ = ["desk", "desk2", "room"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
out_summary = {}

for c, seq in enumerate(SEQ):
    cache = rtl.load_cache(REPO / "eval/results/tune_cache_loop", seq)
    pkl = REPO / f"eval/results/tune_cache_loop/match_cache_{seq}_torch.pkl"
    mc = {}
    if pkl.exists():
        with open(pkl, "rb") as f:
            mc = pickle.load(f)
    diag = {}
    r = rtl.eval_seq(cache, params, cam, matcher, mc, diag=diag)
    stride, gt = cache["stride"], cache["gt_pos"]
    npos, cols = diag["node_pos"], diag["scale_col"]

    rows = []
    for e, ((i, j), sc) in enumerate(zip(diag["edges"], diag["edge_scale"])):
        if cols[e] is None:
            continue
        gt_b = float(np.linalg.norm(gt[j // stride] - gt[i // stride]))
        rows.append((j, j - i, gt_b, sc))
    rows.sort()
    jj = np.array([x[0] for x in rows], float)
    gap = np.array([x[1] for x in rows], float)
    gtb = np.array([x[2] for x in rows], float)
    sc = np.array([x[3] for x in rows], float)
    kf = gap > stride
    keep = kf & (gtb > 0.05)  # avoid ratio blow-up for tiny baselines

    ax = axes[0, c]
    ax.scatter(gtb[~kf], sc[~kf], s=12, alpha=0.4, color="gray",
               label="chain edge")
    ax.scatter(gtb[keep], sc[keep], s=16, alpha=0.7, label="KF edge")
    if keep.sum() > 2:
        A = np.vstack([gtb[keep], np.zeros_like(gtb[keep])]).T
        slope = float(np.linalg.lstsq(A, sc[keep], rcond=None)[0][0])
        xs = np.linspace(0, gtb[keep].max(), 50)
        ax.plot(xs, slope * xs, "r--", lw=1.2,
                label=f"fit y={slope:.2f}·x")
        rho, p = spearmanr(gtb[keep], sc[keep])
        ax.set_title(f"{seq}: scale vs GT baseline  (ATE {r['ATE_median']:.3f})\n"
                     f"Spearman {rho:+.2f}, slope {slope:.2f}")
    ax.set_xlabel("GT baseline between the two frames [m]")
    ax.set_ylabel("optimized edge scale (norm. units)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax2 = axes[1, c]
    ratio = sc[keep] / gtb[keep]
    ax2.scatter(jj[keep], ratio, s=14, alpha=0.7)
    ax2.axhline(np.median(ratio), color="r", ls="--", lw=1.0,
                label=f"median {np.median(ratio):.2f}")
    ax2.set_title(f"{seq}: scale / GT baseline over time "
                  f"(CV {ratio.std()/ratio.mean():.2f})")
    ax2.set_xlabel("time (node id j)")
    ax2.set_ylabel("scale / GT baseline [1/m]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    out_summary[seq] = {"ATE": r["ATE_median"],
                        "slope": slope if keep.sum() > 2 else None,
                        "ratio_median": float(np.median(ratio)),
                        "ratio_cv": float(ratio.std() / ratio.mean())}

fig.suptitle("Estimated edge scale vs GT baseline (top) and the "
             "per-edge ratio over time (bottom); additive kf, single hub")
fig.tight_layout()
p = REPO.parent / "notes/20260919-scale-vs-gt.png"
fig.savefig(p, dpi=130, bbox_inches="tight")
print("wrote", p)
print(json.dumps(out_summary, indent=2))
