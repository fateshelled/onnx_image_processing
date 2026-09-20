"""Analyse the optimized per-edge scales of the additive-keyframe graph.

Questions: are the free edge scales ~constant, or do they vary per edge and
per frame? Do they correlate with camera motion (baseline / speed)? Would an
EMA over the scale sequence stabilise them?

Usage: TORCH_THREADS=4 .venv/bin/python scripts/analyze_scale.py
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
summary = {}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, seq in zip(axes, SEQ):
    cache = rtl.load_cache(REPO / "eval/results/tune_cache_loop", seq)
    pkl = REPO / f"eval/results/tune_cache_loop/match_cache_{seq}_torch.pkl"
    mc = {}
    if pkl.exists():
        with open(pkl, "rb") as f:
            mc = pickle.load(f)
    diag = {}
    r = rtl.eval_seq(cache, params, cam, matcher, mc, diag=diag)

    stride = cache["stride"]
    edges = diag["edges"]
    scales = diag["edge_scale"]
    cols = diag["scale_col"]
    npos = diag["node_pos"]
    gt = cache["gt_pos"]

    def pos(k):
        return np.array(npos[k], float)

    rows = []
    for e, ((i, j), sc) in enumerate(zip(edges, scales)):
        if cols[e] is None:
            continue  # gauge edge (scale fixed = 1)
        gap = j - i
        chain_b = float(np.linalg.norm(pos(j) - pos(i)))          # normalized
        gt_b = float(np.linalg.norm(gt[j // stride] - gt[i // stride]))  # m
        rows.append((j, gap, chain_b, gt_b, sc))

    rows.sort()  # time order by j
    arr = np.array([[r0[1], r0[2], r0[3], r0[4]] for r0 in rows])  # gap, cb, gtb, sc
    gap, cb, gtb, sc = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    chain = gap == stride
    kf = ~chain
    ts = np.array([r0[0] for r0 in rows], float)

    def stat(name, x):
        return (f"{name}: n={len(x)} mean={x.mean():.3f} med={np.median(x):.3f} "
                f"std={x.std():.3f} CV={x.std()/max(abs(x.mean()),1e-9):.3f} "
                f"min={x.min():.3f} max={x.max():.3f}")

    print("=" * 70)
    print(f"seq={seq} ATE={r['ATE_median']:.4f} edges={len(edges)} "
          f"free={len(sc)} (chain={int(chain.sum())}, kf={int(kf.sum())})")
    print(stat("scale(all) ", sc))
    if kf.sum():
        print(stat("scale(KF)  ", sc[kf]))
        print(stat("scale/gap(KF)", sc[kf] / gap[kf]))
        for nm, x in (("gap", gap[kf]), ("chain_base", cb[kf]), ("gt_base", gtb[kf])):
            rho, p = spearmanr(x, sc[kf])
            print(f"  spearman(scale_KF, {nm}) = {rho:+.3f} (p={p:.2e})")
        # lag-1 autocorrelation of the scale sequence in time order (KF only)
        s = sc[kf]
        if len(s) > 2:
            a = s[:-1] - s[:-1].mean()
            b = s[1:] - s[1:].mean()
            ac = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
            print(f"  lag-1 autocorr(scale_KF in time order) = {ac:+.3f}")
        # EMA smoothing error: how close is EMA(alpha) to the raw scale
        for alpha in (0.2, 0.5):
            ema = np.empty_like(s)
            ema[0] = s[0]
            for t in range(1, len(s)):
                ema[t] = alpha * s[t] + (1 - alpha) * ema[t - 1]
            rel = np.abs(ema - s) / np.maximum(np.abs(s), 1e-9)
            print(f"  EMA(alpha={alpha}): mean|rel diff|={rel.mean():.3f} "
                  f"std(EMA)={ema.std():.3f} vs std(raw)={s.std():.3f}")

    summary[seq] = {"ATE": r["ATE_median"], "n_free": int(len(sc)),
                    "scale_all_mean": float(sc.mean()),
                    "scale_all_std": float(sc.std()),
                    "scale_kf_mean": float(sc[kf].mean()) if kf.sum() else None,
                    "scale_kf_std": float(sc[kf].std()) if kf.sum() else None}

    ax.scatter(gap[kf], sc[kf], s=14, alpha=0.6, label="KF edge")
    ax.scatter(gap[chain], sc[chain], s=14, alpha=0.5, color="gray",
               label="chain edge")
    ax.set_xlabel("frame gap (node id difference)")
    ax.set_ylabel("optimized edge scale")
    ax.set_title(f"{seq} (ATE {r['ATE_median']:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("Optimized per-edge scales vs frame gap (additive kf, single hub)")
fig.tight_layout()
out = REPO.parent / "notes/20260919-scale-analysis.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("\nwrote", out)
json.dump(summary, open(REPO / "eval/results/scale_analysis.json", "w"), indent=2)
