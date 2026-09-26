"""Scatter of ATE vs RPE for the ATE-RPE A/B (plus existing study context).

Reads eval/results/ab_ate_rpe_joint.json (dev means), and optionally the
loop-enabled study (tune_rpe.db / rpe_seq_opt1) for context points. Produces
eval/results/ab_ate_rpe_joint.png.
"""

import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna

REPO = Path(__file__).resolve().parents[1]
AB = REPO / "eval/results/ab_ate_rpe_joint.json"
OUT = REPO / "eval/results/ab_ate_rpe_joint.png"
REF = (0.2234, 0.3754)  # baseline ATE mean, RPE2m mean


def mean_ab():
    d = json.loads(AB.read_text())
    by = defaultdict(lambda: {"rpe": [], "ate": [], "nloop": []})
    for k, v in d.items():
        if ":" not in k:
            continue
        name, seq = k.split(":", 1)
        r, a = v.get("RPE_trans_2m_median"), v.get("ATE_median")
        if r is None or a is None or not math.isfinite(r) or not math.isfinite(a):
            continue
        by[name]["rpe"].append(r)
        by[name]["ate"].append(a)
        by[name]["nloop"].append(v.get("n_loop"))
    out = {}
    for name, x in by.items():
        out[name] = {
            "rpe": sum(x["rpe"]) / len(x["rpe"]),
            "ate": sum(x["ate"]) / len(x["ate"]),
            "nloop": x["nloop"],
        }
    return out


def study_points(storage, study_name):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = optuna.load_study(study_name=study_name, storage=f"sqlite:///{storage}")
    pts = []
    for t in s.trials:
        if t.value is None:
            continue
        try:
            ps = json.loads(t.user_attrs["per_seq"])
            r = [ps[q]["RPE_trans_2m_median"] for q in ("desk", "desk2", "room")]
            a = [ps[q]["ATE_median"] for q in ("desk", "desk2", "room")]
        except (KeyError, ValueError):
            continue
        if any(v is None or not math.isfinite(v) for v in r + a):
            continue
        pts.append((sum(a) / 3, sum(r) / 3))
    return pts


def main():
    ab = mean_ab()
    ctx = []
    try:
        ctx = study_points(REPO / "eval/results/tune_rpe.db", "rpe_seq_opt1")
    except Exception as exc:  # context is optional
        print("context study unavailable:", exc)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # --- Panel A: A/B configs only -------------------------------------
    ax = axes[0]
    for name, v in ab.items():
        loop = any(n and n > 0 for n in v["nloop"])
        color = "tab:red" if name.startswith("t205") else "tab:blue"
        marker = "o" if loop else "s"
        ax.scatter(v["ate"], v["rpe"], s=90, c=color, marker=marker,
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(name, (v["ate"], v["rpe"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.scatter(*REF, marker="*", s=300, c="gold", edgecolor="black",
               linewidth=0.8, zorder=4, label="baseline (shipped default)")
    ax.set_xlabel("mean ATE_median [m]  (lower is better)")
    ax.set_ylabel("mean RPE_trans_2m_median [m]  (lower is better)")
    ax.set_title("A/B: odometry x loop x NL-Reg (dev desk/desk2/room)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.annotate("better", xy=(0.05, 0.05), xycoords="axes fraction",
                fontsize=10, color="green")

    # --- Panel B: A/B + loop-enabled study context ---------------------
    ax = axes[1]
    if ctx:
        cx = [p[0] for p in ctx]
        cy = [p[1] for p in ctx]
        ax.scatter(cx, cy, s=12, c="lightgray", label="loop-enabled study (401)")
    for name, v in ab.items():
        ax.scatter(v["ate"], v["rpe"], s=70,
                   c="tab:red" if name.startswith("t205") else "tab:blue",
                   edgecolor="black", linewidth=0.5, zorder=3)
    ax.scatter(*REF, marker="*", s=300, c="gold", edgecolor="black",
               linewidth=0.8, zorder=4, label="baseline")
    ax.set_xlabel("mean ATE_median [m]  (lower is better)")
    ax.set_ylabel("mean RPE_trans_2m_median [m]  (lower is better)")
    ax.set_title("Context: all loop-enabled trials + A/B")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("wrote", OUT)
    for name, v in sorted(ab.items(), key=lambda kv: kv[1]["ate"]):
        print(f"{name:14} ATE={v['ate']:.4f} RPE2m={v['rpe']:.4f} nloop={v['nloop']}")


if __name__ == "__main__":
    main()
