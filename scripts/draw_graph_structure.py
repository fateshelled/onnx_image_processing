"""Draw the current (odom_ref=kf additive) pose-graph topology and the
proposed local-map topology, using the real desk keyframe placement.

Usage: .venv/bin/python scripts/draw_graph_structure.py
"""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from vo.trajectory import Trajectory  # noqa: E402

# Room-best kf parameters (trial 333) drive the keyframe placement.
KF_TRANS, KF_ROT, KF_MIN_GAP, KF_MAX_GAP = 11.265, 40.93, 4, 16
STRIDE = 2

cache = np.load(REPO / "eval/results/tune_cache_loop/desk.npz",
                allow_pickle=True)
odom = list(cache["odom"])

traj = Trajectory()
node_poses = {0: traj.get_current_pose().copy()}
kf_nodes = [0]
last_kf = 0
chain, kf_edges = [], []
for n, o in enumerate(odom):
    a, i = n * STRIDE, (n + 1) * STRIDE
    if not o["ok"]:
        node_poses[i] = node_poses[a].copy()
    else:
        traj.add_relative_pose(o["R"], o["t"])
        node_poses[i] = traj.get_current_pose().copy()
    chain.append((a, i))
    if last_kf != i - STRIDE:
        kf_edges.append((last_kf, i))
    dT = np.linalg.inv(node_poses[i]) @ node_poses[last_kf]
    trans = float(np.linalg.norm(dT[:3, 3]))
    c = (np.trace(dT[:3, :3]) - 1.0) / 2.0
    rot = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    gap = i - last_kf
    if (gap >= KF_MAX_GAP
            or (gap >= KF_MIN_GAP and (trans >= KF_TRANS or rot >= KF_ROT))):
        last_kf = i
        kf_nodes.append(i)

kf_set = set(kf_nodes)
last_kf_of = {}
lk = 0
for n, o in enumerate(odom):
    i = (n + 1) * STRIDE
    if lk != i - STRIDE:
        last_kf_of[i] = lk
    # replicate promotion
    dT = np.linalg.inv(node_poses[i]) @ node_poses[lk]
    trans = float(np.linalg.norm(dT[:3, 3]))
    cc = (np.trace(dT[:3, :3]) - 1.0) / 2.0
    rot = float(np.degrees(np.arccos(np.clip(cc, -1.0, 1.0))))
    if (i - lk >= KF_MAX_GAP
            or (i - lk >= KF_MIN_GAP and (trans >= KF_TRANS or rot >= KF_ROT))):
        lk = i

deg = Counter(e[0] for e in kf_edges)
print(f"nodes={len(node_poses)} chain={len(chain)} additive_kf_edges={len(kf_edges)} "
      f"keyframes={len(kf_nodes)} max_hub_degree={max(deg.values())}")
print("top hub (last_kf) degrees:", deg.most_common(5))
print("first keyframes:", kf_nodes[:6])

# Show only ~3 keyframes so the plot stays readable.
if len(kf_nodes) >= 4:
    WINDOW = kf_nodes[3]
elif len(kf_nodes) >= 2:
    WINDOW = kf_nodes[-1] + STRIDE
else:
    WINDOW = 60
print(f"WINDOW={WINDOW} (keyframes shown: "
      f"{[k for k in kf_nodes if k < WINDOW]})")

nodes = list(range(0, min(WINDOW, max(node_poses) + 1), STRIDE))


def arcs(ax, edges, color, rad, lw=0.8, alpha=0.8, z=1):
    for a, b in edges:
        if b >= WINDOW:
            continue
        ax.add_patch(FancyArrowPatch(
            (a, 0), (b, 0), connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-", color=color, lw=lw, alpha=alpha, zorder=z))


fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# (1) odom_ref=prev baseline: chain + loop candidate window (schematic)
ax = axes[0]
for a, b in chain:
    if b < WINDOW:
        ax.plot([a, b], [0, 0], color="#999999", lw=1.0, zorder=1)
loops = [(kf_nodes[i], kf_nodes[j]) for i in range(len(kf_nodes))
         for j in range(max(0, i - 6), i) if kf_nodes[i] - kf_nodes[j] >= 30
         and kf_nodes[i] < WINDOW]
arcs(ax, loops, "#1f77b4", rad=0.28, lw=0.8, alpha=0.5)
ax.scatter([k for k in kf_nodes if k < WINDOW], [0] * len(
    [k for k in kf_nodes if k < WINDOW]), s=60, marker="o", color="#d62728",
    zorder=3, label="keyframe")
ax.scatter(nodes, [0] * len(nodes), s=8, color="#444444", zorder=2)
ax.set_title("(A) odom_ref=prev  (baseline): consecutive chain + loop candidates "
             "(window; gate-dependent)", loc="left")

# (2) current odom_ref=kf additive: chain + additive keyframe edges (star)
ax = axes[1]
for a, b in chain:
    if b < WINDOW:
        ax.plot([a, b], [0, 0], color="#999999", lw=1.0, zorder=1)
arcs(ax, kf_edges, "#ff7f0e", rad=0.30, lw=0.7, alpha=0.55)
ax.scatter([k for k in kf_nodes if k < WINDOW], [0] * len(
    [k for k in kf_nodes if k < WINDOW]), s=60, marker="o", color="#d62728",
    zorder=3)
ax.scatter(nodes, [0] * len(nodes), s=8, color="#444444", zorder=2)
ax.set_title("(B) odom_ref=kf  (current, additive): every frame -> last keyframe "
             "(orange). Hub/star structure: one keyframe holds many edges.",
             loc="left")

# (3) proposed local map: window of recent keyframes + KF-KF skip edges
ax = axes[2]
for a, b in chain:
    if b < WINDOW:
        ax.plot([a, b], [0, 0], color="#999999", lw=1.0, zorder=1)
local_edges = []
K_RECENT = 3
for n, o in enumerate(odom):
    i = (n + 1) * STRIDE
    if i >= WINDOW:
        continue
    recent = [k for k in kf_nodes if k <= i][-K_RECENT:]
    for k in recent:
        if k != i - STRIDE:
            local_edges.append((k, i))
arcs(ax, local_edges, "#ff7f0e", rad=0.30, lw=0.6, alpha=0.45)
# keyframe-to-keyframe skip edges (adjacent + every 2nd)
kf_skip = [(kf_nodes[i], kf_nodes[j]) for i in range(len(kf_nodes))
           for j in (i - 1, i - 2) if j >= 0 and kf_nodes[i] < WINDOW]
arcs(ax, kf_skip, "#2ca02c", rad=-0.35, lw=1.0, alpha=0.8)
ax.scatter([k for k in kf_nodes if k < WINDOW], [0] * len(
    [k for k in kf_nodes if k < WINDOW]), s=60, marker="o", color="#d62728",
    zorder=3)
ax.scatter(nodes, [0] * len(nodes), s=8, color="#444444", zorder=2)
ax.set_title("(C) proposed local map: each frame -> last 3 keyframes (orange) + "
             "keyframe-keyframe skip edges (green). Richer conditioning, no hub.",
             loc="left")

for ax in axes:
    for k in kf_nodes:
        if k < WINDOW:
            ax.annotate(f"KF{k}", (k, 0.10), ha="center", va="bottom",
                        fontsize=9, color="#d62728", fontweight="bold")
for k in kf_nodes:
    if k < WINDOW:
        axes[1].annotate(f"deg={deg.get(k, 0)}", (k, -0.12), ha="center",
                         va="top", fontsize=8, color="#ff7f0e")

for ax in axes:
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlim(-5, WINDOW)
axes[2].set_xlabel("frame / node index (stride=2)")

legend = [Line2D([0], [0], color="#999999", lw=1.5, label="chain edge (i-2 -> i)"),
          Line2D([0], [0], color="#ff7f0e", lw=1.5, label="additive KF edge / local KF edge"),
          Line2D([0], [0], color="#2ca02c", lw=1.5, label="KF-KF skip edge"),
          Line2D([0], [0], color="#1f77b4", lw=1.5, label="loop candidate"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                 markersize=9, label="keyframe"),
          Line2D([0], [0], marker="o", color="w", markerfacecolor="#444444",
                 markersize=5, label="frame node")]
fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False)
fig.suptitle("TUM fr1 desk pose-graph topology (real keyframe placement, "
             f"first {WINDOW} nodes; kf params = room-best trial333)", y=1.02)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = REPO.parent / "notes/20260919-kfref-graph-structure.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
