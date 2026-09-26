"""Synthetic parity test: legacy `_sequential_kf_priors` vs `OnlinePoseGraph`.

Exact relative measurements, identical graph (chain + spokes + loops), so any
difference in the final keyframe poses is a driver-logic difference (node set,
warm start, NL-Reg anchor, edge parameters, propagation), not matching noise.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

spec = importlib.util.spec_from_file_location("rtl", REPO / "eval/rustuna_tune_loop.py")
rtl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtl)
from vo.online_graph import OnlinePoseGraph  # noqa: E402
from vo.se3 import se3_exp  # noqa: E402
from vo.se3_window import SlidingWindowOptimizer  # noqa: E402

STRIDE, KF_GAP, LOOP_GAP = 1, 5, 30
rng = np.random.default_rng(0)
N = 60
GT = [np.eye(4)]
for _ in range(N - 1):
    GT.append(GT[-1] @ se3_exp(rng.normal(size=6) * 0.05))


def rel(i, j):
    R = GT[j][:3, :3].T @ GT[i][:3, :3]
    t = GT[j][:3, :3].T @ (GT[i][:3, 3] - GT[j][:3, 3])
    return R, t


def m(i, j):
    R, t = rel(i, j)
    return {"ok": True, "R": R, "t": t, "inlier_ratio": 1.0, "n_matches": 300}


def base_params(nl_reg):
    p = dict(rtl.SEQ_OPT1_DEFAULTS)
    p.update(nl_reg=nl_reg, max_keyframes=None, kf_max_gap=KF_GAP,
             kf_min_gap=KF_GAP, kf_trans_thresh=0.0, kf_rot_thresh=0.0,
             loop_min_gap=LOOP_GAP, loop_window=1000, odom_ref="kf")
    return p


def build_opt(params):
    keys = list(range(N))
    kf = keys[::KF_GAP]
    opt = SlidingWindowOptimizer(
        max_iterations=params["loop_iterations"], huber=1.0,
        step_scale_t=params["step_scale_t"], step_scale_r=params["step_scale_t"],
        optimize_scale=True, scale_prior_sigma=params["scale_prior_sigma"],
        tsvd_ratio=0.0)
    for k in keys:
        opt.add_node(k, GT[k].copy())
    added = set()
    last_kf = 0
    for i in keys[STRIDE:]:
        R, t = rel(i - STRIDE, i)
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
        opt.add_edge(i - STRIDE, i, M, scale_free=True)
        added.add((i - STRIDE, i))
        if i != last_kf:
            R, t = rel(last_kf, i)
            M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
            opt.add_edge(last_kf, i, M, scale_free=True)
            added.add((last_kf, i))
        if i - last_kf >= KF_GAP:
            last_kf = i
    # loops (KF-KF, frame gap >= LOOP_GAP)
    for ai, a in enumerate(kf):
        for b in kf[ai + 1:]:
            if b - a < LOOP_GAP or (a, b) in added:
                continue
            R, t = rel(a, b)
            M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
            opt.add_edge(a, b, M, scale_free=True)
    return opt, kf, keys


def ate(poses, idx, gt):
    est = np.array([poses[k][:3, 3] for k in idx])
    g = np.array([gt[k][:3, 3] for k in idx])
    s, R, t = rtl.umeyama(est, g, with_scale=True)
    return float(np.median(np.linalg.norm(s * (est @ R.T) + t - g, axis=1)))


for nl_reg in (True, False):
    params = base_params(nl_reg)
    # legacy
    opt, kf, keys = build_opt(params)
    os.environ["LEGACY_KF_PRIOR"] = "1"  # not needed here; called directly
    red = rtl._sequential_kf_priors(opt, kf, keys, params)
    leg = {k: red.get_pose(k) for k in keys if k in red.pose_ids}
    leg_kf = np.array([leg[k][:3, 3] for k in kf])

    # online
    os.environ.pop("CAUSAL_KF_PRIOR", None)
    g = OnlinePoseGraph(params, None, m)
    for k in keys:
        R, t = rel(k - STRIDE, k)
        od = (R, t, 1.0)
        g.add_frame(k, odom=od)
    on_kf = np.array([g.pose(k)[:3, 3] for k in kf])
    on_all = np.array([g.pose(k)[:3, 3] for k in keys])
    diff_kf = float(np.max(np.linalg.norm(leg_kf - on_kf, axis=1)))

    print(f"nl_reg={nl_reg}: max|legacy_online| KF pos = {diff_kf:.6f}  "
          f"(n_kf legacy={len(kf)} online={g.n_kf})")
    print(f"   ATE_kf legacy={ate(leg, kf, GT):.5f} online={ate({k: g.pose(k) for k in keys}, kf, GT):.5f}")
os.environ.pop("LEGACY_KF_PRIOR", None)
