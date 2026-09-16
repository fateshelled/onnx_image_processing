#!/usr/bin/env python3
"""
TUM RGB-D VO evaluation harness for the Shi-Tomasi+Angle+SparseBAD+Sinkhorn
matcher (incl. pyramid variant).

Modes:
  vo      : end-to-end VO -> trajectory -> Umeyama(SE3, scaled) align to GT -> ATE/RPE.
            pose-source = essential uses monocular RANSAC/MAGSAC;
            pose-source = rgbd-pnp uses registered depth for metric PnP-RANSAC.
  gtcheck : per-pair Sampson residual of extracted matches vs GT-derived fundamental matrix.
            Used to verify geometric consistency on long-baseline pairs (stride sweep).

Usage:
  python eval_tum_vo.py vo --model pyramid.onnx --seq all --stride 2 --method magsac --threshold 1.4
  python eval_tum_vo.py vo --model pyramid.onnx --seq all --stride 2 --pose-source rgbd-pnp --threshold 1.4
  python eval_tum_vo.py gtcheck --model pyramid.onnx --seq desk --stride 8
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root or eval/ dir
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import onnxruntime as ort  # noqa: E402

from pytorch_model.matching.outlier_filters import dustbin_margin_filter  # noqa: E402
from pytorch_model.vo.pose_estimation import (  # noqa: E402
    estimate_pose_ransac,
    estimate_pose_rgbd_pnp,
    CameraIntrinsics,
)
from pytorch_model.vo.trajectory import Trajectory  # noqa: E402
from pytorch_model.vo.se3_window import SlidingWindowOptimizer  # noqa: E402
from eval.sampson_all import sampson_all, build_F_from_pose  # noqa: E402


# --------------------------------------------------------------------------
# TUM dataset
# --------------------------------------------------------------------------
def read_tum_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            t = float(parts[0])
            vals = [float(x) for x in parts[1:8]]
            rows.append((t, vals))
    return rows


def read_path_file(path):
    """Read a TUM timestamp/path index such as rgb.txt or depth.txt."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), parts[1]))
    return rows


def nearest_timestamp(rows, timestamp, max_diff):
    """Return the nearest ``(timestamp, value)`` row within ``max_diff``."""
    if not rows:
        return None
    timestamps = np.fromiter((row[0] for row in rows), dtype=np.float64)
    pos = int(np.searchsorted(timestamps, timestamp))
    candidates = []
    if pos < len(rows):
        candidates.append(rows[pos])
    if pos > 0:
        candidates.append(rows[pos - 1])
    best = min(candidates, key=lambda row: abs(row[0] - timestamp))
    return best if abs(best[0] - timestamp) <= max_diff else None


def associate(rgb_rows, gt_rows, max_diff=0.02):
    """For each rgb frame, find nearest GT by timestamp within max_diff."""
    gt_ts = np.array([r[0] for r in gt_rows])
    pairs = []
    for (ts, _) in rgb_rows:
        idx = np.argmin(np.abs(gt_ts - ts))
        if abs(gt_ts[idx] - ts) <= max_diff:
            pairs.append((ts, gt_ts[idx]))
    return pairs


def quat_to_se3(t, q):
    """TUM quaternion (qx, qy, qz, qw) + translation (tx,ty,tz) -> 4x4."""
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(t)
    return T


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def extract_matches(kpts1, kpts2, P, threshold, max_matches, dbin_margin):
    """Mutual-NN + dustbin-margin filter + top-K. Returns (kpts1, kpts2, scores)."""
    P = P[0]  # (K+1, K+1)
    k1 = kpts1[0]  # (K, 2) (y, x)
    k2 = kpts2[0]
    K = k1.shape[0]
    Pc = P[:K, :K]
    max_j = np.argmax(Pc, axis=1)
    max_i = np.argmax(Pc, axis=0)
    mutual = np.arange(K) == max_i[max_j]
    scores = Pc[np.arange(K), max_j]

    dbin = dustbin_margin_filter(P, dbin_margin)  # mask over K source pts
    pad = (k1[:, 0] >= 0) & (k1[:, 1] >= 0) & (k2[:, 0] >= 0) & (k2[:, 1] >= 0)
    valid = mutual & dbin & pad & (scores >= threshold)

    idx_i = np.where(valid)[0]
    if len(idx_i) == 0:
        return k1[:0], k2[:0], np.array([])
    j = max_j[idx_i]
    sc = scores[idx_i]
    order = np.argsort(sc)[::-1][:max_matches]
    idx_i = idx_i[order]
    j = j[order]
    return k1[idx_i], k2[j], sc


# --------------------------------------------------------------------------
# Umeyama alignment (rigid + uniform scale), numpy
# --------------------------------------------------------------------------
def umeyama(X, Y, with_scale=True):
    """Align X onto Y: ``Y ~ s R X + t``. Optionally fix ``s=1``."""
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    var_x = np.mean(np.sum((X - mu_x) ** 2, axis=1))
    Xc = X - mu_x
    Yc = Y - mu_y
    Sigma = (Xc.T @ Yc) / X.shape[0]
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    s = np.trace(np.diag(D) @ S) / var_x if with_scale else 1.0
    t = mu_y - s * (R @ mu_x)
    return s, R, t


# --------------------------------------------------------------------------
# Pose accumulation / ATE
# --------------------------------------------------------------------------
def estimate_pair(session, img_path1, img_path2, cam, args, depth_path1=None):
    """Run model on a pair, extract matches, estimate pose. Returns dict."""
    a = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return None
    a = cv2.resize(a, (cam.width, cam.height)).astype(np.float32)[None, None]
    b = cv2.resize(b, (cam.width, cam.height)).astype(np.float32)[None, None]

    k1, k2, P = session.run(None, {"image1": a, "image2": b})
    mk1, mk2, sc = extract_matches(k1, k2, P, args.match_threshold,
                                   args.max_matches, args.dbin)

    res = {"n_matches": len(mk1), "ok": False}
    if len(mk1) < 5:
        return res

    if args.pose_source == "rgbd-pnp":
        if depth_path1 is None:
            return res
        depth = cv2.imread(str(depth_path1), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return res
        if depth.shape != (cam.height, cam.width):
            depth = cv2.resize(
                depth, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST
            )
        R, t, mask = estimate_pose_rgbd_pnp(
            mk1,
            mk2,
            depth,
            cam,
            depth_scale=args.depth_scale,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            ransac_threshold=args.threshold,
        )
    else:
        method = cv2.USAC_MAGSAC if args.method == "magsac" else cv2.RANSAC
        R, t, mask = estimate_pose_ransac(
            mk1,
            mk2,
            cam,
            ransac_threshold=args.threshold,
            method=method,
        )
    if R is None:
        return res

    inlier_ratio = float(np.sum(mask) / len(mk1)) if len(mk1) else 0.0
    res.update({"ok": True, "R": R, "t": t, "mask": mask,
                "inlier_ratio": inlier_ratio, "n_inliers": int(np.sum(mask))})

    # Guided retry: low inlier ratio -> Sampson re-selection with R1's E
    if (
        args.pose_source == "essential"
        and args.guided
        and inlier_ratio < args.guided_inlier_thresh
    ):
        K = cam.K
        E = _cross(t.ravel()) @ R
        F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
        h1 = np.concatenate([mk1, np.ones((len(mk1), 1))], axis=1)
        h2 = np.concatenate([mk2, np.ones((len(mk2), 1))], axis=1)
        d = sampson_all(h1, h2, F)
        keep = d < args.guided_sampson
        if keep.sum() >= 5:
            R2, t2, mask2 = estimate_pose_ransac(mk1[keep], mk2[keep], cam,
                                                 ransac_threshold=args.threshold,
                                                 method=method)
            if R2 is not None:
                ir2 = float(np.sum(mask2) / len(mk1[keep]))
                res.update({"R": R2, "t": t2, "mask": mask2,
                            "inlier_ratio": ir2, "n_inliers": int(np.sum(mask2)),
                            "guided": True})
    return res


def _cross(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def relative_from_gt(gt_se3_a, gt_se3_b):
    return np.linalg.inv(gt_se3_a) @ gt_se3_b


# --------------------------------------------------------------------------
# VO mode
# --------------------------------------------------------------------------
def run_vo(args):
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    cam = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                          width=args.width, height=args.height)
    seqs = ["desk", "desk2", "room"] if args.seq == "all" else [args.seq]
    summary = []
    for seq in seqs:
        base = Path(args.dataset_root) / f"rgbd_dataset_freiburg1_{seq}"
        gt_rows = read_tum_file(base / "groundtruth.txt")
        frame_list = read_path_file(base / "rgb.txt")
        depth_rows = (
            read_path_file(base / "depth.txt")
            if args.pose_source == "rgbd-pnp"
            else []
        )
        gt_poses = [
            (ts, quat_to_se3(vals[:3], vals[3:7])) for ts, vals in gt_rows
        ]
        frames = []
        for ts, relative_path in frame_list:
            gt_match = nearest_timestamp(gt_poses, ts, 0.05)
            if gt_match is None:
                continue
            depth_path = None
            if args.pose_source == "rgbd-pnp":
                depth_match = nearest_timestamp(depth_rows, ts, args.depth_max_diff)
                if depth_match is None:
                    continue
                depth_path = base / depth_match[1]
            frames.append((ts, base / relative_path, depth_path, gt_match[1]))

        n_loop = 0
        # process pairs at given stride
        trajectory = Trajectory()
        node_poses: dict[int, np.ndarray] = {0: trajectory.get_current_pose().copy()}
        frame_img: dict[int, str] = {0: str(frames[0][1])}
        odom: list[tuple] = []  # (i, j, R, t) for successful odometry edges
        est_pos = [trajectory.get_current_position().copy()]
        gt_pos = [frames[0][3][:3, 3].copy()]
        n_pairs = 0
        n_ok = 0
        inlier_ratios = []
        for i in range(0, len(frames) - args.stride, args.stride):
            ts_a, pa, da, gta = frames[i]
            ts_b, pb, db, gtb = frames[i + args.stride]
            res = estimate_pair(
                session=session,
                img_path1=str(pa),
                img_path2=str(pb),
                cam=cam,
                args=args,
                depth_path1=da,
            )
            n_pairs += 1
            if res is None or not res.get("ok"):
                # no pose update -> keep previous C (skip)
                est_pos.append(trajectory.get_current_position().copy())
                gt_pos.append(gtb[:3, 3].copy())
                node_poses[i + args.stride] = trajectory.get_current_pose().copy()
                frame_img[i + args.stride] = str(pb)
                continue
            R, t = res["R"], res["t"]
            trajectory.add_relative_pose(R, t)
            est_pos.append(trajectory.get_current_position().copy())
            gt_pos.append(gtb[:3, 3].copy())
            n_ok += 1
            inlier_ratios.append(res["inlier_ratio"])
            node_poses[i + args.stride] = trajectory.get_current_pose().copy()
            frame_img[i + args.stride] = str(pb)
            odom.append((i, i + args.stride, R, t))

        # Loop closure: build the full pose graph, detect loop edges by
        # re-matching keyframe pairs, and re-optimize. Appearance-based
        # detection (matcher inlier ratio) is used instead of a position
        # gate, because the monocular estimate has already drifted and a
        # position gate would miss true loops on these return-trajectories.
        if args.loop_closure:
            opt = SlidingWindowOptimizer(
                window_size=None,
                max_iterations=args.loop_iterations,
                huber=args.loop_huber,
            )
            for idx, T in node_poses.items():
                opt.add_node(idx, T)
            for (i, j, R, t) in odom:
                M = np.eye(4)
                M[:3, :3] = R
                M[:3, 3] = np.asarray(t, float).reshape(3)
                opt.add_edge(i, j, M)

            keys = sorted(node_poses.keys())
            kf = keys[:: max(1, args.keyframe_decim)]
            n_loop = 0
            for ai in range(len(kf)):
                for bi in range(ai + 1, len(kf)):
                    a, b = kf[ai], kf[bi]
                    if b - a < args.loop_min_gap:
                        continue
                    lres = estimate_pair(
                        session=session,
                        img_path1=frame_img[a],
                        img_path2=frame_img[b],
                        cam=cam,
                        args=args,
                        depth_path1=None,
                    )
                    if lres is None or not lres.get("ok"):
                        continue
                    if lres["inlier_ratio"] < args.loop_min_inlier:
                        continue
                    M = np.eye(4)
                    M[:3, :3] = lres["R"]
                    M[:3, 3] = np.asarray(lres["t"], float).reshape(3)
                    opt.add_edge(a, b, M)
                    n_loop += 1
            opt.optimize(verbose=args.loop_verbose)
            for idx in node_poses:
                node_poses[idx] = opt.get_pose(idx)
            # Rebuild est_pos from the optimized node poses, aligned to the
            # original pair endpoints (frame 0 plus one per pair).
            est_pos = [node_poses[0][:3, 3].copy()]
            est_pos += [
                node_poses[(p + 1) * args.stride][:3, 3].copy()
                for p in range(n_pairs)
            ]

        est_pos = np.array(est_pos)
        gt_pos = np.array(gt_pos)
        # Align
        if len(est_pos) >= 3:
            s, R_align, t_align = umeyama(est_pos, gt_pos, with_scale=True)
            aligned = s * (est_pos @ R_align.T) + t_align
            sim3_err = np.linalg.norm(aligned - gt_pos, axis=1)
            ate_rmse = float(np.sqrt(np.mean(sim3_err ** 2)))
            ate_mean = float(np.mean(sim3_err))
            ate_median = float(np.median(sim3_err))

            _, R_metric, t_metric = umeyama(est_pos, gt_pos, with_scale=False)
            metric_aligned = est_pos @ R_metric.T + t_metric
            metric_err = np.linalg.norm(metric_aligned - gt_pos, axis=1)
            metric_rmse = float(np.sqrt(np.mean(metric_err ** 2)))
            metric_mean = float(np.mean(metric_err))
            metric_median = float(np.median(metric_err))
        else:
            ate_rmse = ate_mean = ate_median = float("nan")
            metric_rmse = metric_mean = metric_median = float("nan")
            s = float("nan")

        row = {
            "seq": seq, "pose_source": args.pose_source,
            "method": args.method, "threshold": args.threshold,
            "stride": args.stride, "dbin": args.dbin,
            "guided": args.guided, "max_matches": args.max_matches,
            "loop_closure": args.loop_closure, "n_loop": n_loop,
            "n_pairs": n_pairs, "n_ok": n_ok,
            "ok_rate": n_ok / n_pairs if n_pairs else 0.0,
            "mean_inlier_ratio": float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
            "ATE_RMSE": ate_rmse, "ATE_mean": ate_mean, "ATE_median": ate_median,
            "ATE_metric_RMSE": metric_rmse,
            "ATE_metric_mean": metric_mean,
            "ATE_metric_median": metric_median,
            "alignment_scale": float(s),
        }
        summary.append(row)
        print(f"[{seq}] pairs={n_pairs} ok={n_ok} ok_rate={row['ok_rate']:.2f} "
              f"inl={row['mean_inlier_ratio']:.2f} ATE_RMSE={ate_rmse:.3f}m "
              f"ATE_sim3_med={ate_median:.3f}m "
              f"ATE_metric_med={metric_median:.3f}m scale={s:.3f}"
              + (f" loops={n_loop}" if args.loop_closure else ""))
    return summary


# --------------------------------------------------------------------------
# GT-check mode (long-baseline Sampson consistency)
# --------------------------------------------------------------------------
def run_gtcheck(args):
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    cam = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                          width=args.width, height=args.height)
    seqs = ["desk", "desk2", "room"] if args.seq == "all" else [args.seq]
    out = []
    for seq in seqs:
        base = Path(args.dataset_root) / f"rgbd_dataset_freiburg1_{seq}"
        frame_list = []
        with open(base / "rgb.txt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split()
                if len(p) >= 2:
                    frame_list.append((float(p[0]), base / p[1]))
        gt_by_ts = {}
        for (ts, vals) in read_tum_file(base / "groundtruth.txt"):
            gt_by_ts[ts] = quat_to_se3(vals[:3], vals[3:7])

        rows = []
        for i in range(0, len(frame_list) - args.stride, args.stride):
            ts_a, pa = frame_list[i]
            ts_b, pb = frame_list[i + args.stride]
            gi_a = min(gt_by_ts, key=lambda k: abs(k - ts_a))
            gi_b = min(gt_by_ts, key=lambda k: abs(k - ts_b))
            if abs(gi_a - ts_a) > 0.05 or abs(gi_b - ts_b) > 0.05:
                continue
            a = cv2.resize(cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE),
                           (cam.width, cam.height)).astype(np.float32)[None, None]
            b = cv2.resize(cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE),
                           (cam.width, cam.height)).astype(np.float32)[None, None]
            k1, k2, P = session.run(None, {"image1": a, "image2": b})
            mk1, mk2, sc = extract_matches(k1, k2, P, args.match_threshold,
                                           args.max_matches, args.dbin)
            if len(mk1) < 5:
                rows.append({"n": len(mk1), "gt_sampson": None, "est_sampson": None})
                continue
            # GT relative pose -> F_gt
            T_rel = relative_from_gt(gt_by_ts[gi_a], gt_by_ts[gi_b])
            Rg = T_rel[:3, :3]
            tg = T_rel[:3, 3]
            Fg = build_F_from_pose(Rg, tg, cam.K)
            # estimated E via RANSAC
            R, t, mask = estimate_pose_ransac(mk1, mk2, cam, ransac_threshold=1.0)
            if R is None:
                Fe = None
            else:
                E = _cross(t.ravel()) @ R
                Fe = np.linalg.inv(cam.K).T @ E @ np.linalg.inv(cam.K)
            h1 = np.concatenate([mk1, np.ones((len(mk1), 1))], axis=1)
            h2 = np.concatenate([mk2, np.ones((len(mk2), 1))], axis=1)
            dg = sampson_all(h1, h2, Fg)
            de = sampson_all(h1, h2, Fe) if Fe is not None else np.full(len(mk1), np.nan)
            rows.append({
                "n": len(mk1),
                "gt_sampson_mean": float(np.mean(dg)),
                "gt_sampson_med": float(np.median(dg)),
                "gt_frac_2px": float(np.mean(dg < 2.0)),
                "gt_frac_5px": float(np.mean(dg < 5.0)),
                "est_sampson_mean": float(np.nanmean(de)),
                "est_frac_2px": float(np.nanmean(de < 2.0)),
            })
        # aggregate
        valid = [r for r in rows if r.get("gt_sampson_mean") is not None]
        agg = {
            "seq": seq, "stride": args.stride, "n_pairs": len(valid),
            "gt_sampson_mean": float(np.mean([r["gt_sampson_mean"] for r in valid])),
            "gt_frac_2px": float(np.mean([r["gt_frac_2px"] for r in valid])),
            "gt_frac_5px": float(np.mean([r["gt_frac_5px"] for r in valid])),
            "est_sampson_mean": float(np.nanmean([r["est_sampson_mean"] for r in valid])),
            "est_frac_2px": float(np.nanmean([r["est_frac_2px"] for r in valid])),
        }
        out.append(agg)
        print(f"[{seq}] stride={args.stride} n={len(valid)} "
              f"GT Sampson mean={agg['gt_sampson_mean']:.2f}px "
              f"GT<2px={agg['gt_frac_2px']:.2f} GT<5px={agg['gt_frac_5px']:.2f} "
              f"EST Sampson mean={agg['est_sampson_mean']:.2f}px EST<2px={agg['est_frac_2px']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["vo", "gtcheck"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset-root", default="/home/ubuntu/datasets/tum_rgbd")
    ap.add_argument("--seq", default="all")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--method", choices=["ransac", "magsac"], default="ransac")
    ap.add_argument(
        "--pose-source",
        choices=["essential", "rgbd-pnp"],
        default="essential",
    )
    ap.add_argument("--threshold", type=float, default=1.4)
    ap.add_argument("--dbin", type=float, default=0.1)
    ap.add_argument("--match-threshold", type=float, default=0.1)
    ap.add_argument("--max-matches", type=int, default=1024)
    ap.add_argument("--guided", action="store_true")
    ap.add_argument("--guided-inlier-thresh", type=float, default=0.35)
    ap.add_argument("--guided-sampson", type=float, default=2.0)
    ap.add_argument("--depth-scale", type=float, default=5000.0)
    ap.add_argument("--depth-max-diff", type=float, default=0.02)
    ap.add_argument("--min-depth", type=float, default=0.1)
    ap.add_argument("--max-depth", type=float, default=10.0)
    # Loop closure (pose-graph): appearance-based loop detection + full-graph GN.
    ap.add_argument("--loop-closure", action="store_true",
                    help="Enable loop closure: detect loop edges by re-matching "
                         "keyframe pairs and re-optimize the full pose graph.")
    ap.add_argument("--keyframe-decim", type=int, default=8,
                    help="Keyframe stride for loop candidates (every Nth node).")
    ap.add_argument("--loop-min-gap", type=int, default=30,
                    help="Minimum node-index gap between loop candidates.")
    ap.add_argument("--loop-min-inlier", type=float, default=0.4,
                    help="Minimum inlier ratio to accept a loop edge.")
    ap.add_argument("--loop-iterations", type=int, default=60,
                    help="Gauss-Newton iterations for the full-graph optimize.")
    ap.add_argument("--loop-huber", type=float, default=1.0,
                    help="Huber threshold for pose-graph edge reweighting.")
    ap.add_argument("--loop-verbose", action="store_true",
                    help="Print loop-closure optimize progress.")
    ap.add_argument("--fx", type=float, default=525.0)
    ap.add_argument("--fy", type=float, default=525.0)
    ap.add_argument("--cx", type=float, default=320.0)
    ap.add_argument("--cy", type=float, default=240.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "vo":
        summary = run_vo(args)
    else:
        summary = run_gtcheck(args)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
