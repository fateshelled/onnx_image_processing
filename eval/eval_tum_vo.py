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

from vo.outlier_filters import dustbin_margin_filter  # noqa: E402
from vo.sinkhorn_numpy import NumpySinkhornMatcher  # noqa: E402
from vo.pose_estimation import (  # noqa: E402
    estimate_pose_ransac,
    estimate_pose_rgbd_pnp,
    CameraIntrinsics,
)
from vo.trajectory import Trajectory  # noqa: E402
from vo.se3_window import SlidingWindowOptimizer  # noqa: E402
from vo.loop_closure import (  # noqa: E402
    confirmed_loop_hits,
    edge_key,
    local_candidate,
)
from vo.cycle_consistency import (  # noqa: E402
    chain_residual_deg,
    cumulative_rotations,
)
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
def estimate_pair(session, img_path1, img_path2, cam, args, depth_path1=None,
                  feature_cache=None, frame_a=None, frame_b=None):
    """Run model on a pair, extract matches, estimate pose. Returns dict.

    If the model also outputs per-image descriptors (export
    --with-descriptors: keypoints1, keypoints2, descriptors1, descriptors2,
    matching_probs), they are stored into feature_cache[frame_a] /
    feature_cache[frame_b] so loop closure can re-match without re-detection.
    """
    a = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return None
    a = cv2.resize(a, (cam.width, cam.height)).astype(np.float32)[None, None]
    b = cv2.resize(b, (cam.width, cam.height)).astype(np.float32)[None, None]

    outs = session.run(None, {"image1": a, "image2": b})
    if len(outs) == 5:
        k1, k2, d1, d2, P = outs
        if feature_cache is not None:
            feature_cache[frame_a] = (k1, d1)
            feature_cache[frame_b] = (k2, d2)
    else:
        k1, k2, P = outs
    mk1, mk2, sc = extract_matches(k1, k2, P, args.match_threshold,
                                   args.max_matches, args.dbin)
    return estimate_pose_from_matches(mk1, mk2, cam, args, depth_path1)


def estimate_pose_from_matches(mk1, mk2, cam, args, depth_path1=None):
    """Pose estimation from extracted matches (shared by pair model and
    descriptor-cache paths). Returns the estimate_pair result dict."""
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


def _edge_diag(frames, a, b, R_c, t_c, inlier, n_matches, cum_rot=None):
    """Diagnostics for a closure edge (evaluation only, never used to accept
    the edge). ``R_err_deg`` is against GT; ``R_odom_err_deg`` is the rotation
    disagreement with the odometry chain (the runtime gate signal)."""
    diag = {"a": int(a), "b": int(b),
            "inlier": float(inlier), "n_matches": int(n_matches),
            "R": np.asarray(R_c, float).tolist()}
    R_c = np.asarray(R_c, float)
    t_c = np.asarray(t_c, float).reshape(3)
    if cum_rot is not None and a in cum_rot and b in cum_rot:
        try:
            diag["R_odom_err_deg"] = chain_residual_deg(a, b, R_c, cum_rot)
        except Exception:
            pass
    try:
        T_a_gt = frames[a][3]
        T_b_gt = frames[b][3]
        # M_ab = inv(T_b) T_a (matches the recoverPose R convention used for
        # the loop edge): R_ab = R_b^T R_a, t_ab = R_b^T (t_a - t_b).
        R_gt = T_b_gt[:3, :3].T @ T_a_gt[:3, :3]
        t_gt = T_b_gt[:3, :3].T @ (T_a_gt[:3, 3] - T_b_gt[:3, 3])
        c = (np.trace(R_gt @ R_c.T) - 1.0) / 2.0
        diag["R_err_deg"] = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
        diag["t_err_m"] = float(np.linalg.norm(t_c - t_gt))
        diag["t_gt_norm_m"] = float(np.linalg.norm(t_gt))
    except Exception:
        pass
    return diag


def select_keyframes(keys, node_poses, args):
    """Pick keyframe node ids.

    ``decim``: fixed decimation (every --keyframe-decim-th node), as before.
    ``motion``: promote a node to keyframe when the transform from the last
    keyframe exceeds a translation (normalized units; ~step count) or
    rotation threshold, bounded by --kf-min-gap / --kf-max-gap. This adapts
    the keyframe spacing to the motion instead of the frame index.
    """
    if args.kf_mode == "decim":
        return keys[:: max(1, args.keyframe_decim)]
    kf = [keys[0]]
    last = keys[0]
    for k in keys[1:]:
        gap = k - last
        promote = gap >= args.kf_max_gap
        if not promote and gap >= args.kf_min_gap:
            dT = np.linalg.inv(node_poses[k]) @ node_poses[last]
            trans = float(np.linalg.norm(dT[:3, 3]))
            c = (np.trace(dT[:3, :3]) - 1.0) / 2.0
            rot = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
            promote = trans >= args.kf_trans_thresh or rot >= args.kf_rot_thresh
        if promote:
            kf.append(k)
            last = k
    if kf[-1] != keys[-1]:
        kf.append(keys[-1])
    return kf


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
        # Descriptor cache: filled by the odometry pass itself when the pair
        # model emits descriptors (--with-descriptors export), so keyframe
        # re-matching needs no extra detection.
        feature_cache: dict[int, tuple] = {}
        single_sess = None
        desc_matcher = NumpySinkhornMatcher(
            iterations=args.sink_iterations,
            epsilon=args.sink_epsilon,
            unused_score=args.unused_score,
            distance_type="l2",
        )
        if args.desc_model and args.pose_source == "essential":
            single_sess = ort.InferenceSession(
                args.desc_model, providers=["CPUExecutionProvider"])
        n_cached_match = 0
        n_pair_match = 0

        def lazy_features(idx: int):
            # feature_cache is filled either by the odometry pass (descriptor-
            # emitting pair model) or --desc-model (single-image model).
            if idx in feature_cache:
                return feature_cache[idx]
            if single_sess is None:
                return None  # frame not covered by any odometry pair
            img = cv2.imread(frame_img[idx], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(
                img, (cam.width, cam.height)).astype(np.float32)[None, None]
            kp, desc = single_sess.run(None, {"image": img})
            feature_cache[idx] = (kp, desc)
            return feature_cache[idx]

        def match_keyframes(a: int, b: int):
            # Shared by the additive keyframe edges and loop closure: use the
            # per-frame descriptor cache (numpy Sinkhorn) when both frames have
            # cached features, otherwise the pair model on the two images.
            nonlocal n_cached_match, n_pair_match
            fa, fb = lazy_features(a), lazy_features(b)
            if fa is not None and fb is not None:
                n_cached_match += 1
                kp_a, desc_a = fa
                kp_b, desc_b = fb
                P = desc_matcher.match_probs(desc_a[0], desc_b[0])
                mk1, mk2, _ = extract_matches(
                    kp_a, kp_b, P[None], args.match_threshold,
                    args.max_matches, args.dbin)
                return estimate_pose_from_matches(
                    mk1, mk2, cam, args, depth_path1=None)
            n_pair_match += 1
            return estimate_pair(
                session=session, img_path1=frame_img[a],
                img_path2=frame_img[b], cam=cam, args=args,
                depth_path1=None)

        # process pairs at given stride
        trajectory = Trajectory()
        node_poses: dict[int, np.ndarray] = {0: trajectory.get_current_pose().copy()}
        frame_img: dict[int, str] = {0: str(frames[0][1])}
        odom: list[tuple] = []  # (i, j, R, t) odometry edges
        est_pos = [trajectory.get_current_position().copy()]
        gt_pos = [frames[0][3][:3, 3].copy()]
        n_pairs = 0
        n_ok = 0
        inlier_ratios = []
        # Additive keyframe edges (--odom-ref kf): keep the consecutive chain
        # and additionally add, for every frame, an edge to the last keyframe
        # (longer baseline, better translation conditioning). The redundancy
        # keeps the pose graph and per-edge scale well constrained.
        kf_nodes = [0]
        last_kf = 0

        def kf_additive(i: int):
            nonlocal last_kf
            if args.odom_ref != "kf":
                return
            if last_kf != i - args.stride:
                kres = match_keyframes(last_kf, i)
                if (kres is not None and kres.get("ok")
                        and kres.get("inlier_ratio", 0.0)
                        >= args.kf_edge_min_inlier):
                    odom.append((last_kf, i, kres["R"], kres["t"]))
            dT = np.linalg.inv(node_poses[i]) @ node_poses[last_kf]
            trans = float(np.linalg.norm(dT[:3, 3]))
            c = (np.trace(dT[:3, :3]) - 1.0) / 2.0
            rot = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
            gap = i - last_kf
            if (gap >= args.kf_max_gap
                    or (gap >= args.kf_min_gap
                        and (trans >= args.kf_trans_thresh
                             or rot >= args.kf_rot_thresh))):
                last_kf = i
                kf_nodes.append(i)

        for i in range(args.stride, len(frames), args.stride):
            a = i - args.stride
            _, pa, da, gta = frames[a]
            _, pb, db, gtb = frames[i]
            res = estimate_pair(
                session=session,
                img_path1=str(pa),
                img_path2=str(pb),
                cam=cam,
                args=args,
                depth_path1=da,
                feature_cache=feature_cache,
                frame_a=a,
                frame_b=i,
            )
            n_pairs += 1
            if res is None or not res.get("ok"):
                # no pose update -> keep the previous pose (skip)
                node_poses[i] = node_poses[a].copy()
                est_pos.append(node_poses[i][:3, 3].copy())
                gt_pos.append(gtb[:3, 3].copy())
                frame_img[i] = str(pb)
                kf_additive(i)
                continue
            R, t = res["R"], res["t"]
            trajectory.add_relative_pose(R, t)
            T_i = trajectory.get_current_pose().copy()
            est_pos.append(T_i[:3, 3].copy())
            gt_pos.append(gtb[:3, 3].copy())
            n_ok += 1
            inlier_ratios.append(res["inlier_ratio"])
            node_poses[i] = T_i
            frame_img[i] = str(pb)
            odom.append((a, i, R, t))
            kf_additive(i)

        # Integrated keyframe matching + loop closure: build the full pose
        # graph, then walk the keyframes once. Each new keyframe first tries
        # loop closure against older keyframes in the recent window; if no
        # loop is confirmed it falls back to a local refinement edge with the
        # immediately preceding keyframe. The loop/local stages are exclusive
        # and every added edge is recorded so no constraint is duplicated.
        # Appearance-based detection (matcher inlier ratio) is used instead of
        # a position gate, because the monocular estimate has already drifted
        # and a position gate would miss true loops on these return-trajectories.
        # With --odom-ref kf the pose graph is needed even without loop
        # closure (the additive keyframe edges are optimized jointly).
        if args.loop_closure or args.odom_ref == "kf":
            # Per-edge scale: monocular translations are unit-norm, so allow
            # the optimizer to estimate each edge's translation magnitude
            # (relative to the first edge by default). --scale-loop-only keeps
            # only loop edges free (much smaller problem).
            odom_scale_free = args.edge_scale and not args.scale_loop_only
            opt = SlidingWindowOptimizer(
                window_size=None,
                max_iterations=args.loop_iterations,
                huber=args.loop_huber,
                step_scale_t=args.step_scale_t,
                step_scale_r=args.step_scale_r,
                optimize_scale=args.edge_scale,
                scale_prior_sigma=(args.scale_prior_sigma
                                   if args.edge_scale else 0.0),
            )
            for idx, T in node_poses.items():
                opt.add_node(idx, T)
            added_edges = set()
            for (i, j, R, t) in odom:
                M = np.eye(4)
                M[:3, :3] = R
                M[:3, 3] = np.asarray(t, float).reshape(3)
                opt.add_edge(i, j, M, scale_free=odom_scale_free)
                added_edges.add(edge_key(i, j))

            keys = sorted(node_poses.keys())
            if args.odom_ref == "kf":
                kf = list(kf_nodes)
            else:
                kf = select_keyframes(keys, node_poses, args)
            n_keyframes = len(kf)
            # Keyframe matching runs only with --loop-closure; with
            # --odom-ref kf alone the pose graph is simply optimized.
            if not args.loop_closure:
                kf = []
            gaps = [kf[i + 1] - kf[i] for i in range(len(kf) - 1)]
            kf_step = (max(gaps) if gaps
                       else args.stride * max(1, args.keyframe_decim))
            # Odometry-chain cumulative rotation: the reference path for the
            # rotation cycle-consistency gate. It is independent of any loop
            # decision, and available for every candidate pair.
            odom_rot = {edge_key(i, j): np.asarray(R, float)
                        for (i, j, R, t) in odom}
            cum_rot = cumulative_rotations(keys, odom_rot)
            # Temporal consistency: a loop edge (a, b) is accepted only if the
            # previous --loop-temporal-k keyframes also matched nearly the same
            # old keyframe a' (|a' - a| <= margin). Isolated hits are rejected.
            margin = max(int(args.kf_margin * kf_step), kf_step)
            need = max(1, args.loop_temporal_k)
            window = args.loop_window if args.loop_window > 0 else len(kf)
            n_loop = 0
            n_local = 0
            n_cycle_reject = 0
            loop_edges = []
            local_edges = []
            hits_per_kf: list[list] = [[] for _ in kf]

            for bi, b in enumerate(kf):
                # Loop stage: match older keyframes in the recent window
                # (bounded so the search stays local and cheap).
                hits = []
                for ai in range(max(0, bi - window), bi):
                    a = kf[ai]
                    if b - a < args.loop_min_gap:
                        continue
                    lres = match_keyframes(a, b)
                    if lres is None or not lres.get("ok"):
                        continue
                    if lres["inlier_ratio"] < args.loop_min_inlier:
                        continue
                    hits.append((
                        int(a),
                        np.asarray(lres["R"], float),
                        np.asarray(lres["t"], float).reshape(3),
                        float(lres["inlier_ratio"]),
                        int(lres.get("n_matches", -1)),
                    ))
                hits_per_kf[bi] = hits

                accepted = confirmed_loop_hits(
                    b, hits, bi, hits_per_kf, need, margin, added_edges)
                # Rotation cycle-consistency gate: the loop edge must agree on
                # rotation (scale-free) with the odometry chain between the same
                # nodes. This is the cycle formed by the loop edge and the
                # odometry path, so it is available for every candidate.
                final = []
                for hit in accepted:
                    a_hit, R_hit = int(hit[0]), np.asarray(hit[1], float)
                    if args.cycle_threshold_deg > 0:
                        res = chain_residual_deg(a_hit, b, R_hit, cum_rot)
                        if res > args.cycle_threshold_deg:
                            n_cycle_reject += 1
                            continue
                    final.append(hit)
                if final:
                    sig = {"sigma_t": None, "sigma_r": None}
                    if args.loop_rot_only:
                        # Monocular recoverPose translations are unit-norm per
                        # step, so a long-baseline loop edge's translation is
                        # inconsistent with the odometry chain scale. Constrain
                        # rotation only; let the sim3 evaluation handle scale.
                        sig = {"sigma_t": 1e6, "sigma_r": None}
                        if args.loop_sigma_scale > 0:
                            sig["sigma_r"] = (
                                args.step_scale_r * args.loop_sigma_scale)
                    elif args.loop_sigma_scale > 0:
                        sig = {
                            "sigma_t": args.step_scale_t * args.loop_sigma_scale,
                            "sigma_r": args.step_scale_r * args.loop_sigma_scale,
                        }
                    for (a, R_c, t_c, inl, n_m) in final:
                        M = np.eye(4)
                        M[:3, :3] = R_c
                        M[:3, 3] = t_c
                        opt.add_edge(a, b, M, scale_free=args.edge_scale, **sig)
                        added_edges.add(edge_key(a, b))
                        n_loop += 1
                        loop_edges.append(
                            _edge_diag(frames, a, b, R_c, t_c, inl, n_m,
                                       cum_rot=cum_rot))
                    continue  # exclusive: loop confirmed for this keyframe

                # No loop: local refinement vs the nearest keyframe.
                if not args.local_refine:
                    continue
                a = local_candidate(kf, bi, added_edges)
                if a is None:
                    continue
                lres = match_keyframes(a, b)
                if lres is None or not lres.get("ok"):
                    continue
                if lres["inlier_ratio"] < args.local_min_inlier:
                    continue
                M = np.eye(4)
                M[:3, :3] = np.asarray(lres["R"], float)
                M[:3, 3] = np.asarray(lres["t"], float).reshape(3)
                opt.add_edge(a, b, M, scale_free=odom_scale_free)
                added_edges.add(edge_key(a, b))
                n_local += 1
                local_edges.append(_edge_diag(
                    frames, a, b, lres["R"], lres["t"],
                    float(lres["inlier_ratio"]), int(lres.get("n_matches", -1)),
                    cum_rot=cum_rot))

            print(f"[loop] matched via cache: {n_cached_match}, pair model: "
                  f"{n_pair_match}, loops: {n_loop}, local: {n_local}, "
                  f"cycle-reject: {n_cycle_reject}")
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
            "n_local": n_local if args.loop_closure else 0,
            "n_cycle_reject": n_cycle_reject if args.loop_closure else 0,
            "cycle_threshold_deg": args.cycle_threshold_deg,
            "loop_rot_only": bool(args.loop_rot_only),
            "edge_scale": bool(args.edge_scale),
            "scale_prior_sigma": args.scale_prior_sigma,
            "scale_loop_only": bool(args.scale_loop_only),
            "kf_mode": args.kf_mode,
            "n_keyframes": (n_keyframes
                            if (args.loop_closure or args.odom_ref == "kf")
                            else 0),
            "odom_ref": args.odom_ref,
            **({"loop_edges": loop_edges, "local_edges": local_edges}
               if args.loop_closure else {}),
            "desc_model": bool(args.desc_model),
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
              + (f" loops={n_loop} local={n_local} cyclo={n_cycle_reject}"
                 if args.loop_closure else ""))
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
    ap.add_argument("--kf-mode", choices=["decim", "motion"], default="decim",
                    help="Keyframe selection: fixed decimation, or motion-based "
                         "(translation/rotation since the last keyframe).")
    ap.add_argument("--kf-trans-thresh", type=float, default=8.0,
                    help="Motion mode: promote when the translation from the "
                         "last keyframe (normalized units, ~step count) exceeds "
                         "this.")
    ap.add_argument("--kf-rot-thresh", type=float, default=10.0,
                    help="Motion mode: promote when the rotation from the last "
                         "keyframe (degrees) exceeds this.")
    ap.add_argument("--kf-min-gap", type=int, default=4,
                    help="Motion mode: minimum frame-index gap between keyframes.")
    ap.add_argument("--kf-max-gap", type=int, default=16,
                    help="Motion mode: force a keyframe at this frame-index gap "
                         "(matches --keyframe-decim=8 at stride 2).")
    ap.add_argument("--kf-edge-min-inlier", type=float, default=0.0,
                    help="With --odom-ref kf, minimum inlier ratio to add the "
                         "additive edge to the last keyframe (0 = no gate; "
                         "gating at 0.3 hurt desk, so it is off by default).")
    ap.add_argument("--odom-ref", choices=["prev", "kf"], default="prev",
                    help="prev: consecutive-pair chain only. kf: additive - "
                         "keep the chain and also add, for every frame, an edge "
                         "to the last keyframe (longer baseline, better "
                         "translation conditioning). The pose graph is "
                         "optimized even without --loop-closure.")
    ap.add_argument("--loop-min-gap", type=int, default=30,
                    help="Minimum node-index gap between loop candidates.")
    ap.add_argument("--loop-min-inlier", type=float, default=0.4,
                    help="Minimum inlier ratio to accept a loop edge.")
    ap.add_argument("--loop-window", type=int, default=40,
                    help="Search loop candidates among this many recent "
                         "keyframes (0 = all previous keyframes).")
    ap.add_argument("--local-refine", action="store_true", default=False,
                    help="When no loop is confirmed, also add a refinement edge "
                         "to the nearest keyframe (off by default: it hurt ATE "
                         "on desk/desk2).")
    ap.add_argument("--local-min-inlier", type=float, default=0.4,
                    help="Minimum inlier ratio to accept a local refinement edge.")
    # Rotation cycle-consistency gate: a loop edge must agree (in rotation)
    # with an alternative path through other accepted closure edges, where
    # such a path exists. Scale-free, so it works before RGB-D depth.
    ap.add_argument("--cycle-threshold-deg", type=float, default=0.0,
                    help="Max rotation disagreement (deg) between a loop edge "
                         "and the odometry chain (0 = disable). Off by default: "
                         "on desk it rejects good loops (residual is uncorrelated "
                         "with GT error).")
    ap.add_argument("--loop-rot-only", action="store_true",
                    help="Constrain loop edges by rotation only (ignore their "
                         "unit-norm monocular translation, which conflicts with "
                         "the odometry chain scale).")
    ap.add_argument("--edge-scale", action="store_true", default=False,
                    help="Optimize a per-edge translation scale (similarity "
                         "pose graph) instead of fixing all translations to "
                         "unit norm. First free edge is the scale gauge.")
    ap.add_argument("--scale-prior-sigma", type=float, default=0.5,
                    help="Std of the log-scale prior pulling edge scales "
                         "toward 1 (only with --edge-scale).")
    ap.add_argument("--scale-loop-only", action="store_true", default=False,
                    help="With --edge-scale, keep odometry edges at unit scale "
                         "and free only loop/refinement edges.")
    ap.add_argument("--loop-iterations", type=int, default=60,
                    help="Gauss-Newton iterations for the full-graph optimize.")
    ap.add_argument("--loop-huber", type=float, default=1.0,
                    help="Huber threshold for pose-graph edge reweighting.")
    ap.add_argument("--loop-verbose", action="store_true",
                    help="Print loop-closure optimize progress.")
    # Temporal consistency: an edge is added only when the previous
    # --loop-temporal-k keyframes also matched (nearly) the same old keyframe.
    ap.add_argument("--loop-temporal-k", type=int, default=1,
                    help="Require this many consecutive keyframes to see the "
                         "same old loop spot (1 = off).")
    ap.add_argument("--kf-margin", type=float, default=1.5,
                    help="Old-keyframe matching tolerance in kf-steps for "
                         "temporal consistency.")
    # Edge weighting: loop edges get sigma = odom_scale * this factor
    # (>1 = weaker constraint; 0 = same weight as odometry edges).
    ap.add_argument("--loop-sigma-scale", type=float, default=0.0,
                    help="If >0, loop edge sigma = odometry sigma * scale.")
    ap.add_argument("--step-scale-t", type=float, default=0.05)
    ap.add_argument("--step-scale-r", type=float, default=0.05)
    # Descriptor-cache path: per-frame keypoint+descriptor ONNX model
    # (export with --single-image). NumPy sinkhorn params default to the
    # pyramid_k512_l2 export settings.
    ap.add_argument("--desc-model", default=None,
                    help="Single-image ONNX model outputting keypoints+descriptors "
                         "(export_shi_tomasi_angle_sparse_bad_sinkhorn_pyramid "
                         "--single-image). Uses cached per-frame features for "
                         "loop re-matching (numpy Sinkhorn instead of the pair "
                         "model).")
    ap.add_argument("--sink-epsilon", type=float, default=0.05)
    ap.add_argument("--sink-iterations", type=int, default=20)
    ap.add_argument("--unused-score", type=float, default=1.0)
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
