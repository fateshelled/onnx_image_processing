"""Rustuna tuning of the loop-closure / scale / keyframe stage.

The matching/pose cache (eval/rustuna_tune.py) only covers the plain
consecutive-pair pipeline. This tuner caches per-frame keypoints+descriptors
(wd model) and the consecutive-pair odometry poses once, then evaluates the
full pipeline (chain + additive keyframe edges + loop closure + per-edge
scale) from the cache for each trial (no ONNX inference).

Two phases:
  --build-cache : run the wd pair model once per consecutive pair, cache
                  per-frame features and the odometry relative poses.
  (default)     : Rustuna TPE study over the loop/scale/keyframe parameters.
                  Objective: mean ATE_sim3_med over --seq (default desk,desk2).

Run:
  .venv/bin/python eval/rustuna_tune_loop.py --build-cache
  .venv/bin/python eval/rustuna_tune_loop.py --n-trials 40 --seq desk,desk2
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

import cv2  # noqa: E402
import onnxruntime as ort  # noqa: E402

from eval_tum_vo import (  # noqa: E402
    estimate_pose_from_matches,
    extract_matches,
    select_keyframes,
    nearest_timestamp,
    quat_to_se3,
    read_path_file,
    read_tum_file,
    umeyama,
    resolve_dataset,
    intrinsics_for,
    _trans_consistent,
)
from vo.loop_closure import (  # noqa: E402
    confirmed_loop_hits,
    edge_key,
    local_candidate,
)
from vo.cycle_consistency import chain_residual_deg, cumulative_rotations  # noqa: E402
from vo.loop_sim3_verifier import Sim3LoopVerifier  # noqa: E402
from vo.sinkhorn_numpy import NumpySinkhornMatcher  # noqa: E402
from vo.trajectory import Trajectory  # noqa: E402
from vo.se3_window import SlidingWindowOptimizer  # noqa: E402
from vo.scale_kf import ScaleKF  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

DEFAULT_ARGS = SimpleNamespace(
    pose_source="essential", method="magsac", threshold=1.4, dbin=0.1,
    max_matches=1024, match_threshold=0.1, guided=False,
    guided_inlier_thresh=0.35, guided_sampson=2.0, depth_scale=5000.0,
    min_depth=0.1, max_depth=10.0,
)


def load_frames(dataset_root, seq, gt_max_diff=0.05):
    base, _cam = resolve_dataset(dataset_root, seq)
    gt_rows = read_tum_file(base / "groundtruth.txt")
    gt_poses = [(ts, quat_to_se3(v[:3], v[3:7])) for ts, v in gt_rows]
    frames = []
    for ts, rel in read_path_file(base / "rgb.txt"):
        gm = nearest_timestamp(gt_poses, ts, gt_max_diff)
        if gm is not None:
            frames.append((base / rel, gm[1]))
    return frames


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def build_cache(args):
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    out = Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    seqs = [s for s in args.seq.split(",") if s]
    for seq in seqs:
        fx, fy, cx, cy = intrinsics_for(
            args.dataset_root, seq, (args.fx, args.fy, args.cx, args.cy))
        cam = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy,
                               width=args.width, height=args.height)
        frames = load_frames(args.dataset_root, seq)
        stride = args.stride
        n_pairs = (len(frames) - 1) // stride
        feat_idx, feat_kp, feat_desc = [], [], []
        odom = []
        gt_pos = [frames[0][1][:3, 3].copy()]
        t0 = time.time()
        for n in range(n_pairs):
            i = n * stride
            j = i + stride
            a = cv2.resize(cv2.imread(str(frames[i][0]), cv2.IMREAD_GRAYSCALE),
                           (args.width, args.height)).astype(np.float32)[None, None]
            b = cv2.resize(cv2.imread(str(frames[j][0]), cv2.IMREAD_GRAYSCALE),
                           (args.width, args.height)).astype(np.float32)[None, None]
            outs = sess.run(None, {"image1": a, "image2": b})
            k1, k2, d1, d2, P = outs
            for idx, kp, desc in ((i, k1, d1), (j, k2, d2)):
                if idx not in feat_idx:
                    feat_idx.append(idx)
                    feat_kp.append(kp.astype(np.float32))
                    feat_desc.append(desc.astype(np.float32))
            mk1, mk2, _ = extract_matches(
                k1, k2, P, DEFAULT_ARGS.match_threshold,
                DEFAULT_ARGS.max_matches, DEFAULT_ARGS.dbin)
            res = estimate_pose_from_matches(mk1, mk2, cam, DEFAULT_ARGS)
            odom.append({
                "ok": bool(res.get("ok", False)),
                "R": res.get("R"), "t": res.get("t"),
                "inlier": float(res.get("inlier_ratio", 0.0)),
            })
            gt_pos.append(frames[j][1][:3, 3].copy())
            if n % 100 == 0:
                print(f"  [{seq}] {n}/{n_pairs} {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(
            out / f"{seq}.npz",
            feat_idx=np.array(feat_idx, dtype=np.int64),
            feat_kp=np.array(feat_kp, dtype=np.float32),
            feat_desc=np.array(feat_desc, dtype=np.float32),
            odom=np.array(odom, dtype=object),
            gt_pos=np.array(gt_pos, dtype=np.float64),
            stride=np.int64(stride),
            n_frames=np.int64(len(frames)),
        )
        print(f"[cache] {seq}: {n_pairs} pairs -> {out / f'{seq}.npz'}", flush=True)


def load_cache(cache_dir, seq):
    z = np.load(Path(cache_dir) / f"{seq}.npz", allow_pickle=True)
    feat = {int(i): (k, d) for i, k, d in
            zip(z["feat_idx"], z["feat_kp"], z["feat_desc"])}
    return {"feat": feat, "odom": list(z["odom"]), "gt_pos": z["gt_pos"],
            "stride": int(z["stride"]), "n_frames": int(z["n_frames"])}


# Default parameters for the sequential fixed-lag graph (seq_opt1 + NL-Reg),
# shared with the online graph module. Callers that need the batch graph must
# pass graph_mode="stride" explicitly.
from vo.online_graph import DEFAULT_PARAMS as _SEQ_DEFAULTS  # noqa: E402
from vo.online_graph import OnlinePoseGraph  # noqa: E402

SEQ_OPT1_DEFAULTS = {
    **_SEQ_DEFAULTS,
    "scale_kf_q": 1e-3, "scale_kf_r": 0.1, "scale_kf_sigma": 0.5,
    "scale_kf_adaptive": False, "scale_kf_innov_tau": 3.0,
    "scale_kf_adapt_loop_ratio": 0.0,
    "loop_rot_only": False, "cycle_threshold_deg": 0.0,
    "loop_verifier": "sim3", "loop_verifier_gate": 6,
    "loop_verifier_window": 4, "loop_verifier_abstain": "accept",
    "loop_verifier_keyframes": True,
}


def _sequential_kf_priors(opt, kf, keys, params):
    """Sequential keyframe-graph driver with transient marginalization.

    ``opt`` supplies the measurements (``edges``) and odometry poses only; it
    is not used as the solver. Consecutive keyframes are processed left to
    right. At every step the active window contains the keyframes seen so far,
    the accumulated relative priors, the new keyframe and the transient stride
    frames in between. After optimization the transients are Schur-
    marginalized (scales fixed at their window optimum) into a dense relative
    prior over the two bounding keyframes, and then dropped. The reduced
    graph therefore only ever holds keyframes + priors + KF-KF (loop) edges,
    so its size is bounded by the number of keyframes.
    """
    step_t = params["step_scale_t"]
    max_kf = params.get("max_keyframes")  # None = keep all keyframes active
    all_edges = list(opt.edges)
    kf_set = set(kf)
    kf_index = {nd: m for m, nd in enumerate(kf)}
    loop_edges = [(i, j, M, st, sr) for (i, j, M, st, sr) in all_edges
                  if i in kf_set and j in kf_set
                  and abs(kf_index[i] - kf_index[j]) > 1]
    priors = []
    node_priors = []  # (ids, H, b, x0) from exited keyframes
    est_pose = {kf[0]: opt.get_pose(kf[0])}

    def new_window():
        return SlidingWindowOptimizer(
            window_size=None, max_iterations=params["loop_iterations"],
            huber=1.0, step_scale_t=step_t, step_scale_r=step_t,
            optimize_scale=True, scale_prior_sigma=params["scale_prior_sigma"],
            tsvd_ratio=(0.0 if params.get("nl_reg", False)
                        else params.get("seq_tsvd_ratio", 0.0)))

    for m in range(len(kf) - 1):
        a, b = kf[m], kf[m + 1]
        trans = [i for i in keys if a < i < b]
        trans_set = set(trans)
        if max_kf is None:
            free_kf = kf[:m + 2]
        else:
            free_kf = kf[max(0, (m + 1) - max(1, max_kf) + 1):m + 2]
        free_set = set(free_kf)
        # Option 1 (deferred marginalization): keyframes that left the active
        # window stay in the optimizer until every node on their Markov
        # blanket is active again; only then are they Schur-marginalized and
        # dropped. This keeps the marginal at a fresh linearization point.
        held = set()
        changed = True
        while changed:
            changed = False
            for (ids, *_rest) in node_priors:
                if any(nd in free_set or nd in held for nd in ids):
                    for nd in ids:
                        if (nd not in free_set and nd not in held
                                and nd not in trans_set):
                            held.add(nd)
                            changed = True
            for e in all_edges:
                if (e[0] in free_set or e[0] in held):
                    if e[1] in kf_set and e[1] not in free_set \
                            and e[1] not in held:
                        held.add(e[1])
                        changed = True
                if (e[1] in free_set or e[1] in held):
                    if e[0] in kf_set and e[0] not in free_set \
                            and e[0] not in held:
                        held.add(e[0])
                        changed = True
        node_set = free_set | held | trans_set
        nodes = sorted(node_set)

        act = new_window()
        for nd in nodes:
            act.add_node(nd, est_pose.get(nd, opt.get_pose(nd)))
        for (x, y, G, Om) in priors:
            if x in node_set and y in node_set:
                act.add_edge(x, y, G, omega=Om)
        for (ids, H, bbias, x0) in node_priors:
            if all(nd in node_set for nd in ids):
                act.add_prior_factor(ids, H, bbias, x0)
        seg = {a} | trans_set | {b}
        for (i, j, M, st, sr) in all_edges:
            if i in seg and j in seg and not (i == a and j == b):
                act.add_edge(i, j, M, sigma_t=st, sigma_r=sr, scale_free=True)
        for (i, j, M, st, sr) in all_edges:
            if {i, j} == {a, b}:  # direct keyframe-to-keyframe measurement
                act.add_edge(i, j, M, sigma_t=st, sigma_r=sr, scale_free=True)
        for (i, j, M, st, sr) in loop_edges:
            if i in node_set and j in node_set:
                act.add_edge(i, j, M, sigma_t=st, sigma_r=sr, scale_free=True)
        if params.get("nl_reg", False):
            act.add_nl_regularization(params.get("nl_reg_c", 1.0),
                                      params.get("nl_reg_tau", 1.0),
                                      params.get("nl_reg_length", 1.0))
        act.optimize()
        if trans:
            aa, bb, G, Om = act.marginalize_relative(trans, [a, b],
                                                     fix_scales=True)
            priors.append((aa, bb, G, Om))
        for nd in nodes:
            est_pose[nd] = act.get_pose(nd)
        # Marginalize a held keyframe once its blanket is fully active and no
        # existing prior still refers to it.
        referenced = {nd for (ids, *_r) in node_priors for nd in ids}
        for E in sorted(held):
            if E in referenced or E not in act.pose_ids:
                continue
            nbr = set()
            for (i, j, *_r) in act.edges:
                if i == E:
                    nbr.add(j)
                elif j == E:
                    nbr.add(i)
            for (ids, *_r) in act.prior_factors:
                if E in ids:
                    nbr |= (set(ids) - {E})
            nbr = {x for x in nbr if x in node_set and x not in trans_set}
            if nbr and nbr <= free_set:
                keep_ids, H_r, b_r = act.marginalize_general([E], sorted(nbr))
                node_priors.append((tuple(keep_ids), H_r, b_r,
                                    [act.get_pose(k) for k in keep_ids]))
                held.discard(E)

    if max_kf is None:
        red = new_window()
        for nd in kf:
            red.add_node(nd, est_pose.get(nd, opt.get_pose(nd)))
        for (aa, bb, G, Om) in priors:
            red.add_edge(aa, bb, G, omega=Om)
        for (i, j, M, st, sr) in loop_edges:
            red.add_edge(i, j, M, sigma_t=st, sigma_r=sr, scale_free=True)
        red.optimize()
        return red
    # Bounded mode: report the frozen sequential estimates directly.
    red = new_window()
    for nd in kf:
        red.add_node(nd, est_pose.get(nd, opt.get_pose(nd)))
    return red


def _keyframe_windows(graph, endpoint):
    """Candidate local windows from the nearest previous keyframes (S1).

    Baselines between keyframes are longer than stride windows and reuse the
    chain poses the graph already carries (``graph._T``), so the gauge matches
    the odometry windows.  Failed odometry steps are bridged as identity by
    the chain, as elsewhere in the graph, and longer baselines also accumulate
    more of the unit-norm composition approximation; both are acceptable here
    because a bad window only weakens the Sim(3) evidence.
    """
    endpoint_pose = graph._T.get(endpoint)
    if endpoint_pose is None:
        return []
    previous = [frame for frame in graph.keyframes
                if frame < endpoint and frame in graph._T][-2:]
    windows = []
    for first in previous:
        # ``_T`` is camera-to-world, and the window convention is
        # ``x_endpoint = R @ x_first + t``, so the relative pose is
        # ``inv(T_endpoint) @ T_first``.
        relative = np.linalg.inv(endpoint_pose) @ graph._T[first]
        windows.append((int(first), int(endpoint),
                        relative[:3, :3].copy(), relative[:3, 3].copy(),
                        False))
    return windows


def _eval_online_kf_prior(c, params, cam, match_frames, raw_match_frames=None):
    """Drive ``OnlinePoseGraph`` over a cached sequence (the online driver).

    The precomputed per-frame odometry is injected so the chain is identical to
    the legacy evaluator; every other match (keyframe spokes + loop closures)
    goes through the shared, memoized ``match_frames``. ATE is over all stride
    frames, with intermediate frames propagated by the graph itself.
    """
    stride = int(c["stride"])
    graph = OnlinePoseGraph(params, cam, match_frames)
    if params.get("loop_verifier") == "sim3":
        if raw_match_frames is None:
            raise ValueError("loop_verifier=sim3 requires raw_match_frames")
        window_fn = None
        if params.get("loop_verifier_keyframes",
                      SEQ_OPT1_DEFAULTS["loop_verifier_keyframes"]):
            def window_fn(endpoint):
                return _keyframe_windows(graph, endpoint)
        graph.loop_verifier = Sim3LoopVerifier(
            c["odom"], raw_match_frames, cam.K, stride,
            gate=int(params.get("loop_verifier_gate", 6)),
            window_strides=int(params.get("loop_verifier_window", 4)),
            abstain_policy=str(params.get(
                "loop_verifier_abstain",
                SEQ_OPT1_DEFAULTS["loop_verifier_abstain"])),
            window_fn=window_fn)
    graph.add_frame(0)
    keys = [0]
    for n, o in enumerate(c["odom"]):
        idx = (n + 1) * stride
        # Always pass the odometry slot; ``R is None`` means "failed, keep the
        # previous pose without adding an edge" (legacy behaviour). Passing
        # ``odom=None`` would instead ask the graph to match on its own.
        if o.get("ok"):
            od = (o.get("R"), o.get("t"), float(o.get("inlier", 1.0)))
        else:
            od = (None, None, 0.0)
        graph.add_frame(idx, odom=od)
        keys.append(idx)
    est = np.array([graph.pose(k)[:3, 3] for k in keys])
    gt_pos = c["gt_pos"]
    gt = np.array([gt_pos[k // stride] for k in keys])
    if len(est) >= 3:
        s, R_a, t_a = umeyama(est, gt, with_scale=True)
        aligned = s * (est @ R_a.T) + t_a
        ate = float(np.median(np.linalg.norm(aligned - gt, axis=1)))
    else:
        ate = float("nan")
    # ``n_verifier_*`` are graph-side counters.  The Sim3 verifier folds
    # abstains into its policy verdict (bool), so ``n_verifier_abstained`` is
    # always 0 for it; the real abstain breakdown is in ``verifier_abstain*``.
    result = {"ATE_median": ate, "n_loop": graph.n_loop, "n_kf": graph.n_kf,
              "n_cycle_reject": graph.n_cycle_rejected,
              "n_verifier_rejected": graph.n_verifier_rejected,
              "n_verifier_abstained": graph.n_verifier_abstained,
              "n_robust_downweighted": graph.n_robust_downweighted,
              "n_robust_rot_downweighted": graph.n_robust_rot_downweighted,
              "n_robust_dir_downweighted": graph.n_robust_dir_downweighted,
              "loop_rot_scale": graph.loop_rot_scale,
              "loop_dir_scale": graph.loop_dir_scale,
              "n_mad_samples": len(graph._hist_rot)}
    verifier = graph.loop_verifier
    if verifier is not None:
        result.update({
            "verifier_accept": verifier.n_accept,
            "verifier_reject": verifier.n_reject,
            "verifier_abstain": verifier.n_abstain,
            "verifier_abstain_no_cloud": verifier.n_abstain_no_cloud,
            "verifier_abstain_no_match": verifier.n_abstain_no_match,
            "verifier_abstain_few_tracks": verifier.n_abstain_few_tracks,
        })
    return result


# --------------------------------------------------------------------------
# Evaluator (mirrors run_vo's loop-closure block)
# --------------------------------------------------------------------------
def eval_seq(c, params, cam, desc_matcher, match_cache=None, diag=None):
    params = {**SEQ_OPT1_DEFAULTS, **params}
    stride = c["stride"]
    pa = SimpleNamespace(**vars(DEFAULT_ARGS))
    pa.kf_mode = params["kf_mode"]
    pa.keyframe_decim = params["keyframe_decim"]
    pa.kf_trans_thresh = params["kf_trans_thresh"]
    pa.kf_rot_thresh = params["kf_rot_thresh"]
    pa.kf_min_gap = 4
    pa.kf_max_gap = 16

    def match_frames_uncached(a, b):
        kp_a, desc_a = c["feat"][a]
        kp_b, desc_b = c["feat"][b]
        P = desc_matcher.match_probs(desc_a[0], desc_b[0])
        mk1, mk2, _ = extract_matches(kp_a, kp_b, P[None], pa.match_threshold,
                                      pa.max_matches, pa.dbin)
        return estimate_pose_from_matches(mk1, mk2, cam, pa)

    def match_frames(a, b):
        # Matching + relative-pose estimation depends only on the frame pair:
        # every threshold used here comes from DEFAULT_ARGS, not from the
        # tuned ``params``. So the result is memoized across trials.
        if match_cache is None:
            return match_frames_uncached(a, b)
        key = (a, b)
        cached = match_cache.get(key)
        if cached is not None:
            return cached
        res = match_frames_uncached(a, b)
        if res.get("ok"):
            cached = {"ok": True, "R": res["R"], "t": res["t"],
                      "inlier_ratio": float(res["inlier_ratio"]),
                      "n_matches": int(res.get("n_matches", -1))}
        else:
            cached = {"ok": False, "n_matches": int(res.get("n_matches", -1))}
        match_cache[key] = cached
        return cached

    def raw_match_frames(a, b):
        # Raw correspondences for the optional Sim(3) loop verifier: no pose
        # estimation here, the verifier's RANSAC is the outlier filter.
        if a not in c["feat"] or b not in c["feat"]:
            return None
        kp_a, desc_a = c["feat"][a]
        kp_b, desc_b = c["feat"][b]
        P = desc_matcher.match_probs(desc_a[0], desc_b[0])
        ma, mb, _ = extract_matches(kp_a, kp_b, P[None], pa.match_threshold,
                                    pa.max_matches, pa.dbin)
        return ma, mb

    # The tuned/production graph mode is driven by the shared online graph, so
    # the evaluator and the streaming sample run exactly the same code.
    if params.get("graph_mode") == "kf_prior":
        return _eval_online_kf_prior(c, params, cam, match_frames,
                                     raw_match_frames=raw_match_frames)

    # --- odometry pass (chain) + optional additive keyframe edges ---
    traj = Trajectory()
    node_poses = {0: traj.get_current_pose().copy()}
    odom_edges = []
    est_pos = [traj.get_current_position().copy()]
    kf_nodes = [0]
    last_kf = 0
    for n, o in enumerate(c["odom"]):
        a = n * stride
        i = (n + 1) * stride
        est_pos.append(node_poses[a][:3, 3].copy())
        if not o["ok"]:
            node_poses[i] = node_poses[a].copy()
        else:
            traj.add_relative_pose(o["R"], o["t"])
            node_poses[i] = traj.get_current_pose().copy()
            odom_edges.append((a, i, o["R"], o["t"]))
            est_pos[-1] = node_poses[i][:3, 3].copy()
        if params["odom_ref"] == "kf":
            # Chain-relative transform to the last keyframe (reference for the
            # translation-consistency gate and for keyframe promotion).
            dT = np.linalg.inv(node_poses[i]) @ node_poses[last_kf]
            # Local map: constrain the frame to the last K keyframes (K=1 keeps
            # the previous single-hub behaviour).
            for ref in kf_nodes[-max(1, params.get("kf_local_map_k", 1)):]:
                if ref == i - stride:
                    continue
                dT_ref = (dT if ref == last_kf
                          else np.linalg.inv(node_poses[i]) @ node_poses[ref])
                kres = match_frames(ref, i)
                if (kres is not None and kres.get("ok")
                        and kres.get("inlier_ratio", 0.0)
                        >= params.get("kf_edge_min_inlier", 0.0)
                        and _trans_consistent(
                            dT_ref, kres.get("t"),
                            params.get("trans_gate_deg", 0.0))):
                    odom_edges.append((ref, i, kres["R"], kres["t"]))
            trans = float(np.linalg.norm(dT[:3, 3]))
            co = (np.trace(dT[:3, :3]) - 1.0) / 2.0
            rot = float(np.degrees(np.arccos(np.clip(co, -1.0, 1.0))))
            gap = i - last_kf
            if (gap >= pa.kf_max_gap
                    or (gap >= pa.kf_min_gap
                        and (trans >= pa.kf_trans_thresh or rot >= pa.kf_rot_thresh))):
                last_kf = i
                kf_nodes.append(i)
    gt_pos = c["gt_pos"]

    keys = sorted(node_poses.keys())
    if params["odom_ref"] == "kf":
        kf = list(kf_nodes)
    else:
        kf = select_keyframes(keys, node_poses, pa)

    # Graph structure: "stride" keeps every stride frame as a node (legacy);
    # "kf" keeps only keyframes. Intermediate frames then only carry the
    # odometry chain used for propagation and for the final ATE.
    graph_mode = params.get("graph_mode", "stride")
    if graph_mode in ("kf", "kf_window"):
        if graph_mode == "kf_window":
            # Keyframes plus the first R stride frames after each keyframe
            # (short-baseline spokes), dropping the long mid-segment tail.
            r_frames = int(params.get("kf_window_r", 2)) * stride
            keep = set(kf)
            for k in keys:
                p = kf[max(0, np.searchsorted(kf, k, side="right") - 1)]
                if k - p <= r_frames:
                    keep.add(k)
            graph_nodes = sorted(keep)
        else:
            keep = set(kf)
            graph_nodes = list(kf)
        if params["odom_ref"] == "kf":
            graph_edges = [(i, j, R, t) for (i, j, R, t) in odom_edges
                           if i in keep and j in keep]
        else:
            # No additive keyframe spokes in prev mode: compose the odometry
            # chain into direct keyframe-to-keyframe measurements, and keep
            # any chain edges among the retained frames.
            graph_edges = []
            for a, b in zip(kf[:-1], kf[1:]):
                M = np.linalg.inv(node_poses[b]) @ node_poses[a]
                graph_edges.append((a, b, M[:3, :3].copy(), M[:3, 3].copy()))
            keep_set = set(graph_nodes)
            graph_edges += [(i, j, R, t) for (i, j, R, t) in odom_edges
                            if i in keep_set and j in keep_set]
    else:
        graph_nodes = keys
        graph_edges = odom_edges

    # --- pose graph + loop closure + per-edge scale ---
    opt = SlidingWindowOptimizer(
        window_size=None, max_iterations=params["loop_iterations"], huber=1.0,
        step_scale_t=params["step_scale_t"], step_scale_r=params["step_scale_t"],
        optimize_scale=True, scale_prior_sigma=params["scale_prior_sigma"],
        tsvd_ratio=(0.0 if (params.get("scale_kf_adaptive", False)
                            or params.get("scale_kf_adapt_loop_ratio", 0.0) > 0.0)
                    else params.get("tsvd_ratio", 0.0)))
    for idx in graph_nodes:
        opt.add_node(idx, node_poses[idx])
    added = set()
    for (i, j, R, t) in graph_edges:
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = np.asarray(t, float).reshape(3)
        opt.add_edge(i, j, M, scale_free=True)
        added.add(edge_key(i, j))
    gaps = [kf[i + 1] - kf[i] for i in range(len(kf) - 1)]
    kf_step = max(gaps) if gaps else stride * max(1, pa.keyframe_decim)
    odom_rot = {edge_key(i, j): np.asarray(R, float) for (i, j, R, t) in odom_edges}
    cum_rot = cumulative_rotations(keys, odom_rot)
    margin = max(int(1.5 * kf_step), kf_step)
    need = max(1, params["loop_temporal_k"])
    window = params["loop_window"] if params["loop_window"] > 0 else len(kf)
    hits_per_kf = [[] for _ in kf]
    n_loop = 0
    for bi, b in enumerate(kf):
        hits = []
        for ai in range(max(0, bi - window), bi):
            a = kf[ai]
            if b - a < params["loop_min_gap"]:
                continue
            lres = match_frames(a, b)
            if lres is None or not lres.get("ok"):
                continue
            if lres["inlier_ratio"] < params["loop_min_inlier"]:
                continue
            hits.append((int(a), np.asarray(lres["R"], float),
                         np.asarray(lres["t"], float).reshape(3),
                         float(lres["inlier_ratio"]),
                         int(lres.get("n_matches", -1))))
        hits_per_kf[bi] = hits
        accepted = confirmed_loop_hits(b, hits, bi, hits_per_kf, need, margin, added)
        for (a, R_c, t_c, inl, n_m) in accepted:
            if params["loop_rot_only"]:
                continue
            if params["cycle_threshold_deg"] > 0 and \
                    chain_residual_deg(a, b, R_c, cum_rot) > params["cycle_threshold_deg"]:
                continue
            M = np.eye(4)
            M[:3, :3] = R_c
            M[:3, 3] = t_c
            sig = {}
            if params["loop_sigma_scale"] > 0:
                sig = {"sigma_t": params["step_scale_t"] * params["loop_sigma_scale"],
                       "sigma_r": params["step_scale_t"] * params["loop_sigma_scale"]}
            opt.add_edge(a, b, M, scale_free=True, **sig)
            added.add(edge_key(a, b))
            n_loop += 1

    # Adaptive gate decided BEFORE the first optimize: loop-closure density is
    # known here and determines whether the graph can rely on KF scales
    # (dense loops) or needs TSVD (sparse loops, e.g. desk2 n_loop=1).
    loop_ratio_pre = n_loop / max(len(kf), 1)
    adapt_lr = float(params.get("scale_kf_adapt_loop_ratio", 0.0))
    tsvd_mode = bool(params.get("scale_kf", False)) and adapt_lr > 0.0 \
        and loop_ratio_pre < adapt_lr
    if tsvd_mode:
        opt.tsvd_ratio = float(params.get("tsvd_ratio", 0.0) or 1e-3)

    # Sequential mode owns all optimization: the graph above is only a
    # measurement container (odometry poses + edges), never batch-solved.
    if graph_mode != "kf_prior":
        opt.optimize()

    if graph_mode != "kf_prior" and params.get("scale_kf", False) and not tsvd_mode:
        # Linear-KF pre-pass over the time-ordered edge scales. The measurement
        # is z_e = s_e / m_e (m_e = chain baseline from the odometry poses).
        # The filtered coefficient k and its variance P then re-centre the
        # scale prior at log(k * m_e); a second (warm-start) optimize runs.
        kf_filter = ScaleKF(params.get("scale_kf_q", 1e-3),
                            params.get("scale_kf_r", 0.05))
        adaptive = bool(params.get("scale_kf_adaptive", False))
        tau = float(params.get("scale_kf_innov_tau", 3.0))
        order = []
        for e, (i, j, *_rest) in enumerate(opt.edges):
            if opt.scale_col[e] is None:
                continue  # gauge / fixed scale
            m = float(np.linalg.norm(
                node_poses[j][:3, 3] - node_poses[i][:3, 3]))
            if m < 1e-9:
                continue
            order.append((int(j), e, m, float(opt.edge_scale[e])))
        order.sort()
        mean_by_edge = {}
        innov = []
        for j, e, m, s in order:
            k_hat, p, inov = kf_filter.update(s / m)
            mean_by_edge[e] = float(np.log(max(k_hat * m, 1e-6)))
            if inov != 0.0:
                innov.append(abs(inov))
        for e, mu in mean_by_edge.items():
            opt.scale_prior_mean[e] = mu
        # Adaptive gating: if the scale coefficient does not follow the
        # motion model (large normalised innovation, e.g. desk2), enable TSVD
        # for the second pass; otherwise keep the KF prior only.
        score = float(np.median(innov)) if innov else 0.0
        if diag is not None:
            diag["kf_innov_median"] = score
            diag["kf_innov"] = [float(x) for x in innov]
        # Adaptive gating. Two candidate misfit signals:
        #  - normalised KF innovation (motion-model misfit)
        #  - loop-closure density: few confirmed loops means the graph relies
        #    on free-scale keyframe edges (e.g. desk2: n_loop=1).
        loop_ratio = n_loop / max(len(kf), 1)
        if diag is not None:
            diag["loop_ratio"] = float(loop_ratio)
        use_tsvd = False
        if adaptive and score > tau:
            use_tsvd = True
        lr_thresh = float(params.get("scale_kf_adapt_loop_ratio", 0.0))
        if lr_thresh > 0.0 and loop_ratio < lr_thresh:
            use_tsvd = True
        if use_tsvd:
            opt.tsvd_ratio = float(params.get("tsvd_ratio", 0.0) or 1e-3)
        sig = float(params.get("scale_kf_sigma", 0.5))
        if sig > 0.0:
            opt.scale_prior_sigma = sig
            opt.optimize()

    if graph_mode == "kf_prior":
        # Sequential, bounded graph: forward pass with fixed-lag windows and
        # transient marginalization. No global/batch solve is performed.
        opt = _sequential_kf_priors(opt, kf, keys, params)
        graph_nodes = list(kf)

    if diag is not None:
        # Per-edge diagnostics: optimized scale, endpoints and chain positions.
        diag["edges"] = [(int(i), int(j)) for (i, j, *_rest) in opt.edges]
        diag["edge_scale"] = [float(s) for s in opt.edge_scale]
        diag["scale_col"] = list(opt.scale_col)
        diag["node_pos"] = {int(k): np.asarray(v[:3, 3], float).tolist()
                            for k, v in node_poses.items()}

    if graph_mode in ("kf", "kf_window", "kf_prior"):
        # Frames not in the graph are propagated from the nearest optimized
        # keyframe along the odometry chain; ATE is still over every frame.
        # Per-segment scale: the optimized keyframe spacing may differ from
        # the raw odometry chain (scale-free edges), so rescale each segment's
        # relative translation before propagating the intermediate frames.
        node_set = set(graph_nodes)
        seg_alpha = {}
        for p, q in zip(kf[:-1], kf[1:]):
            d_opt = np.linalg.norm(
                opt.get_pose(q)[:3, 3] - opt.get_pose(p)[:3, 3])
            d_odom = np.linalg.norm(
                node_poses[q][:3, 3] - node_poses[p][:3, 3])
            seg_alpha[p] = d_opt / d_odom if d_odom > 1e-9 else 1.0
        est = []
        for k in keys:
            if k in node_set:
                est.append(opt.get_pose(k)[:3, 3])
                continue
            p = kf[max(0, np.searchsorted(kf, k, side="right") - 1)]
            rel = np.linalg.inv(node_poses[p]) @ node_poses[k]
            rel[:3, 3] *= seg_alpha.get(p, 1.0)
            est.append((opt.get_pose(p) @ rel)[:3, 3])
        est = np.array(est)
        if diag is not None:
            # Diagnostics: error of the optimized keyframes themselves.
            est_k = np.array([opt.get_pose(k)[:3, 3] for k in kf])
            gt_k = np.array([gt_pos[k // stride] for k in kf])
            if len(est_k) >= 3:
                sk, Rk, tk = umeyama(est_k, gt_k, with_scale=True)
                ek = np.linalg.norm(sk * (est_k @ Rk.T) + tk - gt_k, axis=1)
                diag["ate_kf"] = float(np.median(ek))
    else:
        est = np.array([opt.get_pose(k)[:3, 3] for k in keys])
    gt = np.array([gt_pos[k // stride] for k in keys])
    if len(est) >= 3:
        s, R_a, t_a = umeyama(est, gt, with_scale=True)
        aligned = s * (est @ R_a.T) + t_a
        err = np.linalg.norm(aligned - gt, axis=1)
        ate = float(np.median(err))
    else:
        ate = float("nan")
    return {"ATE_median": ate, "n_loop": n_loop, "n_kf": len(kf)}


# --------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------
def run_study(args):
    seqs = [s for s in args.seq.split(",") if s]
    caches = {s: load_cache(args.cache_dir, s) for s in seqs}
    # Match results depend only on the frame pair and the (fixed) matcher, so
    # they are valid across trials AND across process restarts. Persist them
    # to disk (keyed by seq + matcher backend) and reload on startup.
    cache_paths = {
        s: Path(args.cache_dir) / f"match_cache_{s}_{args.matcher}.pkl"
        for s in seqs
    }
    match_caches = {}
    for s in seqs:
        p = cache_paths[s]
        if p.exists():
            with open(p, "rb") as f:
                match_caches[s] = pickle.load(f)
            print(f"[match_cache] loaded {len(match_caches[s])} pairs "
                  f"from {p}", flush=True)
        else:
            match_caches[s] = {}
    cam = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                           width=args.width, height=args.height)
    if getattr(args, "auto_intrinsics", False):
        cams = {}
        for s in seqs:
            fx, fy, cx, cy = intrinsics_for(
                args.dataset_root, s, (args.fx, args.fy, args.cx, args.cy))
            cams[s] = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy,
                                       width=args.width, height=args.height)
    else:
        cams = {s: cam for s in seqs}
    if args.matcher == "torch":
        from torch_sinkhorn import TorchSinkhornMatcher
        desc_matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                            unused_score=1.0,
                                            distance_type="l2")
    else:
        desc_matcher = NumpySinkhornMatcher(iterations=20, epsilon=0.05,
                                            unused_score=1.0, distance_type="l2")

    def save_match_caches():
        for s in seqs:
            p = cache_paths[s]
            tmp = p.with_name(p.name + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(match_caches[s], f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, p)

    import rustuna
    if getattr(args, "sampler", "tpe") == "random":
        sampler = rustuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = rustuna.samplers.TPESampler(seed=args.seed)
    study_kwargs = dict(direction="minimize", sampler=sampler,
                        study_name=args.study_name)
    if args.storage:
        # Rustuna's SQLite storage is Optuna-compatible; optuna-dashboard can
        # read the same file (sqlite:///...). load_if_exists lets a stopped
        # run resume and only add the remaining trials.
        study_kwargs["storage"] = rustuna.storages.SQLite3Storage(args.storage)
        study_kwargs["load_if_exists"] = True
    study = rustuna.create_study(**study_kwargs)

    def objective(trial):
        def sf(name, low, high):
            return trial.suggest_float(name, low, high)

        def si(name, low, high):
            return trial.suggest_int(name, low, high)

        odom_ref = (args.fix_odom_ref or
                    trial.suggest_categorical("odom_ref", ["prev", "kf"]))
        kf_mode = (args.fix_kf_mode or
                   trial.suggest_categorical("kf_mode", ["decim", "motion"]))
        # keyframe_decim is only consulted when odom_ref="prev" AND
        # kf_mode="decim" (see eval_seq / select_keyframes). Likewise
        # kf_trans/rot are only used otherwise. Don't burn search dimensions
        # on parameters that cannot affect the objective.
        use_decim = odom_ref == "prev" and kf_mode == "decim"
        params = {
            "odom_ref": odom_ref,
            "kf_mode": kf_mode,
            "keyframe_decim": (trial.suggest_int("keyframe_decim", 4, 16)
                               if use_decim else 15),
            "kf_trans_thresh": (6.0 if use_decim else
                                sf("kf_trans_thresh", 2.0, 16.0)),
            "kf_rot_thresh": (10.0 if use_decim else
                              sf("kf_rot_thresh", 5.0, 45.0)),
            "loop_window": si("loop_window", 20, 80),
            "loop_min_gap": (args.fix_loop_min_gap if args.fix_loop_min_gap
                             is not None else
                             trial.suggest_int("loop_min_gap", 20, 60)),
            "loop_min_inlier": (args.fix_loop_min_inlier
                                if args.fix_loop_min_inlier is not None else
                                trial.suggest_float("loop_min_inlier",
                                                    0.25, 0.6)),
            "loop_temporal_k": si("loop_temporal_k", 1, 4),
            "scale_prior_sigma": sf("scale_prior_sigma", 0.2, 2.0),
            "step_scale_t": sf("step_scale_t", 0.02, 0.2),
            "loop_sigma_scale": sf("loop_sigma_scale", 0.0, 4.0),
            "loop_iterations": args.tune_iterations,
            "tsvd_ratio": args.tsvd_ratio,
            "scale_kf": args.scale_kf,
            "scale_kf_q": args.scale_kf_q,
            "scale_kf_r": args.scale_kf_r,
            "scale_kf_sigma": args.scale_kf_sigma,
            "scale_kf_adaptive": args.scale_kf_adaptive,
            "scale_kf_innov_tau": args.scale_kf_innov_tau,
            "scale_kf_adapt_loop_ratio": args.scale_kf_adapt_loop_ratio,
            "trans_gate_deg": args.trans_gate_deg,
            "kf_edge_min_inlier": args.kf_edge_min_inlier,
            "kf_local_map_k": args.kf_local_map_k,
            "loop_rot_only": False,
            "cycle_threshold_deg": 0.0,
        }
        per = {s: eval_seq(caches[s], params, cams[s], desc_matcher,
                           match_caches[s]) for s in seqs}
        vals = [per[s]["ATE_median"] for s in seqs]
        mean_med = float(np.mean(vals))
        trial.set_user_attr("per_seq", json.dumps(
            {s: per[s]["ATE_median"] for s in seqs}))
        trial.set_user_attr("eval_params", json.dumps(params))
        print(f"trial {len(study.trials)-1}: mean={mean_med:.4f} "
              + " ".join(f"{s}={per[s]['ATE_median']:.3f}" for s in seqs)
              + f" {json.dumps({k: round(v,3) if isinstance(v,float) else v for k,v in params.items()})}",
              flush=True)
        save_match_caches()
        return mean_med

    if not args.report_only:
        study.optimize(objective, args.n_trials)
        save_match_caches()
    for s in seqs:
        print(f"[match_cache] {s}: {len(match_caches[s])} frame pairs "
              f"memoized -> {cache_paths[s]}", flush=True)
    # FAIL/PRUNED trials (e.g. stale RUNNING ones marked on resume) have
    # value None; exclude them before sorting/comparing.
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        raise RuntimeError("no completed trials to report")
    best = min(completed, key=lambda t: t.value)

    def full_params_of(t):
        ep = t.user_attrs.get("eval_params")
        if ep:
            return json.loads(ep)
        # Older trials (pre eval_params) may omit conditionally-fixed keys.
        p = dict(t.params)
        p.setdefault("keyframe_decim", 15)
        p.setdefault("kf_trans_thresh", 6.0)
        p.setdefault("kf_rot_thresh", 10.0)
        p.setdefault("loop_min_gap", args.fix_loop_min_gap or 30)
        p.setdefault("loop_min_inlier", args.fix_loop_min_inlier or 0.4)
        p.setdefault("loop_iterations", args.tune_iterations)
        p.setdefault("tsvd_ratio", 0.0)
        p.setdefault("scale_kf", False)
        p.setdefault("scale_kf_q", 1e-3)
        p.setdefault("scale_kf_r", 0.05)
        p.setdefault("scale_kf_sigma", 0.5)
        p.setdefault("scale_kf_adaptive", False)
        p.setdefault("scale_kf_innov_tau", 3.0)
        p.setdefault("scale_kf_adapt_loop_ratio", 0.0)
        p.setdefault("trans_gate_deg", 0.0)
        p.setdefault("kf_edge_min_inlier", 0.0)
        p.setdefault("kf_local_map_k", 1)
        p.setdefault("loop_rot_only", False)
        p.setdefault("cycle_threshold_deg", 0.0)
        return p

    trials_sorted = sorted(completed, key=lambda t: t.value)
    top = [{"value": t.value, "params": t.params,
            "per_seq": json.loads(t.user_attrs["per_seq"])}
           for t in trials_sorted[:10]]
    result = {"best_value": best.value, "best_params": full_params_of(best),
              "best_per_seq": json.loads(best.user_attrs["per_seq"]),
              "top10": top, "n_trials": args.n_trials, "seed": args.seed,
              "seqs": seqs, "tune_iterations": args.tune_iterations}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["best_params"], indent=2))
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eval/pyramid_k512_l2_wd.onnx")
    ap.add_argument("--dataset-root", default="/home/ubuntu/datasets/tum_rgbd")
    ap.add_argument("--seq", default="desk,desk2")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune-iterations", type=int, default=20)
    ap.add_argument("--tsvd-ratio", type=float, default=0.0,
                    help="Truncated SVD: drop Hessian directions with "
                         "eigenvalue below this fraction of the max (0=off).")
    ap.add_argument("--scale-kf", action="store_true", default=False,
                    help="Enable the linear-KF scale-coefficient pre-pass "
                         "(re-centres the scale prior at k*motion).")
    ap.add_argument("--scale-kf-q", type=float, default=1e-3,
                    help="KF process variance q (how fast k may drift).")
    ap.add_argument("--scale-kf-r", type=float, default=0.05,
                    help="KF measurement variance r.")
    ap.add_argument("--scale-kf-sigma", type=float, default=0.5,
                    help="Scale-prior sigma used after the KF re-centering.")
    ap.add_argument("--scale-kf-adaptive", action="store_true", default=False,
                    help="Enable adaptive gating: only apply TSVD when the KF "
                         "normalised innovation indicates a motion-model "
                         "misfit.")
    ap.add_argument("--scale-kf-innov-tau", type=float, default=3.0,
                    help="Innovation (median |z|) threshold for adaptive TSVD.")
    ap.add_argument("--scale-kf-adapt-loop-ratio", type=float, default=0.0,
                    help="Adaptive TSVD by loop density: if n_loop/n_kf is "
                         "below this, enable TSVD for the second pass (0=off).")
    ap.add_argument("--trans-gate-deg", type=float, default=0.0,
                    help="With odom_ref=kf, reject additive keyframe edges whose "
                         "translation direction disagrees with the chain by more "
                         "than this angle (degrees). 0 disables.")
    ap.add_argument("--kf-edge-min-inlier", type=float, default=0.0,
                    help="With odom_ref=kf, minimum inlier ratio for additive "
                         "keyframe edges. 0 disables.")
    ap.add_argument("--kf-local-map-k", type=int, default=1,
                    help="With odom_ref=kf, constrain each frame to the last K "
                         "keyframes (local map). 1 = single-hub additive.")
    ap.add_argument("--cache-dir", default="eval/results/tune_cache_loop")
    ap.add_argument("--out", default="eval/results/rustuna_tune_loop.json")
    ap.add_argument("--storage", default=None,
                    help="SQLite file path for study persistence (optuna-"
                         "dashboard compatible). Empty = in-memory.")
    ap.add_argument("--study-name", default="vo_loop_tune")
    ap.add_argument("--fix-odom-ref", default=None, choices=["prev", "kf"],
                    help="Fix odom_ref instead of searching it.")
    ap.add_argument("--fix-kf-mode", default=None, choices=["decim", "motion"],
                    help="Fix kf_mode instead of searching it.")
    ap.add_argument("--fix-loop-min-gap", type=int, default=None,
                    help="Fix loop_min_gap instead of searching it.")
    ap.add_argument("--fix-loop-min-inlier", type=float, default=None,
                    help="Fix loop_min_inlier instead of searching it.")
    ap.add_argument("--matcher", default="torch", choices=["numpy", "torch"],
                    help="Descriptor matcher backend. 'torch' is ~6x faster "
                         "and float32-equivalent (tuning only).")
    ap.add_argument("--sampler", default="tpe", choices=["tpe", "random"],
                    help="Sampler backend. Use 'random' for unbiased exploration.")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="Do not run new trials; just write the result JSON "
                         "from an existing (storage-backed) study.")
    ap.add_argument("--fx", type=float, default=525.0)
    ap.add_argument("--auto-intrinsics", action="store_true", default=False,
                    help="Use per-camera TUM intrinsics (freiburg1/2/3).")
    ap.add_argument("--fy", type=float, default=525.0)
    ap.add_argument("--cx", type=float, default=320.0)
    ap.add_argument("--cy", type=float, default=240.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()
    if args.build_cache:
        build_cache(args)
    else:
        run_study(args)


if __name__ == "__main__":
    main()
