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
)
from vo.loop_closure import (  # noqa: E402
    confirmed_loop_hits,
    edge_key,
    local_candidate,
)
from vo.cycle_consistency import chain_residual_deg, cumulative_rotations  # noqa: E402
from vo.sinkhorn_numpy import NumpySinkhornMatcher  # noqa: E402
from vo.trajectory import Trajectory  # noqa: E402
from vo.se3_window import SlidingWindowOptimizer  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

DEFAULT_ARGS = SimpleNamespace(
    pose_source="essential", method="magsac", threshold=1.4, dbin=0.1,
    max_matches=1024, match_threshold=0.1, guided=False,
    guided_inlier_thresh=0.35, guided_sampson=2.0, depth_scale=5000.0,
    min_depth=0.1, max_depth=10.0,
)


def load_frames(dataset_root, seq, gt_max_diff=0.05):
    base = Path(dataset_root) / f"rgbd_dataset_freiburg1_{seq}"
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
    cam = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                           width=args.width, height=args.height)
    out = Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    seqs = [s for s in args.seq.split(",") if s]
    for seq in seqs:
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


# --------------------------------------------------------------------------
# Evaluator (mirrors run_vo's loop-closure block)
# --------------------------------------------------------------------------
def eval_seq(c, params, cam, desc_matcher, match_cache=None):
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
            if last_kf != i - stride:
                kres = match_frames(last_kf, i)
                if kres is not None and kres.get("ok"):
                    odom_edges.append((last_kf, i, kres["R"], kres["t"]))
            dT = np.linalg.inv(node_poses[i]) @ node_poses[last_kf]
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

    # --- pose graph + loop closure + per-edge scale ---
    opt = SlidingWindowOptimizer(
        window_size=None, max_iterations=params["loop_iterations"], huber=1.0,
        step_scale_t=params["step_scale_t"], step_scale_r=params["step_scale_t"],
        optimize_scale=True, scale_prior_sigma=params["scale_prior_sigma"])
    for idx, T in node_poses.items():
        opt.add_node(idx, T)
    added = set()
    for (i, j, R, t) in odom_edges:
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = np.asarray(t, float).reshape(3)
        opt.add_edge(i, j, M, scale_free=True)
        added.add(edge_key(i, j))

    keys = sorted(node_poses.keys())
    if params["odom_ref"] == "kf":
        kf = list(kf_nodes)
    else:
        kf = select_keyframes(keys, node_poses, pa)
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
    opt.optimize()

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
                                trial.suggest_float("kf_trans_thresh",
                                                    2.0, 16.0)),
            "kf_rot_thresh": (10.0 if use_decim else
                              trial.suggest_float("kf_rot_thresh",
                                                  5.0, 45.0)),
            "loop_window": trial.suggest_int("loop_window", 20, 80),
            "loop_min_gap": trial.suggest_int("loop_min_gap", 20, 60),
            "loop_min_inlier": trial.suggest_float("loop_min_inlier", 0.25, 0.6),
            "loop_temporal_k": trial.suggest_int("loop_temporal_k", 1, 4),
            "scale_prior_sigma": trial.suggest_float("scale_prior_sigma", 0.2, 2.0),
            "step_scale_t": trial.suggest_float("step_scale_t", 0.02, 0.2),
            "loop_sigma_scale": trial.suggest_float("loop_sigma_scale", 0.0, 4.0),
            "loop_iterations": args.tune_iterations,
            "loop_rot_only": False,
            "cycle_threshold_deg": 0.0,
        }
        per = {s: eval_seq(caches[s], params, cam, desc_matcher,
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
        p.setdefault("loop_iterations", args.tune_iterations)
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
    ap.add_argument("--matcher", default="numpy", choices=["numpy", "torch"],
                    help="Descriptor matcher backend. 'torch' is ~6x faster "
                         "and float32-equivalent (tuning only).")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="Do not run new trials; just write the result JSON "
                         "from an existing (storage-backed) study.")
    ap.add_argument("--fx", type=float, default=525.0)
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
