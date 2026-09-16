"""Hyperparameter tuning of the matching/pose stage with Rustuna.

Two-phase design:
1. --build-cache: run the matcher ONNX exactly once per pair and cache the
   config-independent part of match extraction (mutual-NN matches, scores,
   dustbin margins, keypoint coords) plus the GT positions. Trials afterwards
   re-use the cache and skip ONNX inference entirely.
2. Rustuna TPE study over (method, threshold, dbin, guided_inlier_thresh,
   guided_sampson). Objective: mean ATE_med over sequences. The cache-based
   evaluator replicates eval_tum_vo.estimate_pair (essential path) and the
   trajectory chaining in run_vo.

Run:
    .venv/bin/python eval/rustuna_tune.py --model eval/pyramid_k512_l2.onnx \
        --n-trials 100 --out eval/results/rustuna_tune.json
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
import onnxruntime as ort  # noqa: E402

from eval_tum_vo import (  # noqa: E402
    _cross,
    nearest_timestamp,
    quat_to_se3,
    read_path_file,
    read_tum_file,
    umeyama,
)
from pytorch_model.vo.pose_estimation import CameraIntrinsics, estimate_pose_ransac  # noqa: E402
from pytorch_model.vo.trajectory import Trajectory  # noqa: E402
from eval.sampson_all import sampson_all  # noqa: E402


# --------------------------------------------------------------------------
# Data assembly (mirrors run_vo in eval_tum_vo.py, essential path only)
# --------------------------------------------------------------------------
def load_frames(dataset_root, seq, stride, gt_max_diff=0.05):
    base = Path(dataset_root) / f"rgbd_dataset_freiburg1_{seq}"
    gt_rows = read_tum_file(base / "groundtruth.txt")
    frame_list = read_path_file(base / "rgb.txt")
    gt_poses = [(ts, quat_to_se3(vals[:3], vals[3:7])) for ts, vals in gt_rows]
    frames = []
    for ts, relative_path in frame_list:
        gt_match = nearest_timestamp(gt_poses, ts, gt_max_diff)
        if gt_match is None:
            continue
        frames.append((ts, base / relative_path, gt_match[1]))
    return frames


# --------------------------------------------------------------------------
# Phase 1: cache builder
# --------------------------------------------------------------------------
def build_cache(args):
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    cam = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                           width=args.width, height=args.height)
    out_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = ["desk", "desk2", "room"] if args.seq == "all" else [args.seq]
    for seq in seqs:
        frames = load_frames(args.dataset_root, seq, args.stride)
        n_pairs = (len(frames) - 1) // args.stride
        js, scs, d1s, k1s, k2s = [], [], [], [], []
        gt_pos = [frames[0][2][:3, 3].copy()]
        for n in range(n_pairs):
            i = n * args.stride
            pa = frames[i][1]
            pb = frames[i + args.stride][1]
            gt_pos.append(frames[i + args.stride][2][:3, 3].copy())
            a = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
            b = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
            if a is None or b is None:
                raise RuntimeError(f"unreadable image pair: {pa} / {pb}")
            a = cv2.resize(a, (args.width, args.height)).astype(np.float32)[None, None]
            b = cv2.resize(b, (args.width, args.height)).astype(np.float32)[None, None]
            k1, k2, P = session.run(None, {"image1": a, "image2": b})
            js_, scs_, d1_, k1_, k2_ = mutual_matches(k1, k2, P)
            js.append(js_); scs.append(scs_); d1s.append(d1_)
            k1s.append(k1_); k2s.append(k2_)
            if n % 100 == 0:
                print(f"  [{seq}] pair {n}/{n_pairs}", flush=True)
        np.savez_compressed(
            out_dir / f"{seq}.npz",
            j=np.array(js, dtype=object),
            score=np.array(scs, dtype=object),
            d1=np.array(d1s, dtype=object),
            k1=np.array(k1s, dtype=object),
            k2=np.array(k2s, dtype=object),
            gt_pos=np.array(gt_pos, dtype=np.float64),
            stride=np.int64(args.stride),
        )
        print(f"[cache] {seq}: {n_pairs} pairs -> {out_dir / f'{seq}.npz'}", flush=True)


def mutual_matches(kpts1, kpts2, P):
    """Config-independent part of extract_matches: mutual-NN + padding only.

    Returns per pair: (j, score, d1, k1c, k2c) where d1 is the dustbin
    probability of the source point (image1 side), which is all that
    dustbin_margin_filter uses for the mutual subset.
    """
    Pm = P[0]
    k1 = kpts1[0]
    k2 = kpts2[0]
    K = k1.shape[0]
    core = Pm[:K, :K]
    max_j = np.argmax(core, axis=1)
    max_i = np.argmax(core, axis=0)
    mutual = np.arange(K) == max_i[max_j]
    scores = core[np.arange(K), max_j]
    pad = (k1[:, 0] >= 0) & (k1[:, 1] >= 0) & (k2[:, 0] >= 0) & (k2[:, 1] >= 0)
    idx = np.where(mutual & pad)[0]
    j = max_j[idx]
    sc = scores[idx]
    d1 = Pm[idx, K]
    return j.astype(np.int32), sc.astype(np.float32), d1.astype(np.float32), \
        k1[idx].astype(np.float32), k2[j].astype(np.float32)


# --------------------------------------------------------------------------
# Phase 2: cache-based evaluation of one config
# --------------------------------------------------------------------------
def load_cache(cache_dir, seq):
    z = np.load(Path(cache_dir) / f"{seq}.npz", allow_pickle=True)
    return {
        "j": list(z["j"]), "score": list(z["score"]), "d1": list(z["d1"]),
        "k1": list(z["k1"]), "k2": list(z["k2"]),
        "gt_pos": z["gt_pos"], "stride": int(z["stride"]),
    }


def eval_seq(d, params, cam, max_matches=1024):
    """Replicates estimate_pair (essential path) + run_vo chaining for one seq."""
    method = cv2.USAC_MAGSAC if params["method"] == "magsac" else cv2.RANSAC
    traj = Trajectory()
    est_pos = [traj.get_current_position().copy()]
    gt_pos = [d["gt_pos"][0].copy()]
    n_ok = 0
    inls = []
    n_pairs = len(d["j"])
    for n in range(n_pairs):
        sc = d["score"][n]
        valid = (sc - d["d1"][n]) >= params["dbin"]
        order = np.argsort(sc[valid])[::-1][:max_matches]
        mk1 = d["k1"][n][valid][order]
        mk2 = d["k2"][n][valid][order]
        if len(mk1) < 5:
            est_pos.append(traj.get_current_position().copy())
            gt_pos.append(d["gt_pos"][n + 1].copy())
            continue
        R, t, mask = estimate_pose_ransac(
            mk1, mk2, cam, ransac_threshold=params["threshold"], method=method,
        )
        if R is not None and params["guided_inlier_thresh"] > 0.0:
            ir = float(np.sum(mask) / len(mk1))
            if ir < params["guided_inlier_thresh"]:
                Kmat = cam.K
                E = _cross(t.ravel()) @ R
                F = np.linalg.inv(Kmat).T @ E @ np.linalg.inv(Kmat)
                h1 = np.concatenate([mk1, np.ones((len(mk1), 1))], axis=1)
                h2 = np.concatenate([mk2, np.ones((len(mk2), 1))], axis=1)
                dist = sampson_all(h1, h2, F)
                keep = dist < params["guided_sampson"]
                if keep.sum() >= 5:
                    R2, t2, mask2 = estimate_pose_ransac(
                        mk1[keep], mk2[keep], cam,
                        ransac_threshold=params["threshold"], method=method,
                    )
                    if R2 is not None:
                        R, t, mask = R2, t2, mask2
        if R is None:
            est_pos.append(traj.get_current_position().copy())
            gt_pos.append(d["gt_pos"][n + 1].copy())
            continue
        traj.add_relative_pose(R, t)
        est_pos.append(traj.get_current_position().copy())
        gt_pos.append(d["gt_pos"][n + 1].copy())
        n_ok += 1
        inls.append(float(np.sum(mask) / len(mk1)))

    est = np.array(est_pos)
    gt = np.array(gt_pos)
    if len(est) >= 3:
        s, R_a, t_a = umeyama(est, gt, with_scale=True)
        aligned = s * (est @ R_a.T) + t_a
        err = np.linalg.norm(aligned - gt, axis=1)
        ate_med = float(np.median(err))
        ate_rmse = float(np.sqrt(np.mean(err ** 2)))
    else:
        ate_med = ate_rmse = float("nan")
    return {"ATE_median": ate_med, "ATE_RMSE": ate_rmse, "n_pairs": n_pairs,
            "n_ok": n_ok, "mean_inlier": float(np.mean(inls)) if inls else 0.0}


def load_all_cache(cache_dir, seqs):
    return {seq: load_cache(cache_dir, seq) for seq in seqs}


# --------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------
def run_study(args):
    seqs = ["desk", "desk2", "room"] if args.seq == "all" else [args.seq]
    cache = load_all_cache(args.cache_dir, seqs)
    cam = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                           width=args.width, height=args.height)

    import rustuna

    sampler = rustuna.samplers.TPESampler(seed=args.seed)
    study = rustuna.create_study(direction="minimize", sampler=sampler,
                                 study_name="vo_matching_tune")

    def objective(trial):
        params = {
            "method": trial.suggest_categorical("method", ["ransac", "magsac"]),
            "threshold": trial.suggest_float("threshold", 0.5, 3.0),
            "dbin": trial.suggest_float("dbin", 0.0, 0.5),
            "guided_inlier_thresh": trial.suggest_float("guided_inlier_thresh", 0.0, 0.8),
            "guided_sampson": trial.suggest_float("guided_sampson", 1.0, 4.0),
        }
        per_seq = {seq: eval_seq(cache[seq], params, cam) for seq in seqs}
        vals = [per_seq[seq]["ATE_median"] for seq in seqs]
        mean_med = float(np.mean(vals))
        # rustuna 0.1.0 accepts str-only user attrs
        trial.set_user_attr("per_seq", json.dumps(
            {seq: per_seq[seq]["ATE_median"] for seq in seqs}))
        trial.set_user_attr("n_ok", json.dumps(
            {seq: per_seq[seq]["n_ok"] for seq in seqs}))
        print(f"trial {len(study.trials) - 1}: mean={mean_med:.4f} "
              f"desk={vals[0]:.3f} desk2={vals[1]:.3f} room={vals[2]:.3f} "
              f"params={json.dumps(params)}", flush=True)
        return mean_med

    study.optimize(objective, args.n_trials)

    best = study.best_trial
    trials_sorted = sorted(study.trials, key=lambda t: t.value)
    top = []
    for t in trials_sorted[:10]:
        ps = t.user_attrs.get("per_seq")
        top.append({"value": t.value, "params": t.params,
                    "per_seq": json.loads(ps) if isinstance(ps, str) else ps})
    result = {
        "best_value": best.value,
        "best_params": best.params,
        "best_per_seq": (json.loads(best.user_attrs["per_seq"])
                         if isinstance(best.user_attrs.get("per_seq"), str)
                         else best.user_attrs.get("per_seq")),
        "top10": top,
        "n_trials": args.n_trials,
        "seed": args.seed,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["best_params"], indent=2))
    print(f"best mean ATE_med: {best.value:.4f}")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset-root", default="/home/ubuntu/datasets/tum_rgbd")
    ap.add_argument("--seq", default="all")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default="eval/results/tune_cache")
    ap.add_argument("--out", default="eval/results/rustuna_tune.json")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="evaluate fixed known configs from cache and exit")
    ap.add_argument("--fx", type=float, default=525.0)
    ap.add_argument("--fy", type=float, default=525.0)
    ap.add_argument("--cx", type=float, default=320.0)
    ap.add_argument("--cy", type=float, default=240.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    if args.build_cache or args.check:
        if args.build_cache:
            build_cache(args)
        if args.check:
            seqs = ["desk", "desk2", "room"] if args.seq == "all" else [args.seq]
            cache = load_all_cache(args.cache_dir, seqs)
            cam = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                                   width=args.width, height=args.height)
            known = {
                "[1] ransac1.0": {"method": "ransac", "threshold": 1.0, "dbin": 0.3,
                                  "guided_inlier_thresh": 0.0, "guided_sampson": 2.0},
                "[8] magsac1.0+guided": {"method": "magsac", "threshold": 1.0, "dbin": 0.3,
                                         "guided_inlier_thresh": 0.35, "guided_sampson": 2.0},
            }
            for name, params in known.items():
                vals = {seq: eval_seq(cache[seq], params, cam) for seq in seqs}
                print(f"{name}: " + " ".join(
                    f"{seq} med={vals[seq]['ATE_median']:.3f} rmse={vals[seq]['ATE_RMSE']:.3f} "
                    f"ok={vals[seq]['n_ok']}/{vals[seq]['n_pairs']}" for seq in seqs), flush=True)
        return
    run_study(args)


if __name__ == "__main__":
    main()
