"""Categorical tuning of the bounded online pose graph (OnlinePoseGraph).

Objective = mean + worst ATE over the training sequences (lambda=1), matching
notes/20260921-online-retuning-plan.md and the tuning policy. Discrete levels
only (no continuous float search). Structural settings are fixed; the search
covers pose/loop/keyframe dimensions plus the bounding knobs that now matter
(max_keyframes, held_cap, global_opt_period).
"""

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

if os.environ.get("TORCH_THREADS"):
    import torch
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

spec = importlib.util.spec_from_file_location(
    "rtl", REPO / "eval/rustuna_tune_loop.py")
rtl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtl)
from eval_tum_vo import intrinsics_for  # noqa: E402
from torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

DATASET_ROOT = "/home/ubuntu/datasets/tum_rgbd"
CACHE = REPO / "eval/results/tune_cache_loop"

# Discrete levels (4-5 per dimension, per the tuning policy).
SPACE = {
    "step_scale_t": [0.05, 0.10, 0.15, 0.20],
    "scale_prior_sigma": [0.3, 0.6, 1.0, 1.5, 2.0],
    "kf_trans_thresh": [4.0, 8.0, 12.0, 16.0],
    "kf_rot_thresh": [10.0, 20.0, 30.0, 45.0],
    "kf_max_gap": [8, 12, 16, 24],
    "kf_local_map_k": [1, 2, 3],
    "loop_window": [20, 40, 60, 80],
    "loop_min_gap": [20, 30, 40, 60],
    "loop_min_inlier": [0.30, 0.40, 0.50, 0.60],
    "loop_temporal_k": [1, 2, 3, 4],
    "loop_sigma_scale": [0.0, 1.0, 2.0, 3.0],
    "nl_reg_c": [3.0, 10.0, 30.0],
    "nl_reg_tau": [1.0, 3.0, 10.0, 30.0],
    "nl_reg_length": [0.5, 1.0, 2.0],
    "max_keyframes": ["2", "3", "5", "8", "12"],
    "held_cap": ["auto", "4", "8"],
    "global_opt_period": [0, 5, 10, 20],
}

FIXED = {
    "odom_ref": "kf", "kf_mode": "motion", "keyframe_decim": 15,
    "kf_edge_min_inlier": 0.0, "trans_gate_deg": 0.0,
    "loop_iterations": 10, "tsvd_ratio": 0.0,
    "scale_kf": False, "scale_kf_adaptive": False,
    "scale_kf_adapt_loop_ratio": 0.0, "scale_kf_innov_tau": 3.0,
    "scale_kf_q": 1e-3, "scale_kf_r": 0.1, "scale_kf_sigma": 0.5,
    "loop_rot_only": False, "cycle_threshold_deg": 0.0,
    "graph_mode": "kf_prior", "seq_tsvd_ratio": 0.0,
    "nl_reg": True, "global_opt_on_loop": True,
}


def decode(raw):
    params = dict(FIXED)
    params.update(raw)
    params["max_keyframes"] = (None if params["max_keyframes"] == "none"
                               else int(params["max_keyframes"]))
    params["held_cap"] = (None if params["held_cap"] in ("auto", "none")
                          else int(params["held_cap"]))
    if "scale_kf" in params:
        params["scale_kf"] = bool(params["scale_kf"])
    return params


ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="desk,desk2,room")
ap.add_argument("--n-trials", type=int, default=24)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--storage",
                default=str(REPO / "eval/results/tune_online_optuna.db"))
ap.add_argument("--study-name", default="online_bounded")
ap.add_argument("--out", default=str(REPO / "eval/results/tune_online.json"))
ap.add_argument("--with-scale-kf", action="store_true",
                help="also search the sequential Kalman-filter (scale_kf) knobs")
args = ap.parse_args()

if args.with_scale_kf:
    # The Kalman filter replaced the previous fixed scale_kf=False assumption
    # (commit 8075067), so expose it (and its prior sigma, only when on) to the
    # search. Kept out of SPACE because sigma is conditional on scale_kf.
    FIXED.pop("scale_kf", None)

seqs = [s for s in args.seq.split(",") if s]
matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                               unused_score=1.0, distance_type="l2")
caches, cams, mcaches = {}, {}, {}
for s in seqs:
    caches[s] = rtl.load_cache(CACHE, s)
    pkl = CACHE / f"match_cache_{s}_torch.pkl"
    mcaches[s] = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
    cams[s] = CameraIntrinsics(
        *intrinsics_for(DATASET_ROOT, s, (525., 525., 320., 240.)),
        width=640, height=480)

import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed),
    study_name=args.study_name, storage=f"sqlite:///{args.storage}",
    load_if_exists=True)


def objective(trial):
    raw = {name: trial.suggest_categorical(name, levels)
           for name, levels in SPACE.items()}
    if args.with_scale_kf:
        raw["scale_kf"] = trial.suggest_categorical("scale_kf", [0, 1])
        if raw["scale_kf"]:
            raw["scale_kf_sigma"] = trial.suggest_categorical(
                "scale_kf_sigma", [0.3, 0.5, 1.0])
    params = decode(raw)
    per, meta = {}, {}
    t0 = time.time()
    for s in seqs:
        r = rtl.eval_seq(caches[s], params, cams[s], matcher, mcaches[s])
        per[s] = r["ATE_median"]
        meta[s] = {"n_loop": r["n_loop"], "n_kf": r["n_kf"]}
    vals = [per[s] for s in seqs]
    mean = float(np.mean(vals))
    worst = float(np.max(vals))
    score = mean + worst
    trial.set_user_attr("per_seq", json.dumps(per))
    trial.set_user_attr("worst", str(worst))
    trial.set_user_attr("score_mean_plus_worst", str(score))
    trial.set_user_attr("eval_params", json.dumps(params))
    trial.set_user_attr("meta", json.dumps(meta))
    trial.set_user_attr("seconds", str(round(time.time() - t0, 1)))
    print(f"trial {len(study.trials) - 1}: score={score:.4f} mean={mean:.4f} "
          f"worst={worst:.4f} "
          + " ".join(f"{s}={per[s]:.3f}" for s in seqs)
          + f" [kf={raw['max_keyframes']} held={raw['held_cap']} "
          f"gop={raw['global_opt_period']}"
          + (f" skf={raw['scale_kf']}/{raw.get('scale_kf_sigma', '-')}"
             if "scale_kf" in raw else "")
          + "]", flush=True)
    return score


try:
    study.optimize(objective, args.n_trials)
finally:
    # Persist the match cache so repeated pairs stay deterministic across runs.
    for s in seqs:
        pkl = CACHE / f"match_cache_{s}_torch.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(mcaches[s], f, protocol=pickle.HIGHEST_PROTOCOL)

completed = [t for t in study.trials if t.value is not None]
best = min(completed, key=lambda t: t.value)
result = {
    "best_score_mean_plus_worst": best.value,
    "best_per_seq": json.loads(best.user_attrs["per_seq"]),
    "best_params": json.loads(best.user_attrs["eval_params"]),
    "n_trials": len(completed),
}
json.dump(result, open(args.out, "w"), indent=2)
print("wrote", args.out, flush=True)
