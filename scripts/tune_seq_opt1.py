"""Categorical tuning of the sequential fixed-lag graph (seq_opt1 + NL-Reg).

Discrete levels only (see notes/20260920-tuning-policy.md): no continuous
float search. Objective = mean ATE over the training sequences, with the
worst-case logged for overfitting monitoring. Structural settings are fixed
(graph_mode=kf_prior, NL-Reg on with rough constants) so the search only
covers the physically meaningful pose/loop/keyframe dimensions.
"""

import argparse
import importlib.util
import json
import os
import pickle
import sys
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

# Discrete levels (policy: 4-6 physically meaningful levels per dimension).
SPACE = {
    "step_scale_t": [0.05, 0.10, 0.15, 0.20],
    "scale_prior_sigma": [0.3, 0.6, 1.0, 1.5, 2.0],
    "loop_window": [20, 40, 60, 80],
    "loop_min_gap": [20, 30, 40, 60],
    "loop_min_inlier": [0.30, 0.40, 0.50, 0.60],
    "loop_temporal_k": [1, 2, 3, 4],
    "loop_sigma_scale": [0.0, 1.0, 2.0, 3.0],
    "kf_trans_thresh": [4.0, 8.0, 12.0, 16.0],
    "kf_rot_thresh": [10.0, 20.0, 30.0, 45.0],
}

FIXED = {
    "odom_ref": "kf", "kf_mode": "motion", "keyframe_decim": 15,
    "kf_local_map_k": 1, "kf_edge_min_inlier": 0.0, "trans_gate_deg": 0.0,
    "loop_iterations": 10, "tsvd_ratio": 0.0,
    "scale_kf": False, "scale_kf_adaptive": False,
    "scale_kf_adapt_loop_ratio": 0.0, "scale_kf_innov_tau": 3.0,
    "scale_kf_q": 1e-3, "scale_kf_r": 0.1, "scale_kf_sigma": 0.5,
    "loop_rot_only": False, "cycle_threshold_deg": 0.0,
    "graph_mode": "kf_prior", "max_keyframes": 3, "seq_tsvd_ratio": 0.0,
    "nl_reg": True, "nl_reg_c": 10.0, "nl_reg_tau": 10.0, "nl_reg_length": 1.0,
}

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="desk,desk2,room")
ap.add_argument("--n-trials", type=int, default=24)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--storage",
                default=str(REPO / "eval/results/tune_seq_opt1_optuna.db"))
ap.add_argument("--study-name", default="seq_opt1")
ap.add_argument("--out", default=str(REPO / "eval/results/tune_seq_opt1.json"))
args = ap.parse_args()

seqs = [s for s in args.seq.split(",") if s]
matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                               unused_score=1.0, distance_type="l2")
caches, cams, mcaches = {}, {}, {}
for s in seqs:
    caches[s] = rtl.load_cache(CACHE, s)
    pkl = CACHE / f"match_cache_{s}_numpy.pkl"
    mcaches[s] = pickle.load(open(pkl, "rb")) if pkl.exists() else {}
    cams[s] = CameraIntrinsics(
        *intrinsics_for(DATASET_ROOT, s, (525., 525., 320., 240.)),
        width=640, height=480)

import optuna  # noqa: E402

# Optuna (not rustuna) is used here: rustuna stores numeric categorical params
# and category labels in a non-JSON format that optuna-dashboard cannot read.
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed),
    study_name=args.study_name,
    storage=f"sqlite:///{args.storage}",
    load_if_exists=True)


def objective(trial):
    params = dict(FIXED)
    for name, levels in SPACE.items():
        params[name] = trial.suggest_categorical(name, levels)
    per = {s: rtl.eval_seq(caches[s], params, cams[s], matcher, mcaches[s])
           for s in seqs}
    vals = [per[s]["ATE_median"] for s in seqs]
    mean = float(np.mean(vals))
    worst = float(np.max(vals))
    trial.set_user_attr("per_seq", json.dumps({s: per[s]["ATE_median"]
                                               for s in seqs}))
    trial.set_user_attr("worst", str(worst))
    trial.set_user_attr("eval_params", json.dumps(params))
    print(f"trial {len(study.trials)-1}: mean={mean:.4f} worst={worst:.4f} "
          + " ".join(f"{s}={per[s]['ATE_median']:.3f}" for s in seqs)
          + f" {json.dumps({k: params[k] for k in SPACE})}", flush=True)
    return mean


study.optimize(objective, args.n_trials)

completed = [t for t in study.trials if t.value is not None]
best = min(completed, key=lambda t: t.value)
result = {
    "best_value": best.value,
    "best_params": json.loads(best.user_attrs["eval_params"]),
    "best_per_seq": json.loads(best.user_attrs["per_seq"]),
    "n_trials": len(completed),
}
json.dump(result, open(args.out, "w"), indent=2)
print("wrote", args.out, flush=True)
