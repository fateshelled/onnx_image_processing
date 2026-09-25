"""Overnight optuna tuning with a gauge-free RPE objective.

Objective: mean over the training sequences of ``RPE_trans_2m_median`` (the
median relative-translation error over 2 m GT segments, global scale fixed from
the whole trajectory).  Per-sequence RPE/ATE and graph counters are stored as
trial user attributes for post-hoc plateau / Pareto selection.

Follows notes/20260920-tuning-policy.md: discretised categorical space, TPE,
no continuous suggest, holdout sequences are not used here.

Usage:
  .venv/bin/python scripts/tune_rpe.py --n-trials 200 \
      --storage eval/results/tune_rpe.db --study-name rpe_seq_opt1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import optuna

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

spec = importlib.util.spec_from_file_location(
    "rtl", REPO / "eval/rustuna_tune_loop.py")
rtl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtl)

from vo.pose_estimation import CameraIntrinsics  # noqa: E402

DATASET_ROOT = "/home/ubuntu/datasets/tum_rgbd"
CACHE = REPO / "eval/results/tune_cache_loop"
OBJECTIVE_METRIC = "RPE_trans_2m_median"

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
    "kf_max_gap": [8, 12, 16, 24],
    "max_keyframes": [3, 5, 8, 12, 16],
    "cycle_threshold_deg": [0.0, 2.0, 5.0, 10.0],
    "loop_verifier": ["none", "sim3"],
}
CONDITIONAL = {
    "loop_verifier_gate": ([4, 6, 8, 10], "loop_verifier", ["sim3"]),
}

FIXED = {
    "odom_ref": "kf", "kf_mode": "motion", "keyframe_decim": 15,
    "kf_local_map_k": 1, "kf_edge_min_inlier": 0.0, "trans_gate_deg": 0.0,
    "loop_iterations": 10, "tsvd_ratio": 0.0,
    "scale_kf": False, "scale_kf_adaptive": False,
    "scale_kf_adapt_loop_ratio": 0.0, "scale_kf_innov_tau": 3.0,
    "scale_kf_q": 1e-3, "scale_kf_r": 0.1, "scale_kf_sigma": 0.5,
    "loop_rot_only": False, "loop_robust": "none",
    "graph_mode": "kf_prior", "seq_tsvd_ratio": 0.0,
    # NL-Reg constants follow the established fixed tuning set for seq_opt1
    # (notes/20260920-tuning-policy.md), not the runtime default (30/0.5).
    "nl_reg": True, "nl_reg_c": 10.0, "nl_reg_tau": 10.0, "nl_reg_length": 1.0,
    "loop_verifier_window": 4, "loop_verifier_keyframes": True,
    "loop_verifier_abstain": "accept",
}


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="desk,desk2,room")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=None,
                    help="seconds; stops the study when reached")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seqs = [s for s in args.seq.split(",") if s]
    # Tie the study/db/output names to the exact configuration so a changed
    # space or sequence set cannot silently mix trials from different
    # objectives in one study.
    blob = json.dumps({"seqs": seqs, "space": SPACE, "conditional": CONDITIONAL,
                       "fixed": FIXED, "metric": OBJECTIVE_METRIC},
                      sort_keys=True, default=str)
    config_id = hashlib.sha1(blob.encode()).hexdigest()[:8]
    args.storage = Path(args.storage) if args.storage else (
        REPO / f"eval/results/tune_rpe_{config_id}.db")
    args.study_name = args.study_name or f"rpe_{config_id}"
    args.out = Path(args.out) if args.out else (
        REPO / f"eval/results/tune_rpe_{config_id}.json")
    print(f"config_id={config_id} study={args.study_name} "
          f"storage={args.storage}", flush=True)
    from eval.torch_sinkhorn import TorchSinkhornMatcher
    matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")
    caches, cams, mcaches, match_paths = {}, {}, {}, {}
    for s in seqs:
        c = rtl.load_cache(CACHE, s)
        frames = rtl.load_frames(DATASET_ROOT, s)
        if len(frames) != int(c.get("n_frames", len(frames))):
            raise ValueError(f"{s}: frame count {len(frames)} != "
                             f"n_frames {c.get('n_frames')}")
        gt_pose = np.array(
            [frames[i][1] for i in range(0, len(frames), int(c["stride"]))],
            dtype=np.float64)
        if len(gt_pose) != len(c["gt_pos"]):
            raise ValueError(f"{s}: gt_pose {len(gt_pose)} != gt_pos "
                             f"{len(c['gt_pos'])}")
        c["gt_pose"] = gt_pose
        caches[s] = c
        pkl = CACHE / f"match_cache_{s}_torch.pkl"
        if not pkl.exists():
            pkl = CACHE / f"match_cache_{s}_numpy.pkl"
        with pkl.open("rb") as handle:
            mcaches[s] = pickle.load(handle)
        match_paths[s] = pkl
        fx, fy, cx, cy = rtl.intrinsics_for(DATASET_ROOT, s,
                                            (525., 525., 320., 240.))
        cams[s] = CameraIntrinsics(fx, fy, cx, cy, 640, 480)

    args.storage.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name=args.study_name, storage=f"sqlite:///{args.storage}",
        load_if_exists=True)

    def objective(trial):
        params = dict(FIXED)
        for name, levels in SPACE.items():
            params[name] = trial.suggest_categorical(name, levels)
        for name, (levels, dep, values) in CONDITIONAL.items():
            params[name] = (trial.suggest_categorical(name, levels)
                            if params[dep] in values else levels[0])
        per = {}
        try:
            for s in seqs:
                res = rtl.eval_seq(caches[s], params, cams[s], matcher,
                                   mcaches[s])
                per[s] = {
                    "RPE_trans_2m_median": _finite(res.get("RPE_trans_2m_median")),
                    "RPE_trans_1m_median": _finite(res.get("RPE_trans_1m_median")),
                    "RPE_trans_5m_median": _finite(res.get("RPE_trans_5m_median")),
                    "RPE_rot_2m_median_deg": _finite(res.get("RPE_rot_2m_median_deg")),
                    "ATE_median": _finite(res.get("ATE_median")),
                    "n_loop": res.get("n_loop"),
                }
        finally:
            # Persist newly computed matches so a restart/timeout does not
            # change the RANSAC-nondeterministic pairs.
            for s in seqs:
                with match_paths[s].open("wb") as handle:
                    pickle.dump(mcaches[s], handle)
        values = [per[s][OBJECTIVE_METRIC] for s in seqs]
        trial.set_user_attr("per_seq", json.dumps(per))
        trial.set_user_attr("eval_params", json.dumps(params))
        # ``worst`` must stay a finite float: the selection tool does
        # ``float(user_attrs["worst"])`` and would crash on None.
        if any(v is None for v in values):
            trial.set_user_attr("worst", 1e6)
            return 1e6
        trial.set_user_attr("worst", float(max(values)))
        return float(np.mean(values))

    def report(study, trial):
        if trial.value is not None:
            print(f"trial {trial.number}: value={trial.value:.4f} "
                  + json.dumps(trial.params), flush=True)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   callbacks=[report])

    completed = [t for t in study.trials if t.value is not None]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not completed:
        args.out.write_text(json.dumps(
            {"objective_metric": OBJECTIVE_METRIC, "n_completed": 0}))
        print("no completed trials; wrote", args.out, flush=True)
        raise SystemExit(1)
    best = min(completed, key=lambda t: t.value)
    if best.value >= 1e6:
        print("WARNING: best trial hit the nan penalty; no usable config",
              flush=True)
    result = {
        "objective_metric": OBJECTIVE_METRIC,
        "config_id": config_id,
        "study_name": args.study_name,
        "best_trial": best.number,
        "best_value": best.value,
        "best_worst": best.user_attrs.get("worst"),
        "best_params": json.loads(best.user_attrs.get("eval_params", "{}")),
        "best_per_seq": json.loads(best.user_attrs.get("per_seq", "{}")),
        "n_completed": len(completed),
    }
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
