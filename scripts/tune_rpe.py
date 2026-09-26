"""Overnight optuna tuning for the online VO, split by role.

Two phases:

* ``odometry`` (Phase 1): loop closure is disabled (``loop_enable=False``) and
  the objective is the gauge-free RPE (``RPE_trans_2m_median``) so only the
  local odometry / sliding-graph parameters are optimised.
* ``loop`` (Phase 2): loops are enabled and the objective is ``ATE_median``,
  with the odometry parameters fixed from a Phase-1 result
  (``--odometry-json``); RPE is stored alongside as a constraint signal.

Discretised categorical space, TPE, holdout sequences are never used for
selection (notes/20260920-tuning-policy.md).

Usage:
  .venv/bin/python scripts/tune_rpe.py --phase odometry --n-trials 300 \
      --storage eval/results/tune_odom.db --study-name odom_v1  # optional; defaults are phase+hash
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
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

# Shared fixed graph/odometry keys (loop-specific keys are per phase).
ODOMETRY_FIXED = {
    "odom_ref": "kf", "kf_mode": "motion", "keyframe_decim": 15,
    "graph_mode": "kf_prior", "tsvd_ratio": 0.0,
    "nl_reg": True, "nl_reg_c": 10.0, "nl_reg_tau": 10.0, "nl_reg_length": 1.0,
    "kf_local_map_k": 1, "kf_edge_min_inlier": 0.0, "trans_gate_deg": 0.0,
    "loop_iterations": 10, "seq_tsvd_ratio": 0.0,
    "scale_kf": False, "scale_kf_q": 1e-3, "scale_kf_r": 0.1,
    "scale_kf_sigma": 0.5, "scale_kf_adaptive": False,
    "scale_kf_adapt_loop_ratio": 0.0, "scale_kf_innov_tau": 3.0,
}

PHASES = {
    "odometry": {
        "objective": "RPE_trans_2m_median",
        "fixed": {**ODOMETRY_FIXED, "loop_enable": False,
                  "loop_verifier": "none", "cycle_threshold_deg": 0.0},
        "space": {
            "step_scale_t": [0.05, 0.10, 0.15, 0.20],
            "scale_prior_sigma": [0.3, 0.6, 1.0, 1.5, 2.0],
            "kf_trans_thresh": [4.0, 8.0, 12.0, 16.0],
            "kf_rot_thresh": [10.0, 20.0, 30.0, 45.0],
            "kf_max_gap": [8, 12, 16, 24],
            "max_keyframes": [3, 5, 8, 12, 16],
            "kf_local_map_k": [1, 2],
            "kf_edge_min_inlier": [0.0, 0.3],
            "nl_reg_c": [3.0, 10.0, 30.0],
            "nl_reg_tau": [1.0, 10.0, 30.0],
            "nl_reg_length": [0.5, 1.0],
            "loop_iterations": [5, 10, 20],
            "scale_kf": [False, True],
        },
        "conditional": {
            "scale_kf_sigma": ([0.3, 0.5, 1.0], "scale_kf", [True]),
        },
    },
    "loop": {
        "objective": "ATE_median",
        "fixed": {**ODOMETRY_FIXED, "loop_enable": True,
                  "loop_verifier_keyframes": True,
                  "loop_verifier_abstain": "accept"},
        "space": {
            "loop_window": [20, 40, 60, 80],
            "loop_min_gap": [20, 30, 40, 60],
            "loop_min_inlier": [0.3, 0.4, 0.5, 0.6],
            "loop_temporal_k": [1, 2, 3, 4],
            "loop_sigma_scale": [0.0, 1.0, 2.0, 3.0],
            "loop_robust": ["none", "gm", "gnc_gm"],
            "cycle_threshold_deg": [0.0, 2.0, 5.0, 10.0],
            "loop_verifier": ["none", "sim3"],
        },
        "conditional": {
            "loop_verifier_gate": ([4, 6, 8, 10], "loop_verifier", ["sim3"]),
        },
    },
}


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _atomic_pickle(path, obj):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _atomic_text(path, text):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=sorted(PHASES), default="odometry")
    ap.add_argument("--seq", default="desk,desk2,room")
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--odometry-json", default=None,
                    help="Phase-1 result JSON; its best_params are injected "
                         "into the fixed odometry keys (loop phase)")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    phase = PHASES[args.phase]
    seqs = [s for s in args.seq.split(",") if s]
    fixed = dict(phase["fixed"])
    if args.phase == "loop":
        if not args.odometry_json:
            print("WARNING: --phase loop without --odometry-json uses default "
                  "odometry parameters", flush=True)
        else:
            odom = json.loads(Path(args.odometry_json).read_text())
            allowed = (set(ODOMETRY_FIXED)
                       | set(PHASES["odometry"]["space"])
                       | set(PHASES["odometry"]["conditional"]))
            injected = {k: v for k, v in odom.get("best_params", {}).items()
                        if k in allowed}
            fixed.update(injected)
            print(f"injected {len(injected)} odometry params from "
                  f"{args.odometry_json}", flush=True)

    blob = json.dumps({"phase": args.phase, "seqs": seqs,
                       "space": phase["space"], "conditional": phase["conditional"],
                       "fixed": fixed, "objective": phase["objective"]},
                      sort_keys=True, default=str)
    config_id = hashlib.sha1(blob.encode()).hexdigest()[:8]
    storage = (Path(args.storage) if args.storage
               else REPO / f"eval/results/tune_{args.phase}_{config_id}.db")
    study_name = args.study_name or f"{args.phase}_{config_id}"
    out = (Path(args.out) if args.out
           else REPO / f"eval/results/tune_{args.phase}_{config_id}.json")
    print(f"phase={args.phase} config_id={config_id} study={study_name} "
          f"storage={storage}", flush=True)

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

    storage.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name=study_name, storage=f"sqlite:///{storage}",
        load_if_exists=True)

    metric = phase["objective"]

    def objective(trial):
        params = dict(fixed)
        for name, levels in phase["space"].items():
            params[name] = trial.suggest_categorical(name, levels)
        for name, (levels, dep, values) in phase["conditional"].items():
            if params[dep] in values:
                params[name] = trial.suggest_categorical(name, levels)
            # otherwise keep the phase fixed value for ``name``
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
            for s in seqs:
                _atomic_pickle(match_paths[s], mcaches[s])
        values = [per[s][metric] for s in seqs]
        trial.set_user_attr("per_seq", json.dumps(per))
        trial.set_user_attr("eval_params", json.dumps(params))
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
                   callbacks=[report], catch=(Exception,))

    completed = [t for t in study.trials if t.value is not None]
    out.parent.mkdir(parents=True, exist_ok=True)
    if not completed:
        _atomic_text(out, json.dumps({"phase": args.phase,
                                     "objective_metric": metric,
                                     "n_completed": 0}))
        print("no completed trials; wrote", out, flush=True)
        raise SystemExit(1)
    best = min(completed, key=lambda t: t.value)
    if best.value >= 1e6:
        print("WARNING: best trial hit the nan penalty", flush=True)
    result = {
        "phase": args.phase,
        "objective_metric": metric,
        "config_id": config_id,
        "study_name": study_name,
        "best_trial": best.number,
        "best_value": best.value,
        "best_worst": best.user_attrs.get("worst"),
        "best_params": json.loads(best.user_attrs.get("eval_params", "{}")),
        "best_per_seq": json.loads(best.user_attrs.get("per_seq", "{}")),
        "n_completed": len(completed),
    }
    _atomic_text(out, json.dumps(result, indent=2, ensure_ascii=False))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
