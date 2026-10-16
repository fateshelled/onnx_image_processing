"""Stage-A tuning for online VO bounding and Tier-2 parameters.

Run only one instance at a time because match-cache updates are snapshot based.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
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

from eval.torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.online_graph import DEFAULT_PARAMS  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402

DATASET_ROOT = "/home/ubuntu/datasets/tum_rgbd"
CACHE = REPO / "eval/results/tune_cache_loop"
SEQUENCES = ("desk", "desk2", "room")
MATCHER_CONFIG = {"iterations": 20, "epsilon": 0.05,
                  "unused_score": 1.0, "distance_type": "l2"}
BASELINE_ATE = {"desk": 0.167254, "desk2": 0.201919, "room": 0.301095}
BASELINE_RPE_MEAN = 0.375461
SPACE = {
    "global_opt_on_loop": [True, False],
    "global_opt_period": [0, 5, 10, 20, 40],
    "loop_sigma_scale": [1.0, 1.5, 2.0, 3.0, 4.0],
    "max_keyframes": [8, 12, 16],
    "held_cap": [2, 4, 8, "auto"],
}
BASE = {
    "global_opt_on_loop": True,
    "global_opt_period": 10,
    "loop_sigma_scale": 1.0,
    "max_keyframes": 12,
    "held_cap": 4,
}


class TrialEvaluationError(RuntimeError):
    """A single parameter trial failed while the tuning infrastructure is sound."""


def decode(raw):
    params = dict(DEFAULT_PARAMS)
    params.update(raw)
    params["held_cap"] = None if raw["held_cap"] == "auto" else raw["held_cap"]
    params.update(kf_local_map_k=1, loop_robust="none", scale_kf=False,
                  seq_tsvd_ratio=0.0, nl_reg=True, loop_enable=True,
                  loop_verifier="none", cycle_threshold_deg=0.0)
    return params


def ofat_trials():
    yield dict(BASE)
    for name, levels in SPACE.items():
        for value in levels:
            if value != BASE[name]:
                trial = dict(BASE)
                trial[name] = value
                yield trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--storage", default=str(
        REPO / "eval/results/vo_stage_a.db"))
    parser.add_argument("--study-name", default="vo_stage_a_scale_gauge")
    parser.add_argument("--out", default=str(
        REPO / "eval/results/vo_stage_a.json"))
    args = parser.parse_args()

    matcher = TorchSinkhornMatcher(**MATCHER_CONFIG)
    caches = {seq: rtl.load_cache(CACHE, seq) for seq in SEQUENCES}
    cams = {}
    match_caches = {}
    for seq in SEQUENCES:
        frames = rtl.load_frames(DATASET_ROOT, seq)
        stride = int(caches[seq]["stride"])
        gt_pose = np.asarray(
            [frames[i][1] for i in range(0, len(frames), stride)],
            dtype=np.float64)
        if len(gt_pose) != len(caches[seq]["gt_pos"]):
            raise ValueError(f"{seq}: gt_pose length mismatch")
        caches[seq]["gt_pose"] = gt_pose
        fx, fy, cx, cy = rtl.intrinsics_for(
            DATASET_ROOT, seq, (525.0, 525.0, 320.0, 240.0))
        cams[seq] = CameraIntrinsics(fx, fy, cx, cy, 640, 480)
        path = CACHE / f"match_cache_{seq}_torch.pkl"
        with path.open("rb") as handle:
            match_caches[seq] = pickle.load(handle)

    definition = {
        "sequences": SEQUENCES,
        "dataset_root": DATASET_ROOT,
        "cache": str(CACHE.relative_to(REPO)),
        "matcher": MATCHER_CONFIG,
        "space": SPACE,
        "base": BASE,
        "baseline_ate": BASELINE_ATE,
        "baseline_rpe_mean": BASELINE_RPE_MEAN,
        "fixed": {key: decode(BASE)[key] for key in sorted(decode(BASE))
                  if key not in SPACE},
        "objective": "normalized_ate_mean+0.5worst+rpe5+ate2pct",
        "source_sha256": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                REPO / "eval/rustuna_tune_loop.py",
                REPO / "vo/online_graph.py",
                REPO / "vo/se3_window.py",
            )
        },
    }
    definition_hash = hashlib.sha256(json.dumps(
        definition, sort_keys=True).encode()).hexdigest()[:12]
    effective_study_name = f"{args.study_name}_{definition_hash}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name=effective_study_name,
        storage=f"sqlite:///{args.storage}",
        load_if_exists=True)
    existing = set()
    from optuna.trial import TrialState
    for trial in study.trials:
        if trial.state not in (TrialState.COMPLETE, TrialState.WAITING):
            continue
        values = trial.system_attrs.get("fixed_params", trial.params)
        if all(name in values for name in SPACE):
            existing.add(json.dumps(
                {name: values[name] for name in SPACE}, sort_keys=True))
    for params in ofat_trials():
        signature = json.dumps(params, sort_keys=True)
        if signature not in existing:
            study.enqueue_trial(params)

    def save_match_caches():
        for seq, cache in match_caches.items():
            path = CACHE / f"match_cache_{seq}_torch.pkl"
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            with tmp.open("wb") as handle:
                pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)

    def objective(trial):
        raw = {name: trial.suggest_categorical(name, levels)
               for name, levels in SPACE.items()}
        params = decode(raw)
        per = {}
        started = time.perf_counter()
        try:
            for seq in SEQUENCES:
                try:
                    result = rtl.eval_seq(
                        caches[seq], params, cams[seq], matcher,
                        match_caches[seq])
                except Exception as exc:
                    raise TrialEvaluationError(
                        f"{seq} evaluation failed") from exc
                per[seq] = {
                    "ATE": float(result["ATE_median"]),
                    "RPE2m": float(result["RPE_trans_2m_median"]),
                    "n_loop": int(result["n_loop"]),
                    "n_kf": int(result["n_kf"]),
                    "n_dropped_held": int(result["n_dropped_held"]),
                }
        finally:
            save_match_caches()
        ratios = np.array([
            per[seq]["ATE"] / BASELINE_ATE[seq] for seq in SEQUENCES])
        mean_rpe = float(np.mean([per[seq]["RPE2m"] for seq in SEQUENCES]))
        if not np.all(np.isfinite(ratios)) or not np.isfinite(mean_rpe):
            raise TrialEvaluationError(
                f"non-finite metrics: ratios={ratios}, rpe={mean_rpe}")
        objective_value = float(np.mean(ratios) + 0.5 * np.max(ratios))
        objective_value += 5.0 * max(0.0, mean_rpe / BASELINE_RPE_MEAN - 1.0)
        objective_value += float(np.sum(np.maximum(ratios - 1.02, 0.0)))
        trial.set_user_attr("per_seq", json.dumps(per))
        trial.set_user_attr("params", json.dumps(params))
        trial.set_user_attr("mean_rpe2m", mean_rpe)
        trial.set_user_attr("seconds", round(time.perf_counter() - started, 1))
        print(f"trial={trial.number} score={objective_value:.6f} "
              f"mean_rpe={mean_rpe:.6f} ratios={ratios.tolist()} "
              f"params={raw}", flush=True)
        return objective_value

    while True:
        completed_before = [
            trial for trial in study.trials
            if (trial.state == TrialState.COMPLETE and trial.value is not None
                and np.isfinite(trial.value))]
        remaining = args.n_trials - len(completed_before)
        if remaining <= 0:
            break
        study.optimize(objective, n_trials=remaining,
                       catch=(TrialEvaluationError,))
        completed_after = [
            trial for trial in study.trials
            if (trial.state == TrialState.COMPLETE and trial.value is not None
                and np.isfinite(trial.value))]
        if len(completed_after) == len(completed_before):
            raise RuntimeError("no additional trial completed successfully")
    completed = completed_after if 'completed_after' in locals() else completed_before
    best = min(completed, key=lambda trial: trial.value)
    output = {
        "study": effective_study_name,
        "definition_hash": definition_hash,
        "definition": definition,
        "seed": args.seed,
        "n_trials": len(completed),
        "n_failed": sum(trial.state.name == "FAIL" for trial in study.trials),
        "best_trial": best.number,
        "best_score": best.value,
        "best_params": json.loads(best.user_attrs["params"]),
        "best_per_seq": json.loads(best.user_attrs["per_seq"]),
        "best_mean_rpe2m": best.user_attrs["mean_rpe2m"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(output, indent=2))
    os.replace(tmp, out_path)
    save_match_caches()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
