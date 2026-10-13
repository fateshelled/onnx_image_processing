"""Optuna tuner for the adopted PR#36 defaults (soft + bounded canonical + bearing).

Run one instance at a time: match-cache updates are snapshot based.

Search groups (discrete categorical levels, see notes/20260920-tuning-policy.md):
  A solver/soft   : relative_reg_alpha/tau, relative_sigma_r/t/l, loop_iterations
  B cap (ON fixed): canonical_max_keyframes, loop_window
  C bearing       : bearing_sigma_d, bearing_dir_kernel, bearing_dir_delta,
                    bearing_radial, bearing_sigma_radial,
                    bearing_scale_prior_sigma, bearing_scale_prior_mean,
                    step_scale_t, scale_prior_sigma, loop_rot_sigma
  E kf/loop       : kf_trans_thresh, kf_rot_thresh, kf_max_gap, kf_local_map_k,
                    loop_min_gap, loop_min_inlier, loop_temporal_k, loop_sigma_scale

Fixed (not searched): canonical_cap=True, marginalization_backend=canonical_marginal,
canonical_regularization=relative_joint_soft, loop_observation_model=bearing,
nl_reg=False, tsvd_ratio/seq_tsvd_ratio=0, loop_robust=none, scale_kf=False.

Objective (minimize), on desk/desk2/room:
    mean(ATE/ref) + 0.5 * max(ATE/ref)
plus an RPE penalty and a small penalty per sequence whose ratio exceeds 1.02.
`ref` is the ATE of the shipped DEFAULT_PARAMS on the same sequences (measured
once, cached), so the baseline trial scores exactly 1.5.

Stage 1 enqueues the baseline and one-factor-at-a-time (OFAT) trials around the
baseline (conditional params enable their gate). TPE then explores the space.

Dashboard: ``optuna-dashboard sqlite:///<storage>``. The study name embeds a hash
of the sources, space, fixed defaults and limit so changed settings start a
separate study.
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
MATCHER_CONFIG = {"iterations": 20, "epsilon": 0.05,
                  "unused_score": 1.0, "distance_type": "l2"}

GROUPS = {
    "A": {
        "relative_reg_alpha": [1.0, 3.0, 10.0, 30.0],
        "relative_reg_tau": [0.1, 0.3, 1.0, 3.0],
        "relative_sigma_r": [0.5, 1.0, 2.0],
        "relative_sigma_t": [0.5, 1.0, 2.0],
        "relative_sigma_l": [0.5, 1.0, 2.0],
        "loop_iterations": [10, 20, 30],
    },
    "B": {
        "canonical_max_keyframes": [8, 12, 16, 24],
        "loop_window": [40, 60, 80, 100],
    },
    "C": {
        "bearing_dir_kernel": ["gm", "quadratic", "huber"],
        "bearing_sigma_d": [0.15, 0.2, 0.3, 0.4],
        "bearing_dir_delta": [0.5, 1.0, 2.0],
        "bearing_radial": [False, True],
        "bearing_sigma_radial": [0.05, 0.1, 0.2],
        "bearing_scale_prior_sigma": [0.3, 0.6, 1.0],
        "bearing_scale_prior_mean": [0.0, 0.5],
        "step_scale_t": [0.05, 0.1, 0.15, 0.2],
        "scale_prior_sigma": [0.3, 0.6, 1.0, 1.5, 2.0],
        "loop_rot_sigma": [0.05, 0.1, 0.2],
    },
    "E": {
        "kf_trans_thresh": [4.0, 8.0, 12.0, 16.0],
        "kf_rot_thresh": [10.0, 20.0, 30.0, 45.0],
        "kf_max_gap": [8, 12, 16, 24],
        "kf_local_map_k": [1, 2, 3],
        "loop_min_gap": [20, 30, 40, 60],
        "loop_min_inlier": [0.3, 0.4, 0.5, 0.6],
        "loop_temporal_k": [1, 2, 3, 4],
        "loop_sigma_scale": [0.5, 1.0, 2.0, 3.0],
    },
}

# name -> (gate param, gate values for which the param is active).
# bearing_dir_delta only affects the huber direction kernel (bearing_observation
# uses direction_delta in the huber branch only).
CONDITIONAL = {
    "bearing_dir_delta": ("bearing_dir_kernel", {"huber"}),
    "bearing_sigma_radial": ("bearing_radial", {True}),
    "bearing_scale_prior_sigma": ("bearing_radial", {True}),
    "bearing_scale_prior_mean": ("bearing_radial", {True}),
}

SOURCE_FILES = (
    "scripts/tune_bearing_cap.py",
    "eval/rustuna_tune_loop.py",
    "eval/eval_tum_vo.py",
    "eval/torch_sinkhorn.py",
) + tuple(
    path.relative_to(REPO).as_posix()
    for path in sorted((REPO / "vo").rglob("*.py"))
)


class TrialEvaluationError(RuntimeError):
    """A single parameter trial failed while the tuning infrastructure is sound."""


def build_space(groups):
    space = {}
    for group in groups:
        if group not in GROUPS:
            raise SystemExit(f"unknown group {group!r}; choose from "
                             f"{sorted(GROUPS)}")
        space.update(GROUPS[group])
    return space


def order_space(space):
    """Put gate parameters before the params that depend on them."""
    gates = {gate for gate, _ in CONDITIONAL.values()}
    first = {name: space[name] for name in space if name in gates}
    rest = {name: space[name] for name in space if name not in gates}
    return {**first, **rest}


def base_raw(space):
    """Baseline level of every searched parameter.

    Conditional parameters whose shipped default is not itself a search level
    (e.g. ``bearing_sigma_radial=None`` while the levels are positive floats) are
    skipped, so the OFAT enqueue never fixes them to an out-of-grid value that
    would clash with ``suggest_categorical`` when their gate is enabled.
    """
    raw = {}
    for name, levels in space.items():
        if name in DEFAULT_PARAMS and DEFAULT_PARAMS[name] in levels:
            raw[name] = DEFAULT_PARAMS[name]
    for dep, (gate_param, gate_values) in CONDITIONAL.items():
        if dep in raw and raw.get(gate_param) not in gate_values:
            del raw[dep]
    return raw


def build_params(raw):
    return {**DEFAULT_PARAMS, **raw}


def signature(params, space):
    return json.dumps({name: params.get(name) for name in sorted(space)},
                      sort_keys=True, default=str)


def suggest_raw(trial, space):
    raw = {}
    for name, levels in space.items():
        gate = CONDITIONAL.get(name)
        if gate is not None:
            gate_param, gate_values = gate
            if raw.get(gate_param) not in gate_values:
                continue
        raw[name] = trial.suggest_categorical(name, levels)
    return raw


def ofat_signature(values, space):
    """Effective full-key signature of an OFAT trial.

    ``values`` may be partial. Inactive conditional parameters are normalized to
    ``None`` so an explicitly fixed but unused value does not create a duplicate.
    """
    normalized = {}
    for name in sorted(space):
        gate = CONDITIONAL.get(name)
        if gate is not None:
            gate_param, gate_values = gate
            if values.get(gate_param) not in gate_values:
                normalized[name] = None
                continue
        normalized[name] = values.get(name)
    return json.dumps(normalized, sort_keys=True, default=str)


def _fill_active_dependents(trial, space):
    """Pin active conditional params to a deterministic level for OFAT clarity."""
    for dep, (gate_param, gate_values) in CONDITIONAL.items():
        if dep in trial or dep not in space:
            continue
        if trial.get(gate_param) in gate_values:
            trial[dep] = sorted(space[dep], key=str)[0]


def ofat_trials(space):
    base = base_raw(space)
    yield dict(base)
    for name, levels in space.items():
        for value in levels:
            if value == base.get(name):
                continue
            trial = dict(base)
            trial[name] = value
            gate = CONDITIONAL.get(name)
            if gate is not None:
                gate_param, gate_values = gate
                if trial.get(gate_param) not in gate_values:
                    # deterministic representative level for the gate
                    trial[gate_param] = sorted(gate_values, key=str)[0]
            _fill_active_dependents(trial, space)
            yield trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="desk,desk2,room")
    parser.add_argument("--groups", default="A,B,C,E",
                        help="comma list of groups among A,B,C,E")
    parser.add_argument("--n-trials", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None,
                        help="truncate each sequence to the first N frames (smoke)")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--storage", default=str(
        REPO / "eval/results/tune_bearing_cap.db"))
    parser.add_argument("--study-name", default="bearing_cap")
    parser.add_argument("--out", default=str(
        REPO / "eval/results/tune_bearing_cap.json"))
    args = parser.parse_args()
    if args.n_trials < 1:
        parser.error("--n-trials must be >= 1")

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    space = order_space(build_space(groups))
    sequences = [s.strip() for s in args.seq.split(",") if s.strip()]

    matcher = TorchSinkhornMatcher(**MATCHER_CONFIG)
    caches, cams, match_caches = {}, {}, {}
    for seq in sequences:
        cache = rtl.load_cache(CACHE, seq)
        frames = rtl.load_frames(DATASET_ROOT, seq)
        stride = int(cache["stride"])
        cache["gt_pose"] = np.asarray(
            [frames[i][1] for i in range(0, len(frames), stride)], dtype=np.float64)
        if args.limit is not None and args.limit < len(cache["odom"]):
            cache = dict(cache)
            cache["odom"] = cache["odom"][:args.limit]
            cache["gt_pos"] = cache["gt_pos"][:args.limit + 1]
            cache["gt_pose"] = np.asarray(cache["gt_pose"])[:args.limit + 1]
        caches[seq] = cache
        fx, fy, cx, cy = rtl.intrinsics_for(
            DATASET_ROOT, seq, (525.0, 525.0, 320.0, 240.0))
        cams[seq] = CameraIntrinsics(fx, fy, cx, cy, 640, 480)
        mc_path = CACHE / f"match_cache_{seq}_torch.pkl"
        if mc_path.exists():
            with mc_path.open("rb") as handle:
                match_caches[seq] = pickle.load(handle)
        else:
            print(f"[warn] no match cache for {seq}; starting empty", flush=True)
            match_caches[seq] = {}

    def save_match_caches():
        for seq, mc in match_caches.items():
            path = CACHE / f"match_cache_{seq}_torch.pkl"
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            with tmp.open("wb") as handle:
                pickle.dump(mc, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)

    result_cache = {}

    def evaluate(params):
        sig = signature(params, space)
        if sig in result_cache:
            return result_cache[sig]
        per = {}
        try:
            for seq in sequences:
                holder = {}

                def on_frame(idx, graph, _holder=holder):
                    _holder["graph"] = graph

                result = rtl.eval_seq(
                    caches[seq], params, cams[seq], matcher, match_caches[seq],
                    on_frame=on_frame)
                canonical = getattr(holder.get("graph"), "_canonical", None)
                per[seq] = {
                    "ATE": float(result["ATE_median"]),
                    "RPE2m": float(result["RPE_trans_2m_median"]),
                    "n_loop": int(result["n_loop"]),
                    "n_kf": int(result["n_kf"]),
                    "n_absorbed": int(getattr(canonical, "n_absorbed", 0)),
                    "n_retired_loops": int(
                        getattr(canonical, "n_retired_loops", 0)),
                }
        except Exception as exc:
            raise TrialEvaluationError(
                f"{seq} evaluation failed: {type(exc).__name__}: {exc}") from exc
        result_cache[sig] = per
        return per

    baseline = evaluate(build_params(base_raw(space)))
    ref_ate = {seq: baseline[seq]["ATE"] for seq in sequences}
    base_rpes = [baseline[seq]["RPE2m"] for seq in sequences]
    ref_rpe = (float(np.nanmean(base_rpes)) if np.any(np.isfinite(base_rpes))
               else float("nan"))
    if not all(np.isfinite(ref_ate[s]) for s in sequences):
        raise SystemExit(f"baseline ATE is non-finite: {ref_ate}")
    print(f"baseline (DEFAULT_PARAMS): "
          f"{ {s: round(ref_ate[s], 5) for s in sequences} }", flush=True)

    definition = {
        "sequences": sequences,
        "groups": groups,
        "space": space,
        "conditional": {k: [v[0], sorted(v[1])] for k, v in CONDITIONAL.items()},
        "fixed_defaults": {k: DEFAULT_PARAMS[k] for k in sorted(DEFAULT_PARAMS)
                           if k not in space},
        "dataset_root": DATASET_ROOT,
        "cache": str(CACHE.relative_to(REPO)),
        "matcher": MATCHER_CONFIG,
        "limit": args.limit,
        "objective": "mean(ATE/ref)+0.5max(ATE/ref)+rpe+ratio-penalties",
        "source_sha256": {
            path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
            for path in SOURCE_FILES},
    }
    definition_hash = hashlib.sha256(json.dumps(
        definition, sort_keys=True).encode()).hexdigest()[:12]
    study_name = f"{args.study_name}_{definition_hash}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name=study_name,
        storage=f"sqlite:///{args.storage}",
        load_if_exists=True)

    from optuna.trial import TrialState
    existing = set()
    for trial in study.trials:
        if trial.state not in (TrialState.COMPLETE, TrialState.WAITING):
            continue
        values = trial.system_attrs.get("fixed_params", trial.params)
        existing.add(ofat_signature(values, space))
    for raw in ofat_trials(space):
        sig = ofat_signature(raw, space)
        if sig not in existing:
            study.enqueue_trial(raw)
            existing.add(sig)

    def objective(trial):
        raw = suggest_raw(trial, space)
        params = build_params(raw)
        started = time.perf_counter()
        try:
            per = evaluate(params)
        finally:
            save_match_caches()
        ratios = np.array([per[seq]["ATE"] / ref_ate[seq] for seq in sequences])
        rpes = [per[seq]["RPE2m"] for seq in sequences]
        mean_rpe = (float(np.nanmean(rpes)) if np.any(np.isfinite(rpes))
                    else float("nan"))
        if not np.all(np.isfinite(ratios)):
            raise TrialEvaluationError("non-finite ATE")
        value = float(np.mean(ratios) + 0.5 * np.max(ratios))
        if (np.isfinite(mean_rpe) and np.isfinite(ref_rpe) and ref_rpe > 0):
            value += 5.0 * max(0.0, mean_rpe / ref_rpe - 1.0)
        value += float(np.sum(np.maximum(ratios - 1.02, 0.0)))
        trial.set_user_attr("per_seq", json.dumps(per))
        trial.set_user_attr("params", json.dumps(params))
        trial.set_user_attr("raw", json.dumps(raw, default=str))
        trial.set_user_attr(
            "mean_rpe2m", mean_rpe if np.isfinite(mean_rpe) else None)
        trial.set_user_attr("seconds", round(time.perf_counter() - started, 1))
        print(f"trial={trial.number} score={value:.6f} "
              f"rpe={mean_rpe:.6f} ratios={np.round(ratios, 4).tolist()} "
              f"raw={raw}", flush=True)
        return value

    def completed_trials():
        return [t for t in study.trials
                if t.state == TrialState.COMPLETE and t.value is not None
                and np.isfinite(t.value)]

    completed = completed_trials()
    while len(completed) < args.n_trials:
        before = len(completed)
        study.optimize(objective, n_trials=args.n_trials - before,
                       catch=(TrialEvaluationError,))
        completed = completed_trials()
        if len(completed) == before:
            raise RuntimeError("no additional trial completed successfully")

    best = min(completed, key=lambda t: t.value)
    output = {
        "study": study_name,
        "definition_hash": definition_hash,
        "definition": definition,
        "seed": args.seed,
        "n_trials": len(completed),
        "n_failed": sum(t.state.name == "FAIL" for t in study.trials),
        "best_trial": best.number,
        "best_score": best.value,
        "best_params": json.loads(best.user_attrs["params"]),
        "best_raw": json.loads(best.user_attrs["raw"]),
        "best_per_seq": json.loads(best.user_attrs["per_seq"]),
        "best_mean_rpe2m": best.user_attrs.get("mean_rpe2m"),
        "baseline_ate": ref_ate,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(output, indent=2, default=str))
    os.replace(tmp, out_path)
    save_match_caches()
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
