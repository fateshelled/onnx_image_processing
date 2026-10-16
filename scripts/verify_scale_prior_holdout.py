"""Holdout comparison for candidate ``scale_prior_sigma`` defaults.

Candidate values are compared with the previous default (0.6) on seven
sequences that were not used by the desk/desk2/room tuning study. Results and a
dedicated match cache are saved incrementally for resume.
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

CACHE = REPO / "eval/results/tune_cache_loop"
OUT = REPO / "eval/results/scale_prior_holdout_ab.json"
HOLDOUT = [
    "freiburg2_desk",
    "freiburg2_rpy",
    "freiburg2_xyz",
    "freiburg3_long_office_household",
    "freiburg3_sitting_xyz",
    "360",
    "xyz",
]
CONFIGS = {
    "old_0.6": {**DEFAULT_PARAMS, "scale_prior_sigma": 0.6},
    "candidate_0.5": {**DEFAULT_PARAMS, "scale_prior_sigma": 0.5},
    "candidate_0.7": {**DEFAULT_PARAMS, "scale_prior_sigma": 0.7},
    "candidate_0.8": {**DEFAULT_PARAMS, "scale_prior_sigma": 0.8},
    "tuned_1.0": {**DEFAULT_PARAMS, "scale_prior_sigma": 1.0},
}
DEFAULT_CONFIGS = ("candidate_0.5", "candidate_0.7", "candidate_0.8")
SOURCE_FILES = (
    Path(__file__),
    REPO / "eval/rustuna_tune_loop.py",
    REPO / "eval/eval_tum_vo.py",
    REPO / "eval/torch_sinkhorn.py",
) + tuple(sorted((REPO / "vo").rglob("*.py")))
MATCHER_CONFIG = {"iterations": 20, "epsilon": 0.05,
                  "unused_score": 1.0, "distance_type": "l2"}


def _atomic_json(path, value):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _atomic_pickle(path, value):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _source_hash():
    digest = hashlib.sha256()
    for path in SOURCE_FILES:
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _value_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _evaluation_hash(source_hash, input_hash, match_context, seq, config,
                     params, limit):
    payload = {
        "source_hash": source_hash,
        "input_hash": input_hash,
        "match_context": match_context,
        "seq": seq,
        "config": config,
        "params": params,
        "limit": limit,
        "matcher": MATCHER_CONFIG,
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]


def _load_match_cache(seq, input_hash, source_hash, frozen_context=None):
    dedicated = CACHE / f"match_cache_{seq}_scale_prior_ab.pkl"
    seeds = [CACHE / f"match_cache_{seq}_torch.pkl",
             CACHE / f"match_cache_{seq}_numpy.pkl"]
    seed_path = next((path for path in seeds if path.exists()), None)
    seed_hash = _file_hash(seed_path) if seed_path is not None else "empty"
    current_context = {
        "input_hash": input_hash,
        "generation_source_hash": source_hash,
        "matcher": MATCHER_CONFIG,
        # Existing cached measurements are a frozen evaluation input. Hashing
        # the snapshot makes that dependency explicit while isolating A/B from
        # pose-estimation randomness unrelated to scale_prior_sigma.
        "seed_name": seed_path.name if seed_path is not None else "empty",
        "seed_hash": seed_hash,
    }
    context = frozen_context if frozen_context is not None else current_context
    if frozen_context is not None:
        for field in ("input_hash", "matcher", "seed_name", "seed_hash"):
            if frozen_context.get(field) != current_context[field]:
                raise ValueError(
                    f"{seq}: frozen match context {field} mismatch: "
                    f"{frozen_context.get(field)!r} != {current_context[field]!r}")
    if dedicated.exists():
        with dedicated.open("rb") as handle:
            envelope = pickle.load(handle)
        if (isinstance(envelope, dict)
                and envelope.get("context") == context
                and isinstance(envelope.get("entries"), dict)):
            return dedicated, envelope["entries"], context
    if seed_path is not None:
        with seed_path.open("rb") as handle:
            entries = pickle.load(handle)
    else:
        entries = {}
    return dedicated, entries, context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", default=",".join(HOLDOUT))
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--dataset-root", default=os.getenv("TUM_RGBD_ROOT"))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--limit", type=int, default=None,
                        help="truncate to N odometry steps for smoke tests")
    parser.add_argument("--overwrite-existing", action="store_true",
                        help="allow replacing a record whose hash changed")
    args = parser.parse_args()
    if not args.dataset_root or not args.dataset_root.strip():
        parser.error("--dataset-root or TUM_RGBD_ROOT is required")
    args.dataset_root = args.dataset_root.strip()

    seqs = [value for value in args.seqs.split(",") if value]
    configs = [value for value in args.configs.split(",") if value]
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        parser.error(f"unknown configs: {sorted(unknown)}")

    out_path = Path(args.out)
    result = json.loads(out_path.read_text()) if out_path.exists() else {
        "study": "scale_prior_holdout_ab",
        "training_sequences": ["desk", "desk2", "room"],
        "holdout_sequences": seqs,
        "records": {},
    }
    source_hash = _source_hash()
    initial_source_hash = result.get("source_hash")
    for record in result["records"].values():
        if initial_source_hash is not None:
            record.setdefault("source_hash", initial_source_hash)
    result["source_hashes"] = sorted({
        source_hash,
        *(record["source_hash"] for record in result["records"].values()
          if record.get("source_hash") is not None),
    })
    matcher = TorchSinkhornMatcher(**MATCHER_CONFIG)

    for seq in seqs:
        cache = rtl.load_cache(CACHE, seq)
        frames = rtl.load_frames(args.dataset_root, seq)
        stride = int(cache["stride"])
        if len(frames) != int(cache.get("n_frames", len(frames))):
            raise ValueError(f"{seq}: frame count mismatch")
        cache["gt_pose"] = np.asarray(
            [frames[i][1] for i in range(0, len(frames), stride)],
            dtype=np.float64)
        expected = len(cache["odom"]) + 1
        if len(cache["gt_pos"]) != expected or len(cache["gt_pose"]) != expected:
            raise ValueError(
                f"{seq}: cache/GT mismatch: expected {expected}, "
                f"gt_pos={len(cache['gt_pos'])}, gt_pose={len(cache['gt_pose'])}")
        if not np.allclose(cache["gt_pos"], cache["gt_pose"][:, :3, 3],
                           rtol=0.0, atol=1e-9):
            raise ValueError(f"{seq}: cached and current GT positions differ")
        input_hash = hashlib.sha256((
            _file_hash(CACHE / f"{seq}.npz") + _value_hash(cache["gt_pose"])
        ).encode()).hexdigest()
        if args.limit is not None and args.limit < len(cache["odom"]):
            cache = dict(cache)
            cache["odom"] = cache["odom"][:args.limit]
            cache["gt_pos"] = cache["gt_pos"][:args.limit + 1]
            cache["gt_pose"] = cache["gt_pose"][:args.limit + 1]

        fx, fy, cx, cy = rtl.intrinsics_for(
            args.dataset_root, seq, (525.0, 525.0, 320.0, 240.0))
        cam = CameraIntrinsics(fx, fy, cx, cy, 640, 480)
        baseline = result["records"].get(f"old_0.6:{seq}")
        if baseline is None:
            raise ValueError(f"{seq}: missing old_0.6 baseline record")
        if baseline.get("input_hash") != input_hash:
            raise ValueError(f"{seq}: baseline input hash mismatch")
        match_path, match_cache, match_context = _load_match_cache(
            seq, input_hash, source_hash, baseline.get("match_context"))

        for config in configs:
            params = CONFIGS[config]
            evaluation_hash = _evaluation_hash(
                source_hash, input_hash, match_context, seq, config, params,
                args.limit)
            key = f"{config}:{seq}"
            previous = result["records"].get(key)
            if previous is not None and previous.get("evaluation_hash") == evaluation_hash:
                print(f"{key:48} SKIP (done)", flush=True)
                continue
            if previous is not None and not args.overwrite_existing:
                raise ValueError(
                    f"{key}: existing record hash changed; pass "
                    "--overwrite-existing to replace it")

            holder = {}

            def on_frame(_idx, graph):
                holder["graph"] = graph

            started = time.perf_counter()
            values = rtl.eval_seq(cache, params, cam, matcher, match_cache,
                                  on_frame=on_frame,
                                  match_seed_context=f"scale-prior-ab:{seq}")
            elapsed = time.perf_counter() - started
            _atomic_pickle(match_path, {
                "context": match_context,
                "entries": match_cache,
            })
            canonical = getattr(holder.get("graph"), "_canonical", None)
            record = {
                "config": config,
                "seq": seq,
                "scale_prior_sigma": params["scale_prior_sigma"],
                "source_hash": source_hash,
                "evaluation_hash": evaluation_hash,
                "ATE_median": values.get("ATE_median"),
                "ATE_rmse": values.get("ATE_rmse"),
                "ATE_p90": values.get("ATE_p90"),
                "RPE_trans_1m_median": values.get("RPE_trans_1m_median"),
                "RPE_trans_2m_median": values.get("RPE_trans_2m_median"),
                "RPE_trans_5m_median": values.get("RPE_trans_5m_median"),
                "RPE_rot_2m_median_deg": values.get("RPE_rot_2m_median_deg"),
                "n_loop": values.get("n_loop"),
                "n_kf": values.get("n_kf"),
                "n_absorbed": getattr(canonical, "n_absorbed", 0),
                "n_retired_loops": getattr(canonical, "n_retired_loops", 0),
                "seconds": round(elapsed, 1),
                "input_hash": input_hash,
                "match_context": match_context,
            }
            result["records"][key] = record
            _atomic_json(out_path, result)
            print(f"{key:48} ATE={record['ATE_median']:.5f} "
                  f"RPE2m={record['RPE_trans_2m_median']} "
                  f"loops={record['n_loop']} ({elapsed:.0f}s)", flush=True)

    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
