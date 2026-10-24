"""Reproducible accuracy-first A/B for canonical segment boundaries.

Uses the shipped dev evaluator and match cache. Results are atomic and bound
to source hashes, effective parameters, sequence length and cache hashes.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "eval")]
from vo.online_graph import DEFAULT_PARAMS
from vo.pose_estimation import CameraIntrinsics
from eval.torch_sinkhorn import TorchSinkhornMatcher

spec = importlib.util.spec_from_file_location("rtl", REPO / "eval/rustuna_tune_loop.py")
rtl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtl)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


class InputsChangedError(RuntimeError):
    """Exit code 3 distinguishes invalid comparisons from solver failures."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="desk,desk2,room")
    ap.add_argument("--modes", default="legacy,canonical_conditional,canonical_marginal")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--params-json", default="{}")
    ap.add_argument("--dataset-root", type=Path,
                    default=Path.home() / "datasets/tum_rgbd")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-trajectories", action="store_true")
    args = ap.parse_args()
    cache_dir = REPO / "eval/results/tune_cache_loop"
    sources = {str(p.relative_to(REPO)): digest(p)
               for p in sorted((REPO / "vo").glob("*.py"))}
    for p in [Path(__file__), REPO / "eval/rustuna_tune_loop.py",
              REPO / "eval/eval_tum_vo.py"]:
        sources[str(p.relative_to(REPO))] = digest(p)
    definition = dict(source_hashes=sources, max_frames=args.max_frames,
                      params=json.loads(args.params_json),
                      commit=subprocess.check_output(
                          ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
    if "marginalization_backend" in definition["params"]:
        raise ValueError("marginalization_backend must be selected through --modes")
    result = json.loads(args.out.read_text()) if args.out.exists() else {
        "definition": definition, "results": {}}
    if result["definition"] != definition:
        raise ValueError("result file belongs to a different implementation/definition")
    matcher = TorchSinkhornMatcher(iterations=20, epsilon=.05,
                                   unused_score=1., distance_type="l2")
    for seq in args.seqs.split(","):
        dataset, _ = rtl.resolve_dataset(str(args.dataset_root), seq)
        pkl = cache_dir / f"match_cache_{seq}_torch.pkl"
        input_paths = {"npz": cache_dir / f"{seq}.npz",
                       "groundtruth": dataset / "groundtruth.txt",
                       "rgb": dataset / "rgb.txt", "match_cache": pkl}
        input_hashes = {name: digest(path) for name, path in input_paths.items()}
        c = rtl.load_cache(cache_dir, seq)
        frames = rtl.load_frames(str(args.dataset_root), seq)
        gt = np.array([frames[i][1] for i in range(0, len(frames), int(c["stride"]))])
        if len(gt) != len(c["gt_pos"]):
            raise ValueError("GT/cache length mismatch")
        if not np.allclose(gt[:, :3, 3], c["gt_pos"], rtol=0, atol=1e-9):
            raise ValueError("ATE cache GT and RPE dataset GT disagree")
        c["gt_pose"] = gt
        if args.max_frames:
            c["odom"] = c["odom"][:args.max_frames - 1]
            c["gt_pos"] = c["gt_pos"][:args.max_frames]
            c["gt_pose"] = c["gt_pose"][:args.max_frames]
        cache_hash = input_hashes["match_cache"]
        with pkl.open("rb") as f:
            original_cache = pickle.load(f)
        fx, fy, cx, cy = rtl.intrinsics_for(str(args.dataset_root), seq, (525.,525.,320.,240.))
        cam = CameraIntrinsics(fx, fy, cx, cy, 640, 480)
        if {name: digest(path) for name, path in input_paths.items()} != input_hashes:
            raise InputsChangedError("sequence inputs changed while loading")
        inputs = {**input_hashes,
                  "intrinsics": [fx, fy, cx, cy, 640, 480],
                  "stride": int(c["stride"])}
        for mode in args.modes.split(","):
            key = f"{mode}:{seq}"
            if key in result["results"]:
                if result["results"][key]["inputs"] != inputs:
                    raise InputsChangedError("sequence inputs changed since previous result")
                continue
            for old_key, old in result["results"].items():
                if old_key.split(":", 1)[1] == seq and old["inputs"] != inputs:
                    raise InputsChangedError("cannot mix different sequence inputs in A/B")
            params = {**DEFAULT_PARAMS, "loop_verifier": "none",
                      "marginalization_backend": mode, **definition["params"]}
            # Start each variant with exactly the same cache, never mutate the
            # shared cache. Any miss is measured and reported below.
            mc = dict(original_cache)
            np.random.seed(0)
            import cv2
            cv2.setRNGSeed(0)
            last = {}
            def on_frame(idx, graph):
                last["graph"] = graph
                if idx % 100 == 0:
                    print(f"{key} frame={idx} kf={graph.n_kf}", flush=True)
            start = time.perf_counter()
            metrics = rtl.eval_seq(c, params, cam, matcher, mc, on_frame=on_frame)
            if {name: digest(path) for name, path in input_paths.items()} != input_hashes:
                raise InputsChangedError("sequence inputs changed during evaluation")
            graph = last["graph"]
            if args.save_trajectories:
                trajectory = args.out.with_name(f"{args.out.stem}_{mode}_{seq}_trajectory.npz")
                sampled_ids = np.arange(len(c["odom"]) + 1) * int(c["stride"])
                np.savez_compressed(trajectory, sampled_ids=sampled_ids,
                    est_poses=np.array([graph.pose(int(i)) for i in sampled_ids]),
                    gt_poses=c["gt_pose"][:len(sampled_ids)])
                metrics["trajectory_file"] = trajectory.name
                metrics["trajectory_sha256"] = digest(trajectory)
            kfs = graph.keyframes
            est = np.array([graph.pose(i)[:3, 3] for i in kfs])
            truth = c["gt_pos"][[i // int(c["stride"]) for i in kfs]]
            s, R, t = rtl.umeyama(est, truth, with_scale=True)
            metrics["ATE_kf_median"] = float(np.median(np.linalg.norm(s*(est@R.T)+t-truth, axis=1)))
            for name, value in list(metrics.items()):
                if isinstance(value, float) and not np.isfinite(value):
                    if name.startswith("RPE_"):
                        metrics[name] = None  # insufficient travelled distance
                    else:
                        raise ValueError(f"non-finite required metric {name}")
            metrics["sec"] = time.perf_counter() - start
            metrics["new_match_cache_entries"] = len(mc) - len(original_cache)
            metrics["match_cache_sha"] = cache_hash
            metrics["inputs"] = inputs
            metrics["params"] = params
            if graph._canonical is not None:
                graph._canonical.assert_ownership()
                metrics["canonical_revision"] = graph._canonical.revision
                metrics["canonical_sources"] = len(graph._canonical.sources)
            result["results"][key] = metrics
            atomic_json(args.out, result)
            print(key, json.dumps(metrics), flush=True)


if __name__ == "__main__":
    try:
        main()
    except InputsChangedError as exc:
        print(f"INPUTS_CHANGED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(3)
