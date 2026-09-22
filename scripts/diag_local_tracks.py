"""Diagnose P0a feature tracks and covisibility on cached sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from eval.eval_tum_vo import estimate_pose_from_matches, intrinsics_for  # noqa: E402
from eval.rustuna_tune_loop import DEFAULT_ARGS, load_cache  # noqa: E402
from eval.torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.local_tracks import (  # noqa: E402
    PairMatches,
    build_feature_tracks,
    covisibility_counts,
    select_covisible_neighbors,
)
from vo.onnx_matcher import extract_match_indices  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402


def frame_pairs(frames, pair_radius):
    """Enumerate local frame pairs by position in the selected frame list."""
    if pair_radius < 1:
        raise ValueError("pair_radius must be at least 1")
    return [(frames[i], frames[j])
            for i in range(len(frames))
            for j in range(i + 1, min(len(frames), i + pair_radius + 1))]


def histogram(values):
    return {str(value): values.count(value) for value in sorted(set(values))}


def evaluate_sequence(seq, opts, matcher):
    cache = load_cache(opts.cache_dir, seq)
    frames = sorted(cache["feat"])[:opts.max_frames]
    pairs = frame_pairs(frames, opts.pair_radius)
    fx, fy, cx, cy = intrinsics_for(
        opts.dataset_root, seq, (opts.fx, opts.fy, opts.cx, opts.cy))
    cam = CameraIntrinsics(fx, fy, cx, cy, opts.width, opts.height)
    pair_matches = []
    pose_failures = 0
    pair_rows = []
    for frame_i, frame_j in pairs:
        kpts_i, desc_i = cache["feat"][frame_i]
        kpts_j, desc_j = cache["feat"][frame_j]
        probs = matcher.match_probs(desc_i[0], desc_j[0])[None]
        idx_i, idx_j, scores = extract_match_indices(
            kpts_i, kpts_j, probs, DEFAULT_ARGS.match_threshold,
            DEFAULT_ARGS.max_matches, DEFAULT_ARGS.dbin)
        pose = estimate_pose_from_matches(
            kpts_i[0][idx_i], kpts_j[0][idx_j], cam, DEFAULT_ARGS)
        if pose.get("ok"):
            mask = np.asarray(pose["mask"], dtype=bool).reshape(-1)
            if len(mask) != len(idx_i):
                raise RuntimeError(
                    f"pose mask length mismatch for ({frame_i}, {frame_j})")
        else:
            pose_failures += 1
            mask = np.zeros(len(idx_i), dtype=bool)
        pair_matches.append(PairMatches(
            frame_i, frame_j, idx_i, idx_j, scores, mask))
        pair_rows.append({
            "frame_i": int(frame_i), "frame_j": int(frame_j),
            "matches": int(len(idx_i)), "inliers": int(mask.sum()),
            "pose_ok": bool(pose.get("ok", False)),
        })

    built = build_feature_tracks(pair_matches)
    selected_tracks = tuple(
        track for track in built.tracks
        if len(track.observations) >= opts.min_track_length)
    counts = covisibility_counts(selected_tracks)
    observations_per_frame = {
        int(frame): sum(frame in {item[0] for item in track.observations}
                        for track in selected_tracks)
        for frame in frames
    }
    neighbors = {
        str(frame): [
            {"frame": neighbor, "shared_tracks": shared}
            for neighbor, shared in select_covisible_neighbors(
                frame, counts, max_neighbors=opts.max_neighbors,
                min_shared_tracks=opts.min_shared_tracks)
        ]
        for frame in frames
    }
    lengths = [len(track.observations) for track in built.tracks]
    shared_values = list(counts.values())
    return {
        "sequence": seq,
        "frames": [int(frame) for frame in frames],
        "n_pairs": len(pairs),
        "pose_failures": pose_failures,
        "matches": built.n_matches,
        "inlier_matches": built.n_inlier_matches,
        "accepted_edges": built.n_accepted_edges,
        "conflicts": built.n_conflicts,
        "duplicates": built.n_duplicates,
        "tracks": len(built.tracks),
        "selected_tracks": len(selected_tracks),
        "track_length_histogram": histogram(lengths),
        "covisibility_pairs": len(counts),
        "shared_tracks_median": (float(np.median(shared_values))
                                 if shared_values else None),
        "shared_tracks_max": max(shared_values, default=0),
        "observations_per_frame": observations_per_frame,
        "neighbors": neighbors,
        "pair_rows": pair_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="desk,desk2,room")
    parser.add_argument("--cache-dir", default="eval/results/tune_cache_loop")
    parser.add_argument("--dataset-root", default="/home/ubuntu/datasets/tum_rgbd")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--pair-radius", type=int, default=2)
    parser.add_argument("--min-track-length", type=int, default=3)
    parser.add_argument("--min-shared-tracks", type=int, default=3)
    parser.add_argument("--max-neighbors", type=int, default=5)
    parser.add_argument("--fx", type=float, default=525.0)
    parser.add_argument("--fy", type=float, default=525.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path)
    opts = parser.parse_args()
    sequences = [value.strip() for value in opts.seq.split(",") if value.strip()]
    if not sequences:
        parser.error("--seq must contain at least one sequence")
    if (opts.max_frames < 2 or opts.min_track_length < 2
            or opts.pair_radius < 1 or opts.min_shared_tracks < 1
            or opts.max_neighbors < 0):
        parser.error("invalid frame, track, pair, or neighbor limit")
    if (not np.isfinite([opts.fx, opts.fy, opts.cx, opts.cy]).all()
            or opts.fx <= 0 or opts.fy <= 0
            or opts.width <= 0 or opts.height <= 0):
        parser.error("camera intrinsics and image dimensions must be valid")
    matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")
    reports = []
    failed = False
    for seq in sequences:
        try:
            reports.append(evaluate_sequence(seq, opts, matcher))
        except Exception as exc:  # noqa: BLE001 - preserve other sequences
            failed = True
            reports.append({"sequence": seq,
                            "error": f"{type(exc).__name__}: {exc}"})
    text = json.dumps(reports, indent=2, allow_nan=False)
    if opts.output:
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        opts.output.write_text(text + "\n")
    print(text)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
