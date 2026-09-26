"""Diagnose P0a feature tracks and covisibility on cached sequences."""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from eval.eval_tum_vo import estimate_pose_from_matches, intrinsics_for, umeyama  # noqa: E402
from eval.rustuna_tune_loop import DEFAULT_ARGS, load_cache  # noqa: E402
from eval.torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.local_tracks import (  # noqa: E402
    PairMatches,
    build_feature_tracks,
    covisibility_counts,
    select_covisible_neighbors,
    triangulate_feature_track,
)
from vo.local_ba import BAObservation, optimize_local_ba  # noqa: E402
from vo.loop_sim3_verifier import compose_odom  # noqa: E402
from vo.onnx_matcher import extract_match_indices  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402
from vo.se3 import se3_log  # noqa: E402


def frame_pairs(frames, pair_radius):
    """Enumerate local frame pairs by position in the selected frame list."""
    if pair_radius < 1:
        raise ValueError("pair_radius must be at least 1")
    return [(frames[i], frames[j])
            for i in range(len(frames))
            for j in range(i + 1, min(len(frames), i + pair_radius + 1))]


def histogram(values):
    return {str(value): values.count(value) for value in sorted(set(values))}


def percentile_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"median": None, "p90": None}
    return {"median": float(np.median(values)),
            "p90": float(np.percentile(values, 90))}


def _inverse_pose(rotation, translation):
    inverse_rotation = rotation.T
    return inverse_rotation, -inverse_rotation @ translation


def anchor_poses(cache, frames, pair_poses=(), *, return_bridges=False):
    """Build anchor-to-camera poses using an explicit equal-step gauge.

    Cached odometry translations have unit norm per stride.  A fallback pair
    spanning multiple strides is therefore scaled by its stride count.  This
    is only the same equal-step approximation already used by composed cached
    odometry; it is not a metric scale estimate.
    """
    stride = int(cache["stride"])
    if stride < 1 or any(frame % stride for frame in frames):
        raise ValueError("feature frames must align with the positive cache stride")
    frame_set = set(frames)
    graph = {frame: [] for frame in frames}

    def add_edge(frame_i, frame_j, rotation, translation, cost, bridge=False):
        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        translation = np.asarray(translation, dtype=float).reshape(3)
        graph[frame_i].append(
            (cost, frame_j, rotation, translation, bridge, (frame_i, frame_j)))
        inverse = _inverse_pose(rotation, translation)
        graph[frame_j].append(
            (cost, frame_i, inverse[0], inverse[1], bridge, (frame_i, frame_j)))

    for frame_i, frame_j in zip(frames, frames[1:]):
        if frame_j - frame_i != stride:
            continue
        pose = compose_odom(cache["odom"], frame_i // stride,
                            frame_j // stride)
        if pose is not None:
            add_edge(frame_i, frame_j, pose[0], pose[1], 1)
    for frame_i, frame_j, rotation, translation in pair_poses:
        if frame_i in frame_set and frame_j in frame_set:
            steps = abs(frame_j - frame_i) // stride
            if steps < 1:
                continue
            # Prefer any path through cached odometry over an estimated bridge.
            add_edge(frame_i, frame_j, rotation, translation,
                     1000 + steps, bridge=True)
            graph[frame_i][-1] = (*graph[frame_i][-1][:3],
                                  graph[frame_i][-1][3] * steps,
                                  *graph[frame_i][-1][4:])
            graph[frame_j][-1] = (*graph[frame_j][-1][:3],
                                  graph[frame_j][-1][3] * steps,
                                  *graph[frame_j][-1][4:])

    first = frames[0]
    best = {first: (0, (first,))}
    poses = {first: (np.eye(3), np.zeros(3))}
    predecessor = {}
    queue = [(0, (first,), first)]
    while queue:
        cost, path, frame = heapq.heappop(queue)
        if best.get(frame) != (cost, path):
            continue
        R_frame, t_frame = poses[frame]
        for edge_cost, neighbor, R_edge, t_edge, bridge, source_pair in graph[frame]:
            candidate = (cost + edge_cost, path + (neighbor,))
            if neighbor in best and best[neighbor] <= candidate:
                continue
            best[neighbor] = candidate
            poses[neighbor] = (R_edge @ R_frame,
                               R_edge @ t_frame + t_edge)
            predecessor[neighbor] = (frame, bridge, source_pair)
            heapq.heappush(queue, (candidate[0], candidate[1], neighbor))
    missing = sorted(frame_set - poses.keys())
    if missing:
        raise RuntimeError(f"no pose path to frames {missing}")
    used_bridges = set()
    for frame in frames:
        cursor = frame
        while cursor != first:
            parent, bridge, source_pair = predecessor[cursor]
            if bridge:
                used_bridges.add(source_pair)
            cursor = parent
    bridges = [list(pair) for pair in sorted(used_bridges)]
    return (poses, bridges) if return_bridges else poses


def _camera_to_world_poses(anchor_to_camera):
    poses = {}
    for frame, (rotation, translation) in anchor_to_camera.items():
        pose = np.eye(4)
        pose[:3, :3] = rotation.T
        pose[:3, 3] = -rotation.T @ translation
        poses[frame] = pose
    return poses


def _largest_ba_component(poses, landmarks, observations, required_frames=()):
    adjacency = {}
    for item in observations:
        pose_node = ("p", item.frame_id)
        landmark_node = ("l", item.track_id)
        adjacency.setdefault(pose_node, set()).add(landmark_node)
        adjacency.setdefault(landmark_node, set()).add(pose_node)
    components = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        reached = {start}
        frontier = [start]
        while frontier:
            node = frontier.pop()
            for neighbor in adjacency[node] - reached:
                reached.add(neighbor)
                frontier.append(neighbor)
        remaining -= reached
        component_observations = tuple(
            item for item in observations
            if (("p", item.frame_id) in reached
                and ("l", item.track_id) in reached))
        if all(("p", frame) in reached for frame in required_frames):
            components.append((len(component_observations), reached,
                               component_observations))
    if not components:
        return {}, {}, ()
    _count, nodes, selected = max(
        components, key=lambda item: (item[0], -min(node[1]
                                                    for node in item[1])))
    selected_poses = {frame: poses[frame] for kind, frame in nodes if kind == "p"}
    selected_landmarks = {track: landmarks[track]
                          for kind, track in nodes if kind == "l"}
    return selected_poses, selected_landmarks, selected


def _run_ba_diagnostic(anchor_poses, keypoints, tracks, triangulated, cam,
                       opts):
    started = time.perf_counter()
    track_by_id = {track.track_id: track for track in tracks}
    valid_points = {result.track_id: result.point
                    for result in triangulated if result.hard_valid}
    observations = tuple(
        BAObservation(frame, track_id, tuple(keypoints[frame][feature]), feature)
        for track_id in sorted(valid_points)
        for frame, feature in track_by_id[track_id].observations)
    poses, landmarks, observations = _largest_ba_component(
        _camera_to_world_poses(anchor_poses), valid_points, observations)
    base = {
        "poses": len(poses), "landmarks": len(landmarks),
        "observations": len(observations), "iterations": 0,
        "initial_cost": None, "final_cost": None, "cost_ratio": None,
        "rotation_step_deg_median": None, "rotation_step_deg_max": None,
        "translation_step_median": None, "translation_step_max": None,
    }
    if len(poses) < 2 or not landmarks:
        return {"ok": False, "reason": "no connected BA component", **base,
                "elapsed_ms": 1000.0 * (time.perf_counter() - started)}

    result = optimize_local_ba(
        poses, landmarks, observations, cam,
        max_iterations=opts.ba_max_iterations, huber_delta=opts.ba_huber)
    movable = [frame for frame in sorted(poses) if frame != result.pose_ids[0]]
    increments = [se3_log(np.linalg.inv(poses[frame]) @ result.poses[frame])
                  for frame in movable]
    rotation_steps = [float(np.degrees(np.linalg.norm(step[:3])))
                      for step in increments]
    translation_steps = [float(np.linalg.norm(step[3:]))
                         for step in increments]
    return {
        "ok": result.ok, "reason": result.reason, **base,
        "iterations": result.iterations,
        "initial_cost": (result.initial_cost
                         if np.isfinite(result.initial_cost) else None),
        "final_cost": (result.final_cost
                       if np.isfinite(result.final_cost) else None),
        "cost_ratio": (result.final_cost / result.initial_cost
                       if result.ok and result.initial_cost > 0.0 else None),
        "rotation_step_deg_median": (float(np.median(rotation_steps))
                                     if rotation_steps else None),
        "rotation_step_deg_max": max(rotation_steps, default=None),
        "translation_step_median": (float(np.median(translation_steps))
                                    if translation_steps else None),
        "translation_step_max": max(translation_steps, default=None),
        "elapsed_ms": 1000.0 * (time.perf_counter() - started),
    }


def _trajectory_ate(poses, frames, gt_pos, stride):
    indices = [frame // stride for frame in frames]
    if (len(frames) < 3 or max(indices) >= len(gt_pos)
            or not np.isfinite(gt_pos[indices]).all()):
        raise ValueError("at least three aligned finite ground-truth positions required")
    est = np.array([poses[frame][:3, 3] for frame in frames])
    gt = np.asarray(gt_pos[indices], dtype=float)
    if np.mean(np.sum((est - est.mean(axis=0)) ** 2, axis=1)) <= 1e-12:
        raise ValueError("degenerate estimated trajectory")
    scale, rotation, translation = umeyama(est, gt, with_scale=True)
    errors = np.linalg.norm(scale * (est @ rotation.T) + translation - gt, axis=1)
    return {"median": float(np.median(errors)),
            "rmse": float(np.sqrt(np.mean(errors ** 2)))}


def _run_bounded_ba_ab(cache, frames, anchor, keypoints, tracks, cam, opts):
    """Offline window smoothing with full-sequence track/initial-pose prep.

    Only BA observations/triangulation are window-local. Input tracks and
    initial poses can depend on future frames: this is not an online A/B.
    ATE measures the retrospectively smoothed trajectory.
    """
    initial = _camera_to_world_poses(anchor)
    poses = {frame: pose.copy() for frame, pose in initial.items()}
    track_by_id = {track.track_id: track for track in tracks}
    baseline = _trajectory_ate(initial, frames, cache["gt_pos"], cache["stride"])
    rows = []
    started = time.perf_counter()
    for end in range(2, len(frames)):
        window = frames[max(0, end + 1 - opts.ba_window_size):end + 1]
        window_set = set(window)
        # Triangulate from the current window; do not reuse full-sequence points.
        local_anchor = {frame: (pose[:3, :3].T,
                                -pose[:3, :3].T @ pose[:3, 3])
                        for frame, pose in poses.items() if frame in window_set}
        local_tracks = [type(track)(track.track_id, tuple(
            (frame, feature) for frame, feature in track.observations
            if frame in window_set)) for track in tracks]
        local_tracks = [track for track in local_tracks
                        if len(track.observations) >= 3]
        landmarks = {}
        for track in local_tracks:
            triangulated = triangulate_feature_track(
                track, keypoints, local_anchor, cam.K,
                (opts.width, opts.height))
            if triangulated is not None and triangulated.hard_valid:
                landmarks[track.track_id] = triangulated.point
        observations = tuple(
            BAObservation(frame, track_id, tuple(keypoints[frame][feature]), feature)
            for track_id in sorted(landmarks)
            for frame, feature in track_by_id[track_id].observations
            if frame in window_set)
        selected_poses, selected_points, observations = _largest_ba_component(
            {frame: poses[frame] for frame in window}, landmarks, observations,
            required_frames=(window[0], window[1], window[-1]))
        row = {"end_frame": int(frames[end]), "frames": sorted(selected_poses),
               "landmarks": len(selected_points), "observations": len(observations),
               "accepted": False, "reason": "", "cost_ratio": None,
               "rotation_step_deg_max": None, "translation_step_ratio_max": None,
               "pre_ate_rmse": None, "candidate_ate_rmse": None}
        if (len(selected_poses) < 3 or frames[end] not in selected_poses
                or window[0] not in selected_poses
                or window[1] not in selected_poses):
            row["reason"] = "insufficient connected covisibility"
            rows.append(row)
            continue
        start_frame = min(selected_poses)
        # The window's initial baseline is the local scale reference.
        baseline_length = float(np.linalg.norm(
            poses[window[1]][:3, 3] - poses[window[0]][:3, 3]))
        if baseline_length <= 1e-12:
            row["reason"] = "degenerate baseline"
            rows.append(row)
            continue
        result = optimize_local_ba(
            selected_poses, selected_points, observations, cam,
            max_iterations=opts.ba_max_iterations, huber_delta=opts.ba_huber)
        row["cost_ratio"] = (float(result.final_cost / result.initial_cost)
                             if result.ok and result.initial_cost > 0 else None)
        if not result.ok:
            row["reason"] = result.reason
            rows.append(row)
            continue
        # Counterfactual only: GT never influences acceptance. Refit Sim(3)
        # independently for the pre/candidate trajectories over the same frames.
        row["pre_ate_rmse"] = _trajectory_ate(
            poses, frames, cache["gt_pos"], cache["stride"])["rmse"]
        row["candidate_ate_rmse"] = _trajectory_ate(
            {**poses, **result.poses}, frames, cache["gt_pos"],
            cache["stride"])["rmse"]
        steps = [se3_log(np.linalg.inv(poses[frame]) @ result.poses[frame])
                 for frame in sorted(selected_poses) if frame != start_frame]
        rotation_max = max(float(np.degrees(np.linalg.norm(step[:3])))
                           for step in steps)
        translation_max = max(float(np.linalg.norm(step[3:]) / baseline_length)
                              for step in steps)
        row["rotation_step_deg_max"] = rotation_max
        row["translation_step_ratio_max"] = translation_max
        if (not np.isfinite(rotation_max + translation_max)
                or rotation_max > opts.ba_max_rotation_deg
                or translation_max > opts.ba_max_translation_ratio):
            row["reason"] = "pose jump gate"
        else:
            row["accepted"] = True
            poses.update(result.poses)
        rows.append(row)
    return {"evaluation_scope": "offline_full_sequence_prep_retrospective_smoothing",
            "baseline_ate": baseline,
            "retrospective_bounded_ate": _trajectory_ate(
                poses, frames, cache["gt_pos"], cache["stride"]),
            "accepted": sum(row["accepted"] for row in rows),
            "rejected": sum(not row["accepted"] for row in rows),
            "elapsed_ms": 1000.0 * (time.perf_counter() - started),
            "windows": rows}


def evaluate_sequence(seq, opts, matcher):
    cache = load_cache(opts.cache_dir, seq)
    frames = sorted(cache["feat"])[:opts.max_frames]
    pairs = frame_pairs(frames, opts.pair_radius)
    fx, fy, cx, cy = intrinsics_for(
        opts.dataset_root, seq, (opts.fx, opts.fy, opts.cx, opts.cy))
    cam = CameraIntrinsics(fx, fy, cx, cy, opts.width, opts.height)
    pair_matches = []
    pair_poses = []
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
            if pose.get("R") is not None and pose.get("t") is not None:
                pair_poses.append((frame_i, frame_j, pose["R"], pose["t"]))
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
    poses, pose_bridges = anchor_poses(
        cache, frames, pair_poses, return_bridges=True)
    keypoints = {frame: np.asarray(cache["feat"][frame][0][0], dtype=float)
                 for frame in frames}
    triangulated = [
        result for track in selected_tracks
        if (result := triangulate_feature_track(
            track, keypoints, poses, cam.K, (opts.width, opts.height))) is not None
    ]
    ba_report = None
    if getattr(opts, "run_ba", False):
        ba_report = _run_ba_diagnostic(
            poses, keypoints, selected_tracks, triangulated, cam, opts)
    bounded_report = None
    if getattr(opts, "ba_window_ate", False):
        bounded_report = _run_bounded_ba_ab(
            cache, frames, poses, keypoints, built.tracks, cam, opts)
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
    parallax = [result.parallax_deg for result in triangulated]
    condition = [result.condition_ratio for result in triangulated]
    positive = [result.positive_depth_fraction for result in triangulated]
    reprojection_median = [result.reprojection_median_px
                           for result in triangulated]
    reprojection_p90 = [result.reprojection_p90_px
                        for result in triangulated]
    depth_baselines = [result.max_depth_baselines for result in triangulated]
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
        "triangulated_tracks": len(triangulated),
        "hard_valid_tracks": sum(result.hard_valid for result in triangulated),
        "pose_bridges": pose_bridges,
        "pose_bridge_scale_model": "equal unit translation per cache stride",
        "parallax_deg": percentile_summary(parallax),
        "condition_ratio": percentile_summary(condition),
        "positive_depth_fraction": percentile_summary(positive),
        "reprojection_median_px": percentile_summary(reprojection_median),
        "reprojection_p90_px": percentile_summary(reprojection_p90),
        "max_depth_baselines": percentile_summary(depth_baselines),
        "local_ba": ba_report,
        "bounded_ba_ab": bounded_report,
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
    parser.add_argument("--run-ba", action="store_true")
    parser.add_argument("--ba-max-iterations", type=int, default=15)
    parser.add_argument("--ba-huber", type=float, default=3.0)
    parser.add_argument("--ba-window-ate", action="store_true",
                        help="offline bounded-window BA vs initial trajectory ATE")
    parser.add_argument("--ba-window-size", type=int, default=6)
    parser.add_argument("--ba-max-rotation-deg", type=float, default=2.0)
    parser.add_argument("--ba-max-translation-ratio", type=float, default=0.25)
    opts = parser.parse_args()
    sequences = [value.strip() for value in opts.seq.split(",") if value.strip()]
    if not sequences:
        parser.error("--seq must contain at least one sequence")
    if (opts.max_frames < (3 if opts.ba_window_ate else 2)
            or opts.min_track_length < 2
            or opts.pair_radius < 1 or opts.min_shared_tracks < 1
            or opts.max_neighbors < 0 or opts.ba_max_iterations < 1
            or not np.isfinite(opts.ba_huber) or opts.ba_huber <= 0.0
            or opts.ba_window_size < 3
            or not np.isfinite(opts.ba_max_rotation_deg)
            or opts.ba_max_rotation_deg <= 0.0
            or not np.isfinite(opts.ba_max_translation_ratio)
            or opts.ba_max_translation_ratio <= 0.0):
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
