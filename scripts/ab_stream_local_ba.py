"""Evaluate a causal bounded BA overlay on the fixed-cache online pose graph.

The graph runs unchanged. BA reuses the graph's *current* poses as incoming
measurements and propagates accepted corrections into its overlay. Consecutive
mode freezes poses as they leave the window; keyframe mode retains the O(N)
trajectory until the final graph state so keyframe corrections can be applied
consistently to intermediate frames. It does not feed pose/landmark factors
into the graph's marginalized priors; report this as an overlay A/B. BA feature,
match, track, and landmark windows are bounded, while the evaluator's fixed
caches are loaded for the full sequence.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.eval_tum_vo import estimate_pose_from_matches, intrinsics_for  # noqa: E402
from eval.rustuna_tune_loop import SEQ_OPT1_DEFAULTS, eval_seq, load_cache  # noqa: E402
from eval.torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from scripts.diag_local_tracks import _largest_ba_component, _trajectory_ate  # noqa: E402
from vo.local_ba import BAObservation, optimize_local_ba  # noqa: E402
from vo.local_tracks import PairMatches, build_feature_tracks, triangulate_feature_track  # noqa: E402
from vo.onnx_matcher import extract_match_indices  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402
from vo.se3 import se3_log  # noqa: E402
from eval.rustuna_tune_loop import DEFAULT_ARGS  # noqa: E402


def _pair(cache, frame_i, frame_j, matcher, cam):
    kp_i, desc_i = cache["feat"][frame_i]
    kp_j, desc_j = cache["feat"][frame_j]
    probs = matcher.match_probs(desc_i[0], desc_j[0])[None]
    idx_i, idx_j, scores = extract_match_indices(
        kp_i, kp_j, probs, DEFAULT_ARGS.match_threshold,
        DEFAULT_ARGS.max_matches, DEFAULT_ARGS.dbin)
    pose = estimate_pose_from_matches(
        kp_i[0][idx_i], kp_j[0][idx_j], cam, DEFAULT_ARGS)
    mask = (np.asarray(pose["mask"], dtype=bool).reshape(-1)
            if pose.get("ok") else np.zeros(len(idx_i), dtype=bool))
    if len(mask) != len(idx_i):
        raise ValueError(f"inlier mask mismatch at {frame_i}, {frame_j}")
    return PairMatches(frame_i, frame_j, idx_i, idx_j, scores, mask)


def _solve_window(frames, pairs, poses, keypoints, cam, opts,
                  relative_priors=None):
    """One bounded local BA; return a full-window atomic proposal and diagnostics."""
    built = build_feature_tracks(pairs)
    local_anchor = {frame: (poses[frame][:3, :3].T,
                            -poses[frame][:3, :3].T @ poses[frame][:3, 3])
                    for frame in frames}
    points = {}
    observations = []
    for track in built.tracks:
        if len(track.observations) < 3:
            continue
        result = triangulate_feature_track(
            track, keypoints, local_anchor, cam.K, (cam.width, cam.height))
        if result is None or not result.hard_valid:
            continue
        points[track.track_id] = result.point
        observations.extend(BAObservation(
            frame, track.track_id, tuple(keypoints[frame][feature]), feature)
            for frame, feature in track.observations)
    selected, landmarks, observations = _largest_ba_component(
        {frame: poses[frame] for frame in frames}, points, tuple(observations),
        required_frames=(frames[0], frames[1], frames[-1]))
    row = {"end_frame": frames[-1], "poses": len(selected),
           "landmarks": len(landmarks), "observations": len(observations),
           "accepted": False, "reason": "", "cost_ratio": None,
           "rotation_step_deg_max": None, "translation_step_ratio_max": None}
    if len(selected) < 3:
        row["reason"] = "insufficient connected covisibility"
        return {}, row
    relative_priors = {
        edge: prior for edge, prior in (relative_priors or {}).items()
        if edge[0] in selected and edge[1] in selected}
    baseline = float(np.linalg.norm(
        poses[frames[1]][:3, 3] - poses[frames[0]][:3, 3]))
    if not np.isfinite(baseline) or baseline <= 1e-8:
        row["reason"] = "degenerate baseline"
        return {}, row
    result = optimize_local_ba(
        selected, landmarks, observations, cam,
        max_iterations=opts.ba_max_iterations, huber_delta=opts.ba_huber,
        relative_priors=relative_priors,
        prior_rotation_sigma=opts.ba_prior_rotation_sigma,
        prior_direction_sigma=opts.ba_prior_direction_sigma,
        deadline=(time.monotonic() + getattr(opts, "ba_deadline_ms", 0.0) / 1000.0
                  if getattr(opts, "ba_deadline_ms", 0.0) > 0.0 else None))
    if not result.ok:
        row["reason"] = result.reason
        return {}, row
    row["cost_ratio"] = (float(result.final_cost / result.initial_cost)
                         if result.initial_cost > 0 else None)
    steps = [se3_log(np.linalg.inv(poses[f]) @ result.poses[f])
             for f in selected if f != frames[0]]
    rotation_max = max(float(np.degrees(np.linalg.norm(step[:3])))
                       for step in steps)
    translation_max = max(float(np.linalg.norm(step[3:]) / baseline)
                          for step in steps)
    row["rotation_step_deg_max"] = rotation_max
    row["translation_step_ratio_max"] = translation_max
    if (not np.isfinite(rotation_max + translation_max)
            or rotation_max > opts.ba_max_rotation_deg
            or translation_max > opts.ba_max_translation_ratio):
        row["reason"] = "pose jump gate"
        return {}, row
    row["accepted"] = True
    return result.poses, row


def _window_frames(frame, graph, retained, size, mode):
    if mode == "consecutive":
        return [*retained, frame][-(size):]
    previous = [keyframe for keyframe in graph.keyframes
                if keyframe < frame]
    return [*previous[-(size - 1):], frame]


def _apply_keyframe_proposal(proposal, overlay, keyframes):
    """Propagate each optimized anchor correction to its following segment."""
    anchors = sorted(proposal)
    boundaries = sorted(keyframes)
    old = {frame: pose.copy() for frame, pose in overlay.items()}
    for anchor in anchors:
        stop = next((frame for frame in boundaries if frame > anchor), None)
        correction = proposal[anchor] @ np.linalg.inv(old[anchor])
        for frame in overlay:
            if frame >= anchor and (stop is None or frame < stop):
                overlay[frame] = correction @ old[frame]


def evaluate_stream(cache, params, matcher, cam, opts, match_cache=None):
    """Run unchanged graph + bounded BA overlay; freeze both arms at exit."""
    stride = cache["stride"]
    frames = sorted(cache["feat"])
    if (len(frames) != len(cache["odom"]) + 1
            or frames != list(range(0, len(frames) * stride, stride))):
        raise ValueError("stream cache must contain consecutive stride frames")
    graph_state = [None]
    active = []
    pairs = {}
    overlay = {}
    graph_seen = {}
    baseline_frozen = {}
    ba_frozen = {}
    rows = []
    ba_elapsed = 0.0
    started = time.perf_counter()

    def on_frame(frame, graph):
        nonlocal ba_elapsed, pairs
        graph_state[0] = graph
        # Apply each graph correction since the previous callback before
        # freezing an old frame or extending the corrected BA trajectory.
        synced = overlay if opts.window_mode == "keyframe" else active
        for old in synced:
            current = graph.pose(old)
            overlay[old] = current @ np.linalg.inv(graph_seen[old]) @ overlay[old]
            graph_seen[old] = current
        baseline_pose = graph.pose(frame)
        if active:
            prev = max(active)
            # Carry BA's accepted corrections forward using the *current*
            # graph-relative motion; later graph corrections remain independent.
            overlay[frame] = overlay[prev] @ np.linalg.inv(graph.pose(prev)) @ baseline_pose
        else:
            overlay[frame] = baseline_pose
        graph_seen[frame] = baseline_pose
        window = _window_frames(
            frame, graph, active, opts.ba_window_size, opts.window_mode)
        for leaving in set(active) - set(window):
            if opts.window_mode == "consecutive":
                baseline_frozen[leaving] = graph.pose(leaving)
                ba_frozen[leaving] = overlay.pop(leaving)
                del graph_seen[leaving]
        active[:] = window
        wanted_pairs = {
            (earlier, current)
            for index, current in enumerate(window)
            for earlier in window[max(0, index - opts.pair_radius):index]
        }
        pairs = {edge: pair for edge, pair in pairs.items()
                 if edge in wanted_pairs}
        for edge in sorted(wanted_pairs - set(pairs)):
            pairs[edge] = _pair(cache, *edge, matcher, cam)
        if len(active) >= 3:
            keypoints = {f: np.asarray(cache["feat"][f][0][0], dtype=float)
                          for f in window}
            t0 = time.perf_counter()
            solve_args = (
                window, tuple(pairs[edge] for edge in sorted(pairs)),
                overlay, keypoints, cam, opts)
            prior_rotation_sigma = getattr(opts, "ba_prior_rotation_sigma", 0.0)
            prior_direction_sigma = getattr(opts, "ba_prior_direction_sigma", 0.0)
            if prior_rotation_sigma > 0.0:
                prior_poses = {f: graph.pose(f) for f in window}
                relative_priors = {
                    (a, b): np.linalg.inv(prior_poses[a]) @ prior_poses[b]
                    for a, b in zip(window, window[1:])
                    if np.linalg.norm(prior_poses[b][:3, 3]
                                     - prior_poses[a][:3, 3]) > 1e-12}
                proposal, row = _solve_window(
                    *solve_args, relative_priors=relative_priors)
            else:
                proposal, row = _solve_window(*solve_args)
            ba_elapsed += time.perf_counter() - t0
            if opts.window_mode == "keyframe":
                _apply_keyframe_proposal(proposal, overlay, graph.keyframes)
            else:
                overlay.update(proposal)
            row["window_frames"] = window
            rows.append(row)
        if len(baseline_frozen) and len(baseline_frozen) % 50 == 0:
            print(f"[stream] frozen={len(baseline_frozen)}/{len(frames)}",
                  file=sys.stderr, flush=True)

    graph_report = eval_seq(cache, params, cam, matcher, match_cache,
                            on_frame=on_frame)
    graph = graph_state[0]
    if graph is None:
        raise RuntimeError("graph produced no frames")
    remaining = frames if opts.window_mode == "keyframe" else active
    for frame in remaining:
        baseline_frozen[frame] = graph.pose(frame)
        ba_frozen[frame] = overlay[frame]
    return {
        "evaluation_scope": "causal_graph_input_ba_overlay_no_graph_feedback_full_cache_loaded",
        "frames": len(frames), "window_size": opts.ba_window_size,
        "window_mode": opts.window_mode,
        "graph_final_ate_median": graph_report["ATE_median"],
        "baseline_frozen_ate": _trajectory_ate(
            baseline_frozen, frames, cache["gt_pos"], stride),
        "ba_frozen_ate": _trajectory_ate(
            ba_frozen, frames, cache["gt_pos"], stride),
        "accepted": sum(row["accepted"] for row in rows),
        "rejected": sum(not row["accepted"] for row in rows),
        "ba_elapsed_ms": ba_elapsed * 1000,
        "total_elapsed_ms": (time.perf_counter() - started) * 1000,
        "graph_loops": graph_report["n_loop"],
        "windows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="desk,desk2,room")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("eval/results/tune_cache_loop"))
    parser.add_argument("--dataset-root", default="/home/ubuntu/datasets/tum_rgbd")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--ba-window-size", type=int, default=6)
    parser.add_argument("--window-mode", choices=("consecutive", "keyframe"),
                        default="consecutive")
    parser.add_argument("--kf-trans-thresh", type=float,
                        help="override motion keyframe translation threshold")
    parser.add_argument("--kf-rot-thresh", type=float,
                        help="override motion keyframe rotation threshold")
    parser.add_argument("--kf-max-gap", type=int,
                        help="override maximum frame-index gap between keyframes")
    parser.add_argument("--pair-radius", type=int, default=2)
    parser.add_argument("--ba-max-iterations", type=int, default=15)
    parser.add_argument("--ba-huber", type=float, default=3.0)
    parser.add_argument("--ba-prior-rotation-sigma", type=float, default=0.0,
                        help="relative rotation prior sigma in radians; 0=off")
    parser.add_argument("--ba-prior-direction-sigma", type=float, default=0.0,
                        help="relative translation direction prior sigma; 0=off")
    parser.add_argument("--ba-deadline-ms", type=float, default=0.0,
                        help="per-window solver deadline; 0=unbounded")
    parser.add_argument("--ba-max-rotation-deg", type=float, default=2.0)
    parser.add_argument("--ba-max-translation-ratio", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    opts = parser.parse_args()
    if (opts.ba_window_size < 3 or opts.pair_radius < 1
            or opts.ba_max_iterations < 1
            or opts.max_frames is not None and opts.max_frames < 3
            or not np.isfinite([opts.ba_huber, opts.ba_max_rotation_deg,
                                opts.ba_max_translation_ratio]).all()
            or min(opts.ba_huber, opts.ba_max_rotation_deg,
                   opts.ba_max_translation_ratio) <= 0):
        parser.error("invalid window, pair, BA or gate settings")
    if ((opts.ba_prior_rotation_sigma < 0.0
         or opts.ba_prior_direction_sigma < 0.0
         or not np.isfinite([opts.ba_prior_rotation_sigma,
                             opts.ba_prior_direction_sigma]).all())
            or ((opts.ba_prior_rotation_sigma == 0.0)
                != (opts.ba_prior_direction_sigma == 0.0))):
        parser.error("prior sigmas must both be zero or both be positive")
    if (not np.isfinite(opts.ba_deadline_ms) or opts.ba_deadline_ms < 0.0):
        parser.error("ba deadline must be finite and nonnegative")
    if (opts.kf_trans_thresh is not None
            and (not np.isfinite(opts.kf_trans_thresh)
                 or opts.kf_trans_thresh <= 0)
            or opts.kf_rot_thresh is not None
            and (not np.isfinite(opts.kf_rot_thresh)
                 or opts.kf_rot_thresh <= 0)
            or opts.kf_max_gap is not None and opts.kf_max_gap < 1):
        parser.error("invalid keyframe promotion settings")
    sequences = [s.strip() for s in opts.seq.split(",") if s.strip()]
    if not sequences:
        parser.error("--seq must include a sequence")
    matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")
    reports = []
    failed = False
    for seq in sequences:
        try:
            cache = load_cache(opts.cache_dir, seq)
            if opts.max_frames is not None:
                frames = sorted(cache["feat"])[:opts.max_frames]
                cache["feat"] = {f: cache["feat"][f] for f in frames}
                cache["odom"] = cache["odom"][:len(frames) - 1]
                cache["gt_pos"] = cache["gt_pos"][:len(frames)]
            fx, fy, cx, cy = intrinsics_for(
                opts.dataset_root, seq, (525.0, 525.0, 320.0, 240.0))
            cam = CameraIntrinsics(fx, fy, cx, cy, 640, 480)
            match_path = opts.cache_dir / f"match_cache_{seq}_torch.pkl"
            with match_path.open("rb") as handle:
                match_cache = pickle.load(handle)
            params = dict(SEQ_OPT1_DEFAULTS)
            if opts.kf_trans_thresh is not None:
                params["kf_trans_thresh"] = opts.kf_trans_thresh
            if opts.kf_rot_thresh is not None:
                params["kf_rot_thresh"] = opts.kf_rot_thresh
            if opts.kf_max_gap is not None:
                params["kf_max_gap"] = opts.kf_max_gap
            result = evaluate_stream(cache, params, matcher,
                                     cam, opts, match_cache=match_cache)
            reports.append({"sequence": seq, **result})
        except Exception as exc:  # preserve other sequences
            failed = True
            reports.append({"sequence": seq, "error": f"{type(exc).__name__}: {exc}"})
    text = json.dumps(reports, indent=2, allow_nan=False)
    if opts.output is not None:
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        opts.output.write_text(text + "\n")
    print(text)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
