#!/usr/bin/env python3
"""Visual odometry sample: ONNX matcher + the shared online pose graph.

The whole pose-graph logic lives in :mod:`vo.online_graph` (the same code the
offline evaluator drives), so this script only wires the pieces together:

* a frame source (video / image directory / camera),
* the ONNX matching model,
* an index-based ``match_fn(i, j)`` backed by a small frame-image cache,
* trajectory output (npz / plot) and optional live display.

The graph requests every match itself (odometry chain, keyframe spokes, loop
closures); the sample never decides graph structure. Use ``--skip-frames`` to
keep a usable baseline between nodes (consecutive frames are often degenerate
for a monocular Essential-matrix solve).
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent))

from vo import CameraIntrinsics, Trajectory, create_camera  # noqa: E402
from vo.online_graph import DEFAULT_PARAMS, OnlinePoseGraph  # noqa: E402
from vo.onnx_matcher import OnnxSessionMatcher  # noqa: E402
from provider_utils import create_session  # noqa: E402


def load_image(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert an image to model input: (1, 1, H, W) grayscale in [0, 255]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)[np.newaxis, np.newaxis, :, :]


def output_indices(output_names):
    """Resolve model outputs by name (robust to extra outputs / ordering)."""
    low = {n.lower(): i for i, n in enumerate(output_names)}

    def pick(*cands):
        for c in cands:
            if c in low:
                return low[c]
        return None

    return {
        "k1": pick("keypoints1", "keypoints_1", "kp1"),
        "k2": pick("keypoints2", "keypoints_2", "kp2"),
        "probs": pick("matching_probs", "matching_probabilities", "probs"),
    }


class VideoReader:
    """Read frames from a video file, an image sequence, or a camera."""

    def __init__(self, source, is_video=True, is_camera=False,
                 camera_backend="opencv", camera_width=640, camera_height=480,
                 camera_fps=30):
        self.is_video = is_video
        self.is_camera = is_camera
        self.source = source
        self.camera = None
        self.cap = None
        self.frame_idx = 0

        if is_camera:
            try:
                device_id = int(source)
            except (ValueError, TypeError):
                device_id = source
            self.camera = create_camera(
                backend=camera_backend, device_id=device_id,
                width=camera_width, height=camera_height, fps=camera_fps,
                enable_depth=False)
            self.total_frames = float("inf")
            self.fps = self.camera.get_fps()
        elif is_video:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {source}")
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
            self.image_files = []
            for pattern in patterns:
                self.image_files.extend(glob.glob(os.path.join(source, pattern)))
            self.image_files.sort()
            self.total_frames = len(self.image_files)
            self.fps = 30.0
            if self.total_frames == 0:
                raise RuntimeError(f"No images found in: {source}")

    def read(self):
        if self.is_camera:
            return self.camera.read()
        if self.is_video:
            return self.cap.read()
        if self.frame_idx >= self.total_frames:
            return False, None
        frame = cv2.imread(self.image_files[self.frame_idx])
        self.frame_idx += 1
        return True, frame

    def release(self):
        if self.is_camera and self.camera is not None:
            self.camera.release()
        elif self.is_video and self.cap is not None:
            self.cap.release()

    def __len__(self):
        return self.total_frames


def draw_display_info(frame, trajectory, frame_count, stats, last, status,
                      model_width, model_height):
    """Annotate a frame with trajectory/status and the last matched keypoints."""
    info = frame.copy()
    fh, fw = info.shape[:2]
    scale = min(fw / 640.0, fh / 480.0)
    font = 0.7 * scale
    thick = max(1, int(2 * scale))
    lh = int(30 * scale)
    mx = int(10 * scale)

    if last is not None and len(last.get("kpts2", [])) > 0:
        kp2 = last["kpts2"]
        mask = last.get("inlier_mask")
        sx, sy = fw / model_width, fh / model_height
        r = max(1, int(3 * scale))
        for i, (y, x) in enumerate(kp2):
            inl = mask is not None and i < len(mask) and mask[i]
            color = (0, 255, 0) if inl else (0, 0, 255)
            cv2.circle(info, (int(x * sx), int(y * sy)), r + (1 if inl else 0),
                       color, -1)

    pos = trajectory.get_current_position()
    dist = trajectory.get_trajectory_length()
    lines = [
        (f"Frame: {frame_count}", (0, 255, 0)),
        (status if status else "STATUS: OK", (0, 0, 255) if status else (0, 255, 0)),
        (f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]", (0, 255, 0)),
        (f"Distance: {dist:.2f} (norm)", (0, 255, 0)),
        (f"Keyframes: {stats['n_kf']} | Loops: {stats['n_loop']}", (0, 255, 0)),
        (f"Matches: {stats['n_matches']} | ok={stats['ok']}", (0, 255, 0)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(info, text, (mx, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    font, color, thick)
    return info


def run_visual_odometry(session, reader, camera_intrinsics, model_height,
                        model_width, params=None, match_threshold=0.1,
                        ransac_threshold=1.4, max_matches=1024, min_matches=20,
                        skip_frames=0, max_frames=None, verbose=True,
                        display=False):
    """Run the online pose graph over a frame source.

    Returns ``(trajectory, graph)``.
    """
    input_names = [i.name for i in session.get_inputs()]
    oi = output_indices([o.name for o in session.get_outputs()])
    for key in ("k1", "k2", "probs"):
        if oi[key] is None:
            raise RuntimeError(f"Model output for '{key}' not found")

    matcher = OnnxSessionMatcher(
        session, camera_intrinsics, input_names, oi,
        match_threshold=match_threshold, max_matches=max_matches,
        min_matches=min_matches, min_inlier_ratio=0.0)
    matcher.debug_display = display

    images = {}

    def match_fn(i, j):
        if i not in images or j not in images:
            return {"ok": False, "n_matches": 0, "inlier_ratio": 0.0}
        return matcher.match(images[i], images[j])

    graph = OnlinePoseGraph(dict(DEFAULT_PARAMS if params is None else params),
                            camera_intrinsics, match_fn)
    trajectory = Trajectory()

    if reader.is_camera:
        for _ in range(10):  # warm up auto-exposure / auto-focus
            if not reader.read()[0]:
                break

    frame_count = 0
    processed = 0
    idx = 0
    start = time.time()

    while True:
        ret, frame = reader.read()
        if not ret:
            break
        frame_count += 1
        if skip_frames and frame_count % (skip_frames + 1) != 0:
            continue
        processed += 1
        if max_frames is not None and processed > max_frames:
            break

        images[idx] = load_image(frame, model_height, model_width)
        T = graph.add_frame(idx)
        trajectory.add_pose(T)

        # Keep only keyframes + the most recent frames in the image cache.
        keep = set(graph.keyframes) | {idx - 1, idx}
        for k in [k for k in images if k not in keep]:
            images.pop(k, None)

        stats = {
            "ok": graph.last_stats["ok"],
            "n_matches": graph.last_stats["n_matches"],
            "n_kf": graph.n_kf,
            "n_loop": graph.n_loop,
        }
        if verbose and processed % 10 == 0:
            elapsed = time.time() - start
            print(f"Frame {frame_count}: position="
                  f"{trajectory.get_current_position()}, "
                  f"odom_ok={stats['ok']}, kf={stats['n_kf']}, "
                  f"loops={stats['n_loop']}, "
                  f"fps={processed / max(elapsed, 1e-6):.1f}")

        if display:
            info = draw_display_info(
                frame, trajectory, frame_count, stats, matcher.last,
                None if stats["ok"] else "POSE FAILED",
                model_width, model_height)
            cv2.imshow("Visual Odometry", info)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = f"trajectory_{int(time.time())}.npz"
                trajectory.save_to_file(path)
                print(f"Trajectory saved to: {path}")

        idx += 1

    elapsed = time.time() - start
    if verbose:
        print(f"\nProcessing complete: {processed} frames, "
              f"{len(trajectory)} poses, {graph.n_kf} keyframes, "
              f"{graph.n_loop} loops")
        print(f"Total distance: {trajectory.get_trajectory_length():.2f} "
              f"(normalised units, monocular scale ambiguity)")
        if processed:
            print(f"Processing time: {elapsed:.2f}s "
                  f"({processed / elapsed:.1f} fps)")

    return trajectory, graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visual odometry with an ONNX matcher and the online "
                    "keyframe pose graph (vo.online_graph)")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", "-v", type=str, help="Input video file")
    src.add_argument("--image-dir", "-d", type=str, help="Input image directory")
    src.add_argument("--camera", "-c", type=str, help="Camera device ID")

    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Exported ONNX matching model")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--camera-backend", type=str, default="opencv",
                        choices=["opencv", "realsense", "orbbec", "oak"])
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)

    parser.add_argument("--match-threshold", "-t", type=float, default=0.1,
                        help="Minimum match probability (default: 0.1)")
    parser.add_argument("--ransac-threshold", type=float, default=1.4,
                        help="MAGSAC/RANSAC reprojection threshold (default: 1.4)")
    parser.add_argument("--max-matches", type=int, default=1024,
                        help="Maximum matches per pair (default: 1024)")
    parser.add_argument("--min-matches", type=int, default=20,
                        help="Minimum matches/inliers to accept a pose (default: 20)")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Process every N-th frame (0=all, default: 0)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to process")

    parser.add_argument("--save-trajectory", type=str, default=None,
                        help="Save trajectory to *.npz")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Save trajectory plot to *.png")
    parser.add_argument("--plot-3d", action="store_true")
    parser.add_argument("--display", action="store_true",
                        help="Show frames (q quit, s save)")
    parser.add_argument("--quiet", "-q", action="store_true")
    return parser.parse_args()


def read_intrinsics(args, model_width, model_height, reader):
    if args.camera is not None and args.camera_backend in ("realsense", "orbbec", "oak") \
            and args.fx is None:
        if not hasattr(reader.camera, "get_camera_intrinsics"):
            raise RuntimeError("Camera does not support intrinsics auto-detection")
        intr = reader.camera.get_camera_intrinsics()
        if intr is None:
            raise RuntimeError("Failed to get camera intrinsics")
        sx = model_width / intr.width
        sy = model_height / intr.height
        return CameraIntrinsics(fx=intr.fx * sx, fy=intr.fy * sy,
                                cx=intr.cx * sx, cy=intr.cy * sy,
                                width=model_width, height=model_height)
    if None in (args.fx, args.fy, args.cx, args.cy):
        raise ValueError("Camera intrinsics (--fx --fy --cx --cy) are required")
    return CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                            width=model_width, height=model_height)


def main():
    args = parse_args()

    print(f"Loading ONNX model: {args.model}")
    session = create_session(args.model)
    shape = session.get_inputs()[0].shape
    model_height, model_width = shape[2], shape[3]
    print(f"Model input size: {model_height}x{model_width}")
    for o in session.get_outputs():
        print(f"  Output: {o.name} {o.shape}")

    if args.camera is not None:
        reader = VideoReader(args.camera, is_video=False, is_camera=True,
                             camera_backend=args.camera_backend,
                             camera_width=args.camera_width,
                             camera_height=args.camera_height,
                             camera_fps=args.camera_fps)
    elif args.video:
        reader = VideoReader(args.video, is_video=True)
    else:
        reader = VideoReader(args.image_dir, is_video=False)

    camera_intrinsics = read_intrinsics(args, model_width, model_height, reader)
    print(f"Camera intrinsics: {camera_intrinsics}")
    print(f"Total frames: {len(reader)}  FPS: {reader.fps:.2f}")

    try:
        trajectory, _graph = run_visual_odometry(
            session, reader, camera_intrinsics, model_height, model_width,
            match_threshold=args.match_threshold,
            ransac_threshold=args.ransac_threshold,
            max_matches=args.max_matches, min_matches=args.min_matches,
            skip_frames=args.skip_frames, max_frames=args.max_frames,
            verbose=not args.quiet, display=args.display)
    finally:
        reader.release()
        if args.display:
            cv2.destroyAllWindows()

    if args.save_trajectory:
        trajectory.save_to_file(args.save_trajectory)
        print(f"\nTrajectory saved to: {args.save_trajectory}")

    if args.save_plot:
        matplotlib.use("Agg")
        if args.plot_3d:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")
            trajectory.plot_3d(ax, show_orientation=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            trajectory.plot_2d(ax, show_orientation=True)
        plt.tight_layout()
        plt.savefig(args.save_plot, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Trajectory plot saved to: {args.save_plot}")

    print("\nDone!")


if __name__ == "__main__":
    main()
