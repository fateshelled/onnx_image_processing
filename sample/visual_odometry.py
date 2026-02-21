#!/usr/bin/env python3
"""
Visual Odometry sample using ONNX feature matching model.

Estimates camera trajectory from a video or image sequence using the
Shi-Tomasi + Angle + Sparse BAD + Sinkhorn ONNX model for feature matching.

Usage:
    # First, export the ONNX model:
    python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn.py -o matcher.onnx -H 480 -W 640 --max-keypoints 512

    # Run VO on video:
    python sample/visual_odometry.py --model matcher.onnx --video video.mp4 --fx 525 --fy 525 --cx 320 --cy 240

    # Run VO on image sequence:
    python sample/visual_odometry.py --model matcher.onnx --image-dir frames/ --fx 525 --fy 525 --cx 320 --cy 240

    # Run VO on webcam:
    python sample/visual_odometry.py --model matcher.onnx --camera 0 --fx 525 --fy 525 --cx 320 --cy 240 --display

    # Save trajectory and visualization:
    python sample/visual_odometry.py --model matcher.onnx --video video.mp4 --fx 525 --fy 525 --cx 320 --cy 240 --save-trajectory trajectory.npz --save-plot trajectory.png
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

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.vo import (
    CameraIntrinsics,
    estimate_pose_ransac,
    Trajectory,
    create_camera,
)
from provider_utils import create_session


def load_image_from_array(
    image: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Convert an image array to model input format.

    Args:
        image: Input image array (H, W, 3) or (H, W)
        height: Target height
        width: Target width

    Returns:
        Grayscale image array of shape (1, 1, H, W) with values in [0, 255]
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Resize
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)

    # Convert to float32 and add batch/channel dimensions
    arr = resized.astype(np.float32)
    return arr[np.newaxis, np.newaxis, :, :]


def extract_matches(
    matching_probs: np.ndarray,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    threshold: float = 0.1,
    max_matches: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract mutual nearest-neighbor matches from Sinkhorn probability matrix.

    Args:
        matching_probs: Sinkhorn probability matrix of shape (1, K+1, K+1)
        keypoints1: Keypoints in image1 of shape (1, K, 2) as (y, x)
        keypoints2: Keypoints in image2 of shape (1, K, 2) as (y, x)
        threshold: Minimum match probability

    Returns:
        Tuple of:
            - matched_kpts1: (N, 2) matched keypoint coordinates in image1
            - matched_kpts2: (N, 2) matched keypoint coordinates in image2
            - match_scores: (N,) match probability scores
    """
    P = matching_probs[0]  # (K+1, K+1)
    kpts1 = keypoints1[0]  # (K, 2)
    kpts2 = keypoints2[0]  # (K, 2)

    K = kpts1.shape[0]

    # Core probability matrix excluding dustbin
    P_core = P[:K, :K]  # (K, K)

    # Mutual nearest neighbors
    max_j_for_i = np.argmax(P_core, axis=1)
    max_i_for_j = np.argmax(P_core, axis=0)

    # Check mutual consistency (vectorized)
    mutual_mask = np.arange(K) == max_i_for_j[max_j_for_i]

    # Get match probabilities
    match_indices_i = np.where(mutual_mask)[0]
    match_indices_j = max_j_for_i[match_indices_i]
    scores = P_core[match_indices_i, match_indices_j]

    # Apply threshold
    above_threshold = scores >= threshold
    match_indices_i = match_indices_i[above_threshold]
    match_indices_j = match_indices_j[above_threshold]
    scores = scores[above_threshold]

    # Sort by score descending and take top matches
    sort_order = np.argsort(scores)[::-1][:max_matches]
    match_indices_i = match_indices_i[sort_order]
    match_indices_j = match_indices_j[sort_order]
    scores = scores[sort_order]

    matched_kpts1 = kpts1[match_indices_i]
    matched_kpts2 = kpts2[match_indices_j]

    return matched_kpts1, matched_kpts2, scores


class VideoReader:
    """Read frames from video file, image sequence, or webcam."""

    def __init__(
        self,
        source,
        is_video: bool = True,
        is_camera: bool = False,
        camera_backend: str = "opencv",
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
    ):
        """
        Initialize video reader.

        Args:
            source: Video file path, image directory, or camera device ID
            is_video: If True, read from video file; otherwise from image directory
            is_camera: If True, read from webcam (source should be camera ID)
            camera_backend: Camera backend ("opencv" or "realsense")
            camera_width: Camera resolution width
            camera_height: Camera resolution height
            camera_fps: Camera framerate
        """
        self.is_video = is_video
        self.is_camera = is_camera
        self.source = source
        self.camera = None
        self.cap = None

        if is_camera:
            # Open camera using wrapper
            # Try to convert to int for numeric device IDs, otherwise keep as string
            try:
                device_id = int(source)
            except (ValueError, TypeError):
                device_id = source

            self.camera = create_camera(
                backend=camera_backend,
                device_id=device_id,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
                enable_depth=False,
            )
            self.total_frames = float('inf')  # Unlimited for camera
            self.fps = self.camera.get_fps()
        elif is_video:
            # Open video file
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {source}")
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            # Load image file list
            patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            self.image_files = []
            for pattern in patterns:
                self.image_files.extend(glob.glob(os.path.join(source, pattern)))
            self.image_files.sort()
            self.total_frames = len(self.image_files)
            self.fps = 30.0  # Default
            self.frame_idx = 0

            if self.total_frames == 0:
                raise RuntimeError(f"No images found in: {source}")

    def read(self) -> tuple[bool, np.ndarray]:
        """
        Read next frame.

        Returns:
            Tuple of (success, frame)
        """
        if self.is_camera:
            ret, frame = self.camera.read()
            return ret, frame
        elif self.is_video:
            ret, frame = self.cap.read()
            return ret, frame
        else:
            if self.frame_idx >= self.total_frames:
                return False, None
            frame = cv2.imread(self.image_files[self.frame_idx])
            self.frame_idx += 1
            return True, frame

    def release(self):
        """Release resources."""
        if self.is_camera and self.camera is not None:
            self.camera.release()
        elif self.is_video and self.cap is not None:
            self.cap.release()

    def __len__(self) -> int:
        """Return total number of frames."""
        return self.total_frames


def run_visual_odometry(
    session: ort.InferenceSession,
    reader: VideoReader,
    camera_intrinsics: CameraIntrinsics,
    model_height: int,
    model_width: int,
    match_threshold: float = 0.1,
    ransac_threshold: float = 1.0,
    max_matches: int = 100,
    min_matches: int = 20,
    min_inlier_ratio: float = 0.5,
    min_motion_pixels: float = 1.0,
    skip_frames: int = 1,
    max_frames: int = None,
    verbose: bool = True,
    display: bool = False,
) -> Trajectory:
    """
    Run visual odometry on video/image sequence/webcam.

    Args:
        session: ONNX Runtime session
        reader: Video reader
        camera_intrinsics: Camera intrinsic parameters
        model_height: Model input height
        model_width: Model input width
        match_threshold: Minimum match probability
        ransac_threshold: RANSAC reprojection threshold
        max_matches: Maximum number of matches
        min_matches: Minimum number of matches required
        min_inlier_ratio: Minimum ratio of RANSAC inliers to matches (0-1).
            Frames where inlier_count/match_count is below this threshold are
            rejected. A low inlier ratio indicates a degenerate Essential Matrix
            (fitted to noise), leading to large random trajectory jumps.
            Default: 0.5 (require at least 50% inliers).
        min_motion_pixels: Minimum RMS pixel displacement between matched
            keypoints to attempt pose estimation (default: 1.0). When the camera
            is stationary the optical flow is near-zero, causing findEssentialMat
            to fit a degenerate matrix and recoverPose to give unstable inlier
            counts (0-3 or all inliers randomly). Frames below this threshold
            are classified as "no motion" and skipped without updating the pose.
            The reference frame IS updated so stale references are avoided.
        skip_frames: Process every N-th frame
        max_frames: Maximum number of frames to process
        verbose: Print progress information
        display: Display frames and trajectory in real-time

    Returns:
        Trajectory object containing camera poses
    """
    trajectory = Trajectory()

    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    # warm up
    if reader.is_camera:
        for _ in range(10):
            ret, prev_frame = reader.read()

    # Read first frame
    ret, prev_frame = reader.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    prev_image = load_image_from_array(prev_frame, model_height, model_width)
    # display_frame = prev_frame.copy() if display else None

    frame_count = 0
    processed_count = 0
    total_matches = 0
    total_inliers = 0

    if verbose:
        print(f"Processing frames (skip={skip_frames})...")
        if display:
            print("Press 'q' to quit, 's' to save current trajectory")

    start_time = time.time()

    while True:
        # Read next frame
        ret, curr_frame = reader.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames if needed
        if frame_count % (skip_frames + 1) != 0:
            continue

        processed_count += 1

        # Check max frames limit
        if max_frames is not None and processed_count > max_frames:
            break

        # Convert current frame to model input
        curr_image = load_image_from_array(curr_frame, model_height, model_width)

        # Run feature matching
        results = session.run(
            output_names,
            {input_names[0]: prev_image, input_names[1]: curr_image},
        )

        keypoints1 = results[0]  # (1, K, 2)
        keypoints2 = results[1]  # (1, K, 2)
        matching_probs = results[2]  # (1, K+1, K+1)

        # Extract matches
        matched_kpts1, matched_kpts2, _scores = extract_matches(
            matching_probs,
            keypoints1,
            keypoints2,
            threshold=match_threshold,
            max_matches=max_matches,
        )

        num_matches = len(matched_kpts1)
        total_matches += num_matches

        # Initialize status for display
        status_message = None
        pose_updated = False
        inlier_mask = np.zeros(num_matches, dtype=bool)
        num_inliers = 0

        if num_matches < min_matches:
            if verbose:
                print(f"Frame {frame_count}: Insufficient matches ({num_matches} < {min_matches}), skipping...")
            status_message = f"INSUFFICIENT MATCHES ({num_matches}/{min_matches})"
        else:
            # Check for sufficient motion before running pose estimation.
            # When the camera is stationary, optical flow is near-zero and
            # findEssentialMat produces a degenerate Essential Matrix, causing
            # recoverPose to return unstable inlier counts (0-3 or all inliers).
            flow = matched_kpts2 - matched_kpts1  # (N, 2) in (dy, dx)
            rms_flow = float(np.sqrt(np.mean(np.sum(flow ** 2, axis=1))))

            if rms_flow < min_motion_pixels:
                # Insufficient motion: skip pose estimation. Do NOT update
                # prev_image here so that slow continuous motion accumulates
                # across frames and eventually crosses the threshold.
                status_message = f"NO MOTION (rms={rms_flow:.2f}px)"
                if verbose:
                    print(f"Frame {frame_count}: No motion (rms={rms_flow:.2f}px), skipping...")
            else:
                # Estimate pose using RANSAC
                R, t, inlier_mask = estimate_pose_ransac(
                    matched_kpts1,
                    matched_kpts2,
                    camera_intrinsics,
                    ransac_threshold=ransac_threshold,
                )

                num_inliers = np.sum(inlier_mask)
                total_inliers += num_inliers

                inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0
                if R is None or num_inliers < min_matches or inlier_ratio < min_inlier_ratio:
                    if verbose:
                        print(f"Frame {frame_count}: Pose estimation failed "
                              f"(inliers={num_inliers}, ratio={inlier_ratio:.0%}), skipping...")
                    status_message = (f"POSE ESTIMATION FAILED "
                                      f"(inliers={num_inliers}, ratio={inlier_ratio:.0%})")
                else:
                    # Add pose to trajectory
                    trajectory.add_relative_pose(R, t)
                    pose_updated = True

                    # Update previous frame only on success
                    prev_image = curr_image

                    if verbose and processed_count % 10 == 0:
                        elapsed = time.time() - start_time
                        fps = processed_count / elapsed
                        if reader.is_camera or reader.total_frames == float('inf'):
                            print(f"Frame {frame_count}: "
                                  f"matches={num_matches}, inliers={num_inliers}, "
                                  f"position={trajectory.get_current_position()}, "
                                  f"fps={fps:.1f}")
                        else:
                            print(f"Frame {frame_count}/{reader.total_frames}: "
                                  f"matches={num_matches}, inliers={num_inliers}, "
                                  f"position={trajectory.get_current_position()}, "
                                  f"fps={fps:.1f}")

        # Display frame and trajectory in real-time (always update if display is on)
        if display:
            # Draw trajectory info on frame
            info_frame = curr_frame.copy()
            frame_h, frame_w = info_frame.shape[:2]
            pos = trajectory.get_current_position()
            dist = trajectory.get_trajectory_length()

            # Auto-scale font size and thickness based on frame size
            # Reference: 640x480 with font_scale=0.7, thickness=2
            base_width = 640
            base_height = 480
            size_scale = min(frame_w / base_width, frame_h / base_height)
            font_scale = 0.7 * size_scale
            font_thickness = max(1, int(2 * size_scale))

            # Calculate line spacing based on scaled font
            line_height = int(30 * size_scale)
            margin_x = int(10 * size_scale)
            start_y = line_height

            # Scale keypoints from model resolution to frame resolution
            scale_x = frame_w / model_width
            scale_y = frame_h / model_height

            # Scale keypoint radius based on frame size
            base_radius = max(1, int(3 * size_scale))

            # Draw matched keypoints
            if num_matches > 0:
                for (y, x), inlier in zip(matched_kpts2, inlier_mask):
                    # Keypoints are in (y, x) format
                    px = int(x * scale_x)
                    py = int(y * scale_y)

                    # Color based on inlier/outlier status
                    if pose_updated and inlier_mask is not None and inlier:
                        # Inliers: Green
                        color = (0, 255, 0)
                        radius = base_radius + 1
                    elif inlier_mask is not None and not inlier:
                        # Outliers (RANSAC rejected): Red
                        color = (0, 0, 255)
                        radius = base_radius
                    else:
                        # No pose estimate: Yellow
                        color = (0, 255, 255)
                        radius = base_radius

                    cv2.circle(info_frame, (px, py), radius, color, -1)
                    cv2.circle(info_frame, (px, py), radius + 1, (0, 0, 0), 1)

            # Always display the same number of lines to prevent flickering
            # Line 1: Frame number
            cv2.putText(info_frame, f"Frame: {frame_count}",
                       (margin_x, start_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

            # Line 2: Status message (error or OK)
            if status_message:
                cv2.putText(info_frame, status_message,
                           (margin_x, start_y + line_height),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
            else:
                cv2.putText(info_frame, "STATUS: OK",
                           (margin_x, start_y + line_height),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

            # Line 3: Position
            cv2.putText(info_frame, f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]",
                       (margin_x, start_y + line_height * 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

            # Line 4: Distance
            cv2.putText(info_frame, f"Distance: {dist:.2f}m",
                       (margin_x, start_y + line_height * 3),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

            # Line 5: Matches and Inliers
            cv2.putText(info_frame, f"Matches: {num_matches} | Inliers: {num_inliers}",
                       (margin_x, start_y + line_height * 4),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

            cv2.imshow('Visual Odometry', info_frame)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                save_path = f"trajectory_{int(time.time())}.npz"
                trajectory.save_to_file(save_path)
                print(f"\nTrajectory saved to: {save_path}")

    elapsed = time.time() - start_time

    if verbose:
        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Processed frames: {processed_count}")
        print(f"Trajectory length: {len(trajectory)} poses")
        print(f"Average matches: {total_matches / max(1, processed_count):.1f}")
        print(f"Average inliers: {total_inliers / max(1, len(trajectory) - 1):.1f}")
        print(f"Total distance: {trajectory.get_trajectory_length():.2f} meters")
        print(f"Processing time: {elapsed:.2f} seconds ({processed_count / elapsed:.1f} fps)")

    return trajectory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visual Odometry using ONNX feature matching model"
    )

    # Input source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--video", "-v",
        type=str,
        help="Input video file path"
    )
    source_group.add_argument(
        "--image-dir", "-d",
        type=str,
        help="Input image directory path"
    )
    source_group.add_argument(
        "--camera", "-c",
        type=str,
        help="Camera device ID (e.g., '0' for default, or serial number/MxID for RealSense/OAK-D)"
    )

    # Model and camera parameters
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to the exported ONNX model file"
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=None,
        help="Focal length in x direction (pixels). Auto-detected for RealSense cameras."
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=None,
        help="Focal length in y direction (pixels). Auto-detected for RealSense cameras."
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=None,
        help="Principal point x coordinate (pixels). Auto-detected for RealSense cameras."
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=None,
        help="Principal point y coordinate (pixels). Auto-detected for RealSense cameras."
    )
    parser.add_argument(
        "--camera-backend",
        type=str,
        default="opencv",
        choices=["opencv", "realsense", "orbbec", "oak"],
        help="Camera backend (opencv, realsense, orbbec, or oak, default: opencv)"
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Camera resolution width (default: 640)"
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Camera resolution height (default: 480)"
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=30,
        help="Camera framerate (default: 30)"
    )

    # Processing parameters
    parser.add_argument(
        "--match-threshold", "-t",
        type=float,
        default=0.1,
        help="Match probability threshold (default: 0.1)"
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.0,
        help="RANSAC reprojection threshold in pixels (default: 1.0)"
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=300,
        help="Maximum number of matches (default: 300)"
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=10,
        help="Minimum number of matches required (default: 10)"
    )
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.5,
        help="Minimum RANSAC inlier ratio (inliers/matches) to accept a pose estimate. "
             "Frames below this threshold are skipped to prevent degenerate E matrix jumps. "
             "(default: 0.5)"
    )
    parser.add_argument(
        "--min-motion-pixels",
        type=float,
        default=1.0,
        help="Minimum RMS pixel displacement of matched keypoints to attempt pose estimation. "
             "Frames below this threshold are treated as 'no motion' to avoid degenerate "
             "Essential Matrix estimation when the camera is stationary (default: 1.0)"
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Process every N-th frame (0=process all frames, default: 0)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: None, process all)"
    )

    # Output options
    parser.add_argument(
        "--save-trajectory",
        type=str,
        default=None,
        help="Save trajectory to file (*.npz)"
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Save trajectory plot to file (*.png)"
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="Plot 3D trajectory instead of 2D"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display frames and trajectory in real-time (press 'q' to quit, 's' to save)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create ONNX Runtime session
    print(f"Loading ONNX model: {args.model}")
    session = create_session(args.model)
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    # Get model input dimensions
    input_shape = inputs[0].shape  # [B, 1, H, W]
    model_height = input_shape[2]
    model_width = input_shape[3]

    print(f"Model input size: {model_height}x{model_width}")
    for inp in inputs:
        print(f"  Input:  {inp.name} {inp.shape}")
    for out in outputs:
        print(f"  Output: {out.name} {out.shape}")

    # Open video/image/camera source
    if args.camera is not None:
        print(f"\nOpening camera: {args.camera} (backend: {args.camera_backend})")
        reader = VideoReader(
            args.camera,
            is_video=False,
            is_camera=True,
            camera_backend=args.camera_backend,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
        )
    elif args.video:
        print(f"\nOpening video: {args.video}")
        reader = VideoReader(args.video, is_video=True, is_camera=False)
    else:
        print(f"\nOpening image directory: {args.image_dir}")
        reader = VideoReader(args.image_dir, is_video=False, is_camera=False)

    # Create camera intrinsics
    # For RealSense/Orbbec/OAK cameras, auto-detect intrinsics if not provided
    if args.camera is not None and args.camera_backend in ["realsense", "orbbec", "oak"]:
        if args.fx is None or args.fy is None or args.cx is None or args.cy is None:
            print(f"\nAuto-detecting camera intrinsics from {args.camera_backend.upper()}...")
            if hasattr(reader.camera, 'get_camera_intrinsics'):
                camera_intrinsics = reader.camera.get_camera_intrinsics()
                if camera_intrinsics is None:
                    raise RuntimeError(f"Failed to get camera intrinsics from {args.camera_backend.upper()}")
                print(f"Camera intrinsics (auto-detected, native resolution): {camera_intrinsics}")
                # Scale intrinsics from camera native resolution to model input resolution.
                # Essential Matrix estimation requires intrinsics in the same coordinate
                # space as the keypoints (model resolution), not the camera's native resolution.
                scale_x = model_width / camera_intrinsics.width
                scale_y = model_height / camera_intrinsics.height
                if scale_x != 1.0 or scale_y != 1.0:
                    camera_intrinsics = CameraIntrinsics(
                        fx=camera_intrinsics.fx * scale_x,
                        fy=camera_intrinsics.fy * scale_y,
                        cx=camera_intrinsics.cx * scale_x,
                        cy=camera_intrinsics.cy * scale_y,
                        width=model_width,
                        height=model_height,
                    )
                    print(f"Camera intrinsics (scaled to model {model_width}x{model_height}): {camera_intrinsics}")
            else:
                raise RuntimeError("Camera does not support intrinsics auto-detection")
        else:
            # Use manually specified intrinsics
            camera_intrinsics = CameraIntrinsics(
                fx=args.fx,
                fy=args.fy,
                cx=args.cx,
                cy=args.cy,
                width=model_width,
                height=model_height,
            )
            print(f"\nCamera intrinsics (manual): {camera_intrinsics}")
    else:
        # Non-3D-camera: require manual specification
        if args.fx is None or args.fy is None or args.cx is None or args.cy is None:
            raise ValueError(
                "Camera intrinsics (--fx, --fy, --cx, --cy) are required for OpenCV cameras and video files. "
                "Please specify all intrinsic parameters."
            )
        camera_intrinsics = CameraIntrinsics(
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            width=model_width,
            height=model_height,
        )
        print(f"\nCamera intrinsics: {camera_intrinsics}")

    if reader.is_camera:
        print(f"Camera mode (unlimited frames)")
    else:
        print(f"Total frames: {len(reader)}")
    print(f"FPS: {reader.fps:.2f}")

    # Run visual odometry
    try:
        trajectory = run_visual_odometry(
            session,
            reader,
            camera_intrinsics,
            model_height,
            model_width,
            match_threshold=args.match_threshold,
            ransac_threshold=args.ransac_threshold,
            max_matches=args.max_matches,
            min_matches=args.min_matches,
            min_inlier_ratio=args.min_inlier_ratio,
            min_motion_pixels=args.min_motion_pixels,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
            verbose=not args.quiet,
            display=args.display,
        )
    finally:
        reader.release()
        if args.display:
            cv2.destroyAllWindows()

    # Save trajectory if requested
    if args.save_trajectory:
        trajectory.save_to_file(args.save_trajectory)
        print(f"\nTrajectory saved to: {args.save_trajectory}")

    # Plot trajectory if requested
    if args.save_plot:
        print(f"\nGenerating trajectory plot...")
        matplotlib.use('Agg')  # Non-interactive backend

        if args.plot_3d:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            trajectory.plot_3d(ax, show_orientation=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            trajectory.plot_2d(ax, show_orientation=True)

        plt.tight_layout()
        plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to: {args.save_plot}")
        plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
