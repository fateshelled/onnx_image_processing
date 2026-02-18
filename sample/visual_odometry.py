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
import numpy as np
import onnxruntime as ort
from PIL import Image

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.vo import (
    CameraIntrinsics,
    estimate_pose_ransac,
    Trajectory,
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

    # Check mutual consistency
    mutual_mask = np.zeros(K, dtype=bool)
    for i in range(K):
        j = max_j_for_i[i]
        if max_i_for_j[j] == i:
            mutual_mask[i] = True

    # Get match probabilities
    match_indices_i = np.where(mutual_mask)[0]
    match_indices_j = max_j_for_i[match_indices_i]
    scores = P_core[match_indices_i, match_indices_j]

    # Apply threshold
    above_threshold = scores >= threshold
    match_indices_i = match_indices_i[above_threshold]
    match_indices_j = match_indices_j[above_threshold]
    scores = scores[above_threshold]

    matched_kpts1 = kpts1[match_indices_i]
    matched_kpts2 = kpts2[match_indices_j]

    return matched_kpts1, matched_kpts2, scores


class VideoReader:
    """Read frames from video file, image sequence, or webcam."""

    def __init__(self, source, is_video: bool = True, is_camera: bool = False):
        """
        Initialize video reader.

        Args:
            source: Video file path, image directory, or camera device ID
            is_video: If True, read from video file; otherwise from image directory
            is_camera: If True, read from webcam (source should be camera ID)
        """
        self.is_video = is_video
        self.is_camera = is_camera
        self.source = source

        if is_camera:
            # Open webcam
            self.cap = cv2.VideoCapture(int(source))
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera: {source}")
            self.total_frames = float('inf')  # Unlimited for camera
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps == 0:
                self.fps = 30.0  # Default FPS for cameras
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
        if self.is_camera or self.is_video:
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
        if self.is_camera or self.is_video:
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
    min_matches: int = 20,
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
        min_matches: Minimum number of matches required
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

    # Read first frame
    ret, prev_frame = reader.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    prev_image = load_image_from_array(prev_frame, model_height, model_width)
    display_frame = prev_frame.copy() if display else None

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
        matched_kpts1, matched_kpts2, scores = extract_matches(
            matching_probs,
            keypoints1,
            keypoints2,
            threshold=match_threshold,
        )

        num_matches = len(scores)
        total_matches += num_matches

        # Initialize status for display
        status_message = None
        pose_updated = False

        if num_matches < min_matches:
            if verbose:
                print(f"Frame {frame_count}: Insufficient matches ({num_matches} < {min_matches}), skipping...")
            status_message = f"INSUFFICIENT MATCHES ({num_matches}/{min_matches})"
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

            if R is None or num_inliers < min_matches:
                if verbose:
                    print(f"Frame {frame_count}: Pose estimation failed (inliers={num_inliers}), skipping...")
                status_message = f"POSE ESTIMATION FAILED (inliers={num_inliers}/{min_matches})"
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
            pos = trajectory.get_current_position()
            dist = trajectory.get_trajectory_length()

            # Display current status
            cv2.putText(info_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if status_message:
                # Show error/warning message
                cv2.putText(info_frame, status_message, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(info_frame, f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(info_frame, f"Distance: {dist:.2f}m", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Show normal trajectory info
                cv2.putText(info_frame, f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(info_frame, f"Distance: {dist:.2f}m", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(info_frame, f"Matches: {num_matches}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(info_frame, f"Inliers: {num_inliers}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
        type=int,
        help="Webcam device ID (e.g., 0 for default camera)"
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
        required=True,
        help="Focal length in x direction (pixels)"
    )
    parser.add_argument(
        "--fy",
        type=float,
        required=True,
        help="Focal length in y direction (pixels)"
    )
    parser.add_argument(
        "--cx",
        type=float,
        required=True,
        help="Principal point x coordinate (pixels)"
    )
    parser.add_argument(
        "--cy",
        type=float,
        required=True,
        help="Principal point y coordinate (pixels)"
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
        "--min-matches",
        type=int,
        default=20,
        help="Minimum number of matches required (default: 20)"
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

    # Create camera intrinsics
    camera_intrinsics = CameraIntrinsics(
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        width=model_width,
        height=model_height,
    )
    print(f"\nCamera intrinsics: {camera_intrinsics}")

    # Open video/image/camera source
    if args.camera is not None:
        print(f"\nOpening camera: {args.camera}")
        reader = VideoReader(args.camera, is_video=False, is_camera=True)
    elif args.video:
        print(f"\nOpening video: {args.video}")
        reader = VideoReader(args.video, is_video=True, is_camera=False)
    else:
        print(f"\nOpening image directory: {args.image_dir}")
        reader = VideoReader(args.image_dir, is_video=False, is_camera=False)

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
            min_matches=args.min_matches,
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
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

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
