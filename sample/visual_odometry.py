#!/usr/bin/env python3
"""
Visual Odometry sample using ONNX feature matching model.

Estimates camera trajectory from a video or image sequence.  Two model types
are supported and detected automatically from the number of model outputs:

  3-output model (keypoints1, keypoints2, matching_probs):
      Shi-Tomasi + Angle + Sparse BAD + Sinkhorn
      Pose estimation is performed with OpenCV RANSAC (findEssentialMat).

  4-output model (keypoints1, keypoints2, matching_probs, E):
      Shi-Tomasi + Angle + Sparse BAD + Sinkhorn + Essential Matrix
      The Essential Matrix is estimated inside the ONNX model using the
      weighted 8-point algorithm.  Pose recovery (recoverPose) is called
      directly with the ONNX-provided E, skipping RANSAC.

Usage:
    # Export the 3-output model:
    python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn.py -o matcher.onnx -H 480 -W 640 --max-keypoints 512

    # Export the 4-output combined model (Essential Matrix baked in):
    python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn_essential_matrix.py \\
        -o matcher_e.onnx -H 480 -W 640 --max-keypoints 512 \\
        --fx 525 --fy 525 --cx 320 --cy 240

    # Run VO on video:
    python sample/visual_odometry.py --model matcher.onnx --video video.mp4 --fx 525 --fy 525 --cx 320 --cy 240

    # Run VO on image sequence:
    python sample/visual_odometry.py --model matcher.onnx --image-dir frames/ --fx 525 --fy 525 --cx 320 --cy 240

    # Run VO on webcam with live trajectory plot:
    python sample/visual_odometry.py --model matcher.onnx --camera 0 --fx 525 --fy 525 --cx 320 --cy 240 --display --plot-realtime

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


def estimate_pose_from_essential_matrix(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    E: np.ndarray,
    camera_intrinsics,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """
    Recover camera pose from a pre-computed Essential Matrix.

    Used with the 4-output combined ONNX model
    (Shi-Tomasi + Angle + Sparse BAD + Sinkhorn + Essential Matrix), which
    embeds Essential Matrix estimation inside the ONNX graph.  Unlike the
    RANSAC-based ``estimate_pose_ransac``, this function skips
    ``findEssentialMat`` and calls ``recoverPose`` directly with the
    ONNX-provided E.

    Args:
        keypoints1: Matched keypoints in image 1, shape (N, 2) as (y, x).
        keypoints2: Matched keypoints in image 2, shape (N, 2) as (y, x).
        E: Essential Matrix from the ONNX model, shape (3, 3).
        camera_intrinsics: Camera intrinsic parameters (CameraIntrinsics).

    Returns:
        Tuple of:
            - R: Rotation matrix (3, 3) or None if recovery failed.
            - t: Translation vector (3, 1) or None if recovery failed.
            - inlier_mask: Boolean mask of chirality-passing points (N,).
    """
    if len(keypoints1) < 5:
        return None, None, np.zeros(len(keypoints1), dtype=bool)

    # Convert from (y, x) to (x, y) as required by OpenCV
    pts1 = keypoints1[:, [1, 0]].astype(np.float64)
    pts2 = keypoints2[:, [1, 0]].astype(np.float64)

    E_f64 = E.astype(np.float64)

    num_inliers, R, t, pose_mask = cv2.recoverPose(
        E_f64,
        pts1,
        pts2,
        camera_intrinsics.K,
    )

    if num_inliers < 5:
        return None, None, np.zeros(len(keypoints1), dtype=bool)

    inlier_mask = pose_mask.ravel() > 0
    return R, t, inlier_mask


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


def draw_display_info(
    frame: np.ndarray,
    trajectory,  # Trajectory type
    frame_count: int,
    num_matches: int,
    num_inliers: int,
    matched_kpts2: np.ndarray,
    inlier_mask: np.ndarray,
    pose_updated: bool,
    status_message: str,
    model_width: int,
    model_height: int,
) -> np.ndarray:
    """
    Draw visual odometry information on frame.

    Args:
        frame: Input frame to annotate
        trajectory: Trajectory object
        frame_count: Current frame number
        num_matches: Number of feature matches
        num_inliers: Number of RANSAC inliers
        matched_kpts2: Matched keypoints in current frame (N, 2) as (y, x)
        inlier_mask: Boolean mask indicating inliers (N,) or None
        pose_updated: Whether pose was successfully updated
        status_message: Error/warning message or None
        model_width: Model input width for scaling
        model_height: Model input height for scaling

    Returns:
        Annotated frame with trajectory info and keypoints
    """
    info_frame = frame.copy()
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
        for i, (y, x) in enumerate(matched_kpts2):
            # Keypoints are in (y, x) format
            px = int(x * scale_x)
            py = int(y * scale_y)

            # Color based on inlier/outlier status
            if pose_updated and inlier_mask is not None and inlier_mask[i]:
                # Inliers: Green
                color = (0, 255, 0)
                radius = base_radius + 1
            elif inlier_mask is not None and not inlier_mask[i]:
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

    return info_frame


def draw_match_frame(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    matched_kpts1: np.ndarray,
    matched_kpts2: np.ndarray,
    inlier_mask: np.ndarray,
    model_width: int,
    model_height: int,
    rms_flow: float = 0.0,
    mean_flow_mag: float = 0.0,
    max_display_height: int = 480,
) -> np.ndarray:
    """
    Draw a side-by-side keypoint match visualization for debugging.

    Left half: reference (previous) frame.  Right half: current frame.
    Connecting lines are colour-coded:
      Green  = inliers  (chirality-passing after recoverPose)
      Red    = outliers (chirality-failing)
      Yellow = no pose estimate (no motion / estimation failed)

    Args:
        prev_frame: Reference frame BGR (H, W, 3)
        curr_frame: Current frame BGR (H, W, 3)
        matched_kpts1: Matched keypoints in reference frame (N, 2) as (y, x) in model resolution
        matched_kpts2: Matched keypoints in current frame (N, 2) as (y, x) in model resolution
        inlier_mask: Boolean inlier mask (N,); None or all-False = no pose
        model_width: Model input width in pixels
        model_height: Model input height in pixels
        rms_flow: RMS optical-flow magnitude (pixels) shown in info text
        mean_flow_mag: Mean optical-flow magnitude (pixels) shown in info text
        max_display_height: Height each frame is scaled to for display

    Returns:
        Side-by-side BGR canvas ready for cv2.imshow
    """
    fh, fw = prev_frame.shape[:2]
    disp_h = max_display_height
    disp_w = max(1, int(fw * disp_h / max(fh, 1)))

    prev_disp = cv2.resize(prev_frame, (disp_w, disp_h))
    curr_disp = cv2.resize(curr_frame, (disp_w, disp_h))

    canvas = np.zeros((disp_h, disp_w * 2, 3), dtype=np.uint8)
    canvas[:, :disp_w] = prev_disp
    canvas[:, disp_w:] = curr_disp

    # Scale from model resolution to display resolution
    sx = disp_w / max(model_width, 1)
    sy = disp_h / max(model_height, 1)

    n_inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
    has_pose = inlier_mask is not None and n_inliers > 0

    for i, (kp1, kp2) in enumerate(zip(matched_kpts1, matched_kpts2)):
        y1, x1 = kp1
        y2, x2 = kp2
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx) + disp_w, int(y2 * sy))

        if not has_pose:
            color = (0, 200, 200)  # Yellow: no pose estimate
        elif inlier_mask[i]:
            color = (0, 210, 0)    # Green: inlier
        else:
            color = (0, 0, 200)    # Red: outlier

        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1)
        cv2.circle(canvas, p2, 2, color, -1)

    # Vertical divider
    cv2.line(canvas, (disp_w, 0), (disp_w, disp_h - 1), (160, 160, 160), 1)

    # Info text
    info = (f"Matches:{len(matched_kpts1)}  Inliers:{n_inliers}  "
            f"RMS:{rms_flow:.1f}px  Mean:{mean_flow_mag:.2f}px")
    cv2.putText(canvas, info, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Reference", (8, disp_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Current", (disp_w + 8, disp_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    return canvas


def _nice_grid_step(data_range: float) -> float:
    """Compute a human-friendly grid interval for the given data range."""
    if data_range <= 0:
        return 1.0
    rough = data_range / 6.0
    exp = int(np.floor(np.log10(rough)))
    mag = 10.0 ** exp
    frac = rough / mag
    if frac < 1.5:
        return mag * 1.0
    elif frac < 3.5:
        return mag * 2.0
    elif frac < 7.5:
        return mag * 5.0
    else:
        return mag * 10.0


class TrajectoryViewer:
    """
    Real-time trajectory visualization window using OpenCV.

    Supports two display modes toggled with the ``t`` key:

    * **2D** – top-down X-Z plane view (default)
    * **3D** – orthographic projection with configurable azimuth / elevation

    The window is created on construction and destroyed by :meth:`destroy`
    (or implicitly when ``cv2.destroyAllWindows()`` is called).

    A mouse callback is pre-registered for future pan / zoom / rotate support;
    see :meth:`_on_mouse` for the implementation guide.

    Public view-state attributes (can be set before calling :meth:`render`):

    * ``mode``        – ``"2d"`` or ``"3d"``
    * ``azimuth``     – 3-D horizontal orbit angle in degrees (default -60)
    * ``elevation``   – 3-D tilt above horizontal in degrees (default 30)
    * ``zoom``        – multiplicative zoom factor; >1 zooms in (default 1.0)
    * ``pan_x``       – world-space X pan offset in metres (default 0.0)
    * ``pan_z``       – world-space Z pan offset in metres (default 0.0)
    """

    WINDOW_NAME = "Trajectory"

    def __init__(self, canvas_size: int = 600) -> None:
        self.canvas_size = canvas_size
        self.mode: str = "2d"       # "2d" | "3d"

        # 3-D view angles (degrees)
        self.azimuth: float = -60.0     # horizontal orbit around world Y
        self.elevation: float = 30.0    # tilt above horizontal

        # 2-D / 3-D shared pan & zoom (reserved for mouse interaction)
        self.zoom: float = 1.0          # >1 → zoomed in; <1 → zoomed out
        self.pan_x: float = 0.0         # world-space X offset (metres)
        self.pan_z: float = 0.0         # world-space Z offset (metres)

        # Smoothed heading direction (EMA of camera Z-axis in world frame)
        # Lower alpha → smoother arrow but slower to react to real turns.
        self.heading_smoothing: float = 0.15
        self._smoothed_heading: np.ndarray = np.array([0.0, 0.0, 1.0])

        # Internal drag state used by the mouse callback
        self._mouse_pressed: bool = False
        self._mouse_button: int = -1    # 0 = left, 1 = right
        self._drag_start_x: int = 0
        self._drag_start_y: int = 0
        self._mouse_x: int = 0
        self._mouse_y: int = 0

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

    # ------------------------------------------------------------------
    # Mouse interaction – stub; extend here to add interactivity
    # ------------------------------------------------------------------

    def _on_mouse(
        self, event: int, x: int, y: int, flags: int, param
    ) -> None:
        """
        OpenCV mouse callback – currently a stub for future interactivity.

        Planned behaviour (not yet active):

        **2-D mode**

          - Left-drag       pan the view   (update ``pan_x``, ``pan_z``)
          - Scroll up/down  zoom in / out  (update ``zoom``)
          - Double-click    reset pan & zoom to defaults

        **3-D mode**

          - Left-drag       orbit camera   (update ``azimuth``, ``elevation``)
          - Right-drag      pan target     (update ``pan_x``, ``pan_z``)
          - Scroll up/down  zoom in / out  (update ``zoom``)
          - Double-click    reset to default view angles

        Implementation guide – fill in the placeholder ``pass`` statements:

        1. ``EVENT_LBUTTONDOWN`` / ``EVENT_RBUTTONDOWN`` →
           save ``_drag_start_x/y``, set ``_mouse_pressed = True``.
        2. ``EVENT_MOUSEMOVE`` while pressed →
           compute ``dx = x - _drag_start_x``, ``dy = y - _drag_start_y``;
           update view state:

           - 2-D pan:   ``pan_x -= dx / scale``,  ``pan_z += dy / scale``
           - 3-D orbit: ``azimuth  -= dx * 0.5``,  ``elevation = clamp(elevation + dy * 0.5, -89, 89)``

           Update ``_drag_start_x/y`` to ``x, y`` so delta is incremental.
        3. ``EVENT_LBUTTONUP`` / ``EVENT_RBUTTONUP`` →
           clear ``_mouse_pressed``.
        4. ``EVENT_MOUSEWHEEL`` (``flags > 0`` = scroll up = zoom in) →
           ``zoom = clamp(zoom * (1.1 if flags > 0 else 0.9), 0.1, 50)``.
        """
        self._mouse_x, self._mouse_y = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_pressed = True
            self._mouse_button = 0
            self._drag_start_x, self._drag_start_y = x, y
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._mouse_pressed = True
            self._mouse_button = 1
            self._drag_start_x, self._drag_start_y = x, y
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._mouse_pressed = False
            self._mouse_button = -1
        elif event == cv2.EVENT_MOUSEMOVE and self._mouse_pressed:
            pass  # TODO: pan (2-D) / orbit (3-D)
        elif event == cv2.EVENT_MOUSEWHEEL:
            pass  # TODO: zoom (flags > 0 → zoom in, flags < 0 → zoom out)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def toggle_mode(self) -> None:
        """Toggle between '2d' and '3d' display modes."""
        self.mode = "3d" if self.mode == "2d" else "2d"

    def render(self, trajectory) -> None:
        """Render the trajectory to the OpenCV window."""
        # Update heading EMA once per frame (shared by both 2D and 3D paths)
        fwd_raw = trajectory.get_current_pose()[:3, :3][:, 2]
        self._smoothed_heading = (
            self.heading_smoothing * fwd_raw
            + (1.0 - self.heading_smoothing) * self._smoothed_heading
        )
        norm = np.linalg.norm(self._smoothed_heading)
        if norm > 1e-6:
            self._smoothed_heading /= norm

        if self.mode == "2d":
            canvas = self._draw_2d(trajectory)
        else:
            canvas = self._draw_3d(trajectory)
        cv2.imshow(self.WINDOW_NAME, canvas)

    def destroy(self) -> None:
        """Destroy the trajectory window."""
        cv2.destroyWindow(self.WINDOW_NAME)

    # ------------------------------------------------------------------
    # 2-D top-down view (X-Z plane)
    # ------------------------------------------------------------------

    def _draw_2d(self, trajectory) -> np.ndarray:
        """Render top-down X-Z plane view."""
        H = W = self.canvas_size
        mg_top, mg_bottom, mg_left, mg_right = 30, 65, 55, 20
        px0, py0 = mg_left, mg_top
        px1, py1 = W - mg_right, H - mg_bottom
        pw, ph = px1 - px0, py1 - py0

        canvas = np.full((H, W, 3), 25, dtype=np.uint8)

        positions = trajectory.get_positions_array()  # (N, 3)
        xs, zs = positions[:, 0], positions[:, 2]

        x_center = (xs.min() + xs.max()) / 2.0 + self.pan_x
        z_center = (zs.min() + zs.max()) / 2.0 + self.pan_z

        # Apply zoom: larger zoom → smaller visible world range → larger scale
        x_view = max(xs.max() - xs.min(), 1.0) / self.zoom
        z_view = max(zs.max() - zs.min(), 1.0) / self.zoom
        scale = min(pw / x_view, ph / z_view) * 0.82  # px / m

        def to_canvas(x: float, z: float) -> tuple[int, int]:
            cx = int(px0 + pw / 2 + (x - x_center) * scale)
            cy = int(py0 + ph / 2 - (z - z_center) * scale)  # Z+ → up
            return cx, cy

        # Grid
        grid_step = _nice_grid_step(max(x_view, z_view))
        x_vmin = x_center - (pw / 2) / scale
        x_vmax = x_center + (pw / 2) / scale
        z_vmin = z_center - (ph / 2) / scale
        z_vmax = z_center + (ph / 2) / scale

        xg = np.floor(x_vmin / grid_step) * grid_step
        while xg <= x_vmax + grid_step:
            cx, _ = to_canvas(xg, z_center)
            if px0 <= cx <= px1:
                cv2.line(canvas, (cx, py0), (cx, py1), (50, 50, 50), 1)
                cv2.putText(canvas, f"{xg:.1f}", (cx - 14, py1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (110, 110, 110), 1)
            xg += grid_step

        zg = np.floor(z_vmin / grid_step) * grid_step
        while zg <= z_vmax + grid_step:
            _, cy = to_canvas(x_center, zg)
            if py0 <= cy <= py1:
                cv2.line(canvas, (px0, cy), (px1, cy), (50, 50, 50), 1)
                cv2.putText(canvas, f"{zg:.1f}", (2, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (110, 110, 110), 1)
            zg += grid_step

        cv2.rectangle(canvas, (px0, py0), (px1, py1), (80, 80, 80), 1)

        # Trajectory path
        if len(positions) >= 2:
            for i in range(1, len(positions)):
                cv2.line(canvas,
                         to_canvas(xs[i - 1], zs[i - 1]),
                         to_canvas(xs[i], zs[i]),
                         (30, 140, 255), 2)

        # Markers
        cv2.circle(canvas, to_canvas(xs[0], zs[0]), 5, (0, 200, 60), -1)
        cv2.circle(canvas, to_canvas(xs[0], zs[0]), 6, (255, 255, 255), 1)
        cv2.circle(canvas, to_canvas(xs[-1], zs[-1]), 5, (50, 50, 230), -1)
        cv2.circle(canvas, to_canvas(xs[-1], zs[-1]), 6, (255, 255, 255), 1)

        # Current camera heading arrow (EMA-smoothed camera Z-axis, X-Z plane)
        fwd_w = self._smoothed_heading
        arrow_len = grid_step * 0.6
        curr_pt = to_canvas(xs[-1], zs[-1])
        tip_pt  = to_canvas(xs[-1] + fwd_w[0] * arrow_len,
                            zs[-1] + fwd_w[2] * arrow_len)
        cv2.arrowedLine(canvas, curr_pt, tip_pt, (0, 220, 255), 2, tipLength=0.35)

        # Axis labels
        cv2.putText(canvas, "X [m]", (px1 - 36, H - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
        cv2.putText(canvas, "Z [m]", (2, py0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        self._draw_common_labels(canvas, trajectory, "2D  Top View (X-Z)",
                                 px0, py0, px1, py1, H, W, mg_top)
        return canvas

    # ------------------------------------------------------------------
    # 3-D orthographic view
    # ------------------------------------------------------------------

    def _draw_3d(self, trajectory) -> np.ndarray:
        """Render 3-D orthographic projection."""
        H = W = self.canvas_size
        mg_top, mg_bottom, mg_left, mg_right = 30, 65, 20, 20
        px0, py0 = mg_left, mg_top
        px1, py1 = W - mg_right, H - mg_bottom
        pw, ph = px1 - px0, py1 - py0

        canvas = np.full((H, W, 3), 25, dtype=np.uint8)

        positions = trajectory.get_positions_array()  # (N, 3)

        # Rotation matrix: world → screen
        #   Ry: orbit around world Y (azimuth)
        #   Rx: tilt up/down (elevation)
        az = np.radians(self.azimuth)
        el = np.radians(self.elevation)
        Ry = np.array([[ np.cos(az), 0, np.sin(az)],
                       [ 0,          1, 0          ],
                       [-np.sin(az), 0, np.cos(az)]])
        Rx = np.array([[1, 0,             0            ],
                       [0, np.cos(el), -np.sin(el)],
                       [0, np.sin(el),  np.cos(el)]])
        R = Rx @ Ry  # (3, 3)

        center = positions.mean(axis=0)
        rotated = (R @ (positions - center).T).T  # (N, 3)

        # Screen scale: fit the rotated extents with zoom applied
        rot_range = max(
            rotated[:, 0].max() - rotated[:, 0].min(),
            rotated[:, 1].max() - rotated[:, 1].min(),
            1.0,
        ) / self.zoom
        scale = min(pw, ph) / rot_range * 0.82  # px / m

        def to_canvas(rx: float, ry: float) -> tuple[int, int]:
            cx = int(px0 + pw / 2 + rx * scale)
            cy = int(py0 + ph / 2 - ry * scale)  # screen Y downward
            return cx, cy

        # Ground-plane grid (world Y = mean Y)
        xs_w, zs_w = positions[:, 0], positions[:, 2]
        x_range_w = max(xs_w.max() - xs_w.min(), 1.0)
        z_range_w = max(zs_w.max() - zs_w.min(), 1.0)
        grid_step = _nice_grid_step(max(x_range_w, z_range_w))
        y_grid = center[1]

        x_gmin = np.floor(xs_w.min() / grid_step) * grid_step - grid_step
        x_gmax = np.ceil(xs_w.max()  / grid_step) * grid_step + grid_step
        z_gmin = np.floor(zs_w.min() / grid_step) * grid_step - grid_step
        z_gmax = np.ceil(zs_w.max()  / grid_step) * grid_step + grid_step

        xg = x_gmin
        while xg <= x_gmax:
            p1r = R @ (np.array([xg, y_grid, z_gmin]) - center)
            p2r = R @ (np.array([xg, y_grid, z_gmax]) - center)
            cv2.line(canvas, to_canvas(p1r[0], p1r[1]),
                     to_canvas(p2r[0], p2r[1]), (50, 50, 50), 1)
            xg += grid_step

        zg = z_gmin
        while zg <= z_gmax:
            p1r = R @ (np.array([x_gmin, y_grid, zg]) - center)
            p2r = R @ (np.array([x_gmax, y_grid, zg]) - center)
            cv2.line(canvas, to_canvas(p1r[0], p1r[1]),
                     to_canvas(p2r[0], p2r[1]), (50, 50, 50), 1)
            zg += grid_step

        cv2.rectangle(canvas, (px0, py0), (px1, py1), (80, 80, 80), 1)

        # Coordinate axes at start position
        axis_len = grid_step * 0.8
        origin_w = positions[0] - center
        for vec, color, label in [
            (np.array([axis_len, 0, 0]), (60,  60, 220), "X"),  # BGR: red
            (np.array([0, axis_len, 0]), (60, 180,  60), "Y"),  # green
            (np.array([0, 0, axis_len]), (220, 80,  60), "Z"),  # blue
        ]:
            o_r = R @ origin_w
            t_r = R @ (origin_w + vec)
            oc = to_canvas(o_r[0], o_r[1])
            tc = to_canvas(t_r[0], t_r[1])
            cv2.line(canvas, oc, tc, color, 2)
            cv2.putText(canvas, label, (tc[0] + 3, tc[1] + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

        # Trajectory path
        if len(positions) >= 2:
            for i in range(1, len(positions)):
                p1r = R @ (positions[i - 1] - center)
                p2r = R @ (positions[i]     - center)
                cv2.line(canvas,
                         to_canvas(p1r[0], p1r[1]),
                         to_canvas(p2r[0], p2r[1]),
                         (30, 140, 255), 2)

        # Markers
        s_r = R @ (positions[0]  - center)
        e_r = R @ (positions[-1] - center)
        cv2.circle(canvas, to_canvas(s_r[0], s_r[1]), 5, (0, 200, 60), -1)
        cv2.circle(canvas, to_canvas(s_r[0], s_r[1]), 6, (255, 255, 255), 1)
        cv2.circle(canvas, to_canvas(e_r[0], e_r[1]), 5, (50, 50, 230), -1)
        cv2.circle(canvas, to_canvas(e_r[0], e_r[1]), 6, (255, 255, 255), 1)

        # Current camera heading arrow (EMA-smoothed camera Z-axis, projected)
        fwd_w = self._smoothed_heading
        arrow_len = grid_step * 0.6
        tip_w = positions[-1] + fwd_w * arrow_len
        tip_r = R @ (tip_w - center)
        cv2.arrowedLine(canvas,
                        to_canvas(e_r[0], e_r[1]),
                        to_canvas(tip_r[0], tip_r[1]),
                        (0, 220, 255), 2, tipLength=0.35)

        # View-angle info in bottom margin
        cv2.putText(canvas,
                    f"Az:{self.azimuth:.0f}  El:{self.elevation:.0f}",
                    (px0, py1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)

        self._draw_common_labels(canvas, trajectory, "3D  Orthographic",
                                 px0, py0, px1, py1, H, W, mg_top)
        return canvas

    # ------------------------------------------------------------------
    # Shared label helpers
    # ------------------------------------------------------------------

    def _draw_common_labels(
        self,
        canvas: np.ndarray,
        trajectory,
        mode_label: str,
        px0: int, py0: int, px1: int, py1: int,
        H: int, W: int, mg_top: int,
    ) -> None:
        """Draw title, position, pose count, and legend onto the canvas."""
        pos = trajectory.get_current_position()

        # Title + mode + toggle hint
        cv2.putText(canvas,
                    f"Trajectory  [{mode_label}]  (t: toggle 2D/3D)",
                    (px0, mg_top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        # Current position
        cv2.putText(canvas,
                    f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m",
                    (px0, H - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # Pose count
        cv2.putText(canvas, f"Poses: {len(trajectory)}",
                    (px1 - 75, H - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # Legend
        lx, ly = px0, H - 18
        cv2.circle(canvas, (lx + 4, ly), 4, (0, 200, 60), -1)
        cv2.putText(canvas, "Start", (lx + 12, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 150, 150), 1)
        cv2.circle(canvas, (lx + 60, ly), 4, (50, 50, 230), -1)
        cv2.putText(canvas, "Current", (lx + 68, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 150, 150), 1)
        cv2.arrowedLine(canvas, (lx + 130, ly), (lx + 145, ly),
                        (0, 220, 255), 2, tipLength=0.4)
        cv2.putText(canvas, "Heading", (lx + 150, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 150, 150), 1)


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
    max_reference_age: int = 30,
    skip_frames: int = 1,
    max_frames: int = None,
    verbose: bool = True,
    display: bool = False,
    show_matches: bool = False,
    plot_realtime: bool = False,
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
            However, slow continuous motion is accumulated by NOT updating the
            reference frame until motion crosses the threshold or max_reference_age.
        max_reference_age: Maximum number of frames the reference frame can age
            before forced update (default: 30). Prevents reference frame from
            becoming too stale during long periods of sub-threshold motion, while
            still allowing slow continuous motion to accumulate for detection.
        skip_frames: Process every N-th frame
        max_frames: Maximum number of frames to process
        verbose: Print progress information
        display: Display annotated current frame in real-time
        show_matches: Display a side-by-side keypoint match window (Reference |
            Current) with connecting lines colour-coded by inlier/outlier
            status. Useful for debugging matching quality. Requires either
            ``display`` or ``plot_realtime`` to also be True so that the
            cv2.waitKey loop runs.

    Returns:
        Trajectory object containing camera poses
    """
    trajectory = Trajectory()

    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    # Detect model type by number of outputs:
    #   3 outputs → Sinkhorn-only model  (keypoints1, keypoints2, matching_probs)
    #   4 outputs → Combined model       (keypoints1, keypoints2, matching_probs, E)
    has_essential_matrix = len(output_names) >= 4

    # Warm up camera (allow auto-exposure/auto-focus to stabilize)
    if reader.is_camera:
        for _ in range(10):
            ret, _ = reader.read()
            if not ret:
                # Early camera initialization failure detected during warm-up
                # Break immediately to allow subsequent error handling to catch it
                break

    # Read first frame
    ret, prev_frame = reader.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    prev_image = load_image_from_array(prev_frame, model_height, model_width)
    prev_display_frame = prev_frame.copy() if show_matches else None

    frame_count = 0
    processed_count = 0
    total_matches = 0
    total_inliers = 0
    frames_since_last_update = 0  # Track reference frame age

    traj_viewer = TrajectoryViewer() if plot_realtime else None
    paused = False

    if verbose:
        print(f"Processing frames (skip={skip_frames})...")
        if display or show_matches or plot_realtime:
            print("Press 'q' to quit, 's' to save, 't' to toggle 2D/3D, Space to pause/resume")

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

        keypoints1 = results[0]      # (1, K, 2)
        keypoints2 = results[1]      # (1, K, 2)
        matching_probs = results[2]  # (1, K+1, K+1)
        E_onnx = results[3] if has_essential_matrix else None  # (3, 3) or None

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
        rms_flow = 0.0
        mean_flow_mag = 0.0

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
                # Insufficient motion: skip pose estimation to avoid degenerate Essential Matrix.
                # DO NOT update reference frame yet - allow slow continuous motion to accumulate
                # across frames until it crosses the threshold. This ensures we don't miss gradual
                # movements (e.g., slow walking, camera drift).
                frames_since_last_update += 1
                status_message = (f"NO MOTION (rms={rms_flow:.2f}px, "
                                  f"mean={mean_flow_mag:.2f}px, "
                                  f"age={frames_since_last_update})")
                if verbose:
                    print(f"Frame {frame_count}: No motion "
                          f"(rms={rms_flow:.2f}px, mean={mean_flow_mag:.2f}px), skipping... "
                          f"(reference age: {frames_since_last_update} frames)")

                # Safety check: if reference frame becomes too old, force update to prevent
                # large jumps when motion eventually resumes (e.g., after long static period).
                if frames_since_last_update >= max_reference_age:
                    prev_image = curr_image
                    if show_matches:
                        prev_display_frame = curr_frame.copy()
                    frames_since_last_update = 0
                    if verbose:
                        print(f"  → Reference frame forced update (age limit reached)")
            else:
                # Estimate pose
                if has_essential_matrix:
                    # 4-output model: use the Essential Matrix from ONNX directly.
                    # The E matrix was computed inside the model using all keypoints
                    # weighted by Sinkhorn probabilities (weighted 8-point algorithm).
                    # We still call recoverPose with the extracted matches to resolve
                    # the sign ambiguity and obtain the inlier chirality mask.
                    R, t, inlier_mask = estimate_pose_from_essential_matrix(
                        matched_kpts1,
                        matched_kpts2,
                        E_onnx,
                        camera_intrinsics,
                    )
                else:
                    # 3-output model: estimate E via RANSAC then recover pose.
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
                    # Keep reference frame unchanged - may succeed on next frame with more motion
                    frames_since_last_update += 1
                else:
                    # Success: add pose to trajectory and update reference frame
                    trajectory.add_relative_pose(R, t)
                    pose_updated = True
                    prev_image = curr_image
                    if show_matches:
                        prev_display_frame = curr_frame.copy()
                    frames_since_last_update = 0  # Reset age counter

                    if verbose and processed_count % 10 == 0:
                        elapsed = time.time() - start_time
                        fps = processed_count / elapsed
                        if reader.is_camera or reader.total_frames == float('inf'):
                            print(f"Frame {frame_count}: "
                                  f"matches={num_matches}, inliers={num_inliers}, "
                                  f"rms={rms_flow:.1f}px, mean={mean_flow_mag:.2f}px, "
                                  f"position={trajectory.get_current_position()}, "
                                  f"fps={fps:.1f}")
                        else:
                            print(f"Frame {frame_count}/{reader.total_frames}: "
                                  f"matches={num_matches}, inliers={num_inliers}, "
                                  f"rms={rms_flow:.1f}px, mean={mean_flow_mag:.2f}px, "
                                  f"position={trajectory.get_current_position()}, "
                                  f"fps={fps:.1f}")

        # Display frame and/or trajectory plot in real-time
        if display:
            info_frame = draw_display_info(
                frame=curr_frame,
                trajectory=trajectory,
                frame_count=frame_count,
                num_matches=num_matches,
                num_inliers=num_inliers,
                matched_kpts2=matched_kpts2,
                inlier_mask=inlier_mask,
                pose_updated=pose_updated,
                status_message=status_message,
                model_width=model_width,
                model_height=model_height,
            )
            cv2.imshow('Visual Odometry', info_frame)

        if show_matches and prev_display_frame is not None:
            match_canvas = draw_match_frame(
                prev_frame=prev_display_frame,
                curr_frame=curr_frame,
                matched_kpts1=matched_kpts1,
                matched_kpts2=matched_kpts2,
                inlier_mask=inlier_mask,
                model_width=model_width,
                model_height=model_height,
                rms_flow=rms_flow,
                mean_flow_mag=mean_flow_mag,
            )
            cv2.imshow('Matches', match_canvas)

        if traj_viewer is not None:
            traj_viewer.render(trajectory)

        if display or show_matches or plot_realtime:
            # When paused, block until any key is pressed; otherwise poll briefly.
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                save_path = f"trajectory_{int(time.time())}.npz"
                trajectory.save_to_file(save_path)
                print(f"\nTrajectory saved to: {save_path}")
            elif key == ord('t') and traj_viewer is not None:
                traj_viewer.toggle_mode()
                if verbose:
                    print(f"Trajectory view: {traj_viewer.mode.upper()}")
            elif key == ord(' '):
                paused = not paused
                if verbose:
                    print(f"{'Paused' if paused else 'Resumed'} (Space to toggle)")

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
        help="Focal length in x direction (pixels). Required for OpenCV/video, auto-detected for RealSense/Orbbec/OAK."
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=None,
        help="Focal length in y direction (pixels). Required for OpenCV/video, auto-detected for RealSense/Orbbec/OAK."
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=None,
        help="Principal point x coordinate (pixels). Required for OpenCV/video, auto-detected for RealSense/Orbbec/OAK."
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=None,
        help="Principal point y coordinate (pixels). Required for OpenCV/video, auto-detected for RealSense/Orbbec/OAK."
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
        "--max-reference-age",
        type=int,
        default=30,
        help="Maximum number of frames the reference frame can age before forced update. "
             "Prevents stale references during long static periods while still allowing "
             "slow continuous motion to accumulate for detection (default: 30)"
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
        help="Display annotated camera frames with keypoints in real-time (press 'q' to quit)"
    )
    parser.add_argument(
        "--show-matches",
        action="store_true",
        help="Display a side-by-side keypoint match window (Reference | Current) with "
             "colour-coded connecting lines for debugging matching quality. "
             "Green=inlier, Red=outlier, Yellow=no pose estimate."
    )
    parser.add_argument(
        "--plot-realtime",
        action="store_true",
        help="Display a live top-down trajectory plot in a separate window while processing"
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

    has_essential_matrix = len(outputs) >= 4
    model_type = (
        "Sinkhorn + Essential Matrix (4-output)"
        if has_essential_matrix
        else "Sinkhorn (3-output)"
    )
    print(f"Model input size: {model_height}x{model_width}")
    print(f"Model type: {model_type}")
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
                # Always re-calculate to ensure consistency, even if scale factors are 1.0.
                scale_x = model_width / camera_intrinsics.width
                scale_y = model_height / camera_intrinsics.height
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
            max_reference_age=args.max_reference_age,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
            verbose=not args.quiet,
            display=args.display,
            show_matches=args.show_matches,
            plot_realtime=args.plot_realtime,
        )
    finally:
        reader.release()
        if args.display or args.show_matches or args.plot_realtime:
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
