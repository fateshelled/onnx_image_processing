"""
Pose estimation utilities for Visual Odometry.

This module provides functions for estimating camera pose from matched
feature points, including Essential Matrix computation and pose recovery.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class CameraIntrinsics:
    """
    Camera intrinsic parameters.

    Args:
        fx: Focal length in x direction (pixels)
        fy: Focal length in y direction (pixels)
        cx: Principal point x coordinate (pixels)
        cy: Principal point y coordinate (pixels)
        width: Image width (pixels)
        height: Image height (pixels)
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def __repr__(self) -> str:
        return (f"CameraIntrinsics(fx={self.fx}, fy={self.fy}, "
                f"cx={self.cx}, cy={self.cy}, "
                f"width={self.width}, height={self.height})")


def estimate_pose_ransac(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    camera_intrinsics: CameraIntrinsics,
    ransac_threshold: float = 1.4,
    ransac_confidence: float = 0.999,
    method: int = cv2.RANSAC,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Estimate relative camera pose using RANSAC-based Essential Matrix estimation.

    Args:
        keypoints1: Keypoints in first image, shape (N, 2) in (y, x) or (x, y) format
        keypoints2: Keypoints in second image, shape (N, 2) in (y, x) or (x, y) format
        camera_intrinsics: Camera intrinsic parameters
        ransac_threshold: RANSAC reprojection threshold in pixels (default: 1.0)
        ransac_confidence: RANSAC confidence level (default: 0.999)
        method: RANSAC method (default: cv2.RANSAC)

    Returns:
        Tuple of:
            - R: Rotation matrix (3, 3) or None if estimation failed
            - t: Translation vector (3, 1) or None if estimation failed.
              Unit norm (monocular scale ambiguity) — trajectory distances
              are in normalised units, not physical meters.
            - inlier_mask: Boolean mask of points that passed both the
              Essential Matrix RANSAC test AND the chirality check in
              recoverPose. Note this is stricter than the RANSAC inlier
              set alone: points fitting E but failing chirality are
              excluded (they would be behind the camera). On the failure
              path (fewer than 5 RANSAC inliers, ``R=None``), the mask
              returned is the RANSAC inlier set BEFORE the chirality
              check — downstream code discards it together with ``R``,
              but callers relying on the failure-path mask should know.
    """
    if len(keypoints1) < 5 or len(keypoints2) < 5:
        return None, None, np.zeros(len(keypoints1), dtype=bool)

    # Convert keypoints to (x, y) format if needed
    # Assume input is (y, x) format from the model
    pts1 = keypoints1[:, [1, 0]].astype(np.float64)  # (y, x) -> (x, y)
    pts2 = keypoints2[:, [1, 0]].astype(np.float64)

    # Estimate Essential Matrix using RANSAC
    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        camera_intrinsics.K,
        method=method,
        prob=ransac_confidence,
        threshold=ransac_threshold,
    )

    if E is None or mask is None or E.size == 0 or E.shape != (3, 3):
        return None, None, np.zeros(len(keypoints1), dtype=bool)

    inlier_mask = mask.ravel().astype(bool)

    # Recover pose from Essential Matrix
    num_inliers, R, t, pose_mask = cv2.recoverPose(
        E,
        pts1,
        pts2,
        camera_intrinsics.K,
        mask=mask,
    )
    if num_inliers < 5:
        return None, None, inlier_mask

    # Update inlier mask with pose recovery results
    inlier_mask = (mask.ravel() != 0) & (pose_mask.ravel() > 0)

    return R, t, inlier_mask


def estimate_pose_rgbd_pnp(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    depth1: np.ndarray,
    camera_intrinsics: CameraIntrinsics,
    depth_scale: float = 5000.0,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    ransac_threshold: float = 2.0,
    ransac_confidence: float = 0.999,
    ransac_iterations: int = 1000,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Estimate metric relative pose from RGB matches and frame-1 depth.

    Matched pixels in the first frame are back-projected with ``depth1`` and
    registered against their second-frame pixels with PnP-RANSAC.  The return
    convention matches :func:`estimate_pose_ransac` and ``cv2.recoverPose``::

        x_current = R @ x_previous + t

    Unlike monocular Essential Matrix recovery, ``t`` is expressed in meters
    when ``depth1 / depth_scale`` is in meters.

    Args:
        keypoints1: First-frame keypoints, shape (N, 2), in (y, x) order.
        keypoints2: Second-frame keypoints, shape (N, 2), in (y, x) order.
        depth1: Registered first-frame depth image. Its dimensions must match
            the camera intrinsics/model image dimensions.
        camera_intrinsics: Camera calibration for both images.
        depth_scale: Raw depth units per meter (TUM RGB-D uses 5000).
        min_depth: Minimum accepted depth in meters.
        max_depth: Maximum accepted depth in meters.
        ransac_threshold: PnP reprojection threshold in pixels.
        ransac_confidence: PnP-RANSAC confidence.
        ransac_iterations: Maximum PnP-RANSAC iterations.

    Returns:
        ``(R, t, inlier_mask)``. The mask is indexed like the input matches
        and excludes both invalid-depth points and PnP outliers.
    """
    n = len(keypoints1)
    empty_mask = np.zeros(n, dtype=bool)
    if (
        n < 4
        or len(keypoints2) != n
        or depth1.ndim != 2
        or depth_scale <= 0
        or min_depth < 0
        or max_depth <= min_depth
    ):
        return None, None, empty_mask

    height, width = depth1.shape
    y = keypoints1[:, 0].astype(np.float64)
    x = keypoints1[:, 1].astype(np.float64)
    sample_y = np.rint(y).astype(np.int64)
    sample_x = np.rint(x).astype(np.int64)
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (sample_x >= 0)
        & (sample_x < width)
        & (sample_y >= 0)
        & (sample_y < height)
    )

    depth_m = np.full(n, np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(inside)
    depth_m[valid_indices] = (
        depth1[sample_y[valid_indices], sample_x[valid_indices]].astype(np.float64)
        / depth_scale
    )
    valid_depth = (
        inside
        & np.isfinite(depth_m)
        & (depth_m >= min_depth)
        & (depth_m <= max_depth)
    )
    valid_indices = np.flatnonzero(valid_depth)
    if len(valid_indices) < 4:
        return None, None, empty_mask

    z = depth_m[valid_indices]
    object_points = np.column_stack([
        (x[valid_indices] - camera_intrinsics.cx) * z / camera_intrinsics.fx,
        (y[valid_indices] - camera_intrinsics.cy) * z / camera_intrinsics.fy,
        z,
    ]).astype(np.float64)
    image_points = keypoints2[valid_indices][:, [1, 0]].astype(np.float64)

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_intrinsics.K,
            None,
            iterationsCount=ransac_iterations,
            reprojectionError=ransac_threshold,
            confidence=ransac_confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return None, None, empty_mask

    if not ok or inliers is None or len(inliers) < 4:
        return None, None, empty_mask

    local_inliers = inliers.ravel().astype(np.int64)
    # Refine only the consensus pose. Keep the RANSAC result if refinement is
    # unavailable or numerically fails on a degenerate point configuration.
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[local_inliers],
            image_points[local_inliers],
            camera_intrinsics.K,
            None,
            rvec,
            tvec,
        )
    except (cv2.error, AttributeError):
        pass

    R, _ = cv2.Rodrigues(rvec)
    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[valid_indices[local_inliers]] = True
    return R, tvec.reshape(3, 1), inlier_mask


def triangulate_points(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    camera_intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Triangulate 3D points from two views.

    Args:
        keypoints1: Keypoints in first image, shape (N, 2) in (y, x) format
        keypoints2: Keypoints in second image, shape (N, 2) in (y, x) format
        R1: Rotation matrix of first camera (3, 3)
        t1: Translation vector of first camera (3, 1)
        R2: Rotation matrix of second camera (3, 3)
        t2: Translation vector of second camera (3, 1)
        camera_intrinsics: Camera intrinsic parameters

    Returns:
        points_3d: Triangulated 3D points, shape (N, 3)
    """
    # Create projection matrices
    P1 = camera_intrinsics.K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = camera_intrinsics.K @ np.hstack([R2, t2.reshape(3, 1)])

    # Convert keypoints to (x, y) format
    pts1 = keypoints1[:, [1, 0]].astype(np.float64).T  # (2, N)
    pts2 = keypoints2[:, [1, 0]].astype(np.float64).T

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # Convert from homogeneous to 3D coordinates, avoiding division by zero
    # When points lie at infinity or triangulation degenerates (zero parallax),
    # the homogeneous coordinate w can be near-zero, causing NaN/Inf values.
    w = points_4d[3, :]
    mask = np.abs(w) > 1e-9  # Threshold for numerical stability
    points_3d = np.zeros((3, points_4d.shape[1]), dtype=np.float64)
    points_3d[:, mask] = points_4d[:3, mask] / w[mask]
    # Points with w ≈ 0 remain at origin (degenerate cases)

    return points_3d.T  # (N, 3)


def compose_transformation(
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compose two transformations.

    Args:
        R1: First rotation matrix (3, 3)
        t1: First translation vector (3, 1) or (3,)
        R2: Second rotation matrix (3, 3)
        t2: Second translation vector (3, 1) or (3,)

    Returns:
        Tuple of:
            - R: Composed rotation matrix (3, 3)
            - t: Composed translation vector (3, 1)
    """
    t1 = t1.reshape(3, 1) if t1.ndim == 1 else t1
    t2 = t2.reshape(3, 1) if t2.ndim == 1 else t2

    # Correct composition: T = T1 @ T2
    # In block matrix form: [R|t] = [R1|t1] @ [R2|t2]
    R = R1 @ R2
    t = R1 @ t2 + t1

    return R, t


def transformation_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert rotation and translation to 4x4 transformation matrix.

    Args:
        R: Rotation matrix (3, 3)
        t: Translation vector (3, 1) or (3,)

    Returns:
        T: Transformation matrix (4, 4)
    """
    t = t.reshape(3, 1) if t.ndim == 1 else t
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T


def matrix_to_transformation(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract rotation and translation from 4x4 transformation matrix.

    Args:
        T: Transformation matrix (4, 4)

    Returns:
        Tuple of:
            - R: Rotation matrix (3, 3)
            - t: Translation vector (3,)
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t
