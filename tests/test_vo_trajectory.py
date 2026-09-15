#!/usr/bin/env python3
"""
Tests for Visual Odometry trajectory accumulation.

Verifies that ``Trajectory.add_relative_pose`` correctly composes relative
poses (as returned by ``cv2.recoverPose``) into a camera-to-world trajectory,
using synthetic scenes with known ground-truth motion.
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_model.vo import Trajectory, CameraIntrinsics


def _rotation_y(angle_rad: float) -> np.ndarray:
    """Rotation around the Y axis (camera yaw)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


def _rotation_x(angle_rad: float) -> np.ndarray:
    """Rotation around the X axis (camera pitch)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def _make_scene(rng: np.random.Generator, n_points: int = 400) -> np.ndarray:
    """Random 3D scene points in front of the camera."""
    X = rng.uniform(-4, 4, (n_points, 3))
    X[:, 2] += rng.uniform(4, 10, n_points)
    return X


def _project(x_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    p = (K @ x_cam.T).T
    return p[:, :2] / p[:, 2:3]


def _relative_poses_from_scene(
    X: np.ndarray,
    centers: list,
    rotations: list,
    K: np.ndarray,
) -> list:
    """
    Render a synthetic scene at consecutive camera poses and estimate the
    relative pose (R, t) for each consecutive pair, as ``cv2.recoverPose``
    would return it.

    Returns a list of (R, t) tuples, one per frame transition.
    """
    rel_poses = []
    for i in range(len(centers) - 1):
        # x_cam = R_wc^T @ (X - C)
        x1 = (X - centers[i]) @ rotations[i]
        x2 = (X - centers[i + 1]) @ rotations[i + 1]
        pts1 = _project(x1, K).astype(np.float64)
        pts2 = _project(x2, K).astype(np.float64)

        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        assert E is not None, f"findEssentialMat failed at pair {i}"
        num, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        rel_poses.append((R, t.ravel()))
    return rel_poses


def _align_by_scale(P: np.ndarray, G: np.ndarray) -> tuple:
    """
    Align estimated positions P to ground truth G by a single global scale
    (VO scale ambiguity). Returns (scale, error_after_alignment).
    """
    s = np.sum(P * G) / np.sum(P * P)
    return s, float(np.linalg.norm(s * P - G))


class TestTrajectoryAccumulation:
    """Ground-truth comparison for trajectory accumulation with rotation."""

    def setup_method(self):
        self.K = np.array([
            [525, 0, 320],
            [0, 525, 240],
            [0, 0, 1],
        ], dtype=np.float64)
        self.intrinsics = CameraIntrinsics(fx=525, fy=525, cx=320, cy=240,
                                           width=640, height=480)

    def _run_accumulation(self, centers, rotations, rel_poses):
        trajectory = Trajectory()
        for R, t in rel_poses:
            trajectory.add_relative_pose(R, t)
        return trajectory.get_positions_array()

    def test_translation_only_forward(self):
        """Pure forward motion should accumulate correctly."""
        rng = np.random.default_rng(0)
        X = _make_scene(rng)
        centers = [np.zeros(3), np.array([0, 0, 1.0]), np.array([0, 0, 2.0])]
        rotations = [np.eye(3), np.eye(3), np.eye(3)]

        rel_poses = _relative_poses_from_scene(X, centers, rotations, self.K)
        positions = self._run_accumulation(centers, rotations, rel_poses)

        # No rotation: estimated positions must match GT up to scale
        gt = np.array(centers)
        scale, err = _align_by_scale(positions, gt)
        assert scale == pytest.approx(1.0, abs=0.05)
        assert err < 0.05

    def test_yaw_rotation_accumulation(self):
        """Yaw rotation per step: the known failure mode of the old code."""
        rng = np.random.default_rng(1)
        X = _make_scene(rng)
        yaw = np.deg2rad(12)
        dR = _rotation_y(yaw)

        centers = [np.zeros(3)]
        rotations = [np.eye(3)]
        for _ in range(4):
            step = rotations[-1] @ np.array([0.1, 0.05, 1.0])
            centers.append(centers[-1] + step)
            rotations.append(rotations[-1] @ dR)

        rel_poses = _relative_poses_from_scene(X, centers, rotations, self.K)
        positions = self._run_accumulation(centers, rotations, rel_poses)

        gt = np.array(centers)
        scale, err = _align_by_scale(positions, gt)
        assert scale == pytest.approx(1.0, abs=0.05)
        assert err < 0.1

    def test_pitch_rotation_accumulation(self):
        """Pitch rotation per step (non-Y-axis rotation)."""
        rng = np.random.default_rng(2)
        X = _make_scene(rng)
        dR = _rotation_x(np.deg2rad(8))

        centers = [np.zeros(3)]
        rotations = [np.eye(3)]
        for _ in range(3):
            step = rotations[-1] @ np.array([0.0, 0.1, 1.0])
            centers.append(centers[-1] + step)
            rotations.append(rotations[-1] @ dR)

        rel_poses = _relative_poses_from_scene(X, centers, rotations, self.K)
        positions = self._run_accumulation(centers, rotations, rel_poses)

        gt = np.array(centers)
        scale, err = _align_by_scale(positions, gt)
        assert scale == pytest.approx(1.0, abs=0.05)
        assert err < 0.1

    def test_stored_poses_are_camera_to_world(self):
        """Stored pose matrices must map camera coordinates to world."""
        rng = np.random.default_rng(3)
        X = _make_scene(rng)
        centers = [np.zeros(3), np.array([0.1, 0.05, 1.0])]
        dR = _rotation_y(np.deg2rad(15))
        rotations = [np.eye(3), np.eye(3) @ dR]

        rel_poses = _relative_poses_from_scene(X, centers, rotations, self.K)
        trajectory = Trajectory()
        for R, t in rel_poses:
            trajectory.add_relative_pose(R, t)

        pose = trajectory.get_current_pose()
        R_wc = pose[:3, :3]
        C = pose[:3, 3]

        # Rotation must be orthonormal
        assert np.allclose(R_wc @ R_wc.T, np.eye(3), atol=1e-9)

        # Camera center must match GT (up to scale)
        gt_C = centers[1]
        s = np.sum(C * gt_C) / np.sum(gt_C * gt_C)
        assert np.linalg.norm(s * C - gt_C) < 0.05

    def test_rotation_matrix_orientation(self):
        """R_wc must equal the composition of GT world rotations (up to scale-free estimation)."""
        rng = np.random.default_rng(4)
        X = _make_scene(rng)
        yaw = np.deg2rad(10)
        dR = _rotation_y(yaw)

        centers = [np.zeros(3)]
        rotations = [np.eye(3)]
        for _ in range(2):
            step = rotations[-1] @ np.array([0.05, 0.0, 1.0])
            centers.append(centers[-1] + step)
            rotations.append(rotations[-1] @ dR)

        rel_poses = _relative_poses_from_scene(X, centers, rotations, self.K)
        trajectory = Trajectory()
        for R, t in rel_poses:
            trajectory.add_relative_pose(R, t)

        # Accumulated world rotation: R_wc_new = R_wc_prev @ R_rel.T
        R_acc = np.eye(3)
        for R, t in rel_poses:
            R_acc = R_acc @ R.T

        pose = trajectory.get_current_pose()
        assert np.allclose(pose[:3, :3], R_acc, atol=1e-9)

        # And it must match the GT world rotation of the last camera
        assert np.allclose(pose[:3, :3], rotations[-1], atol=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
