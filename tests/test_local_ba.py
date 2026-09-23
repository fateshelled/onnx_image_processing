"""Synthetic tests for offline local bundle adjustment."""

import numpy as np
import pytest

from vo.local_ba import (BAObservation, _linearize,
                         _gauge_basis, _has_exact_sim3_gauge,
                         _observation_residual_jacobians, _schur_reduce,
                         linearize_reduced, optimize_local_ba)
from vo.se3 import se3_exp, skew


K = np.array([[400.0, 0.0, 320.0],
              [0.0, 400.0, 240.0],
              [0.0, 0.0, 1.0]])


def _pose(center, rotation=None):
    T = np.eye(4)
    T[:3, :3] = np.eye(3) if rotation is None else rotation
    T[:3, 3] = center
    return T


def _project(T, point):
    camera_point = T[:3, :3].T @ (point - T[:3, 3])
    pixel = K @ camera_point
    return tuple((pixel[[1, 0]] / pixel[2]).tolist())


def _scene():
    true_poses = {
        0: _pose(np.array([0.0, 0.0, 0.0])),
        1: _pose(np.array([1.0, 0.0, 0.0])),
        2: _pose(np.array([2.0, 0.1, 0.0])),
    }
    landmarks = {
        index: np.array([x, y, z])
        for index, (x, y, z) in enumerate([
            (-0.8, -0.4, 4.0), (-0.2, 0.3, 5.0), (0.5, -0.2, 4.5),
            (1.0, 0.5, 6.0), (1.7, -0.3, 5.5), (2.2, 0.2, 4.2),
        ])
    }
    observations = [
        BAObservation(frame_id, track_id, _project(pose, point))
        for frame_id, pose in true_poses.items()
        for track_id, point in landmarks.items()
    ]
    return true_poses, landmarks, observations


def test_local_ba_reduces_reprojection_error_and_recovers_pose():
    true_poses, true_landmarks, observations = _scene()
    initial_poses = {key: value.copy() for key, value in true_poses.items()}
    initial_poses[1] = initial_poses[1] @ se3_exp(
        [0.0, 0.025, 0.0, 0.0, 0.0, 0.0])
    initial_poses[2] = initial_poses[2] @ se3_exp(
        [0.015, -0.02, 0.01, 0.12, -0.08, 0.04])
    initial_landmarks = {
        key: point + np.array([0.03, -0.02, 0.05])
        for key, point in true_landmarks.items()
    }

    result = optimize_local_ba(
        initial_poses, initial_landmarks, observations, K,
        max_iterations=30)

    assert result.ok, result.reason
    assert result.final_cost < result.initial_cost * 1e-4
    assert np.linalg.norm(result.poses[2][:3, 3]
                          - true_poses[2][:3, 3]) < 2e-2
    assert result.reduced_hessian.shape == (18, 18)
    np.testing.assert_allclose(result.reduced_hessian,
                               result.reduced_hessian.T, atol=1e-8)


def test_huber_limits_one_gross_outlier():
    true_poses, landmarks, observations = _scene()
    corrupted = list(observations)
    item = corrupted[-1]
    corrupted[-1] = BAObservation(
        item.frame_id, item.track_id,
        (item.pixel_yx[0] + 250.0, item.pixel_yx[1] - 200.0))
    initial_poses = {key: value.copy() for key, value in true_poses.items()}
    initial_poses[2] = initial_poses[2] @ se3_exp(
        [0.01, -0.015, 0.0, 0.08, -0.03, 0.02])

    result = optimize_local_ba(
        initial_poses, landmarks, corrupted, K,
        huber_delta=2.0, max_iterations=30)

    assert result.ok, result.reason
    assert np.linalg.norm(result.poses[2][:3, 3]
                          - true_poses[2][:3, 3]) < 0.15


def test_reduced_normal_matches_explicit_landmark_schur():
    poses, landmarks, observations = _scene()
    pose_ids, reduced, gradient, _cost, valid = linearize_reduced(
        poses, landmarks, observations, K, huber_delta=1e6)

    normalized = tuple((item.frame_id, item.track_id,
                        np.asarray(item.pixel_yx)) for item in observations)
    landmark_ids = tuple(sorted(landmarks))
    Hpp, Hpl, Hll, bp, bl, *_ = _linearize(
        poses, landmarks, normalized, K, 1e6, pose_ids, landmark_ids)
    expected_hessian = Hpp.copy()
    expected_gradient = bp.copy()
    for index, block in enumerate(Hll):
        inverse = np.linalg.pinv(block, rcond=1e-12)
        cross = Hpl[:, 3 * index:3 * index + 3]
        expected_hessian -= cross @ inverse @ cross.T
        expected_gradient -= cross @ inverse @ bl[index]

    assert pose_ids == (0, 1, 2)
    assert valid == len(observations)
    np.testing.assert_allclose(reduced, expected_hessian, atol=1e-8)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-8)
    assert np.linalg.norm(gradient) < 1e-8
    eigenvalues = np.linalg.eigvalsh(reduced)
    assert np.sum(np.abs(eigenvalues) < 1e-9 * eigenvalues[-1]) == 7

    nullspace = []
    for axis in np.eye(3):
        translation = np.concatenate([
            np.r_[np.zeros(3), poses[frame][:3, :3].T @ axis]
            for frame in pose_ids
        ])
        nullspace.append(translation)
    for axis in np.eye(3):
        rotation = np.concatenate([
            np.r_[poses[frame][:3, :3].T @ axis,
                  poses[frame][:3, :3].T
                  @ (skew(axis) @ poses[frame][:3, 3])]
            for frame in pose_ids
        ])
        nullspace.append(rotation)
    nullspace.append(np.concatenate([
        np.r_[np.zeros(3), poses[frame][:3, :3].T @ poses[frame][:3, 3]]
        for frame in pose_ids
    ]))
    N = np.column_stack(nullspace)
    assert np.linalg.norm(reduced @ N) < 1e-7 * np.linalg.norm(reduced)


def test_schur_pose_and_landmark_steps_match_full_normal_solve():
    poses, landmarks, observations = _scene()
    poses[2] = poses[2] @ se3_exp([0.01, -0.02, 0.0, 0.05, 0.0, 0.02])
    pose_ids = tuple(sorted(poses))
    landmark_ids = tuple(sorted(landmarks))
    normalized = tuple((item.frame_id, item.track_id,
                        np.asarray(item.pixel_yx)) for item in observations)
    Hpp, Hpl, Hll, bp, bl, *_ = _linearize(
        poses, landmarks, normalized, K, 1e6, pose_ids, landmark_ids)
    landmark_damping = 1e-3
    pose_damping = 2e-3
    reduced, gradient, inverses = _schur_reduce(
        Hpp, Hpl, Hll, bp, bl, damping=landmark_damping)
    free = np.arange(6, 18)
    schur_pose = np.linalg.solve(
        reduced[np.ix_(free, free)] + pose_damping * np.eye(len(free)),
        -gradient[free])
    full_pose_step = np.zeros(18)
    full_pose_step[free] = schur_pose
    schur_landmark = np.concatenate([
        -inverses[index] @ (
            bl[index] + Hpl[:, 3 * index:3 * index + 3].T @ full_pose_step)
        for index in range(len(landmark_ids))
    ])

    Hlandmark = np.zeros((18, 18))
    for index, block in enumerate(Hll):
        section = slice(3 * index, 3 * index + 3)
        Hlandmark[section, section] = block + landmark_damping * np.eye(3)
    full_hessian = np.block([
        [Hpp[np.ix_(free, free)] + pose_damping * np.eye(len(free)),
         Hpl[free]],
        [Hpl[free].T, Hlandmark],
    ])
    full_gradient = np.r_[bp[free], bl.ravel()]
    full_step = np.linalg.solve(full_hessian, -full_gradient)
    np.testing.assert_allclose(schur_pose, full_step[:len(free)], atol=1e-8)
    np.testing.assert_allclose(schur_landmark, full_step[len(free):], atol=1e-8)


def test_reprojection_jacobians_match_finite_differences():
    poses, landmarks, observations = _scene()
    item = observations[7]
    pose = poses[item.frame_id]
    point = landmarks[item.track_id]
    observed = np.asarray(item.pixel_yx)
    residual, J_pose, J_landmark = _observation_residual_jacobians(
        pose, point, observed, K)
    epsilon = 1e-7
    numeric_pose = np.zeros((2, 6))
    numeric_landmark = np.zeros((2, 3))
    for column in range(6):
        step = np.zeros(6)
        step[column] = epsilon
        plus = _observation_residual_jacobians(
            pose @ se3_exp(step), point, observed, K)[0]
        minus = _observation_residual_jacobians(
            pose @ se3_exp(-step), point, observed, K)[0]
        numeric_pose[:, column] = (plus - minus) / (2 * epsilon)
    for column in range(3):
        step = np.zeros(3)
        step[column] = epsilon
        plus = _observation_residual_jacobians(
            pose, point + step, observed, K)[0]
        minus = _observation_residual_jacobians(
            pose, point - step, observed, K)[0]
        numeric_landmark[:, column] = (plus - minus) / (2 * epsilon)
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(J_pose, numeric_pose, atol=1e-5)
    np.testing.assert_allclose(J_landmark, numeric_landmark, atol=1e-5)


def test_already_converged_scene_returns_valid_factor():
    poses, landmarks, observations = _scene()
    result = optimize_local_ba(poses, landmarks, observations, K)
    assert result.ok, result.reason
    assert result.iterations == 0
    np.testing.assert_allclose(result.final_cost, 0.0, atol=1e-15)


def test_gauge_rank_check_is_invariant_to_world_scale():
    poses, landmarks, observations = _scene()
    for scale in (1e-9, 1e-3, 1.0, 1e3, 1e9):
        scaled_poses = {key: value.copy() for key, value in poses.items()}
        for pose in scaled_poses.values():
            pose[:3, 3] *= scale
        scaled_landmarks = {key: value * scale
                            for key, value in landmarks.items()}
        pose_ids, H, _b, _cost, _valid = linearize_reduced(
            scaled_poses, scaled_landmarks, observations, K,
            huber_delta=1e6)
        assert _has_exact_sim3_gauge(H, scaled_poses, pose_ids)
        assert np.linalg.matrix_rank(_gauge_basis(scaled_poses, pose_ids)) == 7


def test_local_ba_rejects_duplicate_disconnected_and_deadline():
    poses, landmarks, observations = _scene()
    with np.testing.assert_raises_regex(ValueError, "duplicate"):
        optimize_local_ba(
            poses, landmarks, observations + [observations[0]], K)

    disconnected_poses = {
        0: poses[0], 1: poses[1], 2: poses[2],
        3: _pose(np.array([3.0, 0.0, 0.0])),
    }
    disconnected = [
        BAObservation(frame_id, track_id,
                      _project(disconnected_poses[frame_id], landmarks[track_id]))
        for frame_id, track_ids in ((0, range(3)), (1, range(3)),
                                    (2, range(3, 6)), (3, range(3, 6)))
        for track_id in track_ids
    ]
    with np.testing.assert_raises_regex(ValueError, "connected"):
        optimize_local_ba(disconnected_poses, landmarks, disconnected, K)

    result = optimize_local_ba(
        poses, landmarks, observations, K, deadline=0.5, clock=lambda: 1.0)
    assert not result.ok
    assert result.reason == "deadline exceeded"


def test_linearization_cancellation_discards_partial_system_and_rolls_back():
    poses, landmarks, observations = _scene()
    normalized = tuple((item.frame_id, item.track_id,
                        np.asarray(item.pixel_yx)) for item in observations)
    calls = {"count": 0}

    def cancel_during_linearization():
        calls["count"] += 1
        return calls["count"] >= 2

    blocks = _linearize(
        poses, landmarks, normalized * 8, K, 3.0,
        tuple(sorted(poses)), tuple(sorted(landmarks)),
        cancelled=cancel_during_linearization)
    assert blocks is None

    ticks = {"calls": 0}

    def advancing_clock():
        ticks["calls"] += 1
        # The first candidate is accepted after call 12; expire when the next
        # iteration starts so rollback is exercised from a changed state.
        return 0.0 if ticks["calls"] <= 12 else 1.0

    initial_poses = {key: value.copy() for key, value in poses.items()}
    initial_poses[2] = initial_poses[2] @ se3_exp(
        [0.01, -0.02, 0.0, 0.08, 0.0, 0.02])
    result = optimize_local_ba(
        initial_poses, landmarks, observations, K,
        deadline=0.5, clock=advancing_clock, max_iterations=30)
    assert not result.ok
    assert result.reason == "deadline exceeded"
    assert ticks["calls"] >= 13
    for frame_id in initial_poses:
        np.testing.assert_allclose(result.poses[frame_id], initial_poses[frame_id])
    for track_id in landmarks:
        np.testing.assert_allclose(result.landmarks[track_id], landmarks[track_id])


def test_local_ba_rejects_invalid_models_and_rank_deficiency():
    poses, landmarks, observations = _scene()
    dead_poses = {**poses, 3: _pose(np.array([3.0, 0.0, 0.0]))}
    with pytest.raises(ValueError, match="each pose"):
        optimize_local_ba(dead_poses, landmarks, observations, K)

    skew_camera = K.copy()
    skew_camera[0, 1] = 1.0
    with pytest.raises(ValueError, match="standard"):
        optimize_local_ba(poses, landmarks, observations, skew_camera)

    reflected = {key: value.copy() for key, value in poses.items()}
    reflected[1][:3, :3] = np.diag([-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="invalid camera-to-world"):
        optimize_local_ba(reflected, landmarks, observations, K)

    degenerate_landmarks = {key: np.array([0.0, 0.0, 4.0])
                            for key in landmarks}
    degenerate_observations = [
        BAObservation(frame_id, track_id,
                      _project(pose, degenerate_landmarks[track_id]))
        for frame_id, pose in poses.items()
        for track_id in degenerate_landmarks
    ]
    result = optimize_local_ba(
        poses, degenerate_landmarks, degenerate_observations, K)
    assert not result.ok
    assert "7 gauge modes" in result.reason


def test_local_ba_safely_falls_back_on_degenerate_scale_anchor():
    poses, landmarks, observations = _scene()
    poses[1] = poses[0].copy()

    result = optimize_local_ba(poses, landmarks, observations, K)

    assert not result.ok
    assert result.reason == "scale anchor baseline is degenerate"
    assert result.iterations == 0
    np.testing.assert_allclose(result.poses[1], poses[1])
