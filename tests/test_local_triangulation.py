"""Tests for multi-view feature-track initialization."""

import numpy as np

from vo.local_tracks import FeatureTrack, triangulate_feature_track


def _project(point, pose, K):
    camera_point = pose[0] @ point + pose[1]
    pixel = K @ camera_point
    return (pixel[[1, 0]] / pixel[2])[None]


def test_triangulate_feature_track_recovers_point_and_multiview_metrics():
    K = np.array([[400.0, 0.0, 320.0],
                  [0.0, 400.0, 240.0],
                  [0.0, 0.0, 1.0]])
    poses = {
        0: (np.eye(3), np.zeros(3)),
        2: (np.eye(3), np.array([-0.5, 0.0, 0.0])),
        4: (np.eye(3), np.array([-1.0, 0.0, 0.0])),
    }
    point = np.array([0.2, -0.1, 4.0])
    keypoints = {frame: _project(point, pose, K)
                 for frame, pose in poses.items()}
    track = FeatureTrack(7, ((0, 0), (2, 0), (4, 0)))

    result = triangulate_feature_track(
        track, keypoints, poses, K, (640, 480))

    assert result is not None
    np.testing.assert_allclose(result.point, point, atol=1e-10)
    assert result.initial_pair == (0, 4)
    assert result.hard_valid
    assert result.positive_depth_fraction == 1.0
    assert result.reprojection_median_px < 1e-10
    assert result.reprojection_p90_px < 1e-10
    assert result.parallax_deg > 10.0
    assert 0.0 < result.condition_ratio < 1.0
    np.testing.assert_allclose(result.max_depth_baselines, 4.0)


def test_triangulate_feature_track_reports_bad_third_view_depth():
    K = np.eye(3)
    poses = {
        0: (np.eye(3), np.zeros(3)),
        1: (np.eye(3), np.array([-1.0, 0.0, 0.0])),
        2: (np.eye(3), np.array([0.0, 0.0, -5.0])),
    }
    point = np.array([0.2, 0.0, 4.0])
    keypoints = {frame: _project(point, pose, K)
                 for frame, pose in poses.items()}
    track = FeatureTrack(0, ((0, 0), (1, 0), (2, 0)))

    result = triangulate_feature_track(
        track, keypoints, poses, K, (10, 10))

    assert result is not None
    assert not result.hard_valid
    np.testing.assert_allclose(result.positive_depth_fraction, 2 / 3)


def test_triangulate_feature_track_prefers_parallax_over_temporal_span():
    K = np.array([[400.0, 0.0, 320.0],
                  [0.0, 400.0, 240.0],
                  [0.0, 0.0, 1.0]])
    poses = {
        0: (np.eye(3), np.zeros(3)),
        1: (np.eye(3), np.array([-1.0, 0.0, 0.0])),
        2: (np.eye(3), np.array([-0.01, 0.0, 0.0])),
    }
    point = np.array([0.2, 0.0, 5.0])
    keypoints = {frame: _project(point, pose, K)
                 for frame, pose in poses.items()}
    track = FeatureTrack(0, ((0, 0), (1, 0), (2, 0)))

    result = triangulate_feature_track(
        track, keypoints, poses, K, (640, 480))

    assert result is not None
    assert result.initial_pair != (0, 2)
    np.testing.assert_allclose(result.point, point, atol=1e-10)


def test_triangulate_feature_track_handles_rotated_camera_poses():
    K = np.array([[300.0, 0.0, 160.0],
                  [0.0, 300.0, 120.0],
                  [0.0, 0.0, 1.0]])
    angle = np.deg2rad(8.0)
    rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(angle), 0.0, np.cos(angle)]])
    poses = {
        10: (np.eye(3), np.zeros(3)),
        20: (rotation, np.array([-0.5, 0.0, 0.05])),
        30: (rotation.T, np.array([-1.0, 0.1, 0.1])),
    }
    point = np.array([0.1, -0.2, 4.0])
    keypoints = {frame: _project(point, pose, K)
                 for frame, pose in poses.items()}
    track = FeatureTrack(3, ((10, 0), (20, 0), (30, 0)))

    result = triangulate_feature_track(
        track, keypoints, poses, K, (320, 240))

    assert result is not None
    np.testing.assert_allclose(result.point, point, atol=1e-10)
    assert result.hard_valid
    assert result.reprojection_p90_px < 1e-10
