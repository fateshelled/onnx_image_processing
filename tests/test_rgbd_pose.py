"""Synthetic tests for metric RGB-D PnP relative-pose estimation."""

import cv2
import numpy as np

from pytorch_model.vo import CameraIntrinsics, estimate_pose_rgbd_pnp


def _make_correspondences():
    cam = CameraIntrinsics(
        fx=525.0,
        fy=525.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    pixels1 = np.array(
        [
            [120, 100], [200, 100], [280, 100], [360, 100], [440, 100],
            [140, 180], [220, 180], [300, 180], [380, 180], [460, 180],
            [120, 260], [200, 260], [280, 260], [360, 260], [440, 260],
            [140, 340], [220, 340], [300, 340], [380, 340], [460, 340],
        ],
        dtype=np.float64,
    )
    depths = np.linspace(1.2, 4.8, len(pixels1))
    points3d = np.column_stack(
        [
            (pixels1[:, 0] - cam.cx) * depths / cam.fx,
            (pixels1[:, 1] - cam.cy) * depths / cam.fy,
            depths,
        ]
    )
    expected_rvec = np.array([0.015, -0.025, 0.01], dtype=np.float64)
    expected_R, _ = cv2.Rodrigues(expected_rvec)
    expected_t = np.array([0.08, -0.025, 0.04], dtype=np.float64)
    points2 = (expected_R @ points3d.T).T + expected_t
    projected2 = (cam.K @ points2.T).T
    pixels2 = projected2[:, :2] / projected2[:, 2:3]

    depth = np.zeros((cam.height, cam.width), dtype=np.uint16)
    for (x, y), z in zip(pixels1.astype(int), depths):
        depth[y, x] = int(round(z * 5000.0))

    # Public VO APIs use model keypoints in (y, x) order.
    keypoints1 = pixels1[:, [1, 0]]
    keypoints2 = pixels2[:, [1, 0]]
    return cam, depth, keypoints1, keypoints2, expected_R, expected_t


def test_rgbd_pnp_recovers_metric_pose():
    cam, depth, keypoints1, keypoints2, expected_R, expected_t = (
        _make_correspondences()
    )

    R, t, mask = estimate_pose_rgbd_pnp(
        keypoints1,
        keypoints2,
        depth,
        cam,
        ransac_threshold=0.5,
    )

    assert R is not None
    assert mask.sum() == len(keypoints1)
    assert np.allclose(R, expected_R, atol=2e-4)
    assert np.allclose(t.ravel(), expected_t, atol=2e-4)


def test_rgbd_pnp_excludes_invalid_depth_from_inlier_mask():
    cam, depth, keypoints1, keypoints2, expected_R, expected_t = (
        _make_correspondences()
    )
    invalid = np.array([1, 7, 13])
    for idx in invalid:
        y, x = np.rint(keypoints1[idx]).astype(int)
        depth[y, x] = 0

    R, t, mask = estimate_pose_rgbd_pnp(
        keypoints1,
        keypoints2,
        depth,
        cam,
        ransac_threshold=0.5,
    )

    assert R is not None
    assert not mask[invalid].any()
    assert mask.sum() == len(keypoints1) - len(invalid)
    assert np.allclose(R, expected_R, atol=2e-4)
    assert np.allclose(t.ravel(), expected_t, atol=2e-4)


def test_rgbd_pnp_fails_cleanly_with_too_few_valid_depths():
    cam, depth, keypoints1, keypoints2, _, _ = _make_correspondences()
    depth[:] = 0
    for idx in range(3):
        y, x = np.rint(keypoints1[idx]).astype(int)
        depth[y, x] = 5000

    R, t, mask = estimate_pose_rgbd_pnp(keypoints1, keypoints2, depth, cam)

    assert R is None
    assert t is None
    assert not mask.any()
