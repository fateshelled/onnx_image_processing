import cv2
import numpy as np
import pytest

from vo.sim3_verification import sim3_ransac, triangulate_local


def _rotation(axis_angle):
    return cv2.Rodrigues(np.asarray(axis_angle, dtype=float))[0]


def test_sim3_ransac_recovers_scale_rotation_translation_with_outliers():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(80, 3))
    scale = 2.4
    rotation = _rotation([0.1, -0.2, 0.05])
    translation = np.array([0.3, -1.2, 0.8])
    target = scale * (source @ rotation.T) + translation
    target += rng.normal(scale=0.002, size=target.shape)
    target[:20] = rng.uniform(-8.0, 8.0, size=(20, 3))

    result = sim3_ransac(source, target, seed=9, residual_fraction=0.02,
                         min_inliers=30)

    assert result.ok
    assert result.scale == pytest.approx(scale, rel=2e-3)
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(result.rotation @ rotation.T)[0])) < 0.2
    assert np.linalg.norm(result.translation - translation) < 0.01
    assert result.inliers.sum() >= 58


def test_sim3_ransac_is_deterministic():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(30, 3))
    target = 1.7 * source + np.array([1.0, 2.0, 3.0])
    a = sim3_ransac(source, target, seed=12, min_inliers=12)
    b = sim3_ransac(source, target, seed=12, min_inliers=12)
    assert a.ok and b.ok
    assert a.scale == b.scale
    np.testing.assert_array_equal(a.inliers, b.inliers)


def test_sim3_inlier_mask_preserves_input_indexing_with_nonfinite_rows():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(30, 3))
    target = 1.4 * source + np.array([0.2, -0.1, 0.8])
    source[5] = np.nan
    target[17] = np.inf
    result = sim3_ransac(source, target, seed=3, min_inliers=12)
    assert result.ok
    assert result.inliers.shape == (30,)
    assert not result.inliers[5]
    assert not result.inliers[17]
    assert result.inliers.sum() == 28


def test_sim3_ransac_rejects_collinear_cloud():
    x = np.linspace(-2.0, 2.0, 20)
    source = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    result = sim3_ransac(source, 2.0 * source, min_inliers=12)
    assert not result.ok
    assert result.reason == "degenerate_source"


def test_sim3_rejects_reflection_and_out_of_range_scale():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(30, 3))
    reflected = source.copy()
    reflected[:, 0] *= -1.0
    assert not sim3_ransac(source, reflected, min_inliers=12).ok
    huge = sim3_ransac(source, 50.0 * source, min_inliers=12)
    assert not huge.ok
    assert huge.reason == "no_consensus"


def test_independent_baseline_gauges_make_relative_scale_observable():
    rng = np.random.default_rng(8)
    world = np.column_stack([
        rng.uniform(-1.0, 1.0, 40),
        rng.uniform(-0.7, 0.7, 40),
        rng.uniform(3.0, 7.0, 40),
    ])
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

    def project(X, t):
        Y = X + np.asarray(t)
        uv = (K @ Y.T).T
        xy = uv[:, :2] / uv[:, 2:]
        return xy[:, ::-1]

    p0 = project(world, [0, 0, 0])
    p1 = project(world, [-1.0, 0, 0])
    qa0 = project(world, [0, 0, 0])
    qa1 = project(world, [-0.4, 0, 0])
    A = triangulate_local(p0, p1, np.eye(3), [-1.0, 0, 0], K)
    B = triangulate_local(qa0, qa1, np.eye(3), [-1.0, 0, 0], K)
    valid = A.valid & B.valid
    result = sim3_ransac(A.points[valid], B.points[valid], min_inliers=12,
                         residual_fraction=0.02)
    assert result.ok
    # B used a unit reconstruction baseline for a physical 0.4 baseline,
    # so its reconstructed cloud is 1/0.4 times larger than A.
    assert result.scale == pytest.approx(2.5, rel=1e-3)


def test_triangulation_rejects_zero_parallax():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    points = np.array([[200.0, 300.0], [220.0, 350.0], [250.0, 400.0]])
    result = triangulate_local(points, points, np.eye(3), [1.0, 0, 0], K)
    assert not result.valid.any()


def test_triangulate_local_roundtrips_a_known_pose():
    rng = np.random.default_rng(11)
    K = np.array([[520.0, 0.0, 318.0], [0.0, 519.0, 241.0], [0.0, 0.0, 1.0]])
    X = np.column_stack([rng.uniform(-1.0, 1.0, 60),
                         rng.uniform(-0.8, 0.8, 60),
                         rng.uniform(2.0, 6.0, 60)])
    R = _rotation([0.05, -0.12, 0.03])
    t = np.array([-0.5, 0.1, 0.05])

    def project(points, rotation, translation):
        Y = (rotation @ points.T).T + translation
        uv = (K @ Y.T).T
        xy = uv[:, :2] / uv[:, 2:]
        return xy[:, ::-1]

    p0 = project(X, np.eye(3), np.zeros(3))
    p1 = project(X, R, t)
    result = triangulate_local(p0, p1, R, t, K, min_parallax_deg=0.5)
    assert result.valid.sum() > 30
    np.testing.assert_allclose(result.points[result.valid], X[result.valid],
                               atol=1e-6)
    assert np.all(result.parallax_deg[result.valid] >= 0.5)


def test_triangulation_rejects_points_behind_second_camera():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    X = np.array([[0.1, 0.05, 1.0], [0.2, 0.1, 2.0], [0.1, -0.2, 3.0]])
    uv = (K @ X.T).T
    p0 = (uv[:, :2] / uv[:, 2:])[:, ::-1]
    X1 = X + np.array([0.0, 0.0, -5.0])
    uv1 = (K @ X1.T).T
    p1 = (uv1[:, :2] / uv1[:, 2:])[:, ::-1]
    # Nonzero parallax (the gate is disabled) so invalidity must come from the
    # negative depth in the second camera, not from the parallax filter.
    result = triangulate_local(p0, p1, np.eye(3), [0.0, 0.0, -5.0], K,
                               min_parallax_deg=0.0)
    assert np.all(result.parallax_deg > 0.0)
    assert not result.valid.any()


def test_sim3_residual_threshold_is_scale_invariant():
    rng = np.random.default_rng(13)
    source = rng.normal(size=(60, 3))
    R = _rotation([0.2, 0.1, -0.15])
    target = 1.5 * (source @ R.T) + np.array([0.4, -0.3, 0.9])
    target = target + rng.normal(scale=0.004, size=target.shape)
    target[:12] = rng.uniform(-6.0, 6.0, size=(12, 3))
    small = sim3_ransac(source, target, seed=5, residual_fraction=0.02,
                        min_inliers=20)
    large = sim3_ransac(source, 100.0 * target, seed=5,
                        residual_fraction=0.02, min_inliers=20,
                        max_scale=1000.0)
    assert small.ok and large.ok
    assert large.scale == pytest.approx(100.0 * small.scale, rel=1e-9)
    assert large.median_residual == pytest.approx(
        100.0 * small.median_residual, rel=1e-9)
    np.testing.assert_array_equal(small.inliers, large.inliers)


def test_sim3_rejects_scale_below_min_scale():
    rng = np.random.default_rng(15)
    source = rng.normal(size=(30, 3))
    result = sim3_ransac(source, 0.01 * source, seed=2, min_inliers=12)
    assert not result.ok
    assert result.reason == "no_consensus"


def test_sim3_accepts_a_minimal_three_point_cloud():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.0, 0.0]])
    target = 2.0 * source + np.array([1.0, -0.5, 0.3])
    result = sim3_ransac(source, target, seed=1, min_inliers=3)
    assert result.ok
    assert result.scale == pytest.approx(2.0, rel=1e-9)
    assert result.inliers.sum() == 3


def test_sim3_fails_gracefully_for_empty_and_small_inputs():
    empty = np.empty((0, 3))
    result = sim3_ransac(empty, empty, min_inliers=3)
    assert not result.ok
    assert result.reason == "too_few_points"
    rng = np.random.default_rng(17)
    few = rng.normal(size=(10, 3))
    result = sim3_ransac(few, few, min_inliers=12)
    assert not result.ok
    assert result.reason == "too_few_points"
    with pytest.raises(ValueError):
        sim3_ransac(few, few, min_inliers=2)


def test_sim3_fits_coplanar_cloud():
    rng = np.random.default_rng(19)
    source = np.column_stack([rng.uniform(-2.0, 2.0, 50),
                              rng.uniform(-1.0, 1.0, 50), np.zeros(50)])
    R = _rotation([0.0, 0.0, 0.4])
    target = 1.8 * (source @ R.T) + np.array([0.5, -0.2, 0.1])
    result = sim3_ransac(source, target, seed=3, min_inliers=12)
    assert result.ok
    assert result.scale == pytest.approx(1.8, rel=1e-6)
    angle = np.degrees(np.linalg.norm(
        cv2.Rodrigues(result.rotation @ R.T)[0]))
    assert angle < 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
