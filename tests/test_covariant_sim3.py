from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts.diag_covariant_sim3 import summarize_rows
from vo.covariant_sim3 import (
    _optimizer_failure_reason, _robust_rho, refine_sim3, simplify_covariance,
    triangulation_covariance,
)
from vo.sim3_verification import sim3_ransac, triangulate_local


def test_optimizer_failure_reason_mapping():
    assert _optimizer_failure_reason(
        SimpleNamespace(success=True, status=0), 1.0) is None
    assert _optimizer_failure_reason(
        SimpleNamespace(success=False, status=1), 1.0) == "maxiter_reached"
    assert _optimizer_failure_reason(
        SimpleNamespace(success=False, status=2), 1.0) == "optimizer_failed"
    assert _optimizer_failure_reason(
        SimpleNamespace(success=True, status=0), 1e30) == "singular_covariance"
    assert _optimizer_failure_reason(
        SimpleNamespace(success=False, status=0), np.nan) == "singular_covariance"


def _project(K, X):
    return (K @ np.asarray(X, float))[:2] / np.asarray(X, float)[2]


def test_triangulation_covariance_pixel_noise_scaling():
    K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    point = np.array([0.2, -0.1, 5.])
    p0 = _project(K, point)
    p1 = _project(K, point + [-1., 0., 0.])
    a = triangulation_covariance(p0[::-1], p1[::-1], np.eye(3),
                                 [-1., 0., 0.], K)
    b = triangulation_covariance(p0[::-1], p1[::-1], np.eye(3),
                                 [-1., 0., 0.], K, sigma_px=2.)
    assert a is not None
    assert a[2, 2] > 30 * a[1, 1]
    np.testing.assert_allclose(b, 4 * a, atol=1e-9)


def test_triangulation_covariance_matches_monte_carlo():
    """The finite-difference covariance must equal the empirical one.

    This validates that the numerical Jacobian differentiates the same
    triangulation used in production, including the strong depth/lateral
    anisotropy, without relying on an analytic derivation.
    """
    K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    X = np.array([0.25, -0.15, 3.5])
    R = cv2.Rodrigues(np.array([0.04, -0.09, 0.02]))[0]
    t = np.array([-0.6, 0.03, 0.07])
    p0, p1 = _project(K, X), _project(K, R @ X + t)
    predicted = triangulation_covariance(p0[::-1], p1[::-1], R, t, K)

    rng = np.random.default_rng(0)
    noise = rng.normal(scale=1.0, size=(6000, 4))
    observed = triangulate_local((p0 + noise[:, :2])[:, ::-1],
                                 (p1 + noise[:, 2:])[:, ::-1], R, t, K)
    empirical = np.cov(observed.points.T)
    assert np.linalg.norm(empirical - predicted) / np.linalg.norm(predicted) < 0.05


def test_covariance_refit_anisotropic_beats_isotropic():
    """Depth-direction noise must be down-weighted relative to lateral noise."""
    rng = np.random.default_rng(17)
    A = rng.normal(size=(60, 3)) + [0., 0., 5.]
    R = cv2.Rodrigues(np.array([0.06, -0.11, 0.08]))[0]
    t = np.array([0.7, -0.2, 0.3])
    scale = 1.3
    B = scale * (A @ R.T) + t + rng.normal(
        scale=[0.004, 0.004, 0.05], size=A.shape)
    Ca = np.tile(np.diag([.004 ** 2, .004 ** 2, .05 ** 2]), (len(A), 1, 1))
    Cb = Ca.copy()
    fit = sim3_ransac(A, B, seed=1, residual_fraction=.1, min_inliers=20)
    assert fit.ok

    anisotropic = refine_sim3(A, B, Ca, Cb, fit)
    isotropic = refine_sim3(A, B, np.tile(np.eye(3) * .05 ** 2, (len(A), 1, 1)),
                            np.tile(np.eye(3) * .05 ** 2, (len(A), 1, 1)), fit)
    assert anisotropic["ok"] and isotropic["ok"]

    def angle_deg(model):
        a = model["translation"] / np.linalg.norm(model["translation"])
        b = t / np.linalg.norm(t)
        return float(np.degrees(np.arccos(np.clip(a @ b, -1., 1.))))

    assert angle_deg(anisotropic) < angle_deg(isotropic)
    assert abs(anisotropic["scale"] - scale) < abs(isotropic["scale"] - scale)


def test_covariance_refit_known_anisotropic_sim3():
    rng = np.random.default_rng(17)
    A = rng.normal(size=(35, 3)) + [0., 0., 5.]
    R = cv2.Rodrigues(np.array([0.06, -0.11, 0.08]))[0]
    t = np.array([0.7, -0.2, 0.3])
    B = 1.3 * (A @ R.T) + t + rng.normal(scale=[0.005, 0.005, 0.03],
                                         size=A.shape)
    Ca = np.tile(np.diag([.005 ** 2, .005 ** 2, .03 ** 2]), (len(A), 1, 1))
    Cb = Ca.copy()
    fit = sim3_ransac(A, B, seed=1, residual_fraction=.1, min_inliers=15)
    assert fit.ok
    weighted = refine_sim3(A, B, Ca, Cb, fit)
    assert weighted["ok"]
    assert weighted["cost_after"] <= weighted["cost_before"] + 1e-6
    assert abs(weighted["scale"] - 1.3) < .1


def test_summarize_rows_keeps_both_arms_on_the_same_population():
    """Covariance dropout must not silently improve the reported baseline."""
    def row(fit, cov_fit, gt, base_angle, cov_angle, **extra):
        data = {"fit": fit, "cov_fit": cov_fit, "cov_reason": "",
                "gt_distance_m": gt, "baseline_angle_deg": base_angle,
                "cov_angle_deg": cov_angle,
                "baseline_rotation_error_deg": base_angle,
                "cov_rotation_error_deg": cov_angle,
                "recover_angle_deg": 5.0}
        data.update(extra)
        return data

    rows = [
        row(True, True, 0.10, 40.0, 10.0),
        row(True, True, 0.20, 20.0, 30.0),
        row(True, False, 0.30, 100.0, None, cov_reason="optimizer_failed"),
        row(True, True, 0.01, 50.0, 1.0),
        {"fit": False, "cov_fit": False, "cov_reason": ""},
    ]
    summary = summarize_rows(rows)
    assert summary["n_candidates"] == 5
    assert summary["n_fit"] == 4
    assert summary["n_cov_fit"] == 3
    assert summary["n_cov_dropout"] == 1
    assert summary["cov_dropout_reasons"] == {"optimizer_failed": 1}
    assert summary["cov_dropout_baseline_angle_median_deg"] == 100.0

    ge5 = summary["eval"]["gt_ge_5cm"]
    assert ge5["n"] == 2
    assert ge5["baseline_angle_median_deg"] == 30.0
    assert ge5["cov_angle_median_deg"] == 20.0
    assert ge5["cov_better_count"] == 1
    assert summary["eval"]["gt_ge_10cm"]["n"] == 2


def test_simplify_covariance_modes():
    rotation = cv2.Rodrigues(np.array([0.3, -0.5, 0.2]))[0]
    cov = rotation @ np.diag([0.25, 0.36, 4.0]) @ rotation.T
    ray = rotation @ np.array([0.2, 0.1, 5.0])
    ray /= np.linalg.norm(ray)
    lateral = 0.1
    np.testing.assert_allclose(simplify_covariance(cov, ray, "full", lateral), cov)
    np.testing.assert_allclose(simplify_covariance(cov, ray, "none", lateral),
                               np.eye(3))
    np.testing.assert_allclose(simplify_covariance(cov, ray, "iso", lateral),
                               np.eye(3) * lateral)
    ran = simplify_covariance(cov, ray, "fixed_lateral", lateral)
    depth = float(ray @ cov @ ray)
    expected = lateral * (np.eye(3) - np.outer(ray, ray)) + depth * np.outer(ray, ray)
    np.testing.assert_allclose(ran, expected)
    eigenvalues = np.linalg.eigvalsh(ran)
    assert abs(eigenvalues[-1] - depth) < 1e-12
    np.testing.assert_allclose(eigenvalues[:2], [lateral, lateral], atol=1e-9)
    assert simplify_covariance(None, ray, "fixed_lateral", lateral) is None
    with pytest.raises(ValueError):
        simplify_covariance(cov, ray, "fixed_lateral", None)
    with pytest.raises(ValueError):
        simplify_covariance(cov, ray, "iso", 0.0)
    with pytest.raises(ValueError):
        simplify_covariance(cov, ray, "bogus", lateral)


def test_simplify_covariance_fixed_depth_ratio():
    rotation = cv2.Rodrigues(np.array([0.3, -0.5, 0.2]))[0]
    cov = rotation @ np.diag([0.25, 0.36, 4.0]) @ rotation.T
    ray = rotation @ np.array([0.2, 0.1, 5.0])
    ray /= np.linalg.norm(ray)
    lateral = 0.04
    for ratio in (10.0, 100.0, 1000.0):
        result = simplify_covariance(cov, ray, "fixed_depth", lateral, ratio)
        expected = (lateral * (np.eye(3) - np.outer(ray, ray))
                    + lateral * ratio ** 2 * np.outer(ray, ray))
        np.testing.assert_allclose(result, expected)
        eigenvalues = np.linalg.eigvalsh(result)
        np.testing.assert_allclose(eigenvalues[-1], lateral * ratio ** 2, rtol=1e-9)
        np.testing.assert_allclose(eigenvalues[:2], [lateral, lateral], atol=1e-12)
    assert simplify_covariance(cov, np.zeros(3), "fixed_depth", lateral, 10.0) is None
    with pytest.raises(ValueError):
        simplify_covariance(cov, ray, "fixed_depth", lateral, None)
    with pytest.raises(ValueError):
        simplify_covariance(cov, ray, "fixed_depth", lateral, 0.0)
    with pytest.raises(ValueError):
        simplify_covariance(cov, ray, "fixed_depth", lateral, np.inf)


def test_robust_kernels_and_depth_cap():
    norms = np.array([0.0, 1.0, 3.0, 10.0])
    robust = _robust_rho(norms, "huber", 3.0)
    assert robust[1] == 1.0
    assert np.isclose(robust[3], 2 * 3.0 * 10.0 - 9.0)
    cauchy = _robust_rho(norms, "cauchy", 3.0)
    tukey = _robust_rho(norms, "tukey", 3.0)
    # redescending kernels must stop growing past the scale
    assert cauchy[3] > cauchy[2]
    assert np.isclose(tukey[3], 9.0 / 6.0)
    assert tukey[3] <= tukey[2] + 1e-12
    gm = _robust_rho(norms, "gm", 3.0)
    # Geman-McClure is bounded by delta**2 and monotone increasing
    assert gm[0] == 0.0
    assert np.all(gm <= 9.0 + 1e-12)
    assert np.all(np.diff(gm) >= 0.0)
    assert np.isclose(gm[-1], 9.0 * 100.0 / (9.0 + 100.0))
    with pytest.raises(ValueError):
        _robust_rho(norms, "bogus", 3.0)

    # depth cap on the parallax-derived covariance
    ray = np.array([0., 0., 1.])
    cov = np.diag([0.04, 0.04, 100.0])  # depth/lateral ratio of 50
    capped = simplify_covariance(cov, ray, "fixed_lateral", 0.04,
                                 max_depth_ratio=10.0)
    assert np.isclose(capped[2, 2], 0.04 * 10.0 ** 2)
    np.testing.assert_allclose(np.diag(capped)[:2], [0.04, 0.04])
    full_capped = simplify_covariance(cov, ray, "full", 0.04,
                                      max_depth_ratio=10.0)
    assert np.isclose(full_capped[2, 2], 0.04 * 10.0 ** 2)
    with pytest.raises(ValueError):
        simplify_covariance(cov, ray, "fixed_lateral", 0.04,
                            max_depth_ratio=0.0)


def test_direction_std_scales_with_covariance_scale():
    rng = np.random.default_rng(23)
    A = rng.normal(size=(40, 3)) + [0., 0., 5.]
    R = cv2.Rodrigues(np.array([0.05, -0.1, 0.07]))[0]
    t = np.array([0.6, -0.2, 0.25])
    B = A @ R.T + t + rng.normal(scale=[0.003, 0.003, 0.04], size=A.shape)
    base = np.tile(np.diag([.003 ** 2, .003 ** 2, .04 ** 2]), (len(A), 1, 1))
    fit = sim3_ransac(A, B, seed=1, residual_fraction=.2, min_inliers=15)
    assert fit.ok
    one = refine_sim3(A, B, base, base, fit)
    four = refine_sim3(A, B, 4 * base, 4 * base, fit)
    assert one["ok"] and four["ok"]
    assert one["direction_std_deg"] is not None
    assert four["direction_std_deg"] is not None
    # doubling sigma_px doubles the predicted direction sigma
    np.testing.assert_allclose(four["direction_std_deg"],
                               2 * one["direction_std_deg"], rtol=1e-3)


def test_refine_sim3_reports_dropout_reason():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(20, 3)) + [0., 0., 5.]
    B = A + rng.normal(scale=.01, size=A.shape)
    fit = sim3_ransac(A, B, seed=1, residual_fraction=.2, min_inliers=10)
    assert fit.ok
    Ca = np.tile(np.eye(3), (len(A), 1, 1))
    bad = Ca.copy()
    bad[np.flatnonzero(fit.inliers)[0]] = np.nan
    result = refine_sim3(A, B, bad, Ca, fit)
    assert result["ok"] is False
    assert result["reason"] == "nonfinite_covariance"


def test_refine_sim3_singular_covariance_reason():
    rng = np.random.default_rng(9)
    A = rng.normal(size=(20, 3)) + [0., 0., 5.]
    B = A + rng.normal(scale=.01, size=A.shape)
    fit = sim3_ransac(A, B, seed=1, residual_fraction=.2, min_inliers=10)
    assert fit.ok
    zero = np.zeros((len(A), 3, 3))
    result = refine_sim3(A, B, zero, zero, fit)
    assert result["ok"] is False
    assert result["reason"] == "singular_covariance"
