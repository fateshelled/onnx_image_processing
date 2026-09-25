"""Unit tests for the covariance-weighted Sim(3) loop verifier."""

import cv2
import numpy as np
import pytest

from vo.covariant_sim3_verifier import CovariantSim3LoopVerifier
from vo.sim3_verification import Sim3Result

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
STRIDE = 10
STRUCTURE = np.array([
    [0.3, 0.1, 2.0], [0.6, -0.2, 2.5], [0.1, 0.3, 3.0], [0.5, 0.4, 3.5],
    [0.2, -0.3, 4.0], [0.7, 0.2, 4.5], [0.4, -0.1, 5.0], [0.0, 0.2, 2.2],
    [0.8, -0.4, 2.8], [0.3, 0.5, 3.2], [0.6, -0.5, 4.2], [0.1, -0.1, 4.8],
])


def _odom(count):
    return [{"ok": True, "R": np.eye(3), "t": np.array([1.0, 0.0, 0.0])}
            for _ in range(count)]


def _keypoints(frame):
    camera = STRUCTURE + np.array([frame / STRIDE, 0.0, 0.0])
    uv = (K @ camera.T).T
    return (uv[:, :2] / uv[:, 2:])[:, ::-1]


def _match_fn(a, b):
    return _keypoints(a), _keypoints(b)


def _fit(inliers=12, translation=(1.0, 0.0, 0.0)):
    mask = np.zeros(20, dtype=bool)
    mask[:inliers] = True
    return Sim3Result(True, 1.0, np.eye(3), np.array(translation, float),
                      mask, 0.0)


def _decide(verifier, fit):
    return verifier._decide(10, 40, fit, [], [], [], {}, {})


def test_consistency_gate_accepts_and_rejects():
    verifier = CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                         gate=8, window_strides=2)
    verifier.last_consistency = {"n": 12, "f_in": 0.9, "rho_med": 1.0,
                                 "direction_std_deg": 1.0}
    assert _decide(verifier, _fit()) is True
    verifier.last_consistency = {"n": 12, "f_in": 0.2, "rho_med": 1.0,
                                 "direction_std_deg": 1.0}
    assert _decide(verifier, _fit()) is False
    assert verifier.n_inconsistent_rejected == 1
    verifier.last_consistency = {"n": 12, "f_in": 0.9, "rho_med": 9.0,
                                 "direction_std_deg": 1.0}
    assert _decide(verifier, _fit()) is False
    assert verifier.n_inconsistent_rejected == 2


def test_consistency_gate_falls_back_without_covariance():
    verifier = CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                         gate=8, window_strides=2)
    verifier.last_consistency = None
    assert _decide(verifier, _fit(inliers=12)) is True
    assert _decide(verifier, _fit(inliers=4)) is False


def test_direction_z_gate_rejects_only_beyond_threshold():
    pose_fn = lambda a, b: {"ok": True, "t": np.array([0.0, 0.0, 1.0])}
    verifier = CovariantSim3LoopVerifier(
        _odom(5), _match_fn, K, STRIDE, gate=8, window_strides=2,
        pose_fn=pose_fn, max_direction_z=3.0)
    verifier.last_consistency = {"n": 12, "f_in": 1.0, "rho_med": 0.1,
                                 "direction_std_deg": 1.0}
    assert _decide(verifier, _fit(translation=(1.0, 0.0, 0.0))) is False
    assert verifier.n_direction_z_rejected == 1
    verifier.last_consistency["direction_std_deg"] = 100.0
    assert _decide(verifier, _fit(translation=(1.0, 0.0, 0.0))) is True


def test_real_covariance_cloud_accepts_consistent_loop():
    verifier = CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                         gate=8, window_strides=2)
    assert verifier(10, 40) is True
    assert verifier.n_accept == 1
    assert verifier.last_consistency is not None
    assert verifier.last_consistency["f_in"] == 1.0


def test_fixed_angle_gate_is_applied():
    pose_fn = lambda a, b: {"ok": True, "t": np.array([0.0, 0.0, 1.0])}
    verifier = CovariantSim3LoopVerifier(
        _odom(5), _match_fn, K, STRIDE, gate=8, window_strides=2,
        pose_fn=pose_fn, max_translation_angle_deg=1.0)
    verifier.last_consistency = {"n": 12, "f_in": 1.0, "rho_med": 0.1,
                                 "direction_std_deg": 1.0}
    assert _decide(verifier, _fit(translation=(1.0, 0.0, 0.0))) is False
    assert verifier.n_direction_rejected == 1


def test_missing_direction_sigma_abstains():
    pose_fn = lambda a, b: {"ok": True, "t": np.array([0.0, 0.0, 1.0])}
    verifier = CovariantSim3LoopVerifier(
        _odom(5), _match_fn, K, STRIDE, gate=8, window_strides=2,
        pose_fn=pose_fn, max_direction_z=3.0)
    verifier.last_consistency = {"n": 12, "f_in": 1.0, "rho_med": 0.1,
                                 "direction_std_deg": None}
    assert _decide(verifier, _fit()) is False  # abstain_policy='reject'
    assert verifier.n_uncertain_abstained == 1


def test_missing_covariance_falls_back(monkeypatch):
    import vo.covariant_sim3_verifier as module
    monkeypatch.setattr(module, "triangulation_covariance",
                        lambda *args, **kwargs: None)
    verifier = CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                         gate=8, window_strides=2)
    assert verifier(10, 40) is True
    assert verifier.last_consistency is None


def test_partial_missing_covariance_falls_back():
    from vo.loop_sim3_verifier import joined_tracks
    verifier = CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                         gate=8, window_strides=2)
    cloud_a, cloud_b = verifier._cloud(10), verifier._cloud(40)
    pa, pb = _match_fn(10, 40)
    joined, Xa, Xb = joined_tracks(pa, pb, cloud_a, cloud_b)
    base = verifier._fit(10, 40, Xa, Xb, joined)
    assert base.ok and verifier.last_consistency is not None
    inlier = int(np.flatnonzero(base.inliers)[0])
    verifier.covariances[10][joined[inlier][0]] = None
    refit = verifier._fit(10, 40, Xa, Xb, joined)
    assert refit.ok and verifier.last_consistency is None


def test_inlier_gate_precedes_consistency_gate():
    verifier = CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                         gate=20, window_strides=2)
    verifier.last_consistency = {"n": 12, "f_in": 1.0, "rho_med": 0.1,
                                 "direction_std_deg": 1.0}
    assert _decide(verifier, _fit(inliers=12)) is False
    assert verifier.n_inconsistent_rejected == 0


def test_small_cache_size_does_not_crash():
    verifier = CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                         gate=8, window_strides=2, cache_size=1)
    assert verifier(10, 40) is True
    # a's covariances were evicted, so the gate falls back to the raw inliers
    assert verifier.last_consistency is None


def test_whitened_iterations_run_and_validate():
    verifier = CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                         gate=8, window_strides=2,
                                         whitened_iterations=5)
    assert verifier(10, 40) is True
    assert verifier.last_consistency is not None
    with pytest.raises(ValueError):
        CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                  whitened_iterations=-1)


def test_whitened_irls_reclassifies_depth_outlier(monkeypatch):
    rng = np.random.default_rng(7)
    A = rng.normal(size=(25, 3)) + [0.0, 0.0, 5.0]
    rotation = cv2.Rodrigues(np.array([0.05, -0.1, 0.07]))[0]
    translation = np.array([0.6, -0.2, 0.25])
    B = A @ rotation.T + translation
    B[0] = B[0] + [0.0, 0.0, 8.0]  # depth-direction outlier
    covariance = np.tile(np.diag([1e-4, 1e-4, 100.0]), (25, 1, 1))

    verifier = CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE)
    monkeypatch.setattr(
        verifier, "_cov_array",
        lambda endpoint, keys, points: (covariance if endpoint == 10
                                        else covariance))
    joined = [(i, i) for i in range(25)]
    base = verifier._fit(10, 40, A.tolist(), B.tolist(), joined)
    assert base.ok and not base.inliers[0]  # raw Euclidean RANSAC drops it
    verifier.whitened_iterations = 1
    reclassified = verifier._fit(10, 40, A.tolist(), B.tolist(), joined)
    assert reclassified.ok
    # whitened RANSAC pulls in the depth-uncertain track that Euclidean RANSAC
    # dropped, so the inlier set actually changes
    assert reclassified.inliers[0]
    assert not np.array_equal(reclassified.inliers, base.inliers)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE, sigma_px=0.0)
    with pytest.raises(ValueError):
        CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                  weight_mode="bogus")
    with pytest.raises(ValueError):
        CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE, kernel="bogus")
    with pytest.raises(ValueError):
        CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                  min_consistent_fraction=0.0)
    with pytest.raises(ValueError):
        CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                  max_direction_z=1.0)  # needs pose_fn
    with pytest.raises(ValueError):
        CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                  weight_mode="fixed_depth")  # needs depth_ratio
    with pytest.raises(ValueError):
        CovariantSim3LoopVerifier(_odom(5), _match_fn, K, STRIDE,
                                  max_depth_ratio=float("nan"))
