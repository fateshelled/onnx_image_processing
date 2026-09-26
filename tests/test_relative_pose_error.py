"""Tests for the gauge-free RPE metric with whole-trajectory scale."""

import cv2
import numpy as np
import pytest

from eval.eval_tum_vo import relative_pose_errors


def _se3(rotation, translation):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def _straight_trajectory(n, step=0.2):
    return np.array([_se3(np.eye(3), [step * i, 0.0, 0.0]) for i in range(n)])


def test_rpe_is_zero_for_exact_similarity():
    gt = _straight_trajectory(50)
    rotation = cv2.Rodrigues(np.array([0.1, -0.2, 0.05]))[0]
    scale, translation = 3.7, np.array([1.0, -2.0, 0.5])
    est = np.array([_se3(rotation @ g[:3, :3],
                         scale * (rotation @ g[:3, 3]) + translation)
                    for g in gt])
    metrics = relative_pose_errors(est, gt, lengths_m=(1.0, 2.0, 5.0))
    assert metrics["RPE_trans_1m_median"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["RPE_rot_1m_median_deg"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["RPE_trans_5m_median"] == pytest.approx(0.0, abs=1e-9)


def test_rpe_detects_accumulating_drift():
    gt = _straight_trajectory(50)
    est = gt.copy()
    # lateral drift that accumulates with travelled distance: every segment is
    # corrupted, so the median (not just an RMSE tail) must grow with length
    est[:, 1, 3] += 0.02 * np.arange(len(est))
    metrics = relative_pose_errors(est, gt, lengths_m=(1.0, 5.0))
    assert metrics["RPE_trans_1m_median"] > 0.005
    assert metrics["RPE_trans_5m_median"] > 0.02
    assert (metrics["RPE_trans_5m_median"]
            > metrics["RPE_trans_1m_median"])


def test_rpe_returns_nan_for_unreachable_length():
    gt = _straight_trajectory(10, step=0.2)  # total 1.8 m
    metrics = relative_pose_errors(gt.copy(), gt, lengths_m=(1.0, 5.0))
    assert np.isfinite(metrics["RPE_trans_1m_median"])
    assert np.isnan(metrics["RPE_trans_5m_median"])
