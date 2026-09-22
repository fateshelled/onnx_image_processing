"""Scale-observable Sim(3) primitives for monocular loop verification.

The two point clouds passed to :func:`sim3_ransac` must be reconstructed
independently (for example from a short local two-view track on each side of
a loop).  A single essential-matrix pair has no observable scale and must not
be used to manufacture a Sim(3) measurement from the pose graph itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TriangulationResult:
    points: np.ndarray
    valid: np.ndarray
    parallax_deg: np.ndarray


@dataclass(frozen=True)
class Sim3Result:
    ok: bool
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inliers: np.ndarray
    median_residual: float
    reason: str = ""


def triangulate_local(
    points0_yx: np.ndarray,
    points1_yx: np.ndarray,
    rotation_10: np.ndarray,
    translation_10: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    min_parallax_deg: float = 1.0,
    max_depth_baselines: float = 100.0,
) -> TriangulationResult:
    """Triangulate a local two-view track in camera-0 baseline units.

    ``rotation_10, translation_10`` follow ``x1 = R10 @ x0 + t10``.
    Returned points are expressed in camera-0 coordinates.  Validity requires
    finite homogeneous division, positive depth in both views, sufficient
    ray parallax and bounded depth relative to the input baseline.
    """
    p0 = np.asarray(points0_yx, dtype=np.float64)
    p1 = np.asarray(points1_yx, dtype=np.float64)
    R = np.asarray(rotation_10, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation_10, dtype=np.float64).reshape(3)
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    n = len(p0)
    if p0.shape != (n, 2) or p1.shape != (n, 2):
        raise ValueError("point arrays must both have shape (N, 2)")
    if min_parallax_deg < 0.0 or max_depth_baselines <= 0.0:
        raise ValueError("invalid triangulation thresholds")
    if (not np.all(np.isfinite(K)) or abs(float(np.linalg.det(K))) <= 1e-12
            or K[0, 0] <= 0.0 or K[1, 1] <= 0.0):
        raise ValueError("camera_matrix must be finite, invertible and have "
                         "positive focal lengths")
    if n == 0:
        return TriangulationResult(np.empty((0, 3)), np.zeros(0, bool),
                                   np.empty(0))

    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t[:, None]])
    h = cv2.triangulatePoints(P0, P1, p0[:, ::-1].T, p1[:, ::-1].T)
    w = h[3]
    finite_w = np.isfinite(w) & (np.abs(w) > 1e-12)
    X0 = np.full((n, 3), np.nan, dtype=np.float64)
    X0[finite_w] = (h[:3, finite_w] / w[finite_w]).T
    X1 = (R @ X0.T).T + t

    Kinv = np.linalg.inv(K)
    r0 = (Kinv @ np.column_stack([p0[:, 1], p0[:, 0], np.ones(n)]).T).T
    r1_cam1 = (Kinv @ np.column_stack([p1[:, 1], p1[:, 0], np.ones(n)]).T).T
    r1 = (R.T @ r1_cam1.T).T
    r0 /= np.linalg.norm(r0, axis=1, keepdims=True)
    r1 /= np.linalg.norm(r1, axis=1, keepdims=True)
    cosine = np.clip(np.sum(r0 * r1, axis=1), -1.0, 1.0)
    parallax = np.degrees(np.arccos(cosine))

    baseline = float(np.linalg.norm(t))
    depth_limit = max_depth_baselines * baseline
    valid = (
        finite_w
        & np.all(np.isfinite(X0), axis=1)
        & (X0[:, 2] > 0.0)
        & (X1[:, 2] > 0.0)
        & (parallax >= min_parallax_deg)
        & (baseline > 1e-12)
        & (X0[:, 2] <= depth_limit)
        & (X1[:, 2] <= depth_limit)
    )
    return TriangulationResult(X0, valid, parallax)


def _umeyama_sim3(source: np.ndarray, target: np.ndarray):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if len(source) < 3:
        return None
    mx, my = source.mean(axis=0), target.mean(axis=0)
    X, Y = source - mx, target - my
    variance = float(np.sum(X * X) / len(source))
    if not np.isfinite(variance) or variance <= 1e-12:
        return None
    U, singular, Vt = np.linalg.svd((Y.T @ X) / len(source))
    sign = np.ones(3)
    if np.linalg.det(U @ Vt) < 0.0:
        sign[-1] = -1.0
    R = U @ np.diag(sign) @ Vt
    scale = float(np.sum(singular * sign) / variance)
    translation = my - scale * (R @ mx)
    if scale <= 0.0 or not np.all(np.isfinite([scale, *translation.ravel()])):
        return None
    return scale, R, translation


def _cloud_well_conditioned(points: np.ndarray, ratio: float) -> bool:
    """Reject collinear/zero-extent clouds while accepting planar (rank-2) ones.

    A coplanar track still pins scale and in-plane rotation: the second
    singular value ratio bounds the in-plane extent, and the SVD's arbitrary
    out-of-plane direction does not enter the residuals of coplanar points.
    Only clouds without any well-defined plane (collinear or coincident
    points) are refused here.
    """
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return bool(len(singular) == 3 and singular[0] > 1e-9
                and singular[1] / singular[0] >= ratio)


def sim3_ransac(
    source: np.ndarray,
    target: np.ndarray,
    *,
    residual_fraction: float = 0.05,
    max_iterations: int = 1000,
    seed: int = 0,
    min_inliers: int = 12,
    min_scale: float = 0.1,
    max_scale: float = 10.0,
    min_condition_ratio: float = 1e-2,
) -> Sim3Result:
    """Fit ``target ~= scale * R @ source + t`` with deterministic RANSAC."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if residual_fraction <= 0.0 or max_iterations < 1 or min_inliers < 3:
        raise ValueError("invalid RANSAC parameters")
    input_n = len(source)
    finite = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(target), axis=1)
    finite_indices = np.flatnonzero(finite)
    source, target = source[finite], target[finite]
    n = len(source)
    # Umeyama needs three non-collinear points; tiny clouds therefore use a
    # 3-point minimal sample instead of the usual 4-point one.
    sample_size = min(4, n)
    empty = np.zeros(input_n, dtype=bool)
    fail = lambda reason: Sim3Result(False, np.nan, np.eye(3), np.zeros(3),
                                     empty.copy(), np.inf, reason)
    if n < min_inliers:
        return fail("too_few_points")
    if not _cloud_well_conditioned(source, min_condition_ratio):
        return fail("degenerate_source")
    if not _cloud_well_conditioned(target, min_condition_ratio):
        return fail("degenerate_target")

    # Residuals are measured in target units, so derive their threshold from
    # target spread. This remains invariant to the relative reconstruction
    # gauge estimated by Sim(3), including scales far from one.
    pair_distances = np.linalg.norm(target[:, None] - target[None, :], axis=2)
    nonzero = pair_distances[pair_distances > 1e-9]
    if len(nonzero) == 0:
        return fail("zero_extent")
    threshold = residual_fraction * float(np.median(nonzero))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(max_iterations):
        idx = rng.choice(n, size=sample_size, replace=False)
        if not _cloud_well_conditioned(source[idx], min_condition_ratio):
            continue
        model = _umeyama_sim3(source[idx], target[idx])
        if model is None:
            continue
        s, R, t = model
        if not min_scale <= s <= max_scale:
            continue
        residual = np.linalg.norm(target - (s * (source @ R.T) + t), axis=1)
        inliers = residual <= threshold
        score = (int(inliers.sum()), -float(np.median(residual[inliers]))
                 if inliers.any() else -np.inf)
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[1].sum() < min_inliers:
        return fail("no_consensus")
    inliers = best[1]
    # Refit and re-select inliers until the mask stabilizes (bounded so the
    # result stays deterministic and cheap).
    for _ in range(3):
        if (not _cloud_well_conditioned(source[inliers], min_condition_ratio)
                or not _cloud_well_conditioned(target[inliers],
                                               min_condition_ratio)):
            return fail("degenerate_refit")
        model = _umeyama_sim3(source[inliers], target[inliers])
        if model is None:
            return fail("refit_failed")
        s, R, t = model
        if not min_scale <= s <= max_scale:
            return fail("invalid_refit")
        residual = np.linalg.norm(target - (s * (source @ R.T) + t), axis=1)
        refined = residual <= threshold
        if np.array_equal(refined, inliers):
            break
        if refined.sum() < min_inliers:
            return fail("invalid_refit")
        inliers = refined
    else:
        # Defensive: the mask still changes after the iteration cap, so the
        # model cannot be returned consistently with its own inlier set.  This
        # branch is not covered by a synthetic regression test yet.
        return fail("refit_oscillation")
    if inliers.sum() < min_inliers:
        return fail("invalid_refit")
    full_inliers = np.zeros(input_n, dtype=bool)
    full_inliers[finite_indices] = inliers
    return Sim3Result(True, s, R, t, full_inliers,
                      float(np.median(residual[inliers])))
