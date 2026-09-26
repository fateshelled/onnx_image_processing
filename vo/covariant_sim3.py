"""Conditional two-view point covariance and fixed-inlier Sim(3) diagnostic.

The local relative camera pose is held fixed; shared pose errors between
tracks are deliberately outside this independent-observation model.
"""

import cv2
import numpy as np
from scipy.optimize import minimize

from vo.sim3_verification import triangulate_local

REFINE_MAXITER = 500
REFINE_FTOL = 1e-9
REFINE_HUBER_DELTA = 3.0
REFINE_MIN_SCALE = 0.1
REFINE_MAX_SCALE = 10.0


def _optimizer_failure_reason(result, fun):
    """Map a scipy result to a dropout reason, or None on success."""
    if not np.isfinite(fun) or fun >= 1e29:
        return "singular_covariance"
    if not result.success:
        return "maxiter_reached" if result.status == 1 else "optimizer_failed"
    return None


def triangulation_covariance(p0, p1, rotation, translation, K, *, sigma_px=1.0,
                             min_parallax_deg=1.0, max_depth_baselines=100.0):
    """Return per-track 3x3 covariance in the first camera, or None.

    Coordinates are y,x in both views. Central differences differentiate the
    *same* OpenCV DLT triangulation used by the unweighted verifier, with the
    caller's parallax/depth gates so a point is dropped for the same reason in
    both arms. The validity gate is checked only at the unperturbed
    observation: a 1px perturbation crossing a parallax threshold is not
    itself a missing point.
    """
    p0 = np.asarray(p0, float).reshape(2)
    p1 = np.asarray(p1, float).reshape(2)
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError("sigma_px must be positive and finite")
    obs = np.r_[p0, p1]
    if not np.all(np.isfinite(obs)):
        return None
    tri = triangulate_local(p0[None], p1[None], rotation, translation, K,
                            min_parallax_deg=min_parallax_deg,
                            max_depth_baselines=max_depth_baselines)
    if not tri.valid[0]:
        return None
    step = 0.05  # pixels; keep steps below the measurement noise
    J = np.empty((3, 4))
    for col in range(4):
        plus, minus = obs.copy(), obs.copy()
        plus[col] += step
        minus[col] -= step
        a = triangulate_local(plus[:2][None], plus[2:][None],
                              rotation, translation, K,
                              min_parallax_deg=min_parallax_deg,
                              max_depth_baselines=max_depth_baselines).points[0]
        b = triangulate_local(minus[:2][None], minus[2:][None],
                              rotation, translation, K,
                              min_parallax_deg=min_parallax_deg,
                              max_depth_baselines=max_depth_baselines).points[0]
        J[:, col] = (a - b) / (2 * step)
    cov = sigma_px ** 2 * (J @ J.T)
    if not np.all(np.isfinite(cov)):
        return None
    eig = np.linalg.eigvalsh(cov)
    if eig[0] <= 0 or eig[-1] / eig[0] > 1e12:
        return None
    return cov


def _robust_rho(norms, kernel, delta):
    if kernel == "huber":
        return np.where(norms <= delta, norms ** 2,
                        2 * delta * norms - delta ** 2)
    if kernel == "cauchy":
        return delta ** 2 * np.log1p((norms / delta) ** 2)
    if kernel == "tukey":
        ratio = (norms / delta) ** 2
        return np.where(ratio < 1.0,
                        (delta ** 2 / 6.0) * (1.0 - (1.0 - ratio) ** 3),
                        delta ** 2 / 6.0)
    if kernel == "gm":  # Geman-McClure, bounded by delta**2
        return delta ** 2 * norms ** 2 / (delta ** 2 + norms ** 2)
    raise ValueError(f"unknown kernel: {kernel}")


def simplify_covariance(cov, point, mode, lateral_variance, depth_ratio=None,
                        max_depth_ratio=None):
    """Ablate a per-point covariance for diagnostics.

    ``full`` keeps the finite-difference covariance. ``fixed_lateral`` keeps
    the along-ray (depth) variance from the full covariance but replaces the
    two lateral variances by a per-endpoint constant, removing the lateral
    depth scaling and the cross terms. ``fixed_depth`` instead fixes the
    anisotropy ratio: lateral ``lateral_variance`` and along-ray
    ``lateral_variance * depth_ratio**2``, independent of parallax (the GICP
    planar-eigenvalue idea). ``iso`` uses the lateral constant isotropically
    and ``none`` is the identity control.
    """
    if cov is None:
        return None
    if mode == "none":
        return np.eye(3)
    if mode not in ("full", "fixed_lateral", "fixed_depth", "iso"):
        raise ValueError(f"unknown covariance mode: {mode}")
    if mode != "full" and (lateral_variance is None
                           or not np.isfinite(lateral_variance)
                           or lateral_variance <= 0.0):
        raise ValueError("lateral_variance must be positive and finite")
    if max_depth_ratio is not None and (not np.isfinite(max_depth_ratio)
                                        or max_depth_ratio <= 0.0):
        raise ValueError("max_depth_ratio must be positive and finite")

    ray = np.asarray(point, float)
    norm = float(np.linalg.norm(ray))
    if mode in ("fixed_lateral", "fixed_depth") and norm < 1e-12:
        return None
    unit = ray / norm
    perp = np.eye(3) - np.outer(unit, unit)
    if mode == "full":
        result = np.asarray(cov, float)
    elif mode == "iso":
        result = np.eye(3) * lateral_variance
    elif mode == "fixed_depth":
        if depth_ratio is None or not np.isfinite(depth_ratio) or depth_ratio <= 0.0:
            raise ValueError("depth_ratio must be positive and finite")
        result = (lateral_variance * perp
                  + lateral_variance * depth_ratio ** 2 * np.outer(unit, unit))
    else:  # fixed_lateral
        depth = float(unit @ cov @ unit)
        result = lateral_variance * perp + depth * np.outer(unit, unit)

    if max_depth_ratio is not None and mode in ("full", "fixed_lateral"):
        depth = float(unit @ result @ unit)
        lateral = (float(np.trace(result)) - depth) / 2.0
        cap = max_depth_ratio ** 2 * lateral
        if depth > cap:
            result = result + (cap - depth) * np.outer(unit, unit)
    return result


def refine_sim3(source, target, cov_source, cov_target, fit, *,
                min_scale=REFINE_MIN_SCALE, max_scale=REFINE_MAX_SCALE,
                huber_delta=REFINE_HUBER_DELTA, kernel="huber"):
    """Robust Mahalanobis refit on the original Euclidean RANSAC inliers.

    The objective is the Huber-robust negative log likelihood with the
    covariance log determinant, so scale cannot grow just to inflate the
    covariance. Scale bounds match the original RANSAC; no new acceptance gate
    is added. Always returns a dict with ``ok`` and, on failure, ``reason``.
    """
    mask = fit.inliers
    A, B = np.asarray(source)[mask], np.asarray(target)[mask]
    Ca, Cb = np.asarray(cov_source)[mask], np.asarray(cov_target)[mask]
    if len(A) < 5:
        return {"ok": False, "reason": "too_few_inliers"}
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
        return {"ok": False, "reason": "nonfinite_points"}
    if not (np.all(np.isfinite(Ca)) and np.all(np.isfinite(Cb))):
        return {"ok": False, "reason": "nonfinite_covariance"}
    rotvec = cv2.Rodrigues(fit.rotation)[0].ravel()
    x0 = np.r_[np.log(fit.scale), rotvec, fit.translation]

    def objective(x):
        s = np.exp(x[0])
        R = cv2.Rodrigues(x[1:4])[0]
        residual = B - (s * (A @ R.T) + x[4:7])
        cov = Cb + s * s * (R @ Ca @ R.T)
        try:
            chol = np.linalg.cholesky(cov)
            whitened = np.linalg.solve(chol, residual[..., None])[..., 0]
        except np.linalg.LinAlgError:
            return 1e30
        norms = np.linalg.norm(whitened, axis=1)
        robust = _robust_rho(norms, kernel, huber_delta)
        return float(np.sum(robust + 2 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)))

    if not np.isfinite(objective(x0)):
        return {"ok": False, "reason": "singular_covariance"}
    result = minimize(objective, x0, method="L-BFGS-B",
                      bounds=[(np.log(min_scale), np.log(max_scale))]
                      + [(None, None)] * 6,
                      options={"maxiter": REFINE_MAXITER, "ftol": REFINE_FTOL})
    failure = _optimizer_failure_reason(result, result.fun)
    if failure is not None:
        return {"ok": False, "reason": failure}
    x = result.x
    if (x[0] <= np.log(min_scale) + 1e-4 or x[0] >= np.log(max_scale) - 1e-4):
        return {"ok": False, "reason": "scale_at_bound"}
    if np.linalg.norm(x[4:7]) < 1e-8:
        return {"ok": False, "reason": "zero_translation"}

    def whitened_residuals(y):
        s = np.exp(y[0])
        R = cv2.Rodrigues(y[1:4])[0]
        cov = Cb + s * s * (R @ Ca @ R.T)
        return np.linalg.solve(np.linalg.cholesky(cov),
                               (B - s * (A @ R.T) - y[4:7])[..., None])[:, :, 0].ravel()

    eps = 1e-5
    J = np.column_stack([(whitened_residuals(x + eps * np.eye(7)[k])
                          - whitened_residuals(x - eps * np.eye(7)[k])) / (2 * eps)
                         for k in range(7)])
    singular = np.linalg.svd(J, compute_uv=False)
    if not np.all(np.isfinite(singular)):
        return {"ok": False, "reason": "nonfinite_jacobian"}
    if singular[-1] < 1e-8 or singular[-1] / singular[0] < 1e-5:
        return {"ok": False, "reason": "ill_conditioned"}
    # Calibration diagnostics. The whitened residual is measured on the fitted
    # inliers at the optimum, so it is a conditional lower bound, not the
    # unconditional noise scale. ``direction_std_deg`` is a Gauss-Newton
    # approximation: (J^T J)^-1 assumes unit-variance independent whitened
    # residuals, and it ignores the covariance's parameter dependence and the
    # Huber weights, so it is optimistic. Comparing it to the GT angle error
    # exposes miscalibration, but shared pose error and triangulation bias also
    # move the ratio.
    whitened = whitened_residuals(x).reshape(-1, 3)
    direction_std_deg = None
    try:
        covariance = np.linalg.inv(J.T @ J)
        translation = x[4:7]
        norm = float(np.linalg.norm(translation))
        if norm > 1e-9:
            tangent = np.eye(3) - np.outer(translation / norm, translation / norm)
            projected = tangent @ covariance[4:7, 4:7] @ tangent / norm ** 2
            direction_std_deg = float(
                np.degrees(np.sqrt(max(float(np.trace(projected)), 0.0))))
    except np.linalg.LinAlgError:
        direction_std_deg = None
    return {"ok": True, "reason": "",
            "scale": float(np.exp(x[0])),
            "rotation": cv2.Rodrigues(x[1:4])[0],
            "translation": x[4:7],
            "cost_before": objective(x0), "cost_after": float(result.fun),
            "jacobian_condition": float(singular[0] / singular[-1]),
            "whitened_norm_sq_inlier_mean": float(
                (whitened ** 2).sum(axis=1).mean()),
            "direction_std_deg": direction_std_deg}
