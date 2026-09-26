"""Covariance-weighted Sim(3) loop verifier (online accept/reject gate).

Same local-cloud Sim(3) consistency check as :class:`Sim3LoopVerifier`, but the
loop fit is refit with per-track triangulation covariances under a robust
kernel, and the accept decision adds an uncertainty-normalized consistency
gate.  Depth-uncertain tracks therefore contribute less, and a candidate whose
direction is poorly observed can abstain instead of being accepted.

The extra gate uses:

* ``f_in``     fraction of tracks with a computable covariance whose whitened
               residual norm is within ``consistency_chi`` (a chi-square radius
               for 3 dof);
* ``rho_med``  median whitened residual norm over those tracks;
* ``z_dir``    translation-direction disagreement against ``pose_fn`` divided by
               the predicted direction sigma (only when ``max_direction_z > 0``).

With ``whitened_iterations > 0`` the fitted inlier set is reclassified in the
whitened space (whitened RANSAC / IRLS), so it can grow or shrink relative to
the raw Euclidean RANSAC inliers and the inlier-count gate is applied to that
reclassified set.  With ``whitened_iterations == 0`` the raw inliers are kept
and only the robust refit changes the model.  The parent fixed-angle gate is
still applied, but its comparison uses the covariance-refined translation
whereas the parent uses the raw RANSAC translation.  This verifier does not
modify the loop edge, only the accept/reject verdict.
"""

from __future__ import annotations

import numpy as np

from vo.covariant_sim3 import (
    REFINE_HUBER_DELTA, refine_sim3, simplify_covariance,
    triangulation_covariance,
)
from vo.loop_sim3_verifier import (
    SCALE_RATIO_BIN_EDGES, Sim3LoopVerifier,
    build_cloud_from_windows_detailed, point_key, translation_angle_deg,
)
from vo.sim3_verification import Sim3Result

WEIGHT_MODES = ("full", "fixed_lateral", "fixed_depth", "iso", "none")
KERNELS = ("huber", "cauchy", "tukey", "gm")


class CovariantSim3LoopVerifier(Sim3LoopVerifier):
    def __init__(self, *args, sigma_px=1.0, weight_mode="full", kernel="gm",
                 kernel_delta=REFINE_HUBER_DELTA, depth_ratio=0.0,
                 max_depth_ratio=0.0, consistency_chi=2.80,
                 min_consistent_fraction=0.7, max_median_residual=3.0,
                 max_direction_z=0.0, whitened_iterations=0, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(sigma_px) or sigma_px <= 0.0:
            raise ValueError("sigma_px must be positive and finite")
        if weight_mode not in WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}")
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        if not np.isfinite(kernel_delta) or kernel_delta <= 0.0:
            raise ValueError("kernel_delta must be positive and finite")
        for name, value in (("depth_ratio", depth_ratio),
                            ("max_depth_ratio", max_depth_ratio)):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if weight_mode == "fixed_depth" and depth_ratio <= 0.0:
            raise ValueError("weight_mode=fixed_depth requires depth_ratio > 0")
        if not np.isfinite(consistency_chi) or consistency_chi <= 0.0:
            raise ValueError("consistency_chi must be positive and finite")
        if not 0.0 < min_consistent_fraction <= 1.0:
            raise ValueError("min_consistent_fraction must be in (0, 1]")
        if not np.isfinite(max_median_residual) or max_median_residual <= 0.0:
            raise ValueError("max_median_residual must be positive and finite")
        if max_direction_z < 0.0 or not np.isfinite(max_direction_z):
            raise ValueError("max_direction_z must be finite and non-negative")
        if max_direction_z > 0.0 and self.pose_fn is None:
            raise ValueError("direction z gate requires pose_fn")
        if (not np.isfinite(whitened_iterations)
                or float(whitened_iterations) != int(whitened_iterations)
                or int(whitened_iterations) < 0):
            raise ValueError("whitened_iterations must be a non-negative integer")
        self.sigma_px = float(sigma_px)
        self.weight_mode = weight_mode
        self.kernel = kernel
        self.kernel_delta = float(kernel_delta)
        self.depth_ratio = None if depth_ratio <= 0.0 else float(depth_ratio)
        self.max_depth_ratio = (None if max_depth_ratio <= 0.0
                                else float(max_depth_ratio))
        self.consistency_chi = float(consistency_chi)
        self.min_consistent_fraction = float(min_consistent_fraction)
        self.max_median_residual = float(max_median_residual)
        self.max_direction_z = float(max_direction_z)
        self.whitened_iterations = int(whitened_iterations)
        self.covariances = {}
        self.lateral_variance = {}
        self.last_consistency = None
        self.n_inconsistent_rejected = 0
        self.n_direction_z_rejected = 0
        self.n_uncertain_abstained = 0

    def _cloud(self, endpoint):
        cached = self._clouds.get(endpoint)
        if cached is not None or endpoint in self._clouds:
            self._clouds.move_to_end(endpoint)
            return cached
        cloud, chosen = build_cloud_from_windows_detailed(
            self._candidate_windows(endpoint), self.match_fn,
            self.camera_matrix, min_parallax_deg=self.min_parallax_deg)
        covariances = {}
        lateral_variance = None
        if chosen is not None:
            (_first, _last, rotation, translation, forward,
             p_first, p_last, tri) = chosen
            lateral = []
            for keypoint_first, keypoint_last, valid in zip(p_first, p_last,
                                                            tri.valid):
                if not valid:
                    continue
                cov = triangulation_covariance(
                    keypoint_first, keypoint_last, rotation, translation,
                    self.camera_matrix, sigma_px=self.sigma_px,
                    min_parallax_deg=self.min_parallax_deg)
                key = point_key(keypoint_first if forward else keypoint_last)
                if cov is not None and not forward:
                    cov = rotation @ cov @ rotation.T
                covariances[key] = cov
                point = cloud.get(key)
                if cov is not None and point is not None:
                    norm = float(np.linalg.norm(point))
                    if norm >= 1e-12:
                        ray = point / norm
                        lateral.append((float(np.trace(cov))
                                        - float(ray @ cov @ ray)) / 2.0)
            lateral_variance = float(np.median(lateral)) if lateral else None
        self._clouds[endpoint] = cloud
        self.covariances[endpoint] = covariances
        self.lateral_variance[endpoint] = lateral_variance
        while len(self._clouds) > self.cache_size:
            old, _ = self._clouds.popitem(last=False)
            self.covariances.pop(old, None)
            self.lateral_variance.pop(old, None)
        return cloud

    def _cov_array(self, endpoint, keys, points):
        lateral = self.lateral_variance.get(endpoint)
        if self.weight_mode != "full" and self.weight_mode != "none" and lateral is None:
            return np.full((len(keys), 3, 3), np.nan)
        covariances = self.covariances.get(endpoint, {})
        out = []
        for key, point in zip(keys, points):
            cov = covariances.get(key)
            if cov is None:
                out.append(np.full((3, 3), np.nan))
                continue
            simplified = simplify_covariance(
                cov, np.asarray(point, dtype=float), self.weight_mode,
                lateral, self.depth_ratio, self.max_depth_ratio)
            out.append(simplified if simplified is not None
                       else np.full((3, 3), np.nan))
        return np.array(out)

    def _whitened_norms(self, A, B, Ca, Cb, scale, rotation, translation,
                        valid):
        """Whitened residual norms, ``inf`` for tracks without a covariance."""
        residual = B - (scale * (A @ rotation.T) + translation)
        cov = Cb + scale * scale * (rotation @ Ca @ rotation.T)
        norms = np.full(len(A), np.inf)
        for index in np.flatnonzero(valid):
            try:
                chol = np.linalg.cholesky(cov[index])
                whitened = np.linalg.solve(chol, residual[index])
            except np.linalg.LinAlgError:
                continue  # leave this track as an outlier, keep the rest
            value = float(np.linalg.norm(whitened))
            if np.isfinite(value):
                norms[index] = value
        return norms

    def _refine_mask(self, A, B, Ca, Cb, mask, scale, rotation, translation):
        start = Sim3Result(True, scale, rotation, translation, mask, 0.0, "")
        return refine_sim3(A, B, Ca, Cb, start, kernel=self.kernel,
                           huber_delta=self.kernel_delta)

    def _fit(self, a, b, Xa, Xb, joined):
        self.last_consistency = None
        base = super()._fit(a, b, Xa, Xb, joined)
        if not base.ok:
            return base
        keys_a = [key for key, _ in joined]
        keys_b = [key for _, key in joined]
        Ca = self._cov_array(a, keys_a, Xa)
        Cb = self._cov_array(b, keys_b, Xb)
        valid = (np.isfinite(Ca).all(axis=(1, 2))
                 & np.isfinite(Cb).all(axis=(1, 2)))
        # Keep the comparison population fixed: if any fitted inlier lacks a
        # usable covariance, keep the base fit and fall back to the raw gate.
        if not np.all(valid[base.inliers]):
            return base
        A = np.asarray(Xa, dtype=float)
        B = np.asarray(Xb, dtype=float)
        refined = refine_sim3(A, B, Ca, Cb, base, kernel=self.kernel,
                              huber_delta=self.kernel_delta)
        if not refined["ok"]:
            return base
        scale = refined["scale"]
        rotation = refined["rotation"]
        translation = refined["translation"]
        direction_std_deg = refined["direction_std_deg"]
        inlier_mask = base.inliers
        # Optional whitened RANSAC / IRLS: reclassify inliers by their
        # covariance-whitened residual and refit, so depth-uncertain tracks are
        # judged in the same metric as the objective. The set may grow (a
        # depth-uncertain Euclid-outlier is kept) or shrink, so the returned
        # inlier count is not bounded by the raw Euclidean RANSAC inliers.
        for _ in range(self.whitened_iterations):
            norms = self._whitened_norms(A, B, Ca, Cb, scale, rotation,
                                         translation, valid)
            candidate = valid & (norms <= self.consistency_chi)
            if int(candidate.sum()) < self.min_tracks:
                break
            step = self._refine_mask(A, B, Ca, Cb, candidate, scale, rotation,
                                     translation)
            if not step["ok"]:
                break
            scale = step["scale"]
            rotation = step["rotation"]
            translation = step["translation"]
            direction_std_deg = step["direction_std_deg"]
            if np.array_equal(candidate, inlier_mask):
                break
            inlier_mask = candidate
        # Re-derive the mask from the returned model so the two agree even if
        # the loop stopped without a fixed point.  With whitened_iterations=0
        # the raw inliers are kept so the non-IRLS path is unchanged.
        norms = self._whitened_norms(A, B, Ca, Cb, scale, rotation, translation,
                                     valid)
        finite_norms = norms[np.isfinite(norms)]
        if len(finite_norms) == 0:
            return base
        if self.whitened_iterations > 0:
            candidate = valid & (norms <= self.consistency_chi)
            if int(candidate.sum()) >= self.min_tracks:
                inlier_mask = candidate
        if int(inlier_mask.sum()) == 0:
            return base
        self.last_consistency = {
            "n": int(len(finite_norms)),
            "n_skipped": int(len(joined) - len(finite_norms)),
            "f_in": float(np.mean(finite_norms <= self.consistency_chi)),
            "rho_med": float(np.median(finite_norms)),
            "direction_std_deg": direction_std_deg,
        }
        residual = B - (scale * (A @ rotation.T) + translation)
        median_residual = float(np.median(np.linalg.norm(
            residual[inlier_mask], axis=1)))
        return Sim3Result(True, scale, rotation, translation, inlier_mask,
                          median_residual, "")

    def _abstain(self, reason):
        if reason == "uncertain":
            self.n_abstain += 1
            self.n_uncertain_abstained += 1
            return self.abstain_policy == "accept"
        return super()._abstain(reason)

    def _decide(self, a, b, fit, Xa, Xb, joined, cloud_a, cloud_b):
        if not fit.ok:
            self.n_reject += 1
            return False
        self.last_scale = float(fit.scale)
        self.scale_ratio_hist[sum(self.last_scale >= edge for edge in
                                  SCALE_RATIO_BIN_EDGES)] += 1
        self.last_translation_angle_deg = None
        # Inlier-count gate, applied to the (possibly whitened-reclassified)
        # inliers returned by ``_fit``.
        if int(fit.inliers.sum()) < self.gate:
            self.n_reject += 1
            return False
        stats = self.last_consistency
        if stats is not None and (
                stats["f_in"] < self.min_consistent_fraction
                or stats["rho_med"] > self.max_median_residual):
            self.n_reject += 1
            self.n_inconsistent_rejected += 1
            return False
        direction_gate = (self.max_translation_angle_deg is not None
                          or self.max_direction_z > 0.0)
        if direction_gate:
            pose = self.pose_fn(a, b)
            if pose is None or not pose.get("ok") or pose.get("t") is None:
                return self._abstain("no_pose")
            angle = translation_angle_deg(fit.translation, pose.get("t"))
            self.last_translation_angle_deg = angle
            if angle is None:
                return self._abstain("no_pose")
            if (self.max_translation_angle_deg is not None
                    and angle > self.max_translation_angle_deg):
                self.n_reject += 1
                self.n_direction_rejected += 1
                return False
            if self.max_direction_z > 0.0:
                sigma = stats["direction_std_deg"] if stats is not None else None
                if sigma is None or not np.isfinite(sigma) or sigma <= 0.0:
                    return self._abstain("uncertain")
                if angle / sigma > self.max_direction_z:
                    self.n_reject += 1
                    self.n_direction_z_rejected += 1
                    return False
        self.n_accept += 1
        return True

    def __call__(self, a, b):
        self.last_consistency = None
        return super().__call__(a, b)
