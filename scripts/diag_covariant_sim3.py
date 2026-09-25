"""Replay the online loop candidates and refit their fixed inliers with covariance.

GT is read only for scoring. The production verifier and graph are unchanged.
Usage: .venv/bin/python scripts/diag_covariant_sim3.py --output notes/cov-sim3.json
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

import eval.rustuna_tune_loop as evaluator  # noqa: E402
from eval.eval_tum_vo import intrinsics_for  # noqa: E402
from eval.torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.covariant_sim3 import (  # noqa: E402
    REFINE_FTOL, REFINE_HUBER_DELTA, REFINE_MAXITER, REFINE_MAX_SCALE,
    REFINE_MIN_SCALE, refine_sim3, simplify_covariance,
    triangulation_covariance,
)
from vo.loop_sim3_verifier import (  # noqa: E402
    Sim3LoopVerifier, local_windows, point_key, translation_angle_deg,
)
from vo.pose_estimation import CameraIntrinsics  # noqa: E402
from vo.sim3_verification import sim3_ransac, triangulate_local  # noqa: E402


class DiagnosticVerifier(Sim3LoopVerifier):
    poses = None
    rows = None
    sigma_px = 1.0
    weight_mode = "full"
    depth_ratio = None
    max_depth_ratio = None
    kernel = "huber"
    kernel_delta = None

    def _cloud(self, endpoint):
        if endpoint in self._clouds:
            self._clouds.move_to_end(endpoint)
            return self._clouds[endpoint]
        windows = list(self.window_fn(endpoint)) if self.window_fn else []
        odom_windows = local_windows(self.odom, endpoint, self.stride,
                                     self.window_strides)
        if self.backward_only:
            odom_windows = [w for w in odom_windows if not w[4]]
        seen = {(w[0], w[1]) for w in windows}
        windows.extend(w for w in odom_windows if (w[0], w[1]) not in seen)
        best, chosen = None, None
        for first, last, R, t, forward in windows:
            pair = self.match_fn(first, last)
            if pair is None:
                continue
            p0, p1 = pair
            tri = triangulate_local(p0, p1, R, t, self.camera_matrix,
                                    min_parallax_deg=self.min_parallax_deg)
            points = tri.points if forward else (R @ tri.points.T).T + t
            keys = p0 if forward else p1
            cloud = {point_key(key): point for key, point, valid in
                     zip(keys, points, tri.valid) if valid}
            if best is None or len(cloud) > len(best):
                best, chosen = cloud, (first, last, R, t, forward, p0, p1, tri)
        if chosen is not None:
            first, last, R, t, forward, p0, p1, tri = chosen
            covs = {}
            for a, b, ok in zip(p0, p1, tri.valid):
                if not ok:
                    continue
                cov = triangulation_covariance(
                    a, b, R, t, self.camera_matrix, sigma_px=self.sigma_px,
                    min_parallax_deg=self.min_parallax_deg)
                key = point_key(a if forward else b)
                covs[key] = (cov if forward or cov is None else R @ cov @ R.T)
            self.covariances[endpoint] = covs
            lateral = []
            for key, cov in covs.items():
                point = best[key]
                norm = float(np.linalg.norm(point))
                if cov is None or norm < 1e-12:
                    continue
                ray = point / norm
                lateral.append((float(np.trace(cov)) - float(ray @ cov @ ray)) / 2.0)
            self.lateral_variance[endpoint] = (
                float(np.median(lateral)) if lateral else None)
            self.windows[endpoint] = {"first": first, "last": last,
                                      "forward": forward, "baseline": float(np.linalg.norm(t)),
                                      "n_valid": len(best),
                                       "n_cov_valid": sum(c is not None for c in covs.values()),
                                       "lateral_variance": self.lateral_variance[endpoint],
                                      "median_parallax_deg": float(np.median(tri.parallax_deg[tri.valid])) if tri.valid.any() else None,
                                      "median_depth_baselines": float(np.median(tri.points[tri.valid, 2]) / np.linalg.norm(t)) if tri.valid.any() else None}
            self.observations[endpoint] = chosen
        self._clouds[endpoint] = best or None
        while len(self._clouds) > self.cache_size:
            old, _ = self._clouds.popitem(last=False)
            self.covariances.pop(old, None)
            self.windows.pop(old, None)
            self.observations.pop(old, None)
            self.lateral_variance.pop(old, None)
        return self._clouds[endpoint]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.covariances = {}
        self.windows = {}
        self.observations = {}
        self.lateral_variance = {}

    def _oracle_cloud(self, endpoint, *, metric=False):
        """Triangulate the *same* selected observations with GT local pose.

        GT translation is normalized to the original window baseline length so
        the endpoint gauge and Sim(3) scale constraint are preserved. With
        ``metric=True`` the raw GT baseline is used instead, as a separate
        upper-bound reference that changes the gauge. Returns keypoint-indexed
        points.
        """
        first, last, _R, t, forward, p0, p1, _tri = self.observations[endpoint]
        relative = np.linalg.inv(self.poses[last]) @ self.poses[first]
        gt_t = relative[:3, 3]
        if np.linalg.norm(gt_t) < 1e-9 or np.linalg.norm(t) < 1e-9:
            return {}
        if not metric:
            gt_t = gt_t * (np.linalg.norm(t) / np.linalg.norm(gt_t))
        R = relative[:3, :3]
        tri = triangulate_local(p0, p1, R, gt_t, self.camera_matrix,
                                min_parallax_deg=self.min_parallax_deg)
        points = tri.points if forward else (R @ tri.points.T).T + gt_t
        return {point_key(key): point for key, point, ok in
                zip(p0 if forward else p1, points, tri.valid) if ok}

    def _covariances_for(self, endpoint, keys):
        """Per-point covariances in the endpoint frame, ablated by weight mode.

        A point whose full covariance is missing stays missing in every mode so
        the compared population is identical across ablation arms.
        """
        out = []
        lateral = self.lateral_variance.get(endpoint)
        for key in keys:
            cov = self.covariances[endpoint].get(key)
            point = self._clouds[endpoint][key]
            if cov is None or lateral is None or np.linalg.norm(point) < 1e-12:
                out.append(np.full((3, 3), np.nan))
                continue
            simplified = simplify_covariance(cov, point, self.weight_mode,
                                             lateral, self.depth_ratio,
                                             self.max_depth_ratio)
            out.append(simplified if simplified is not None
                       else np.full((3, 3), np.nan))
        return np.array(out)

    def __call__(self, a, b):
        started = time.perf_counter()
        # First use the unmodified decision path. Replaying the exact RANSAC
        # seed below gives the fitted model and inlier mask for both arms.
        decision = super().__call__(a, b)
        row = {"a": int(a), "b": int(b), "decision": bool(decision),
               "window_a": self.windows.get(a), "window_b": self.windows.get(b),
               "fit": False, "cov_fit": False, "cov_reason": ""}
        if self._clouds.get(a) and self._clouds.get(b):
            pair = self.match_fn(a, b)
            if pair is not None:
                pa, pb = pair
                joined = [(ka, kb) for x, y in zip(pa, pb)
                          if (ka := point_key(x)) in self._clouds[a]
                          and (kb := point_key(y)) in self._clouds[b]]
                if len(joined) >= self.min_tracks:
                    A = np.array([self._clouds[a][ka] for ka, _ in joined])
                    B = np.array([self._clouds[b][kb] for _, kb in joined])
                    from vo.loop_sim3_verifier import candidate_seed
                    fit = sim3_ransac(A, B, seed=candidate_seed(self.seed, a, b),
                                      residual_fraction=self.residual_fraction,
                                      max_iterations=self.iterations,
                                      min_inliers=self.min_tracks,
                                      min_condition_ratio=self.min_condition_ratio)
                    row.update(n_tracks=len(joined), fit=bool(fit.ok),
                               reason=fit.reason, n_inliers=int(fit.inliers.sum()))
                    oa, ob = self._oracle_cloud(a), self._oracle_cloud(b)
                    common = [(ka, kb) for ka, kb in joined if ka in oa and kb in ob]
                    row["n_oracle_common"] = len(common)
                    if len(common) >= self.min_tracks:
                        common_seed = 0
                        original_common = sim3_ransac(
                            np.array([self._clouds[a][ka] for ka, _ in common]),
                            np.array([self._clouds[b][kb] for _, kb in common]),
                            seed=common_seed, residual_fraction=self.residual_fraction,
                            max_iterations=self.iterations,
                            min_inliers=self.min_tracks,
                            min_condition_ratio=self.min_condition_ratio)
                        oracle = sim3_ransac(
                            np.array([oa[ka] for ka, _ in common]),
                            np.array([ob[kb] for _, kb in common]),
                            seed=common_seed, residual_fraction=self.residual_fraction,
                            max_iterations=self.iterations,
                            min_inliers=self.min_tracks,
                            min_condition_ratio=self.min_condition_ratio)
                        row["original_common_fit"] = original_common.ok
                        row["oracle_fit"] = oracle.ok
                        if oracle.ok and original_common.ok:
                            gt = np.linalg.inv(self.poses[b]) @ self.poses[a]
                            row["original_common_angle_deg"] = translation_angle_deg(
                                original_common.translation, gt[:3, 3])
                            row["oracle_angle_deg"] = translation_angle_deg(oracle.translation, gt[:3, 3])
                        # Metric-GT baseline upper-bound reference. This changes
                        # the endpoint gauge and is reported separately.
                        ma, mb = (self._oracle_cloud(a, metric=True),
                                  self._oracle_cloud(b, metric=True))
                        mcommon = [(ka, kb) for ka, kb in joined if ka in ma and kb in mb]
                        if len(mcommon) >= self.min_tracks:
                            metric_fit = sim3_ransac(
                                np.array([ma[ka] for ka, _ in mcommon]),
                                np.array([mb[kb] for _, kb in mcommon]),
                                seed=0, residual_fraction=self.residual_fraction,
                                max_iterations=self.iterations,
                                min_inliers=self.min_tracks,
                                min_condition_ratio=self.min_condition_ratio)
                            if metric_fit.ok:
                                gt = np.linalg.inv(self.poses[b]) @ self.poses[a]
                                row["oracle_metric_angle_deg"] = translation_angle_deg(
                                    metric_fit.translation, gt[:3, 3])
                    if fit.ok:
                        Ca = self._covariances_for(a, [ka for ka, _ in joined])
                        Cb = self._covariances_for(b, [kb for _, kb in joined])
                        valid = np.isfinite(Ca).all(axis=(1, 2)) & np.isfinite(Cb).all(axis=(1, 2))
                        row["n_cov_inliers"] = int((valid & fit.inliers).sum())
                        row["n_inliers"] = int(fit.inliers.sum())
                        # Hold the exact original inlier population fixed.
                        # If a covariance cannot be computed, report dropout
                        # rather than silently changing the comparison set.
                        if not np.all(valid[fit.inliers]):
                            weighted = {"ok": False, "reason": "missing_covariance"}
                        else:
                            weighted = refine_sim3(
                                A, B, Ca, Cb, fit, kernel=self.kernel,
                                huber_delta=(REFINE_HUBER_DELTA
                                             if self.kernel_delta is None
                                             else self.kernel_delta))
                        if weighted["ok"]:
                            row["cov_fit"] = True
                            row["cov_scale"] = weighted["scale"]
                            row["cov_translation_norm"] = float(np.linalg.norm(weighted["translation"]))
                            row["cov_jacobian_condition"] = weighted["jacobian_condition"]
                            row["cost_before"] = weighted["cost_before"]
                            row["cost_after"] = weighted["cost_after"]
                            row["whitened_norm_sq_inlier_mean"] = weighted["whitened_norm_sq_inlier_mean"]
                            row["cov_direction_std_deg"] = weighted["direction_std_deg"]
                        else:
                            row["cov_reason"] = weighted["reason"]
                        Ta, Tb = self.poses[a], self.poses[b]
                        gt = np.linalg.inv(Tb) @ Ta
                        distance = float(np.linalg.norm(gt[:3, 3]))
                        row["gt_distance_m"] = distance
                        row["baseline_angle_deg"] = translation_angle_deg(fit.translation, gt[:3, 3])
                        row["baseline_scale"] = float(fit.scale)
                        if weighted["ok"]:
                            row["cov_angle_deg"] = translation_angle_deg(weighted["translation"], gt[:3, 3])
                            row["cov_rotation_error_deg"] = float(np.degrees(np.linalg.norm(
                                cv2.Rodrigues(weighted["rotation"] @ gt[:3, :3].T)[0])))
                        row["baseline_rotation_error_deg"] = float(np.degrees(np.linalg.norm(
                            cv2.Rodrigues(fit.rotation @ gt[:3, :3].T)[0])))
                        recover = self.pose_fn(a, b)
                        row["recover_angle_deg"] = translation_angle_deg(
                            recover["t"], gt[:3, 3]) if recover.get("ok") and recover.get("t") is not None else None
        row["elapsed_sec"] = time.perf_counter() - started
        self.rows.append(row)
        return decision


def _median(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(finite)) if finite else None


def summarize_rows(rows):
    """Paired comparison restricted to candidates where both arms fit.

    The baseline arm is scored on the *same* rows as the covariance arm, so
    covariance dropout cannot masquerade as an improvement. Dropped candidates
    and their baseline quality are reported separately.
    """
    fitted = [r for r in rows if r.get("fit")]
    cov_fitted = [r for r in rows if r.get("cov_fit")]
    dropped = [r for r in fitted if not r.get("cov_fit")]
    reasons = sorted({r.get("cov_reason") for r in dropped})
    summary = {
        "n_candidates": len(rows),
        "n_fit": len(fitted),
        "n_cov_fit": len(cov_fitted),
        "n_cov_dropout": len(dropped),
        "cov_dropout_reasons": {
            reason: sum(1 for r in dropped if r.get("cov_reason") == reason)
            for reason in reasons},
        "cov_dropout_baseline_angle_median_deg": _median(
            [r.get("baseline_angle_deg") for r in dropped]),
        "whitened_norm_sq_inlier_median": _median(
            [r.get("whitened_norm_sq_inlier_mean") for r in cov_fitted]),
        "lateral_variance_median": _median(
            [w.get("lateral_variance")
             for r in cov_fitted for w in (r.get("window_a"), r.get("window_b"))
             if w and w.get("lateral_variance") is not None]),
        "calibration": {},
        "eval": {},
    }
    for tag, threshold in (("gt_ge_5cm", 0.05), ("gt_ge_10cm", 0.10)):
        subset = [r for r in cov_fitted if r.get("gt_distance_m", 0.0) >= threshold]
        if not subset:
            summary["eval"][tag] = {"n": 0}
            continue
        base_angle = [r["baseline_angle_deg"] for r in subset]
        cov_angle = [r["cov_angle_deg"] for r in subset]
        recover = [r.get("recover_angle_deg") for r in subset]
        base_rot = [r["baseline_rotation_error_deg"] for r in subset]
        cov_rot = [r["cov_rotation_error_deg"] for r in subset]

        def better(pairs):
            return sum(a is not None and b is not None and a < b for a, b in pairs)

        summary["eval"][tag] = {
            "n": len(subset),
            "baseline_angle_median_deg": _median(base_angle),
            "cov_angle_median_deg": _median(cov_angle),
            "recover_angle_median_deg": _median(recover),
            "cov_better_count": better(zip(cov_angle, base_angle)),
            "baseline_rotation_median_deg": _median(base_rot),
            "cov_rotation_median_deg": _median(cov_rot),
            "cov_rotation_better_count": better(zip(cov_rot, base_rot)),
        }
        predicted = [r.get("cov_direction_std_deg") for r in subset]
        pairs = [(a, p) for a, p in zip(cov_angle, predicted)
                 if a is not None and p is not None and p > 0.0]
        if pairs:
            summary["calibration"][tag] = {
                "n": len(pairs),
                "direction_std_median_deg": _median([p for _, p in pairs]),
                "coverage_ratio_median": _median([a / p for a, p in pairs]),
                "within_1sigma_count": sum(a <= p for a, p in pairs),
            }
    oracle = [r for r in rows if r.get("oracle_fit") and r.get("original_common_fit")]
    summary["oracle"] = {
        "n": len(oracle),
        "n_original_common_fit": sum(1 for r in rows if r.get("original_common_fit")),
        "n_oracle_fit": sum(1 for r in rows if r.get("oracle_fit")),
        "n_oracle_only": sum(1 for r in rows
                             if r.get("oracle_fit") and not r.get("original_common_fit")),
        "n_original_only": sum(1 for r in rows
                               if r.get("original_common_fit") and not r.get("oracle_fit")),
    }
    if oracle:
        base = [r["original_common_angle_deg"] for r in oracle]
        orc = [r["oracle_angle_deg"] for r in oracle]
        metric = [r.get("oracle_metric_angle_deg") for r in oracle]
        summary["oracle"].update({
            "original_common_angle_median_deg": _median(base),
            "oracle_angle_median_deg": _median(orc),
            "oracle_metric_angle_median_deg": _median(metric),
            "oracle_better_count": sum(o < b for o, b in zip(orc, base)),
        })
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="desk,desk2,room")
    ap.add_argument("--dataset-root", default="/home/ubuntu/datasets/tum_rgbd")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "eval/results/tune_cache_loop")
    ap.add_argument("--sigma-px", type=float, default=1.0)
    ap.add_argument("--weight-mode", default="full",
                    choices=["full", "fixed_lateral", "fixed_depth", "iso", "none"],
                    help="per-point covariance ablation for diagnostics")
    ap.add_argument("--depth-ratio", type=float, default=None,
                    help="sigma_depth/sigma_lateral for --weight-mode fixed_depth")
    ap.add_argument("--max-depth-ratio", type=float, default=None,
                    help="cap sigma_depth/sigma_lateral for full/fixed_lateral")
    ap.add_argument("--kernel", default="huber",
                    choices=["huber", "cauchy", "tukey", "gm"])
    ap.add_argument("--kernel-delta", type=float, default=None,
                    help="scale parameter for the robust kernel (default 3.0)")
    ap.add_argument("--baseline", type=Path, default=ROOT / "notes/20260923-p0a-gt-direction-comparison.json")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not np.isfinite(args.sigma_px) or args.sigma_px <= 0:
        ap.error("--sigma-px must be finite and positive")
    if args.weight_mode == "fixed_depth" and (
            args.depth_ratio is None or not np.isfinite(args.depth_ratio)
            or args.depth_ratio <= 0):
        ap.error("--depth-ratio must be finite and positive for fixed_depth")
    if args.max_depth_ratio is not None and (
            not np.isfinite(args.max_depth_ratio) or args.max_depth_ratio <= 0):
        ap.error("--max-depth-ratio must be finite and positive")
    if args.kernel_delta is not None and (
            not np.isfinite(args.kernel_delta) or args.kernel_delta <= 0):
        ap.error("--kernel-delta must be finite and positive")
    matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")
    original = evaluator.Sim3LoopVerifier
    baseline = json.loads(args.baseline.read_text())
    reports = {}
    for seq in args.seq.split(","):
        seq = seq.strip()
        c = evaluator.load_cache(args.cache_dir, seq)
        fx, fy, cx, cy = intrinsics_for(args.dataset_root, seq,
                                        (525., 525., 320., 240.))
        cam = CameraIntrinsics(fx, fy, cx, cy, 640, 480)
        path = args.cache_dir / f"match_cache_{seq}_torch.pkl"
        if not path.exists():
            path = args.cache_dir / f"match_cache_{seq}_numpy.pkl"
        with path.open("rb") as handle:
            cached = pickle.load(handle)
        frames = evaluator.load_frames(args.dataset_root, seq)
        DiagnosticVerifier.poses = {i: frames[i][1] for i in c["feat"]}
        DiagnosticVerifier.rows = []
        DiagnosticVerifier.sigma_px = args.sigma_px
        DiagnosticVerifier.weight_mode = args.weight_mode
        DiagnosticVerifier.depth_ratio = args.depth_ratio
        DiagnosticVerifier.max_depth_ratio = args.max_depth_ratio
        DiagnosticVerifier.kernel = args.kernel
        DiagnosticVerifier.kernel_delta = args.kernel_delta
        evaluator.Sim3LoopVerifier = DiagnosticVerifier
        try:
            result = evaluator.eval_seq(c, evaluator.SEQ_OPT1_DEFAULTS, cam, matcher, cached)
        finally:
            evaluator.Sim3LoopVerifier = original
        rows = DiagnosticVerifier.rows
        if seq in baseline:
            expected = [(r["a"], r["b"], r["decision"]) for r in baseline[seq]["rows"]]
            actual = [(r["a"], r["b"], r["decision"]) for r in rows]
            if actual != expected:
                raise RuntimeError(f"{seq}: fixed candidate order/decisions differ from baseline: "
                                   f"expected {len(expected)}, actual {len(actual)}; "
                                   f"first mismatch {next(((x, y) for x, y in zip(expected, actual) if x != y), None)}")
            by_key = {(r["a"], r["b"]): r for r in baseline[seq]["rows"]}
            mismatch = [(r["a"], r["b"], r["n_inliers"], by_key[(r["a"], r["b"])].get("sim3_inliers"))
                        for r in rows
                        if r.get("fit") and by_key[(r["a"], r["b"])].get("sim3_fit")
                        and r["n_inliers"] != by_key[(r["a"], r["b"])].get("sim3_inliers")]
            if mismatch:
                raise RuntimeError(f"{seq}: inlier counts differ from baseline: {mismatch[:3]}")
        reports[seq] = {"ATE_median": result["ATE_median"],
                        "verifier_accept": result["verifier_accept"],
                        "verifier_reject": result["verifier_reject"],
                        "prior_candidate_match": seq in baseline,
                        "settings": {"sigma_px": args.sigma_px,
                                     "sigma_px_calibrated": False,
                                     "weight_mode": args.weight_mode,
                                     "depth_ratio": args.depth_ratio,
                                     "max_depth_ratio": args.max_depth_ratio,
                                     "kernel": args.kernel,
                                     "kernel_delta": args.kernel_delta,
                                     "cache_dir": str(args.cache_dir),
                                     "baseline": str(args.baseline),
                                     "objective": "huber_negloglik_logdet",
                                     "refine": {"maxiter": REFINE_MAXITER,
                                                "ftol": REFINE_FTOL,
                                                "huber_delta": REFINE_HUBER_DELTA,
                                                "min_scale": REFINE_MIN_SCALE,
                                                "max_scale": REFINE_MAX_SCALE},
                                     "verifier_defaults": dict(evaluator.SEQ_OPT1_DEFAULTS)},
                        "summary": summarize_rows(rows),
                        "rows": rows}
        print(seq, "candidates", len(rows), "fitted", sum(r["fit"] for r in rows),
              "cov_fitted", sum(r["cov_fit"] for r in rows), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
