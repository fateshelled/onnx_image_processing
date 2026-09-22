"""Measure whether local-track Sim(3) separates true and false loop matches.

This is deliberately offline: ground truth is used only to label candidates
after estimation.  Each loop endpoint is reconstructed from its own local
feature track: the densest backward/forward window (up to ``--local-strides``
stride pairs, cached odometry composed per window; ties go to the longer
baseline).  Only one window is used because the cached per-stride translations
are unit-norm essential-matrix directions, so windows of different lengths do
not share one metric gauge.  The fitted similarity scale is therefore
observable and never reuses the pose graph's scale estimate. (Within one
window the composed pose is only a uniform scaling of the physical pose when
the per-step physical translations have equal norm; the Sim(3) fit absorbs
that single scale.)  A single stride pair has too little parallax on these
sequences, which is why longer windows are preferred when equally dense.

Candidates are pre-filtered by the cached appearance inlier ratio
(``--appearance-min``, the graph's loop gate) and then subsampled uniformly at
random per class, so the selection is not ranked by the Sim(3) score itself.
RANSAC searches with a five-inlier floor (one more than its four-point sample,
so an exact minimal fit alone earns nothing) and the best model's inlier count
is the continuous score, so the reported AUC covers rejected candidates
(score 0) and is not conditioned on acceptance; ``auc_inlier_count`` is the
primary metric (the inlier ratio is not comparable across candidates), an AUC
conditional on candidates that produced any model is also reported
(``*_fitted``), and ``--min-inliers`` is the deployment-style accept gate.
Loop correspondences use all raw matches (the
pair pose is only a sanity gate; Sim(3) RANSAC rejects outliers), because
essential-matrix inliers alone are too few to overlap the local tracks
reliably.  The match cache must be stride-aligned with the ``.npz`` cache (see
``build_cache``); otherwise the GT labels and odometry slots silently shift.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from eval.rustuna_tune_loop import DEFAULT_ARGS, load_cache  # noqa: E402
from eval.eval_tum_vo import estimate_pose_from_matches, intrinsics_for  # noqa: E402
from eval.torch_sinkhorn import TorchSinkhornMatcher  # noqa: E402
from vo.onnx_matcher import extract_matches  # noqa: E402
from vo.pose_estimation import CameraIntrinsics  # noqa: E402
from vo.sim3_verification import sim3_ransac, triangulate_local  # noqa: E402


def _details(c, matcher, cam, a, b, args, require_pose=True):
    if a not in c["feat"] or b not in c["feat"]:
        return None
    ka, da = c["feat"][a]
    kb, db = c["feat"][b]
    P = matcher.match_probs(da[0], db[0])
    pa, pb, _ = extract_matches(ka, kb, P[None], args.match_threshold,
                                args.max_matches, args.dbin)
    if not require_pose:
        return pa, pb, None
    pose = estimate_pose_from_matches(pa, pb, cam, args)
    if not pose.get("ok"):
        return None
    # Keep all raw matches: the pair pose is only a sanity gate, and the
    # Sim(3) RANSAC below is the stage that has to reject outliers.
    return pa, pb, pose


def _point_key(point):
    # The loop pair and the local stride pair share keypoint arrays, so the
    # rounded coordinates act as a join key across float32/float64 copies.
    # Collisions resolve last-wins, which is harmless for identical tracks.
    return tuple(np.round(np.asarray(point, dtype=float), 4))


def _compose_odom(odom, first_slot, last_slot):
    """Compose cached odometry into the pose from first_slot to last_slot.

    Each entry ``odom[n]`` maps frame ``n * stride`` to ``(n + 1) * stride``
    with ``x_next = R @ x_prev + t``.  Returns ``None`` when any entry in the
    window is missing or invalid.
    """
    rotation = np.eye(3)
    translation = np.zeros(3)
    for slot in range(first_slot, last_slot):
        if slot < 0 or slot >= len(odom):
            return None
        entry = odom[slot]
        if (not entry.get("ok") or entry.get("R") is None
                or entry.get("t") is None):
            return None
        R = np.asarray(entry["R"], dtype=float).reshape(3, 3)
        t = np.asarray(entry["t"], dtype=float).reshape(3)
        rotation = R @ rotation
        translation = R @ translation + t
    return rotation, translation


def _local_windows(c, endpoint, stride, window_strides):
    """List local track windows around an endpoint, longest baseline first.

    Each window is either backward (earlier frame -> endpoint) or forward
    (endpoint -> later frame) with up to ``window_strides`` stride pairs, with
    its composed odometry in the cached (unit-norm) step gauge.  Windows of
    different lengths do not share one metric gauge, so the caller must use a
    single window; this helper only enumerates candidates.  Windows whose
    odometry is invalid are skipped.
    """
    endpoint_slot = endpoint // stride
    windows = []
    for steps in range(window_strides, 0, -1):
        first_slot = endpoint_slot - steps
        if first_slot >= 0:
            pose = _compose_odom(c["odom"], first_slot, endpoint_slot)
            if pose is not None:
                windows.append((endpoint - steps * stride, endpoint,
                                pose[0], pose[1], False))
        last_slot = endpoint_slot + steps
        pose = _compose_odom(c["odom"], endpoint_slot, last_slot)
        if pose is not None:
            windows.append((endpoint, endpoint + steps * stride,
                            pose[0], pose[1], True))
    return windows


def _local_cloud(c, matcher, cam, endpoint, stride, window_strides, args,
                 min_parallax_deg=1.0):
    """Reconstruct the endpoint from its densest local window.

    Exactly one window feeds the cloud: the one with the most valid tracks
    (ties go to the longer baseline, because windows are enumerated longest
    first and only strictly larger clouds replace the current best).  Using
    one window keeps a single scale gauge, since the cached per-stride
    translations are unit-norm essential-matrix directions rather than
    metrically consistent displacements, and clouds from windows of different
    lengths cannot be treated as the same gauge.  All candidate windows are
    scanned so that a sparse long window does not hide a denser short one.
    Baseline conditioning is not part of the score; among equally dense
    windows the longer baseline wins (enumeration order).

    The cached odometry is used deliberately (instead of the re-matched pair
    pose): an essential-matrix translation is scale-free, so only the cache
    keeps the local cloud comparable between endpoints.
    """
    best = None
    for first_frame, last_frame, rotation, translation, forward in _local_windows(
            c, endpoint, stride, window_strides):
        pair = _details(c, matcher, cam, first_frame, last_frame, args,
                        require_pose=False)
        if pair is None:
            continue
        p_first, p_last, _ = pair
        tri = triangulate_local(p_first, p_last, rotation, translation, cam.K,
                                min_parallax_deg=min_parallax_deg)
        if forward:
            # Camera 0 is the endpoint itself, so the points are already in
            # the endpoint frame and the endpoint keypoints are the first
            # argument.
            points_endpoint = tri.points
            keys = p_first
        else:
            # Move points from the previous camera into the endpoint camera.
            # The map key is the endpoint keypoint shared with the loop
            # correspondence.
            points_endpoint = (rotation @ tri.points.T).T + translation
            keys = p_last
        cloud = {_point_key(keypoint): point
                 for keypoint, point, valid in
                 zip(keys, points_endpoint, tri.valid) if valid}
        if best is None or len(cloud) > len(best):
            best = cloud
    return best or None


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _jsonable_options(opts):
    return {key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(opts).items()}


def _candidate_seed(seed, a, b):
    """Derive an independent RANSAC seed per candidate pair."""
    return int((seed * 1_000_003 + a * 1_009 + b) % (2 ** 31 - 1))


def _error_report(seq, exc, opts):
    """Per-sequence failure report with the same keys as a success report.

    Counters and metrics that cannot be evaluated are ``None`` (unknown),
    which is distinct from the zero values a successful empty run would give.
    """
    return {
        "sequence": seq, "error": f"{type(exc).__name__}: {exc}",
        "match_cache": None, "elapsed_sec": None,
        "options": _jsonable_options(opts),
        "candidate_pool": None, "eligible": None, "selected": 0,
        "evaluated": 0, "fitted": 0, "true_fitted": 0, "false_fitted": 0,
        "accepted": 0, "true_accepted": 0, "false_accepted": 0,
        "true_rejected": 0, "false_rejected": 0, "accept_precision": None,
        "auc_inlier_count": None, "auc_inlier_count_fitted": None,
        "auc_inlier_ratio": None, "auc_inlier_ratio_fitted": None,
        "auc_negative_residual_fitted": None, "rows": [],
    }


def evaluate_sequence(seq, cache_dir, dataset_root, matcher, opts):
    started = time.perf_counter()
    c = load_cache(cache_dir, seq)
    stride = int(c["stride"])
    fx, fy, cx, cy = intrinsics_for(dataset_root, seq,
                                    (opts.fx, opts.fy, opts.cx, opts.cy))
    cam = CameraIntrinsics(fx, fy, cx, cy, opts.width, opts.height)
    match_path = Path(cache_dir) / f"match_cache_{seq}_torch.pkl"
    if not match_path.exists():
        # Some sequences only have the numpy matcher cache; same schema.
        match_path = Path(cache_dir) / f"match_cache_{seq}_numpy.pkl"
    with match_path.open("rb") as handle:
        cached_matches = pickle.load(handle)
    pose_args = SimpleNamespace(**vars(DEFAULT_ARGS))

    candidates = []
    for (a, b), result in cached_matches.items():
        a, b = int(a), int(b)
        if a % stride or b % stride:
            raise ValueError(
                "match cache keys must be stride-aligned "
                f"(stride={stride}): ({a}, {b})")
        if (not result.get("ok") or b - a < opts.min_gap
                or a - stride < 0):
            continue
        distance = float(np.linalg.norm(
            c["gt_pos"][a // stride] - c["gt_pos"][b // stride]))
        if distance <= opts.true_distance:
            label = True
        elif distance >= opts.false_distance:
            label = False
        else:
            continue
        candidates.append((float(result.get("inlier_ratio", 0.0)), a, b,
                           label, distance))
    # Decision-relevant population: candidates the appearance gate would let
    # through, then a uniform random subsample per class (seeded) instead of a
    # score-ranked top list, so the selection is not biased towards pairs that
    # already look easy to the Sim(3) score.
    eligible = [cand for cand in candidates if cand[0] >= opts.appearance_min]
    sample_rng = np.random.default_rng(opts.seed)
    class_cap = max(1, opts.max_candidates // 2)
    by_class = {True: [], False: []}
    for candidate in eligible:
        by_class[candidate[3]].append(candidate)
    selected = []
    for label in (True, False):
        pool = by_class[label]
        if len(pool) > class_cap:
            take = sample_rng.choice(len(pool), size=class_cap, replace=False)
            selected.extend(pool[int(index)] for index in sorted(take))
        else:
            selected.extend(pool)
    selected.sort(key=lambda candidate: (candidate[1], candidate[2]))

    local_cache = {}
    rows = []
    for _appearance, a, b, label, gt_distance in selected:
        loop = _details(c, matcher, cam, a, b, pose_args)
        if loop is None:
            rows.append({"a": a, "b": b, "label": label,
                         "gt_distance": float(gt_distance), "ok": False,
                         "reason": "no_pose", "n_tracks": 0,
                         "n_inliers": 0, "inlier_ratio": 0.0})
            continue
        pa, pb, _pose = loop
        if a not in local_cache:
            local_cache[a] = _local_cloud(c, matcher, cam, a, stride,
                                          opts.local_strides, pose_args,
                                          opts.min_parallax_deg)
        if b not in local_cache:
            local_cache[b] = _local_cloud(c, matcher, cam, b, stride,
                                          opts.local_strides, pose_args,
                                          opts.min_parallax_deg)
        cloud_a, cloud_b = local_cache[a], local_cache[b]
        if not cloud_a or not cloud_b:
            rows.append({"a": a, "b": b, "label": label,
                         "gt_distance": float(gt_distance), "ok": False,
                         "reason": "no_local_cloud", "n_tracks": 0,
                         "n_inliers": 0, "inlier_ratio": 0.0})
            continue
        Xa, Xb = [], []
        for point_a, point_b in zip(pa, pb):
            ka, kb = _point_key(point_a), _point_key(point_b)
            if ka in cloud_a and kb in cloud_b:
                Xa.append(cloud_a[ka])
                Xb.append(cloud_b[kb])
        if len(Xa) < opts.min_inliers:
            rows.append({"a": a, "b": b, "label": label,
                         "gt_distance": float(gt_distance), "ok": False,
                         "reason": "too_few_tracks", "n_tracks": len(Xa),
                         "n_inliers": 0, "inlier_ratio": 0.0})
            continue
        # Ask RANSAC for the best model it can find and treat the inlier count
        # as a continuous score.  At least five inliers are required so that a
        # trivial exact fit to one four-point sample does not give junk
        # candidates a nonzero floor; the deployment accept decision is the
        # separate gate below.
        fit = sim3_ransac(np.asarray(Xa), np.asarray(Xb),
                          seed=_candidate_seed(opts.seed, a, b),
                          residual_fraction=opts.residual_fraction,
                          max_iterations=opts.iterations,
                          min_inliers=5,
                          min_condition_ratio=opts.min_condition_ratio)
        n_inliers = int(fit.inliers.sum()) if fit.ok else 0
        accepted = bool(fit.ok and n_inliers >= opts.min_inliers)
        rows.append({"a": a, "b": b, "label": label,
                     "gt_distance": float(gt_distance), "ok": accepted,
                     "fit_ok": bool(fit.ok), "reason": fit.reason,
                     "n_tracks": len(Xa), "n_inliers": n_inliers,
                     "inlier_ratio": (n_inliers / len(Xa)) if n_inliers else 0.0,
                     "scale": _finite_or_none(fit.scale),
                     "residual": _finite_or_none(fit.median_residual)})

    labels = [row["label"] for row in rows]
    accepted = [row for row in rows if row["ok"]]
    fitted = [row for row in rows if row.get("fit_ok")]
    fitted_labels = [row["label"] for row in fitted]
    fitted_residual = _auc(
        fitted_labels, [-row["residual"] for row in fitted])
    true_accepted = int(sum(1 for row in rows
                            if row["label"] and row["ok"]))
    false_accepted = int(sum(1 for row in rows
                             if not row["label"] and row["ok"]))
    summary = {
        "sequence": seq,
        "match_cache": match_path.name,
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "options": _jsonable_options(opts),
        "candidate_pool": len(candidates),
        "eligible": len(eligible),
        "selected": len(selected),
        "evaluated": len(rows),
        "fitted": len(fitted),
        "true_fitted": int(sum(fitted_labels)),
        "false_fitted": int(len(fitted) - sum(fitted_labels)),
        "accepted": len(accepted),
        "true_accepted": true_accepted,
        "false_accepted": false_accepted,
        "true_rejected": int(sum(1 for row in rows
                                 if row["label"] and not row["ok"])),
        "false_rejected": int(sum(1 for row in rows
                                  if not row["label"] and not row["ok"])),
        "accept_precision": (true_accepted / len(accepted)
                             if accepted else None),
        # Absolute inlier counts are the primary separation score: the ratio
        # is not comparable across candidates (a 5/5 weak fit scores 1.0).
        "auc_inlier_count": _finite_or_none(
            _auc(labels, [row["n_inliers"] for row in rows])),
        "auc_inlier_count_fitted": _finite_or_none(
            _auc(fitted_labels, [row["n_inliers"] for row in fitted])),
        "auc_inlier_ratio": _finite_or_none(
            _auc(labels, [row["inlier_ratio"] for row in rows])),
        "auc_inlier_ratio_fitted": _finite_or_none(
            _auc(fitted_labels, [row["inlier_ratio"] for row in fitted])),
        "auc_negative_residual_fitted":
            _finite_or_none(fitted_residual),
        "rows": rows,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether local-track Sim(3) separates true and "
                    "false loop matches (GT used for labels only).")
    parser.add_argument("--seq", default="desk,desk2,room")
    parser.add_argument("--cache-dir", default="eval/results/tune_cache_loop")
    parser.add_argument("--dataset-root", default="/home/ubuntu/datasets/tum_rgbd")
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--min-gap", type=int, default=30,
                        help="minimum raw-frame gap between loop endpoints "
                             "(the tuner's loop_min_gap is in node units)")
    parser.add_argument("--appearance-min", type=float, default=0.2,
                        help="keep candidates whose cached appearance "
                             "inlier ratio passes this gate. The graph's 0.5 "
                             "gate leaves no false candidates on desk/desk2, "
                             "so the diagnostic targets the marginal band "
                             "where both classes exist")
    parser.add_argument("--local-strides", type=int, default=2,
                        help="maximum stride pairs per local track window; "
                             "backward/forward candidates are all scanned and "
                             "the densest single window (ties: longer "
                             "baseline) is used")
    parser.add_argument("--min-parallax-deg", type=float, default=1.0)
    parser.add_argument("--min-condition-ratio", type=float, default=1e-2,
                        help="reject near-collinear 4-point samples (1e-3 was "
                             "too permissive; matches the library default)")
    parser.add_argument("--true-distance", type=float, default=0.5)
    parser.add_argument("--false-distance", type=float, default=1.0)
    parser.add_argument("--min-inliers", type=int, default=8,
                        help="accept gate on the best model's inlier count "
                             "(the score/AUC uses the raw count, not this)")
    parser.add_argument("--residual-fraction", type=float, default=0.1,
                        help="RANSAC residual threshold as a fraction of the "
                             "median pairwise target distance; 0.05 turned "
                             "out too tight for short local tracks")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fx", type=float, default=525.0,
                        help="freiburg1 fallback intrinsics; mirror the "
                             "tuner/cache-build flags for other cameras")
    parser.add_argument("--fy", type=float, default=525.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path)
    opts = parser.parse_args()
    matcher = TorchSinkhornMatcher(iterations=20, epsilon=0.05,
                                   unused_score=1.0, distance_type="l2")
    reports = []
    for seq in (name.strip() for name in opts.seq.split(",")):
        if not seq:
            continue
        try:
            reports.append(evaluate_sequence(seq, opts.cache_dir,
                                             opts.dataset_root, matcher, opts))
        except Exception as exc:  # noqa: BLE001 - keep other sequences going
            reports.append(_error_report(seq, exc, opts))
    text = json.dumps(reports, indent=2, allow_nan=False)
    if opts.output:
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        opts.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
