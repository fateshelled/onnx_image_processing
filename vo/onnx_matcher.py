"""ONNX Sinkhorn matcher mirroring the eval pipeline's matching/pose setup.

The eval harness (``eval/eval_tum_vo.py``) uses mutual-NN + a dustbin-margin
filter + top-K matches, then a MAGSAC Essential-matrix solve. This module
provides the same matching as a reusable callable so the online graph
(``vo.online_graph``) and the sample script can reproduce the eval behaviour
instead of duplicating (and diverging from) it.
"""

import numpy as np
import cv2

from .outlier_filters import dustbin_margin_filter
from .pose_estimation import estimate_pose_ransac


def extract_match_indices(kpts1, kpts2, P, threshold=0.1, max_matches=1024,
                          dbin_margin=0.1):
    """Return canonical feature indices after the eval match filters.

    ``P`` is the (1, K+1, K+1) Sinkhorn matrix; keypoints are (1, K, 2) as
    (y, x).  The returned indices refer to those original per-frame arrays and
    therefore remain suitable as feature IDs for multi-pair track building.
    """
    kpts1 = np.asarray(kpts1)
    kpts2 = np.asarray(kpts2)
    P = np.asarray(P)
    if (kpts1.ndim != 3 or kpts2.ndim != 3 or kpts1.shape[0] != 1
            or kpts2.shape[0] != 1 or kpts1.shape[2] != 2
            or kpts2.shape[2] != 2):
        raise ValueError("keypoints must have shape (1, K, 2)")
    if P.ndim != 3 or P.shape[0] != 1:
        raise ValueError("Sinkhorn probabilities must have shape (1, K+1, K+1)")
    if not np.isfinite(threshold) or not np.isfinite(dbin_margin):
        raise ValueError("match thresholds must be finite")
    if not isinstance(max_matches, (int, np.integer)) or max_matches < 0:
        raise ValueError("max_matches must be a non-negative integer")
    if not np.all(np.isfinite(P)):
        raise ValueError("Sinkhorn probabilities must be finite")
    P = P[0]
    k1 = kpts1[0]
    k2 = kpts2[0]
    K = k1.shape[0]
    if k2.shape[0] != K or P.shape != (K + 1, K + 1):
        raise ValueError("keypoint and Sinkhorn dimensions must agree")
    if K == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=float))
    Pc = P[:K, :K]
    max_j = np.argmax(Pc, axis=1)
    max_i = np.argmax(Pc, axis=0)
    mutual = np.arange(K) == max_i[max_j]
    scores = Pc[np.arange(K), max_j]
    dbin = dustbin_margin_filter(P, dbin_margin)
    valid_k1 = np.all(np.isfinite(k1), axis=1) & np.all(k1 >= 0, axis=1)
    valid_k2 = np.all(np.isfinite(k2), axis=1) & np.all(k2 >= 0, axis=1)
    pad = valid_k1 & valid_k2[max_j]
    valid = mutual & dbin & pad & (scores >= threshold)
    idx_i = np.where(valid)[0]
    if len(idx_i) == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=float))
    j = max_j[idx_i]
    sc = scores[idx_i]
    order = np.lexsort((j, idx_i, -sc))[:max_matches]
    idx_i = idx_i[order]
    j = j[order]
    return idx_i, j, sc[order]


def extract_matches(kpts1, kpts2, P, threshold=0.1, max_matches=1024,
                    dbin_margin=0.1):
    """Mutual-NN + dustbin-margin filter + top-K (eval convention)."""
    idx_i, idx_j, scores = extract_match_indices(
        kpts1, kpts2, P, threshold, max_matches, dbin_margin)
    return kpts1[0][idx_i], kpts2[0][idx_j], scores


class OnnxSessionMatcher:
    """Callable matcher: ``match(img_a, img_b) -> dict``.

    ``output_indices`` maps ``k1``/``k2``/``probs`` to the session output
    order (see ``sample.visual_odometry._output_indices``). The returned dict
    has keys ``ok``, ``R``, ``t``, ``inlier_ratio`` and ``n_matches``.
    """

    def __init__(self, session, cam, input_names, output_indices,
                 match_threshold=0.1, max_matches=1024, dbin=0.1,
                 method="magsac", ransac_threshold=1.4, min_matches=20,
                 min_inlier_ratio=0.0):
        self.session = session
        self.cam = cam
        self.in0, self.in1 = input_names[0], input_names[1]
        self.oi = output_indices
        self.match_threshold = match_threshold
        self.max_matches = max_matches
        self.dbin = dbin
        self.method = method
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches
        self.min_inlier_ratio = min_inlier_ratio
        # When set, the most recent match stores keypoints/mask for display.
        self.debug_display = False
        self.last = None

    def match(self, img_a, img_b):
        outs = self.session.run(None, {self.in0: img_a, self.in1: img_b})
        k1 = outs[self.oi["k1"]]
        k2 = outs[self.oi["k2"]]
        P = outs[self.oi["probs"]]
        mk1, mk2, _sc = extract_matches(
            k1, k2, P, self.match_threshold, self.max_matches, self.dbin)
        n = len(mk1)
        if n < self.min_matches:
            self.last = {"kpts2": mk2, "inlier_mask": np.zeros(n, bool),
                         "n_matches": n}
            return {"ok": False, "n_matches": n, "inlier_ratio": 0.0}
        method = cv2.USAC_MAGSAC if self.method == "magsac" else cv2.RANSAC
        R, t, mask = estimate_pose_ransac(
            mk1, mk2, self.cam, ransac_threshold=self.ransac_threshold,
            method=method)
        if R is None:
            self.last = {"kpts2": mk2, "inlier_mask": np.zeros(n, bool),
                         "n_matches": n}
            return {"ok": False, "n_matches": n, "inlier_ratio": 0.0}
        n_inl = int(np.sum(mask))
        ratio = n_inl / n if n else 0.0
        ok = n_inl >= self.min_matches and ratio >= self.min_inlier_ratio
        self.last = {"kpts2": mk2, "inlier_mask": mask, "n_matches": n_inl}
        return {"ok": ok, "R": R, "t": t,
                "inlier_ratio": ratio, "n_matches": n_inl}
