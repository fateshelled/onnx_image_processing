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


def extract_matches(kpts1, kpts2, P, threshold=0.1, max_matches=1024,
                    dbin_margin=0.1):
    """Mutual-NN + dustbin-margin filter + top-K (eval convention).

    ``P`` is the (1, K+1, K+1) Sinkhorn matrix; keypoints are (1, K, 2) as
    (y, x). Returns ``(kpts1, kpts2, scores)``.
    """
    P = P[0]
    k1 = kpts1[0]
    k2 = kpts2[0]
    K = k1.shape[0]
    Pc = P[:K, :K]
    max_j = np.argmax(Pc, axis=1)
    max_i = np.argmax(Pc, axis=0)
    mutual = np.arange(K) == max_i[max_j]
    scores = Pc[np.arange(K), max_j]
    dbin = dustbin_margin_filter(P, dbin_margin)
    pad = (k1[:, 0] >= 0) & (k1[:, 1] >= 0) & (k2[:, 0] >= 0) & (k2[:, 1] >= 0)
    valid = mutual & dbin & pad & (scores >= threshold)
    idx_i = np.where(valid)[0]
    if len(idx_i) == 0:
        return k1[:0], k2[:0], np.array([])
    j = max_j[idx_i]
    sc = scores[idx_i]
    order = np.argsort(sc)[::-1][:max_matches]
    idx_i = idx_i[order]
    j = j[order]
    return k1[idx_i], k2[j], sc


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

    def match(self, img_a, img_b):
        outs = self.session.run(None, {self.in0: img_a, self.in1: img_b})
        k1 = outs[self.oi["k1"]]
        k2 = outs[self.oi["k2"]]
        P = outs[self.oi["probs"]]
        mk1, mk2, _sc = extract_matches(
            k1, k2, P, self.match_threshold, self.max_matches, self.dbin)
        n = len(mk1)
        if n < self.min_matches:
            return {"ok": False, "n_matches": n, "inlier_ratio": 0.0}
        method = cv2.USAC_MAGSAC if self.method == "magsac" else cv2.RANSAC
        R, t, mask = estimate_pose_ransac(
            mk1, mk2, self.cam, ransac_threshold=self.ransac_threshold,
            method=method)
        if R is None:
            return {"ok": False, "n_matches": n, "inlier_ratio": 0.0}
        n_inl = int(np.sum(mask))
        ratio = n_inl / n if n else 0.0
        ok = n_inl >= self.min_matches and ratio >= self.min_inlier_ratio
        return {"ok": ok, "R": R, "t": t,
                "inlier_ratio": ratio, "n_matches": n_inl}
