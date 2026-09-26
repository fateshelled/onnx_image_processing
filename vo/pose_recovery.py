"""Permissive pose-recovery match set (4-output ONNX models).

Some exported models emit an Essential matrix computed inside the ONNX graph
(weighted 8-point over all keypoints). For those, ``recoverPose``'s chirality
check should see a point set consistent with that estimate, i.e. a *permissive*
mutual-NN set (looser threshold, no top-N cut) rather than the tight top-N set
used for display/gating. These helpers build that set and remap the chirality
mask back onto the top-N matches.

The online pose graph and the evaluator use :func:`vo.onnx_matcher.extract_matches`
+ MAGSAC instead; this module is kept for the ONNX-E path and its tests.
"""

import numpy as np


def extract_matches(
    matching_probs: np.ndarray,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    threshold: float = 0.1,
    max_matches: int = 100,
    pose_recovery_threshold: float | None = None,
):
    """Extract mutual nearest-neighbour matches from a Sinkhorn matrix.

    Args:
        matching_probs: Sinkhorn probability matrix of shape (1, K+1, K+1).
        keypoints1: Keypoints in image1 of shape (1, K, 2) as (y, x).
        keypoints2: Keypoints in image2 of shape (1, K, 2) as (y, x).
        threshold: Minimum match probability.
        max_matches: Maximum number of matches (after threshold + sorting).
        pose_recovery_threshold: If set, also return a larger, permissive match
            set for pose recovery. Must be ``<= threshold``. Pad keypoints
            ((y, x) == (-1, -1)) are excluded from the pose set.

    Returns:
        Tuple of:
            - matched_kpts1 / matched_kpts2: (N, 2) top-N matches.
            - scores: (N,) match probabilities.
            - pose_kpts1 / pose_kpts2: (M, 2) permissive pose set.
            - pose_map: (N,) index into the pose set for each top-N match,
              or -1 if that match is not part of the pose set.
    """
    P = matching_probs[0]  # (K+1, K+1)
    kpts1 = keypoints1[0]  # (K, 2)
    kpts2 = keypoints2[0]  # (K, 2)
    K = kpts1.shape[0]

    P_core = P[:K, :K]
    max_j_for_i = np.argmax(P_core, axis=1)
    max_i_for_j = np.argmax(P_core, axis=0)
    mutual_mask = np.arange(K) == max_i_for_j[max_j_for_i]

    match_indices_i = np.where(mutual_mask)[0]
    match_indices_j = max_j_for_i[match_indices_i]
    scores_full = P_core[match_indices_i, match_indices_j]

    if pose_recovery_threshold is not None:
        if pose_recovery_threshold > threshold:
            raise ValueError(
                f"pose_recovery_threshold ({pose_recovery_threshold}) must be "
                f"<= threshold ({threshold})"
            )
        pose_sel = scores_full >= pose_recovery_threshold
        pose_idx_i = match_indices_i[pose_sel]
        pose_idx_j = match_indices_j[pose_sel]
        pose_valid = (
            (kpts1[pose_idx_i, 0] >= 0) & (kpts1[pose_idx_i, 1] >= 0)
            & (kpts2[pose_idx_j, 0] >= 0) & (kpts2[pose_idx_j, 1] >= 0)
        )
        pose_idx_i = pose_idx_i[pose_valid]
        pose_idx_j = pose_idx_j[pose_valid]
        pose_kpts1 = kpts1[pose_idx_i]
        pose_kpts2 = kpts2[pose_idx_j]
    else:
        pose_idx_i = None

    above_threshold = scores_full >= threshold
    match_indices_i = match_indices_i[above_threshold]
    match_indices_j = match_indices_j[above_threshold]
    scores = scores_full[above_threshold]

    sort_order = np.argsort(scores)[::-1][:max_matches]
    match_indices_i = match_indices_i[sort_order]
    match_indices_j = match_indices_j[sort_order]
    scores = scores[sort_order]

    matched_kpts1 = kpts1[match_indices_i]
    matched_kpts2 = kpts2[match_indices_j]

    if pose_recovery_threshold is not None:
        pose_pos = {int(k): p for p, k in enumerate(pose_idx_i)}
        pose_map = np.array(
            [pose_pos.get(int(k), -1) for k in match_indices_i], dtype=np.int64
        )
    else:
        pose_kpts1 = matched_kpts1
        pose_kpts2 = matched_kpts2
        pose_map = np.arange(len(matched_kpts1), dtype=np.int64)

    return matched_kpts1, matched_kpts2, scores, pose_kpts1, pose_kpts2, pose_map


def remap_pose_mask(pose_map: np.ndarray, pose_mask: np.ndarray,
                    num_matches: int) -> np.ndarray:
    """Map a chirality mask computed on the pose set back to the top-N set."""
    inlier_mask = np.zeros(num_matches, dtype=bool)
    pose_map = np.asarray(pose_map, dtype=np.int64)
    passing = np.asarray(pose_mask).ravel() > 0
    mapped = pose_map >= 0
    inlier_mask[mapped] = passing[pose_map[mapped]]
    return inlier_mask


# Backwards-compatible alias (was a private helper in the sample).
_remap_pose_mask = remap_pose_mask
