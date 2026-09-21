"""
Tests for the permissive pose-recovery match set in extract_matches
and the chirality mask remapping (_remap_pose_mask).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vo.pose_recovery import extract_matches, _remap_pose_mask


def make_sinkhorn_matrix(core_scores: np.ndarray) -> np.ndarray:
    """Build a (1, K+1, K+1) probability matrix with a diagonally-dominant core.

    core_scores[i] is set on the core diagonal (mutual-NN match with that
    score); all off-diagonal core entries are kept strictly below both the
    row and column diagonal values so only the diagonal pairs are mutual.
    """
    K = len(core_scores)
    P = np.full((K + 1, K + 1), 0.001, dtype=np.float64)
    for i, s in enumerate(core_scores):
        P[i, i] = s
    # Conservative off-diagonal values guaranteed below each row/column max
    off_diag = np.minimum(core_scores.max(), 1.0) * 0.5
    for i in range(K):
        for j in range(K):
            if i != j:
                P[i, j] = min(off_diag, P[i, i] * 0.2, P[j, j] * 0.2)
    # Dustbin row/column excluded from the core
    P[K, :] = np.linspace(0.1, 0.9, K + 1).min() * 0.01
    P[:, K] = np.linspace(0.1, 0.9, K + 1).min() * 0.01
    P[K, K] = 0.01
    return P[None, :, :]


def make_keypoints(K: int, pad_last: bool = False) -> tuple[np.ndarray, np.ndarray]:
    kpts1 = np.stack([np.array([10.0 * i, 20.0 + i]) for i in range(K)])
    kpts2 = kpts1 + np.array([0.0, 1.0])
    if pad_last:
        kpts1 = kpts1.copy()
        kpts2 = kpts2.copy()
        kpts1[-1] = (-1.0, -1.0)
        kpts2[-1] = (-1.0, -1.0)
    return kpts1[None, :, :], kpts2[None, :, :]


class TestExtractMatchesPoseSet:
    """Pose set must be built pre-threshold/pre-top-N (wider than top-N)."""

    def test_pose_set_is_wider_than_top_n(self):
        # Three mutual matches: 2 above match threshold (0.1) and 1
        # pose-recovery-only match (0.03, below 0.1, above 0.01)
        P = make_sinkhorn_matrix(np.array([0.5, 0.4, 0.03]))
        kpts1, kpts2 = make_keypoints(3)

        matched1, _m2, scores, pose1, pose2, pose_map = extract_matches(
            P, kpts1, kpts2,
            threshold=0.1, max_matches=100, pose_recovery_threshold=0.01,
        )

        assert len(matched1) == 2
        assert scores.tolist() == pytest.approx([0.5, 0.4])
        # Pose set contains the extra low-score mutual match (i=2 → (y,x)=(20,22))
        assert len(pose1) == 3
        assert (20.0, 22.0) in set(map(tuple, pose1.astype(float)))
        assert np.array_equal(pose2[2], [20.0, 23.0])
        # The pose-only match maps to its pose-set position; both top-N
        # matches map into the pose set
        assert isinstance(pose_map, np.ndarray)
        assert len(pose_map) == 2
        assert all(0 <= p < len(pose1) for p in pose_map)

    def test_top_n_and_pose_set_differ_when_pose_below_threshold(self):
        P = make_sinkhorn_matrix(np.array([0.5, 0.4, 0.03]))
        kpts1, kpts2 = make_keypoints(3)

        matched1, matched2, _s, pose1, pose2, pose_map = extract_matches(
            P, kpts1, kpts2,
            threshold=0.1, max_matches=100, pose_recovery_threshold=0.01,
        )

        # Pose set is NOT identical to the top-N set (the no-op regression):
        # it contains a point absent from the top-N set
        pose_set = set(map(tuple, pose1.astype(float)))
        for k in matched1.astype(float):
            assert tuple(k) in pose_set
        extra = list(pose_set.difference(map(tuple, matched1.astype(float))))
        assert len(extra) == 1
        # And every top-N match has a valid pose-set index
        assert (pose_map >= 0).all()

    def test_pose_set_excludes_pad_keypoints(self):
        # 4 keypoints, last one is a pad (-1, -1) on both images with a high
        # mutual score: the pad pair must be excluded from the pose set
        P = make_sinkhorn_matrix(np.array([0.5, 0.4, 0.03, 0.9]))
        kpts1, kpts2 = make_keypoints(4, pad_last=True)

        matched1, _m2, _s, pose1, pose2, pose_map = extract_matches(
            P, kpts1, kpts2,
            threshold=0.1, max_matches=100, pose_recovery_threshold=0.01,
        )

        pose_points = set(map(tuple, pose1.astype(float)))
        assert (-1.0, -1.0) not in pose_points
        assert len(pose1) == 3
        # Top-N still contains the (degenerate) pad match:
        # pose_map marks only that match as unmapped
        assert len(matched1) == 3
        unmapped = pose_map == -1
        assert unmapped.sum() == 1
        # The unmapped top-N match is exactly the pad one
        assert tuple(matched1[unmapped][0]) == (-1.0, -1.0)

    def test_pose_disabled_returns_top_n_and_identity_map(self):
        P = make_sinkhorn_matrix(np.array([0.5, 0.4, 0.03]))
        kpts1, kpts2 = make_keypoints(3)

        matched1, matched2, _s, pose1, pose2, pose_map = extract_matches(
            P, kpts1, kpts2,
            threshold=0.1, max_matches=100, pose_recovery_threshold=None,
        )

        assert np.array_equal(pose1, matched1)
        assert np.array_equal(pose2, matched2)
        assert np.array_equal(pose_map, np.arange(len(matched1)))

    def test_pose_recovery_threshold_above_match_threshold_raises(self):
        P = make_sinkhorn_matrix(np.array([0.5, 0.4, 0.03]))
        kpts1, kpts2 = make_keypoints(3)

        with pytest.raises(ValueError):
            extract_matches(
                P, kpts1, kpts2,
                threshold=0.1, max_matches=100,
                pose_recovery_threshold=0.5,
            )

    def test_max_matches_cut_keeps_full_mapping_indices(self):
        # top-N cut (max_matches=1) must not eat into the pose set
        P = make_sinkhorn_matrix(np.array([0.5, 0.4, 0.03]))
        kpts1, kpts2 = make_keypoints(3)

        matched1, _m2, _s, pose1, pose2, pose_map = extract_matches(
            P, kpts1, kpts2,
            threshold=0.1, max_matches=1, pose_recovery_threshold=0.01,
        )

        assert len(matched1) == 1
        assert len(pose1) == 3
        assert pose_map[0] == 0  # top score (i=0) maps to first pose entry


class TestRemapPoseMask:
    def test_basic_mapping(self):
        pose_map = np.array([-1, 0, 1])
        pose_mask = np.array([0, 1, 1])
        mask = _remap_pose_mask(pose_map, pose_mask, num_matches=3)
        assert mask.tolist() == [False, False, True]

    def test_unmapped_matches_fail_chirality(self):
        pose_map = np.array([2, -1, 0])
        pose_mask = np.array([1, 0, 1])
        mask = _remap_pose_mask(pose_map, pose_mask, num_matches=3)
        # i0 -> pose 2 (passing), i1 -> unmapped, i2 -> pose 0 (passing)
        assert mask.tolist() == [True, False, True]

    def test_all_false_when_pose_mask_empty(self):
        pose_map = np.array([0, 1, -1])
        pose_mask = np.zeros(2)
        mask = _remap_pose_mask(pose_map, pose_mask, num_matches=3)
        assert mask.tolist() == [False, False, False]

    def test_shape(self):
        mask = _remap_pose_mask(np.array([0, 1]), np.array([1, 1]), num_matches=2)
        assert mask.dtype == np.bool_
        assert mask.shape == (2,)
