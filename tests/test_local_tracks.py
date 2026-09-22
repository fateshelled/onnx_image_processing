"""Tests for canonical feature IDs and deterministic local tracks."""

import numpy as np
import pytest

from vo.local_tracks import (
    PairMatches,
    build_feature_tracks,
    covisibility_counts,
    pair_matches_from_sinkhorn,
    select_covisible_neighbors,
)
from vo.onnx_matcher import extract_match_indices, extract_matches


def _sinkhorn(core):
    core = np.asarray(core, dtype=float)
    size = core.shape[0]
    probs = np.full((size + 1, size + 1), 0.001)
    probs[:size, :size] = core
    return probs[None]


def _keypoints(count):
    points = np.column_stack([np.arange(count), np.arange(count) + 10.0])
    return points[None]


def _pair(frame_i, frame_j, feature_i, feature_j, scores, inliers=None):
    if inliers is None:
        inliers = np.ones(len(scores), dtype=bool)
    return PairMatches(frame_i, frame_j, feature_i, feature_j, scores, inliers)


def test_extract_match_indices_preserves_ids_and_score_order():
    keypoints = _keypoints(3)
    probs = _sinkhorn([[0.2, 0.01, 0.01],
                       [0.01, 0.9, 0.01],
                       [0.01, 0.01, 0.5]])
    idx_i, idx_j, scores = extract_match_indices(
        keypoints, keypoints, probs, threshold=0.1, dbin_margin=0.0)
    assert idx_i.tolist() == [1, 2, 0]
    assert idx_j.tolist() == [1, 2, 0]
    assert scores.tolist() == pytest.approx([0.9, 0.5, 0.2])
    matched_i, matched_j, old_scores = extract_matches(
        keypoints, keypoints, probs, threshold=0.1, dbin_margin=0.0)
    np.testing.assert_array_equal(matched_i, keypoints[0][idx_i])
    np.testing.assert_array_equal(matched_j, keypoints[0][idx_j])
    np.testing.assert_array_equal(old_scores, scores)


def test_extract_match_indices_uses_canonical_tie_break_at_top_k():
    keypoints = _keypoints(2)
    probs = _sinkhorn([[0.5, 0.01], [0.01, 0.5]])
    idx_i, idx_j, scores = extract_match_indices(
        keypoints, keypoints, probs, threshold=0.1, max_matches=1,
        dbin_margin=0.0)
    assert idx_i.tolist() == [0]
    assert idx_j.tolist() == [0]
    assert scores.tolist() == [0.5]


def test_extract_match_indices_checks_matched_padding():
    keypoints_i = _keypoints(2)
    keypoints_j = _keypoints(2)
    keypoints_j[0, 1] = -1.0
    probs = _sinkhorn([[0.01, 0.9], [0.8, 0.01]])
    idx_i, idx_j, _ = extract_match_indices(
        keypoints_i, keypoints_j, probs, threshold=0.1, dbin_margin=0.0)
    assert idx_i.tolist() == [1]
    assert idx_j.tolist() == [0]


def test_pair_matches_from_sinkhorn_keeps_canonical_indices():
    keypoints = _keypoints(2)
    pair = pair_matches_from_sinkhorn(
        10, 20, keypoints, keypoints,
        _sinkhorn([[0.8, 0.01], [0.01, 0.7]]), dbin_margin=0.0,
        inlier_mask=np.array([True, False]))
    assert pair.feature_i == (0, 1)
    assert pair.feature_j == (0, 1)
    assert pair.inlier_mask == (True, False)


def test_pair_matches_from_sinkhorn_requires_inlier_mask():
    keypoints = _keypoints(1)
    with pytest.raises(TypeError, match="inlier_mask"):
        pair_matches_from_sinkhorn(0, 1, keypoints, keypoints,
                                   _sinkhorn([[0.8]]))


def test_build_tracks_links_pairs_and_ignores_outliers():
    result = build_feature_tracks([
        _pair(0, 1, [2, 4], [3, 5], [0.9, 0.8], [True, False]),
        _pair(1, 2, [3], [7], [0.7]),
    ])
    assert [track.observations for track in result.tracks] == [
        ((0, 2), (1, 3), (2, 7)),
    ]
    assert result.n_matches == 3
    assert result.n_inlier_matches == 2
    assert result.n_accepted_edges == 2


def test_conflicts_prefer_score_and_are_pair_order_independent():
    pairs = [
        _pair(0, 1, [0], [0], [0.9]),
        _pair(1, 2, [0], [0], [0.8]),
        _pair(0, 2, [1], [0], [0.7]),
    ]
    forward = build_feature_tracks(pairs)
    reverse = build_feature_tracks(reversed(pairs))
    assert forward == reverse
    assert [track.observations for track in forward.tracks] == [
        ((0, 0), (1, 0), (2, 0)),
    ]
    assert forward.n_conflicts == 1


def test_equal_score_conflicts_use_canonical_tie_break():
    result = build_feature_tracks([
        _pair(0, 1, [1], [0], [0.5]),
        _pair(0, 1, [0], [0], [0.5]),
    ])
    assert [track.observations for track in result.tracks] == [
        ((0, 0), (1, 0)),
    ]
    assert result.n_conflicts == 1


def test_pair_validation_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="aligned"):
        _pair(0, 1, [0, 1], [0], [0.5])
    with pytest.raises(ValueError, match="different frames"):
        _pair(0, 0, [0], [1], [0.5])


def test_pair_validation_rejects_lossy_indices_and_masks():
    with pytest.raises(ValueError, match="feature indices"):
        _pair(0, 1, [1.9], [0], [0.5])
    with pytest.raises(ValueError, match="inlier mask"):
        _pair(0, 1, [1], [0], [0.5], [np.nan])
    with pytest.raises(ValueError, match="inlier mask"):
        _pair(0, 1, [1], [0], [0.5], [2])


@pytest.mark.parametrize("frame_id", [0.1, np.nan, True, -1])
def test_pair_validation_rejects_invalid_frame_ids(frame_id):
    with pytest.raises(ValueError, match="frame IDs"):
        _pair(frame_id, 2, [0], [1], [0.5])


def test_pair_data_is_immutable_and_detached_from_inputs():
    feature_i = np.array([1])
    scores = np.array([0.5])
    pair = _pair(0, 1, feature_i, [2], scores)
    feature_i[0] = 9
    scores[0] = 0.1
    assert pair.feature_i == (1,)
    assert pair.scores == (0.5,)


def test_extract_match_indices_handles_empty_and_invalid_inputs():
    empty_keypoints = np.empty((1, 0, 2))
    empty_probs = np.ones((1, 1, 1))
    indices = extract_match_indices(empty_keypoints, empty_keypoints,
                                    empty_probs)
    assert all(len(array) == 0 for array in indices)
    with pytest.raises(ValueError, match="shape"):
        extract_match_indices(np.empty((0, 2)), empty_keypoints, empty_probs)
    with pytest.raises(ValueError, match="finite"):
        extract_match_indices(_keypoints(1), _keypoints(1),
                              np.full((1, 2, 2), np.nan))
    with pytest.raises(ValueError, match="max_matches"):
        extract_match_indices(_keypoints(1), _keypoints(1),
                              _sinkhorn([[0.8]]), max_matches=-1)


def test_covisibility_counts_and_neighbor_selection():
    tracks = (
        type("Track", (), {"observations": ((0, 0), (2, 1), (4, 2))})(),
        type("Track", (), {"observations": ((0, 3), (2, 4))})(),
        type("Track", (), {"observations": ((0, 5), (4, 6))})(),
    )
    counts = covisibility_counts(tracks)
    assert counts == {(0, 2): 2, (0, 4): 2, (2, 4): 1}
    assert select_covisible_neighbors(
        0, counts, max_neighbors=2, min_shared_tracks=1) == ((2, 2), (4, 2))
    assert select_covisible_neighbors(
        2, counts, max_neighbors=3, min_shared_tracks=2) == ((0, 2),)


def test_covisibility_rejects_duplicate_frames_and_invalid_limits():
    track = type("Track", (), {"observations": ((0, 0), (0, 1))})()
    with pytest.raises(ValueError, match="duplicate frame"):
        covisibility_counts([track])
    with pytest.raises(ValueError, match="selection limits"):
        select_covisible_neighbors(0, {}, max_neighbors=-1,
                                    min_shared_tracks=1)
