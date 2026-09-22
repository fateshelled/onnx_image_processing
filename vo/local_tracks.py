"""Deterministic multi-frame feature tracks for local bundle adjustment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .onnx_matcher import extract_match_indices


Observation = tuple[int, int]


@dataclass(frozen=True)
class PairMatches:
    """Canonical feature-index correspondences for one frame pair."""

    frame_i: int
    frame_j: int
    feature_i: tuple[int, ...]
    feature_j: tuple[int, ...]
    scores: tuple[float, ...]
    inlier_mask: tuple[bool, ...]

    def __post_init__(self):
        frame_i = _frame_id(self.frame_i)
        frame_j = _frame_id(self.frame_j)
        if frame_i == frame_j:
            raise ValueError("a match pair must use two different frames")
        feature_i = _feature_ids(self.feature_i)
        feature_j = _feature_ids(self.feature_j)
        scores = _scores(self.scores)
        inlier_mask = _mask(self.inlier_mask)
        values = (feature_i, feature_j, scores, inlier_mask)
        if len({len(value) for value in values}) != 1:
            raise ValueError("match arrays must be one-dimensional and aligned")
        object.__setattr__(self, "frame_i", frame_i)
        object.__setattr__(self, "frame_j", frame_j)
        object.__setattr__(self, "feature_i", feature_i)
        object.__setattr__(self, "feature_j", feature_j)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "inlier_mask", inlier_mask)


def _frame_id(value):
    array = np.asarray(value)
    if (array.ndim != 0 or np.issubdtype(array.dtype, np.bool_)
            or not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array)):
        raise ValueError("frame IDs must be finite non-negative integers")
    numeric = float(array)
    if not np.isfinite(numeric) or numeric < 0 or numeric != np.floor(numeric):
        raise ValueError("frame IDs must be finite non-negative integers")
    return int(numeric)


def _one_dimensional(value, name):
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _feature_ids(value):
    array = _one_dimensional(value, "feature indices")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("feature indices must be finite non-negative integers")
    numeric = np.asarray(array, dtype=float)
    if (not np.all(np.isfinite(numeric)) or np.any(numeric < 0)
            or not np.all(numeric == np.floor(numeric))):
        raise ValueError("feature indices must be finite non-negative integers")
    return tuple(int(item) for item in numeric)


def _scores(value):
    array = _one_dimensional(value, "scores")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("match scores must be finite")
    numeric = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("match scores must be finite")
    return tuple(float(item) for item in numeric)


def _mask(value):
    array = _one_dimensional(value, "inlier mask")
    if np.issubdtype(array.dtype, np.bool_):
        return tuple(bool(item) for item in array)
    if (not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array)
            or not np.all(np.isfinite(array))
            or not np.all((array == 0) | (array == 1))):
        raise ValueError("inlier mask must contain only bool or 0/1 values")
    return tuple(bool(item) for item in array)


@dataclass(frozen=True)
class FeatureTrack:
    track_id: int
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class TrackBuildResult:
    tracks: tuple[FeatureTrack, ...]
    n_matches: int
    n_inlier_matches: int
    n_accepted_edges: int
    n_conflicts: int
    n_duplicates: int


def pair_matches_from_sinkhorn(frame_i, frame_j, kpts_i, kpts_j, probs, *,
                               inlier_mask, threshold=0.1, max_matches=1024,
                               dbin_margin=0.1):
    """Create a track pair while preserving the canonical feature indices."""
    idx_i, idx_j, scores = extract_match_indices(
        kpts_i, kpts_j, probs, threshold, max_matches, dbin_margin)
    return PairMatches(frame_i, frame_j, idx_i, idx_j, scores, inlier_mask)


def build_feature_tracks(pair_matches, *, min_observations=2):
    """Build order-independent tracks without duplicate frames per track.

    Inlier correspondences are processed globally by descending score.  Ties
    use their canonical observation IDs.  An edge is rejected when joining its
    two components would place two different features from the same frame in
    one track.
    """
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2")

    candidates = []
    n_matches = 0
    for pair in pair_matches:
        n_matches += len(pair.scores)
        for feature_i, feature_j, score, inlier in zip(
                pair.feature_i, pair.feature_j, pair.scores, pair.inlier_mask):
            if not inlier:
                continue
            left = (int(pair.frame_i), int(feature_i))
            right = (int(pair.frame_j), int(feature_j))
            if right < left:
                left, right = right, left
            candidates.append((-float(score), left, right))
    candidates.sort()

    parent = {}
    component_frames = {}

    def add(observation):
        if observation not in parent:
            parent[observation] = observation
            component_frames[observation] = {observation[0]: observation[1]}

    def root(observation):
        while parent[observation] != observation:
            parent[observation] = parent[parent[observation]]
            observation = parent[observation]
        return observation

    accepted = conflicts = duplicates = 0
    for _negative_score, left, right in candidates:
        add(left)
        add(right)
        root_left, root_right = root(left), root(right)
        if root_left == root_right:
            duplicates += 1
            continue
        frames_left = component_frames[root_left]
        frames_right = component_frames[root_right]
        if any(frames_left[frame] != frames_right[frame]
               for frame in frames_left.keys() & frames_right.keys()):
            conflicts += 1
            continue
        if root_right < root_left:
            root_left, root_right = root_right, root_left
            frames_left, frames_right = frames_right, frames_left
        parent[root_right] = root_left
        frames_left.update(frames_right)
        del component_frames[root_right]
        accepted += 1

    components = {}
    for observation in parent:
        components.setdefault(root(observation), []).append(observation)
    observations = sorted(
        tuple(sorted(component)) for component in components.values()
        if len(component) >= min_observations)
    tracks = tuple(FeatureTrack(track_id, component)
                   for track_id, component in enumerate(observations))
    return TrackBuildResult(
        tracks=tracks,
        n_matches=n_matches,
        n_inlier_matches=len(candidates),
        n_accepted_edges=accepted,
        n_conflicts=conflicts,
        n_duplicates=duplicates,
    )
