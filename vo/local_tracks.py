"""Deterministic multi-frame feature tracks for local bundle adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .onnx_matcher import extract_match_indices
from .sim3_verification import triangulate_local


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


@dataclass(frozen=True)
class TriangulatedTrack:
    track_id: int
    point: np.ndarray
    initial_pair: tuple[int, int]
    hard_valid: bool
    parallax_deg: float
    condition_ratio: float
    positive_depth_fraction: float
    reprojection_median_px: float
    reprojection_p90_px: float
    max_depth_baselines: float


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


def covisibility_counts(tracks):
    """Count tracks shared by each unordered frame pair."""
    counts = {}
    for track in tracks:
        frames = sorted(frame for frame, _feature in track.observations)
        if len(frames) != len(set(frames)):
            raise ValueError("a track cannot contain duplicate frame IDs")
        for frame_i, frame_j in combinations(frames, 2):
            key = (frame_i, frame_j)
            counts[key] = counts.get(key, 0) + 1
    return counts


def select_covisible_neighbors(frame_id, counts, *, max_neighbors,
                                min_shared_tracks):
    """Select strongest covisible frames with deterministic tie-breaking."""
    frame_id = _frame_id(frame_id)
    if (not isinstance(max_neighbors, (int, np.integer)) or max_neighbors < 0
            or not isinstance(min_shared_tracks, (int, np.integer))
            or min_shared_tracks < 1):
        raise ValueError("invalid covisibility selection limits")
    neighbors = []
    for (frame_i, frame_j), shared in counts.items():
        if frame_i == frame_id:
            neighbor = frame_j
        elif frame_j == frame_id:
            neighbor = frame_i
        else:
            continue
        if shared >= min_shared_tracks:
            neighbors.append((int(neighbor), int(shared)))
    neighbors.sort(key=lambda item: (-item[1], item[0]))
    return tuple(neighbors[:max_neighbors])


def _relative_pose(pose_i, pose_j):
    """Return i-to-j pose from anchor-to-camera poses."""
    R_i, t_i = pose_i
    R_j, t_j = pose_j
    rotation = R_j @ R_i.T
    return rotation, t_j - rotation @ t_i


def _dlt_condition_ratio(point_i_yx, point_j_yx, rotation, translation,
                         camera_matrix):
    """Second-smallest DLT singular value relative to its largest value."""
    Kinv = np.linalg.inv(camera_matrix)
    xy_i = Kinv @ np.array([point_i_yx[1], point_i_yx[0], 1.0])
    xy_j = Kinv @ np.array([point_j_yx[1], point_j_yx[0], 1.0])
    P_i = np.hstack([np.eye(3), np.zeros((3, 1))])
    P_j = np.hstack([rotation, translation[:, None]])
    A = np.vstack([
        xy_i[0] * P_i[2] - P_i[0],
        xy_i[1] * P_i[2] - P_i[1],
        xy_j[0] * P_j[2] - P_j[0],
        xy_j[1] * P_j[2] - P_j[1],
    ])
    singular = np.linalg.svd(A, compute_uv=False)
    return float(singular[-2] / singular[0]) if singular[0] > 0.0 else 0.0


def triangulate_feature_track(track, keypoints, poses, camera_matrix,
                              image_size):
    """Initialize one multi-view track in the common anchor camera gauge.

    ``poses[frame]`` maps anchor coordinates into that camera.  The viable
    frame pair with maximum ray parallax is used; all observations then
    contribute to the quality metrics.
    """
    K = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    width, height = image_size
    observations = tuple(sorted(track.observations))
    if len(observations) < 2:
        raise ValueError("triangulation requires at least two observations")
    frames = [frame for frame, _feature in observations]
    if len(frames) != len(set(frames)):
        raise ValueError("a track cannot contain duplicate frame IDs")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    pixels = {}
    for frame, feature in observations:
        if frame not in poses or frame not in keypoints:
            raise ValueError(f"missing pose or keypoints for frame {frame}")
        frame_points = np.asarray(keypoints[frame], dtype=float)
        if (frame_points.ndim != 2 or frame_points.shape[1] != 2
                or feature < 0 or feature >= len(frame_points)):
            raise ValueError(f"invalid feature {feature} for frame {frame}")
        pixels[frame] = frame_points[feature]

    candidates = []
    for (frame_i, _), (frame_j, _) in combinations(observations, 2):
        rotation, translation = _relative_pose(poses[frame_i], poses[frame_j])
        tri = triangulate_local(
            pixels[frame_i][None], pixels[frame_j][None], rotation,
            translation, K, min_parallax_deg=0.0,
            max_depth_baselines=np.inf)
        if tri.valid[0]:
            candidates.append((frame_i, frame_j, rotation, translation, tri))
    if not candidates:
        return None

    selected = min(
        candidates,
        key=lambda item: (-float(item[4].parallax_deg[0]),
                          -abs(item[1] - item[0]), item[0], item[1]))

    frame_i, frame_j, rotation, translation, tri = selected
    R_i, t_i = poses[frame_i]
    point = R_i.T @ (tri.points[0] - t_i)
    errors = []
    positive = 0
    in_image = True
    depths = []
    for frame, _feature in observations:
        pixel = pixels[frame]
        in_image &= bool(0.0 <= pixel[1] < width and 0.0 <= pixel[0] < height)
        R, t = poses[frame]
        camera_point = R @ point + t
        depths.append(float(camera_point[2]))
        positive += camera_point[2] > 0.0
        if camera_point[2] <= 0.0 or not np.all(np.isfinite(camera_point)):
            errors.append(np.inf)
            continue
        projected = K @ camera_point
        projected_yx = projected[[1, 0]] / projected[2]
        errors.append(float(np.linalg.norm(projected_yx - pixel)))

    errors = np.asarray(errors)
    finite_errors = errors[np.isfinite(errors)]
    baseline = float(np.linalg.norm(translation))
    finite_point = bool(np.all(np.isfinite(point)))
    hard_valid = bool(finite_point and in_image and positive == len(observations))
    return TriangulatedTrack(
        track_id=int(track.track_id), point=point,
        initial_pair=(frame_i, frame_j), hard_valid=hard_valid,
        parallax_deg=float(tri.parallax_deg[0]),
        condition_ratio=_dlt_condition_ratio(
            pixels[frame_i], pixels[frame_j], rotation, translation, K),
        positive_depth_fraction=float(positive / len(observations)),
        reprojection_median_px=(float(np.median(finite_errors))
                                if len(finite_errors) else np.inf),
        reprojection_p90_px=(float(np.percentile(finite_errors, 90))
                             if len(finite_errors) else np.inf),
        max_depth_baselines=(float(max(depths) / baseline)
                             if baseline > 1e-12 else np.inf),
    )
