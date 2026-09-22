"""Shared Sim(3) loop-verification primitives for offline and online paths.

Both the offline diagnostic (``scripts/diag_loop_sim3.py``) and the online
``Sim3LoopVerifier`` reconstruct the local structure around a loop endpoint the
same way: the densest backward/forward window of cached per-step odometry is
triangulated independently on each side, and the raw loop correspondences are
checked for Sim(3) consistency.  The helpers live here so the two paths stay
numerically identical.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from vo.sim3_verification import sim3_ransac, triangulate_local


def point_key(point):
    # The loop pair and the local stride pair share keypoint arrays, so the
    # rounded coordinates act as a join key across float32/float64 copies.
    # Collisions resolve last-wins, which is harmless for identical tracks.
    return tuple(np.round(np.asarray(point, dtype=float), 4))


def compose_odom(odom, first_slot, last_slot):
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


def local_windows(odom, endpoint, stride, window_strides):
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
            pose = compose_odom(odom, first_slot, endpoint_slot)
            if pose is not None:
                windows.append((endpoint - steps * stride, endpoint,
                                pose[0], pose[1], False))
        last_slot = endpoint_slot + steps
        pose = compose_odom(odom, endpoint_slot, last_slot)
        if pose is not None:
            windows.append((endpoint, endpoint + steps * stride,
                            pose[0], pose[1], True))
    return windows


def build_cloud_from_windows(windows, match_fn, camera_matrix, *,
                             min_parallax_deg=1.0,
                             triangulate_fn=triangulate_local):
    """Reconstruct an endpoint cloud from candidate two-view windows.

    Exactly one window feeds the cloud: the one with the most valid tracks
    (ties go to the first candidate, so callers pass longer baselines first).
    Using one window keeps a single scale gauge, since the cached per-stride
    translations are unit-norm essential-matrix directions rather than
    metrically consistent displacements.  All candidates are scanned so that
    a sparse long window does not hide a denser short one.

    Each window is ``(first_frame, last_frame, rotation, translation,
    forward)`` with ``x_last = R @ x_first + t``; ``match_fn(first, last)``
    must return raw ``(points_first, points_last)`` matches or ``None``.
    """
    best = None
    for first_frame, last_frame, rotation, translation, forward in windows:
        pair = match_fn(first_frame, last_frame)
        if pair is None:
            continue
        p_first, p_last = pair
        tri = triangulate_fn(p_first, p_last, rotation, translation,
                             camera_matrix, min_parallax_deg=min_parallax_deg)
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
        cloud = {point_key(keypoint): point
                 for keypoint, point, valid in
                 zip(keys, points_endpoint, tri.valid) if valid}
        if best is None or len(cloud) > len(best):
            best = cloud
    return best or None


def build_local_cloud(odom, endpoint, stride, window_strides, match_fn,
                      camera_matrix, *, min_parallax_deg=1.0,
                      backward_only=False, triangulate_fn=triangulate_local):
    """Reconstruct the endpoint from its densest odometry-window track.

    ``match_fn(first_frame, last_frame)`` must return raw
    ``(points_first, points_last)`` matches or ``None``; the caller owns
    whatever feature storage it needs.  ``backward_only`` skips forward
    windows, which online would need not-yet-computed future frames.
    """
    windows = local_windows(odom, endpoint, stride, window_strides)
    if backward_only:
        windows = [window for window in windows if not window[4]]
    return build_cloud_from_windows(
        windows, match_fn, camera_matrix, min_parallax_deg=min_parallax_deg,
        triangulate_fn=triangulate_fn)


def loop_tracks(pa, pb, cloud_a, cloud_b):
    """Join raw loop correspondences against both endpoint clouds."""
    Xa, Xb = [], []
    for point_a, point_b in zip(pa, pb):
        key_a, key_b = point_key(point_a), point_key(point_b)
        if key_a in cloud_a and key_b in cloud_b:
            Xa.append(cloud_a[key_a])
            Xb.append(cloud_b[key_b])
    return Xa, Xb


def candidate_seed(seed, a, b):
    """Derive an independent RANSAC seed per candidate pair."""
    return int((seed * 1_000_003 + a * 1_009 + b) % (2 ** 31 - 1))


class Sim3LoopVerifier:
    """Accept/reject loop hits by independent local Sim(3) consistency.

    ``__call__(a, b)`` returns ``True`` (accept) or ``False`` (reject).  When
    the candidate cannot be evaluated (no local cloud, no loop match, or fewer
    than ``min_tracks`` joined tracks) the verdict follows ``abstain_policy``:
    ``"reject"`` (default) matches the offline gate calibration, where
    non-evaluable candidates never become accepted edges; ``"accept"`` keeps
    the caller's current decision (fail-open).  ``odom`` and ``match_fn`` are
    owned by the caller, so the pose graph itself stores no features.

    ``window_fn(endpoint)`` optionally returns extra candidate windows (e.g.
    keyframe-to-keyframe baselines from the graph) in the same form as
    :func:`local_windows`; they are tried before the odometry windows.
    """

    def __init__(self, odom, match_fn, camera_matrix, stride, *, gate=6,
                 window_strides=4, min_parallax_deg=1.0, min_tracks=5,
                 min_condition_ratio=1e-2, residual_fraction=0.1,
                 iterations=2000, seed=0, cache_size=64, backward_only=True,
                 abstain_policy="reject", window_fn=None):
        if gate < 1 or min_tracks < 3 or window_strides < 1 or cache_size < 1:
            raise ValueError("invalid Sim3LoopVerifier parameters")
        if abstain_policy not in ("reject", "accept"):
            raise ValueError("abstain_policy must be 'reject' or 'accept'")
        self.odom = odom
        self.match_fn = match_fn
        self.camera_matrix = camera_matrix
        self.stride = int(stride)
        self.gate = int(gate)
        self.window_strides = int(window_strides)
        self.min_parallax_deg = float(min_parallax_deg)
        self.min_tracks = int(min_tracks)
        self.min_condition_ratio = float(min_condition_ratio)
        self.residual_fraction = float(residual_fraction)
        self.iterations = int(iterations)
        self.seed = int(seed)
        self.cache_size = int(cache_size)
        self.backward_only = bool(backward_only)
        self.abstain_policy = abstain_policy
        self.window_fn = window_fn
        self._clouds = OrderedDict()
        self.n_accept = 0
        self.n_reject = 0
        self.n_abstain = 0
        self.n_abstain_no_cloud = 0
        self.n_abstain_no_match = 0
        self.n_abstain_few_tracks = 0

    def _cloud(self, endpoint):
        cached = self._clouds.get(endpoint)
        if cached is not None or endpoint in self._clouds:
            self._clouds.move_to_end(endpoint)
            return cached
        windows = []
        if self.window_fn is not None:
            windows.extend(self.window_fn(endpoint))
        odom_windows = local_windows(self.odom, endpoint, self.stride,
                                     self.window_strides)
        if self.backward_only:
            odom_windows = [window for window in odom_windows
                            if not window[4]]
        seen = {(window[0], window[1]) for window in windows}
        windows.extend(window for window in odom_windows
                       if (window[0], window[1]) not in seen)
        cloud = build_cloud_from_windows(
            windows, self.match_fn, self.camera_matrix,
            min_parallax_deg=self.min_parallax_deg)
        self._clouds[endpoint] = cloud
        while len(self._clouds) > self.cache_size:
            self._clouds.popitem(last=False)
        return cloud

    def __call__(self, a, b):
        a, b = int(a), int(b)
        cloud_a, cloud_b = self._cloud(a), self._cloud(b)
        if not cloud_a or not cloud_b:
            self.n_abstain += 1
            self.n_abstain_no_cloud += 1
            return self.abstain_policy == "accept"
        pair = self.match_fn(a, b)
        if pair is None:
            self.n_abstain += 1
            self.n_abstain_no_match += 1
            return self.abstain_policy == "accept"
        pa, pb = pair
        Xa, Xb = loop_tracks(pa, pb, cloud_a, cloud_b)
        if len(Xa) < self.min_tracks:
            self.n_abstain += 1
            self.n_abstain_few_tracks += 1
            return self.abstain_policy == "accept"
        fit = sim3_ransac(np.asarray(Xa), np.asarray(Xb),
                          seed=candidate_seed(self.seed, a, b),
                          residual_fraction=self.residual_fraction,
                          max_iterations=self.iterations,
                          min_inliers=self.min_tracks,
                          min_condition_ratio=self.min_condition_ratio)
        if fit.ok and int(fit.inliers.sum()) >= self.gate:
            self.n_accept += 1
            return True
        self.n_reject += 1
        return False
