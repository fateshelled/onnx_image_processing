"""Pure local bundle adjustment for camera-to-world poses and landmarks."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .se3 import se3_exp, skew


@dataclass(frozen=True)
class BAObservation:
    frame_id: int
    track_id: int
    pixel_yx: tuple[float, float]
    feature_id: int = -1


@dataclass(frozen=True)
class LocalBAResult:
    ok: bool
    reason: str
    poses: dict[int, np.ndarray]
    landmarks: dict[int, np.ndarray]
    initial_cost: float
    final_cost: float
    iterations: int
    pose_ids: tuple[int, ...]
    reduced_hessian: np.ndarray
    reduced_gradient: np.ndarray
    linearization_poses: tuple[np.ndarray, ...]
    n_observations: int


def _camera_matrix(camera):
    value = camera.K if hasattr(camera, "K") else camera
    K = np.asarray(value, dtype=float)
    if (K.shape != (3, 3) or not np.all(np.isfinite(K))
            or K[0, 0] <= 0.0 or K[1, 1] <= 0.0
            or not np.allclose(K[2], [0.0, 0.0, 1.0])
            or not np.allclose([K[0, 1], K[1, 0]], 0.0)):
        raise ValueError("camera matrix must be a standard finite pinhole matrix")
    return K


def _copy_state(poses, landmarks):
    return ({int(key): np.asarray(value, dtype=float).copy()
             for key, value in poses.items()},
            {int(key): np.asarray(value, dtype=float).reshape(3).copy()
             for key, value in landmarks.items()})


def _validate_state(poses, landmarks, observations):
    if len(poses) < 2:
        raise ValueError("local BA requires at least two poses")
    for frame_id, pose in poses.items():
        if (pose.shape != (4, 4) or not np.all(np.isfinite(pose))
                or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3),
                                   atol=1e-7)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0,
                                  atol=1e-7)):
            raise ValueError(f"invalid camera-to-world pose {frame_id}")
    if not landmarks:
        raise ValueError("local BA requires landmarks")
    counts = {track_id: 0 for track_id in landmarks}
    pose_counts = {frame_id: 0 for frame_id in poses}
    seen = set()
    adjacency = {("p", frame_id): set() for frame_id in poses}
    adjacency.update({("l", track_id): set() for track_id in landmarks})
    normalized = []
    for item in observations:
        if item.frame_id not in poses:
            raise ValueError(f"observation uses unknown frame {item.frame_id}")
        if item.track_id not in landmarks:
            raise ValueError(f"observation uses unknown track {item.track_id}")
        pixel = np.asarray(item.pixel_yx, dtype=float)
        if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
            raise ValueError("observation pixels must be finite yx pairs")
        key = (int(item.frame_id), int(item.track_id))
        if key in seen:
            raise ValueError("duplicate frame-landmark observation")
        seen.add(key)
        counts[item.track_id] += 1
        pose_counts[item.frame_id] += 1
        adjacency[("p", item.frame_id)].add(("l", item.track_id))
        adjacency[("l", item.track_id)].add(("p", item.frame_id))
        normalized.append((int(item.frame_id), int(item.track_id), pixel))
    if any(count < 2 for count in counts.values()):
        raise ValueError("each landmark requires at least two observations")
    if any(count == 0 for count in pose_counts.values()):
        raise ValueError("each pose requires at least one observation")
    start = next(iter(adjacency))
    reached = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    if len(reached) != len(adjacency):
        raise ValueError("pose-landmark observation graph must be connected")
    return tuple(normalized)


def _huber(norm, delta):
    if norm <= delta:
        return 1.0, 0.5 * norm * norm
    return delta / max(norm, 1e-12), delta * (norm - 0.5 * delta)


def _observation_residual_jacobians(pose, landmark, observed, K):
    R, center = pose[:3, :3], pose[:3, 3]
    camera_point = R.T @ (landmark - center)
    x, y, z = camera_point
    if not np.all(np.isfinite(camera_point)) or z <= 0.0:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    predicted = np.array([fy * y / z + cy, fx * x / z + cx])
    J_project = np.array([
        [0.0, fy / z, -fy * y / (z * z)],
        [fx / z, 0.0, -fx * x / (z * z)],
    ])
    J_pose = J_project @ np.hstack([skew(camera_point), -np.eye(3)])
    J_landmark = J_project @ R.T
    return predicted - observed, J_pose, J_landmark


def _filter_initial_outliers(poses, landmarks, observations, K, huber_delta):
    errors = []
    for frame_id, track_id, observed in observations:
        value = _observation_residual_jacobians(
            poses[frame_id], landmarks[track_id], observed, K)
        errors.append(np.inf if value is None
                      else float(np.linalg.norm(value[0])))
    finite = np.asarray(errors)[np.isfinite(errors)]
    if not len(finite):
        return observations
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    threshold = max(5.0 * huber_delta, median + 5.0 * 1.4826 * mad)
    filtered = tuple(item for item, error in zip(observations, errors)
                     if error <= threshold)
    if len(filtered) == len(observations):
        return observations
    candidate = tuple(BAObservation(frame_id, track_id, tuple(pixel))
                      for frame_id, track_id, pixel in filtered)
    try:
        return _validate_state(poses, landmarks, candidate)
    except ValueError:
        return observations


def _linearize(poses, landmarks, observations, K, huber_delta, pose_ids,
               landmark_ids, cancelled=None):
    pose_index = {frame_id: index for index, frame_id in enumerate(pose_ids)}
    landmark_index = {track_id: index
                      for index, track_id in enumerate(landmark_ids)}
    n_pose, n_landmark = len(pose_ids), len(landmark_ids)
    Hpp = np.zeros((6 * n_pose, 6 * n_pose))
    Hpl = np.zeros((6 * n_pose, 3 * n_landmark))
    Hll = np.zeros((n_landmark, 3, 3))
    bp = np.zeros(6 * n_pose)
    bl = np.zeros((n_landmark, 3))
    cost = 0.0
    valid = 0

    for observation_index, (frame_id, track_id, observed) in enumerate(observations):
        if (cancelled is not None and observation_index % 64 == 0
                and cancelled()):
            return None
        T = poses[frame_id]
        linearized = _observation_residual_jacobians(
            T, landmarks[track_id], observed, K)
        if linearized is None:
            continue
        residual, J_pose, J_landmark = linearized
        weight, robust_cost = _huber(float(np.linalg.norm(residual)),
                                     huber_delta)
        root_weight = np.sqrt(weight)
        J_pose = root_weight * J_pose
        J_landmark = root_weight * J_landmark
        weighted_residual = root_weight * residual
        pi = pose_index[frame_id]
        li = landmark_index[track_id]
        ps = slice(6 * pi, 6 * pi + 6)
        ls = slice(3 * li, 3 * li + 3)
        Hpp[ps, ps] += J_pose.T @ J_pose
        Hpl[ps, ls] += J_pose.T @ J_landmark
        Hll[li] += J_landmark.T @ J_landmark
        bp[ps] += J_pose.T @ weighted_residual
        bl[li] += J_landmark.T @ weighted_residual
        cost += robust_cost
        valid += 1
    return Hpp, Hpl, Hll, bp, bl, float(cost), valid


def _schur_reduce(Hpp, Hpl, Hll, bp, bl, damping=0.0):
    reduced_hessian = Hpp.copy()
    reduced_gradient = bp.copy()
    inverses = []
    for index, block in enumerate(Hll):
        regularized = block + damping * np.eye(3)
        inverse = np.linalg.pinv(regularized, rcond=1e-12)
        cross = Hpl[:, 3 * index:3 * index + 3]
        reduced_hessian -= cross @ inverse @ cross.T
        reduced_gradient -= cross @ inverse @ bl[index]
        inverses.append(inverse)
    return (0.5 * (reduced_hessian + reduced_hessian.T),
            reduced_gradient, inverses)


def _gauge_basis(poses, pose_ids):
    directions = []
    for axis in np.eye(3):
        directions.append(np.concatenate([
            np.r_[np.zeros(3), poses[frame][:3, :3].T @ axis]
            for frame in pose_ids
        ]))
    for axis in np.eye(3):
        directions.append(np.concatenate([
            np.r_[poses[frame][:3, :3].T @ axis,
                  poses[frame][:3, :3].T
                  @ (skew(axis) @ poses[frame][:3, 3])]
            for frame in pose_ids
        ]))
    directions.append(np.concatenate([
        np.r_[np.zeros(3), poses[frame][:3, :3].T @ poses[frame][:3, 3]]
        for frame in pose_ids
    ]))
    return np.column_stack(directions)


def _has_exact_sim3_gauge(hessian, poses, pose_ids):
    """Check that only the analytic seven-dimensional gauge is unobservable."""
    hessian = 0.5 * (hessian + hessian.T)
    diagonal = np.diag(hessian)
    if np.any(diagonal <= 0.0):
        return False
    whitening = 1.0 / np.sqrt(diagonal)
    hessian = whitening[:, None] * hessian * whitening[None, :]
    # x = D y for H_y = D H_x D, hence y_gauge = D^-1 x_gauge.
    gauge = _gauge_basis(poses, pose_ids) / whitening[:, None]
    norms = np.linalg.norm(gauge, axis=0)
    if np.any(norms == 0.0):
        return False
    normalized_gauge = gauge / norms
    singular = np.linalg.svd(normalized_gauge, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-10:
        return False
    Q, _ = np.linalg.qr(normalized_gauge, mode="complete")
    scale = max(float(np.linalg.norm(hessian)), 1.0)
    if np.linalg.norm(hessian @ Q[:, :7]) > 1e-7 * scale:
        return False
    observable = Q[:, 7:].T @ hessian @ Q[:, 7:]
    if not len(observable):
        return True
    observable_diagonal = np.diag(observable)
    if np.any(observable_diagonal <= 0.0):
        return False
    observable_whitening = 1.0 / np.sqrt(observable_diagonal)
    normalized = (observable_whitening[:, None] * observable
                  * observable_whitening[None, :])
    eigenvalues = np.linalg.eigvalsh(0.5 * (normalized + normalized.T))
    return bool(eigenvalues[0] > 1e-9)


def _normalized_full_gradient(Hpp, Hll, bp, bl):
    diagonal = np.r_[np.diag(Hpp),
                     np.concatenate([np.diag(block) for block in Hll])]
    gradient = np.r_[bp, bl.ravel()]
    positive = diagonal > 0.0
    if np.any(~positive & (np.abs(gradient) > 0.0)):
        return np.inf
    scaled = np.zeros_like(gradient)
    scaled[positive] = np.abs(gradient[positive]) / np.sqrt(diagonal[positive])
    return float(np.max(scaled, initial=0.0))


def linearize_reduced(poses, landmarks, observations, camera, *,
                      huber_delta=3.0):
    """Return the ungauged landmark-Schur pose normal equation."""
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive")
    K = _camera_matrix(camera)
    pose_state, landmark_state = _copy_state(poses, landmarks)
    normalized = _validate_state(pose_state, landmark_state, observations)
    normalized = _filter_initial_outliers(
        pose_state, landmark_state, normalized, K, huber_delta)
    pose_ids = tuple(sorted(pose_state))
    landmark_ids = tuple(sorted(landmark_state))
    blocks = _linearize(pose_state, landmark_state, normalized, K,
                        huber_delta, pose_ids, landmark_ids)
    H, b, _ = _schur_reduce(*blocks[:5])
    return pose_ids, H, b, blocks[5], blocks[6]


def _state_cost(poses, landmarks, observations, K, huber_delta,
                anchor_id, scale_id, scale_target, scale_sigma,
                cancelled=None):
    blocks = _linearize(
        poses, landmarks, observations, K, huber_delta,
        tuple(sorted(poses)), tuple(sorted(landmarks)), cancelled=cancelled)
    if blocks is None:
        return np.inf, -1
    _, _, _, _, _, cost, valid = blocks
    baseline = np.linalg.norm(poses[scale_id][:3, 3]
                              - poses[anchor_id][:3, 3])
    scale_residual = (baseline - scale_target) / scale_sigma
    return cost + 0.5 * scale_residual * scale_residual, valid


def optimize_local_ba(poses, landmarks, observations, camera, *,
                      pose_anchor_id=None, max_iterations=15,
                      huber_delta=3.0, lambda_init=1e-3,
                      scale_sigma=1e-4, min_observations=6,
                      deadline=None, clock=time.monotonic):
    """Optimize a local map and export its ungauged reduced pose factor.

    Poses are camera-to-world matrices and receive right SE(3) increments.
    The oldest pose is fixed and the initial baseline to the next pose is a
    temporary scale gauge.  Neither temporary gauge is included in the
    returned reduced normal equation.
    """
    if (max_iterations < 1 or huber_delta <= 0.0 or lambda_init <= 0.0
            or scale_sigma <= 0.0 or min_observations < 1):
        raise ValueError("invalid local BA solver settings")
    K = _camera_matrix(camera)
    pose_state, landmark_state = _copy_state(poses, landmarks)
    normalized = _validate_state(pose_state, landmark_state, observations)
    normalized = _filter_initial_outliers(
        pose_state, landmark_state, normalized, K, huber_delta)
    pose_ids = tuple(sorted(pose_state))
    landmark_ids = tuple(sorted(landmark_state))
    anchor_id = pose_ids[0] if pose_anchor_id is None else int(pose_anchor_id)
    if anchor_id not in pose_state:
        return _failure(pose_state, landmark_state, "pose anchor is not present")
    free_pose_ids = tuple(frame_id for frame_id in pose_ids
                          if frame_id != anchor_id)
    scale_id = free_pose_ids[0]
    scale_target = float(np.linalg.norm(
        pose_state[scale_id][:3, 3] - pose_state[anchor_id][:3, 3]))
    if not np.isfinite(scale_target) or scale_target <= 1e-8:
        return _failure(pose_state, landmark_state,
                        "scale anchor baseline is degenerate")

    initial_poses, initial_landmarks = _copy_state(pose_state, landmark_state)
    cancelled = (lambda: deadline is not None and clock() >= deadline)
    if cancelled():
        return _failure(initial_poses, initial_landmarks,
                        "deadline exceeded")
    initial_cost, required_valid = _state_cost(
        pose_state, landmark_state, normalized, K, huber_delta,
        anchor_id, scale_id, scale_target, scale_sigma,
        cancelled=cancelled)
    if cancelled():
        return _failure(initial_poses, initial_landmarks,
                        "deadline exceeded")
    if required_valid < min_observations or not np.isfinite(initial_cost):
        return _failure(initial_poses, initial_landmarks,
                        "insufficient valid observations")
    initial_blocks = _linearize(
        pose_state, landmark_state, normalized, K, huber_delta,
        pose_ids, landmark_ids, cancelled=cancelled)
    if initial_blocks is None or cancelled():
        return _failure(initial_poses, initial_landmarks,
                        "deadline exceeded", initial_cost=initial_cost)
    initial_reduced, _, _ = _schur_reduce(*initial_blocks[:5])
    if cancelled():
        return _failure(initial_poses, initial_landmarks,
                        "deadline exceeded", initial_cost=initial_cost)
    if not _has_exact_sim3_gauge(initial_reduced, pose_state, pose_ids):
        return _failure(initial_poses, initial_landmarks,
                        "reduced system does not have exactly 7 gauge modes",
                        initial_cost=initial_cost)

    free_indices = np.concatenate([
        np.arange(6 * pose_ids.index(frame_id),
                  6 * pose_ids.index(frame_id) + 6)
        for frame_id in free_pose_ids
    ])
    damping = float(lambda_init)
    accepted = 0
    converged = False
    current_cost = initial_cost
    for _iteration in range(max_iterations):
        if cancelled():
            return _failure(initial_poses, initial_landmarks,
                            "deadline exceeded", initial_cost=initial_cost)
        blocks = _linearize(pose_state, landmark_state, normalized, K,
                            huber_delta, pose_ids, landmark_ids,
                            cancelled=cancelled)
        if blocks is None or cancelled():
            return _failure(initial_poses, initial_landmarks,
                            "deadline exceeded", initial_cost=initial_cost)
        Hpp, Hpl, Hll, bp, bl = blocks[:5]
        if _normalized_full_gradient(Hpp, Hll, bp, bl) < 1e-8:
            converged = True
            break
        Hred, bred, inverses = _schur_reduce(
            Hpp, Hpl, Hll, bp, bl, damping=damping)

        scale_pose_index = pose_ids.index(scale_id)
        scale_slice = slice(6 * scale_pose_index, 6 * scale_pose_index + 6)
        delta_center = (pose_state[scale_id][:3, 3]
                        - pose_state[anchor_id][:3, 3])
        baseline = float(np.linalg.norm(delta_center))
        direction = delta_center / baseline
        J_scale = np.zeros(6)
        J_scale[3:] = direction @ pose_state[scale_id][:3, :3] / scale_sigma
        residual_scale = (baseline - scale_target) / scale_sigma
        Hred[scale_slice, scale_slice] += np.outer(J_scale, J_scale)
        bred[scale_slice] += J_scale * residual_scale

        Hfree = Hred[np.ix_(free_indices, free_indices)]
        bfree = bred[free_indices]
        diagonal = np.maximum(np.diag(Hfree), 1.0)
        try:
            step_free = np.linalg.solve(
                Hfree + damping * np.diag(diagonal), -bfree)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue
        if not np.all(np.isfinite(step_free)):
            damping *= 10.0
            continue
        step_pose = np.zeros(6 * len(pose_ids))
        step_pose[free_indices] = step_free
        candidate_poses, candidate_landmarks = _copy_state(
            pose_state, landmark_state)
        for index, frame_id in enumerate(pose_ids):
            delta = step_pose[6 * index:6 * index + 6]
            candidate_poses[frame_id] = pose_state[frame_id] @ se3_exp(delta)
        landmark_steps = []
        for index, track_id in enumerate(landmark_ids):
            cross = Hpl[:, 3 * index:3 * index + 3]
            delta = -inverses[index] @ (bl[index] + cross.T @ step_pose)
            candidate_landmarks[track_id] = landmark_state[track_id] + delta
            landmark_steps.append(float(np.linalg.norm(delta)))
        combined_step = max(float(np.linalg.norm(step_free)),
                            max(landmark_steps, default=0.0))
        if cancelled():
            return _failure(initial_poses, initial_landmarks,
                            "deadline exceeded", initial_cost=initial_cost)
        candidate_cost, candidate_valid = _state_cost(
            candidate_poses, candidate_landmarks, normalized, K, huber_delta,
            anchor_id, scale_id, scale_target, scale_sigma,
            cancelled=cancelled)
        if cancelled():
            return _failure(initial_poses, initial_landmarks,
                            "deadline exceeded", initial_cost=initial_cost)
        if (candidate_valid == required_valid and np.isfinite(candidate_cost)
                and candidate_cost < current_cost):
            pose_state, landmark_state = candidate_poses, candidate_landmarks
            current_cost = candidate_cost
            accepted += 1
            damping = max(damping / 3.0, 1e-12)
            if combined_step < 1e-7:
                converged = True
                break
        else:
            damping *= 10.0

    if accepted == 0 and not converged:
        return _failure(initial_poses, initial_landmarks,
                        "no cost-decreasing step", initial_cost=initial_cost)
    final_blocks = _linearize(pose_state, landmark_state, normalized, K,
                              huber_delta, pose_ids, landmark_ids,
                              cancelled=cancelled)
    if final_blocks is None or cancelled():
        return _failure(initial_poses, initial_landmarks,
                        "deadline exceeded", initial_cost=initial_cost)
    reduced_hessian, reduced_gradient, _ = _schur_reduce(*final_blocks[:5])
    if cancelled():
        return _failure(initial_poses, initial_landmarks,
                        "deadline exceeded", initial_cost=initial_cost)
    if not _has_exact_sim3_gauge(
            reduced_hessian, pose_state, pose_ids):
        return _failure(initial_poses, initial_landmarks,
                        "final reduced system does not have exactly 7 gauge modes",
                        initial_cost=initial_cost)
    if cancelled():
        return _failure(initial_poses, initial_landmarks,
                        "deadline exceeded", initial_cost=initial_cost)
    return LocalBAResult(
        ok=True, reason="", poses=pose_state, landmarks=landmark_state,
        initial_cost=float(initial_cost), final_cost=float(current_cost),
        iterations=accepted, pose_ids=pose_ids,
        reduced_hessian=reduced_hessian,
        reduced_gradient=reduced_gradient,
        linearization_poses=tuple(pose_state[key].copy() for key in pose_ids),
        n_observations=required_valid,
    )


def _failure(poses, landmarks, reason, *, initial_cost=np.inf):
    pose_ids = tuple(sorted(poses))
    size = 6 * len(pose_ids)
    return LocalBAResult(
        ok=False, reason=reason, poses=poses, landmarks=landmarks,
        initial_cost=float(initial_cost), final_cost=float(initial_cost),
        iterations=0, pose_ids=pose_ids,
        reduced_hessian=np.zeros((size, size)),
        reduced_gradient=np.zeros(size),
        linearization_poses=tuple(poses[key].copy() for key in pose_ids),
        n_observations=0,
    )
