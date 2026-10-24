"""Square-root priors over poses and persistent log-scales.

Experimental canonical-graph backend. Marginalization is a row projection,
not a pose-only submatrix extraction. No damping or gauge rows are stored.
"""
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .se3 import se3_inv, se3_log, se3_right_jacobian_inv
from .se3_window import SlidingWindowOptimizer


@dataclass(frozen=True)
class JointPrior:
    keys: tuple
    x0: tuple
    A: np.ndarray
    r0: np.ndarray
    sources: frozenset
    # Provenance, not a numerical rank inference at the current estimate.
    # False is conservative for externally supplied absolute/partial priors.
    relative_pose: bool = False
    pose_components: tuple = ()

    def __post_init__(self):
        width = sum(6 if kind == "pose" else 1 for kind, _ in self.keys)
        if (len(set(self.keys)) != len(self.keys)
                or len(self.x0) != len(self.keys)
                or self.A.ndim != 2 or self.A.shape[1] != width
                or self.r0.shape != (self.A.shape[0],)):
            raise ValueError("invalid joint prior dimensions/keys")
        if not np.isfinite(self.A).all() or not np.isfinite(self.r0).all():
            raise ValueError("non-finite joint prior")
        if self.pose_components:
            flat = [p for group in self.pose_components for p in group]
            poses = {k for kind, k in self.keys if kind == "pose"}
            if (any(not group for group in self.pose_components)
                    or len(flat) != len(set(flat)) or set(flat) != poses):
                raise ValueError("invalid pose component provenance")


def project_rows(J, r, eliminate_columns, keep_columns, rtol=1e-10):
    """min_e ||J_e e + J_s s + r||², including rank-deficient J_e.

    Equilibrate eliminated columns before rank detection, since poses contain
    radians/metres and scales are logarithmic. No artificial diagonal jitter.
    """
    Je = J[:, eliminate_columns]
    A = J[:, keep_columns].copy()
    residual = r.copy()
    if Je.shape[1]:
        norms = np.linalg.norm(Je, axis=0)
        U, s, _ = sla.svd(Je / np.maximum(norms, 1e-15), full_matrices=False)
        rank = int(np.count_nonzero(s > rtol * s[0])) if s.size else 0
        Q = U[:, :rank]
        A -= Q @ (Q.T @ A)
        residual -= Q @ (Q.T @ residual)
    # Compress rows without dropping the orthogonal residual constant.
    if A.shape[0] > A.shape[1] + 1:
        _, R = sla.qr(np.column_stack((A, residual)), mode="economic")
        A, residual = R[:, :-1], R[:, -1]
    return A, residual


class JointWindowOptimizer(SlidingWindowOptimizer):
    """Legacy edge model plus priors referencing stable scalar identities."""

    def __init__(self, pose_gauge_policy="rank", **kwargs):
        if pose_gauge_policy not in ("rank", "physical"):
            raise ValueError("pose_gauge_policy must be rank or physical")
        self.pose_gauge_policy = pose_gauge_policy
        super().__init__(**kwargs)
        self.joint_priors = []
        self.scalar_col = {}
        self.scalar_value = {}
        self.scalar_edge = {}
        self._regularization_priors = []
        # Scale keys whose regularization term is suppressed (their information
        # comes from an explicit prior instead, e.g. bearing loop scales).
        self.scale_regularizer_exclude = set()

    def add_nl_regularization(self, *args, **kwargs):
        before = len(self.prior_factors)
        super().add_nl_regularization(*args, **kwargs)
        self._regularization_priors.extend(self.prior_factors[before:])

    def pop_last_prior_factor(self):
        if not self.prior_factors:
            return False
        removed = self.prior_factors[-1]
        self._regularization_priors = [p for p in self._regularization_priors
                                       if p is not removed]
        return super().pop_last_prior_factor()

    def add_scalar(self, key, value=0.0):
        if key in self.scalar_col:
            raise ValueError(f"duplicate scalar {key}")
        self.scalar_col[key] = self.n_scale
        self.scalar_value[key] = float(value)
        self.n_scale += 1

    def add_named_edge(self, key, i, j, M, **kwargs):
        if key in self.scalar_col:
            raise ValueError(f"duplicate edge scale {key}")
        super().add_edge(i, j, M, scale_free=True, **kwargs)
        e = len(self.edges) - 1
        self.scalar_col[key] = self.scale_col[e]
        self.scalar_edge[key] = e

    def log_scale(self, key):
        if key in self.scalar_edge:
            return float(np.log(self.edge_scale[self.scalar_edge[key]]))
        return self.scalar_value[key]

    def add_joint_prior(self, prior):
        for kind, key in prior.keys:
            if kind == "pose" and key not in self.T:
                raise KeyError(key)
            if kind == "scale" and key not in self.scalar_col:
                raise KeyError(key)
            if kind not in ("pose", "scale"):
                raise ValueError(kind)
        self.joint_priors.append(prior)

    def variable_keys(self):
        keys = [("pose", p) for p in self.pose_ids if p not in self.fixed_ids]
        keys += [("scale", k) for k in sorted(self.scalar_col,
                                              key=self.scalar_col.get)]
        return keys

    def _key_columns(self):
        out, pos = {}, 0
        for key in self.variable_keys():
            n = 6 if key[0] == "pose" else 1
            out[key] = list(range(pos, pos + n))
            pos += n
        return out

    def _joint_delta(self, fac):
        if not fac.keys:
            return np.zeros(0)  # fully eliminated component: constant cost only
        return np.concatenate([
            se3_log(se3_inv(x0) @ self.T[key]) if kind == "pose"
            else np.array([self.log_scale(key) - x0])
            for (kind, key), x0 in zip(fac.keys, fac.x0)])

    def _residuals(self):
        base = super()._residuals()
        return np.concatenate([base] + [f.A @ self._joint_delta(f) + f.r0
                                        for f in self.joint_priors])

    def _huber_weights(self):
        return np.concatenate([super()._huber_weights()] +
                              [np.ones(f.A.shape[0]) for f in self.joint_priors])

    def _jacobian_coo(self, eps=1e-6):
        rows, cols, vals, shape = super()._jacobian_coo(eps)
        rows, cols, vals = list(rows), list(cols), list(vals)
        base = shape[0] - sum(f.A.shape[0] for f in self.joint_priors)
        columns = self._key_columns()
        for fac in self.joint_priors:
            offset = 0
            for key, x0 in zip(fac.keys, fac.x0):
                width = 6 if key[0] == "pose" else 1
                block = fac.A[:, offset:offset + width]
                offset += width
                if key not in columns:  # explicit fixed pose
                    continue
                if key[0] == "pose":
                    delta = se3_log(se3_inv(x0) @ self.T[key[1]])
                    block = block @ se3_right_jacobian_inv(delta)
                rr, cc = np.nonzero(block)
                rows.extend(base + rr)
                cols.extend(np.asarray(columns[key])[cc])
                vals.extend(block[rr, cc])
            base += fac.A.shape[0]
        return (np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64), np.asarray(vals), shape)

    def _pose_components(self):
        """Structural scopes survive row compression of disconnected priors."""
        adjacency = {p: set() for p in self.pose_ids}
        absolute = set(self.fixed_ids)
        groups = [(i, j) for i, j, *_ in self.edges]
        for f in self.joint_priors:
            poses = [k for kind, k in f.keys if kind == "pose"]
            groups.extend(f.pose_components or (poses,))
            if not f.relative_pose:
                absolute.update(poses)
        for factor in self.prior_factors:
            if any(factor is p for p in self._regularization_priors):
                continue  # temporary damping is not an absolute observation
            groups.append(factor[0])
            absolute.update(factor[0])
        for group in groups:
            for p in group:
                adjacency[p].update(group)
        remaining = set(adjacency)
        components = []
        while remaining:
            component, stack = set(), [min(remaining)]
            while stack:
                p = stack.pop()
                if p not in component:
                    component.add(p)
                    stack.extend(adjacency[p] - component)
            remaining -= component
            components.append(component)
        return components, absolute

    def _gauge_anchor_columns(self, free, n_node):
        # Fix physical pose gauges before resolving other numerical nulls.
        components, absolute = (self._pose_components() if self.pose_gauge_policy == "physical"
                                else ([], set()))
        anchors = set()
        offsets = {p: 6*i for i, p in enumerate(free)}
        for component in components:
            if not component & absolute:
                start = offsets[min(component)]
                anchors.update(range(start, start + 6))
        # Absolute/partially constrained components are handled conservatively
        # by rank detection; never freeze their whole first pose by assumption.
        J = self._jacobian() * np.sqrt(self._huber_weights())[:, None]
        columns = np.array([i for i in range(J.shape[1]) if i not in anchors], dtype=int)
        if not J.size:
            return set(range(J.shape[1]))
        if not columns.size:
            return anchors
        reduced = J[:, columns]
        norms = np.linalg.norm(reduced, axis=0)
        _, R, piv = sla.qr(reduced / np.maximum(norms, 1e-15), pivoting=True,
                           mode="economic")
        d = np.abs(np.diag(R))
        rank = int(np.count_nonzero(d > 1e-10 * d.max())) if d.size else 0
        anchors.update(int(columns[c]) for c in piv[rank:])
        return anchors

    def _capture_scales(self):
        return super()._capture_scales(), dict(self.scalar_value)

    def _restore_scales(self, state):
        super()._restore_scales(state[0])
        self.scalar_value = state[1]

    def _apply_scale_step(self, dx):
        super()._apply_scale_step(dx)
        for key in self.scalar_value:
            self.scalar_value[key] += float(dx[self.scalar_col[key]])

    def marginalize_joint(self, eliminate, sources=frozenset(),
                          condition_scales=False):
        """Summarize this *owned subgraph* over every remaining variable.

        Caller must supply only consumed factors, at the parent solve's state.
        condition_scales is a deliberately overconfident diagnostic variant.
        """
        columns = self._key_columns()
        eliminate = set(eliminate)
        if not eliminate <= columns.keys():
            raise ValueError("elimination contains absent/fixed variables")
        keep = [k for k in columns if k not in eliminate
                and not (condition_scales and k[0] == "scale")]
        ecols = [c for k in columns if k in eliminate
                 and not (condition_scales and k[0] == "scale")
                 for c in columns[k]]
        kcols = [c for k in keep for c in columns[k]]
        weights = np.sqrt(self._huber_weights())
        A, r0 = project_rows(self._jacobian() * weights[:, None],
                             self._residuals() * weights, ecols, kcols)
        x0 = tuple(self.T[k].copy() if kind == "pose" else self.log_scale(k)
                   for kind, k in keep)
        relative_pose = (not self.fixed_ids and not self.prior_factors
                         and all(f.relative_pose or not any(kind == "pose" for kind, _ in f.keys)
                                 for f in self.joint_priors))
        kept_poses = {k for kind, k in keep if kind == "pose"}
        components, _ = self._pose_components()
        scopes = tuple(tuple(sorted(c & kept_poses)) for c in components if c & kept_poses)
        return JointPrior(tuple(keep), x0, A, r0, frozenset(sources), relative_pose, scopes)
