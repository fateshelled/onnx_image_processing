"""Accuracy-first canonical graph used to validate segment marginalization.

Every closed segment has disjoint measurement ownership. All historical KFs
remain free and every accepted loop is present in every solve: the first
implementation deliberately has no local/global information handoff or cap.
The raw backend additionally keeps every transient pose for a reference solve.
"""
from dataclasses import dataclass

import numpy as np

from .joint_prior import JointPrior
from .loop_closure import edge_key


MODES = {"canonical_raw", "canonical_conditional", "canonical_marginal",
         "canonical_retained", "canonical_shared"}


@dataclass
class Measurement:
    key: tuple
    a: int
    b: int
    M: np.ndarray
    kwargs: dict


class CanonicalGraph:
    def __init__(self, graph, mode):
        self.graph = graph
        self.mode = mode
        self.raw = {}  # FactorId -> measurement, never coexists with its summary
        self.priors = []
        self.poses = {}
        self.scales = {}
        self.sources = set()
        self.previous_lambda = None
        self.revision = 0

    def _window(self, measurements, priors, poses):
        g = self.graph
        act = g._new_window()
        ids = set(poses)
        for e in measurements:
            ids.update((e.a, e.b))
        for f in priors:
            ids.update(k for kind, k in f.keys if kind == "pose")
        correction = None
        for k in sorted(ids):
            if k in self.poses:
                pose = self.poses[k]
            elif k in g._est:
                pose = g._est[k]
            else:
                # Raw odometry stays in its original frame (also used by KF
                # gates and output reconstruction). Initialize only new poses
                # relative to the latest optimized segment boundary instead
                # of mixing raw and optimized world frames in this window.
                if correction is None:
                    anchor = g._kf[-1]
                    boundary = self.poses.get(anchor, g._est.get(anchor, g._T[anchor]))
                    correction = boundary @ np.linalg.inv(g._T[anchor])
                pose = correction @ g._T[k]
            act.add_node(k, pose)
        # Stable names survive both changes in edge order and removal of an edge.
        for e in measurements:
            act.add_named_edge(e.key, e.a, e.b, e.M,
                               scale=np.exp(self.scales.get(e.key, 0.0)),
                               **e.kwargs)
        scalar_keys = {k for f in priors for kind, k in f.keys if kind == "scale"}
        for key in sorted(scalar_keys):
            if key not in act.scalar_col:
                act.add_scalar(key, self.scales.get(key, 0.0))
        for f in priors:
            act.add_joint_prior(f)
        return act

    @staticmethod
    def _scalar_factor(keys, coefficients, sigma, source):
        return JointPrior(tuple(("scale", k) for k in keys),
                          tuple(0.0 for _ in keys),
                          np.array([coefficients], float) / sigma,
                          np.zeros(1), frozenset({source}))

    def assert_ownership(self):
        owners = set(self.raw)
        for fac in self.priors:
            if owners & fac.sources:
                raise AssertionError("observation counted more than once")
            owners.update(fac.sources)
        if owners != self.sources:
            raise AssertionError("lost canonical observation")

    def close_segment(self, b):
        g = self.graph
        a = g._kf[-1]
        trans = [i for i in g._seg if i not in (a, b)]
        measurements, scale_factors = [], []
        shared = self.mode == "canonical_shared"
        lam = None
        previous = self.previous_lambda
        if shared:
            sigma = float(g.p.get("segment_local_sigma", g.p.get("scale_prior_sigma", .6)))
            transition = float(g.p.get("segment_transition_sigma", .2))
            anchor = float(g.p.get("segment_anchor_sigma", .6))
            if not np.isfinite([sigma, transition, anchor]).all() or min(sigma, transition, anchor) <= 0:
                raise ValueError("segment scale sigmas must be positive")
        for v in trans + [b]:
            if v in g._odom:
                p = g._prev[v]
                key = ("odom", p, v)
                kwargs = {"scale_prior_sigma": 0.0} if shared else {}
                measurements.append(Measurement(key, p, v, g._meas(*g._odom[v]), kwargs))
                if shared:
                    if lam is None:
                        # A failed chain edge starts a new scale process, even
                        # inside one KF segment. Spokes/loops can reconnect the
                        # geometry, but do not invent temporal scale continuity.
                        lam = ("segment", a, b, v)
                        self.scales[lam] = self.scales.get(previous, 0.0)
                        if previous is None:
                            scale_factors.append(self._scalar_factor(
                                [lam], [1.0], anchor, ("scale_anchor", a, b, v)))
                        else:
                            scale_factors.append(self._scalar_factor(
                                [previous, lam], [-1.0, 1.0], transition,
                                ("scale_transition", a, b, v)))
                    scale_factors.append(self._scalar_factor(
                        [key, lam], [1.0, -1.0], sigma, ("scale_local", p, v)))
                g._added.add(edge_key(p, v))
            elif shared:
                lam = previous = None
            for ref, R, t, _inl, _n in g._spoke.get(v, []):
                if ref == g._prev.get(v):
                    continue
                key = ("spoke", ref, v)
                measurements.append(Measurement(key, ref, v, g._meas(R, t), {}))
                g._added.add(edge_key(ref, v))
        # The loop gate sees exactly the same measurements as the legacy path.
        nloops = len(g._loops)
        g._add_loops(b)
        new_loops = [Measurement(("loop", x, y), x, y, g._meas(R, t),
                                 {"robust": True, **sig})
                     for x, y, R, t, sig in g._loops[nloops:]]
        if len({e.key for e in measurements + new_loops}) != len(measurements + new_loops):
            raise ValueError("duplicate canonical observation")
        segment_sources = {e.key for e in measurements}
        for f in scale_factors:
            segment_sources.update(f.sources)
        new_sources = segment_sources | {e.key for e in new_loops}
        if self.sources & new_sources:
            raise ValueError("canonical observation already registered")
        all_raw = list(self.raw.values()) + measurements + new_loops
        all_priors = self.priors + scale_factors
        nodes = set(g._kf) | {b} | set(trans)
        if self.mode == "canonical_raw":
            nodes |= set(self.poses)
        act = self._window(all_raw, all_priors, nodes)
        g.last_n_nodes = len(act.pose_ids)
        if g.p.get("nl_reg", True):
            act.add_nl_regularization(g.p.get("nl_reg_c", 10.0),
                                      g.p.get("nl_reg_tau", 10.0),
                                      g.p.get("nl_reg_length", 1.0))
        act.optimize()
        if g.p.get("nl_reg", True):
            act.pop_last_prior_factor()
        harvested = act.harvest_robust_residuals()
        g.n_robust_downweighted += act.n_robust_downweighted
        g.n_robust_rot_downweighted += act.n_robust_rot_downweighted
        g.n_robust_dir_downweighted += act.n_robust_dir_downweighted

        # Construct the replacement before publishing a new graph revision.
        # Pose/scale estimates here are scratch linearization states.
        old_poses, old_scales = self.poses, dict(self.scales)
        self.poses = {k: act.get_pose(k) for k in act.pose_ids}
        self.scales.update({k: act.log_scale(k) for k in act.scalar_col})
        try:
            summary = None
            if self.mode != "canonical_raw" and segment_sources:
                sub = self._window(measurements, scale_factors, set())
                eliminate = {("pose", k) for k in trans if k in sub.T}
                if self.mode != "canonical_retained":
                    eliminate.update(("scale", e.key) for e in measurements)
                # Lambda states (and all separator KF poses) remain explicit.
                summary = sub.marginalize_joint(
                    eliminate, segment_sources,
                    condition_scales=self.mode == "canonical_conditional")
        except Exception:
            self.poses, self.scales = old_poses, old_scales
            raise
        if self.mode == "canonical_raw":
            self.raw.update({e.key: e for e in measurements})
        elif summary is not None:
            self.priors.append(summary)
        self.raw.update({e.key: e for e in new_loops})
        self.sources.update(new_sources)
        self.assert_ownership()
        if self.mode != "canonical_raw":
            for k in trans:
                self.poses.pop(k, None)
            if self.mode != "canonical_retained":
                for e in measurements:
                    self.scales.pop(e.key, None)
        for k in list(g._kf) + [b]:
            g._est[k] = act.get_pose(k)
        # Keep the existing output reconstruction fixed across the A/B variants.
        for k in trans:
            g._rel[k] = (a, np.linalg.inv(g._T[a]) @ g._T[k])
            g._spoke.pop(k, None)
        g._kf.append(b)
        g._seg = []
        g._pending_global = False  # every solve already includes all loop factors
        g._last_global_kf = len(g._kf)
        if g.p.get("loop_scale_mode", "fixed") == "mad":
            g._commit_loop_history(harvested)
        if shared:
            self.previous_lambda = lam
        self.revision += 1
