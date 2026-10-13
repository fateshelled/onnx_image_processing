"""Accuracy-first canonical graph used to validate segment marginalization.

Every closed segment has disjoint measurement ownership. By default the
canonical backend is bounded (``canonical_cap``): active keyframes are limited
to W=max(canonical_max_keyframes, loop_window) and exited keyframes are
Schur-absorbed; with the cap off all historical KFs remain free and every
accepted loop is present in every solve. The raw backend additionally keeps
every transient pose for a reference solve.
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
        # Bounding (opt-in via graph param "canonical_cap"): active keyframes are
        # the free nodes of the reduced graph. Exited keyframes are Schur
        # absorbed into their neighbours and dropped from the solve.
        self.active = None  # set of active keyframe ids; None until first close
        self.absorbed = set()
        self.n_over_cap = 0
        self.n_absorbed = 0
        self.n_retired_loops = 0

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

    # -- bounding (opt-in) ---------------------------------------------------

    def _loop_window(self):
        # Must match the loop *detection* window in OnlinePoseGraph._add_loops
        # (default 80) so condition B keeps every detectable endpoint active.
        # The caller rejects non-positive values (see close_segment).
        return int(self.graph.p.get("loop_window", 80))

    def _cap_window(self):
        g = self.graph
        w = max(int(g.p.get("canonical_max_keyframes")
                    or g.p.get("max_keyframes", 12) or 12),
                self._loop_window())
        return w

    def _has_incident_loop(self, E):
        return any(key[0] == "loop" and E in (m.a, m.b)
                   for key, m in self.raw.items())

    def _blanket_active(self, E):
        """True when E has incident factors and all their pose endpoints are
        active. A keyframe with no incident factor (measurement-free segment)
        is not absorbable and returns False so the caller can fall back.
        """
        found = False
        for pr in self.priors:
            if ("pose", E) in pr.keys:
                found = True
                for kind, k in pr.keys:
                    if kind == "pose" and k not in self.active:
                        return False
        for key, m in self.raw.items():
            if key[0] == "loop" and E in (m.a, m.b):
                found = True
                if (m.a if m.b == E else m.b) not in self.active:
                    return False
        return found

    def absorb(self, E):
        """Schur-absorb active keyframe E into its active neighbours.

        Combines E's incident priors (and SE3 loop edges) into a single joint
        prior over the neighbours, removes the consumed factors, freezes E's
        pose for output and drops it from the active set. Bearing loop edges
        cannot be marginalized, so the caller must exclude such E.
        """
        g = self.graph
        if E not in self.poses:
            raise KeyError(f"absorb target {E} is not an active pose")
        inc_pri = [pr for pr in self.priors if ("pose", E) in pr.keys]
        inc_loop = [(key, m) for key, m in self.raw.items()
                    if key[0] == "loop" and E in (m.a, m.b)]
        if inc_loop and g.p.get("loop_observation_model", "se3") == "bearing":
            raise ValueError("cannot absorb a keyframe with bearing loops")
        nbr = set()
        for pr in inc_pri:
            nbr |= {k for kind, k in pr.keys if kind == "pose" and k != E}
        for _key, m in inc_loop:
            nbr.add(m.a if m.b == E else m.b)
        if not nbr <= self.active:
            raise RuntimeError(f"absorb {E}: blanket is not active")
        # nbr may be empty when E only carries unary (single-node) priors; the
        # summary is then a constant over no variable and E is safely frozen
        # and dropped. (Constant summaries are retained for ownership and
        # pruned only when loop/observation retirement lands.)
        sub = g._new_window()
        for nd in sorted(nbr | {E}):
            pose = self.poses.get(nd)
            sub.add_node(nd, pose if pose is not None else g._est[nd])
        scale_keys = {k for pr in inc_pri for kind, k in pr.keys
                      if kind == "scale"}
        for key in sorted(scale_keys):
            sub.add_scalar(key, self.scales.get(key, 0.0))
        for pr in inc_pri:
            sub.add_joint_prior(pr)
        for key, m in inc_loop:
            sub.add_named_edge(
                m.key, m.a, m.b, m.M,
                scale=float(np.exp(self.scales.get(m.key, 0.0))), **m.kwargs)
        eliminate = {("pose", E)}
        eliminate |= {("scale", k) for k in scale_keys}
        eliminate |= {("scale", m.key) for _key, m in inc_loop}
        sources = set()
        for pr in inc_pri:
            sources |= pr.sources
        for key, _m in inc_loop:
            sources.add(key)
        summary = sub.marginalize_joint(eliminate, sources)
        # Publish: freeze E's pose, drop it, replace consumed factors.
        g._est[E] = self.poses[E].copy()
        self.poses.pop(E, None)
        inc_pri_ids = {id(pr) for pr in inc_pri}
        self.priors = [pr for pr in self.priors if id(pr) not in inc_pri_ids]
        for key, _m in inc_loop:
            self.raw.pop(key, None)
            self.scales.pop(key, None)
            g._loops = [lp for lp in g._loops
                        if edge_key(lp[0], lp[1]) != key[1:]]
        self.priors.append(summary)
        self.active.discard(E)
        self.absorbed.add(E)
        self.n_absorbed += 1

    def _retire_oldest_bearing_loop(self):
        """Drop the oldest retained bearing loop to free its held endpoints.

        Bearing loops cannot be Schur-absorbed, so their endpoints stay active;
        retiring a loop (dropping the constraint) is the only way to bound the
        active set. SE3 loops are absorbable and never retired. The dropped
        constraint is an approximation, counted in ``n_retired_loops``.
        """
        g = self.graph
        if g.p.get("loop_observation_model", "se3") != "bearing":
            return False  # SE3 loops are absorbable and must not be retired
        loops = [(key, m) for key, m in self.raw.items() if key[0] == "loop"]
        if not loops:
            return False
        key, m = min(loops, key=lambda kv: (min(kv[1].a, kv[1].b),
                                            max(kv[1].a, kv[1].b)))
        self.raw.pop(key, None)
        self.sources.discard(key)
        self.scales.pop(key, None)
        g._loops = [lp for lp in g._loops if edge_key(lp[0], lp[1]) != key[1:]]
        self.n_retired_loops += 1
        return True

    def _absorb_overflow(self):
        # Runs after the segment solve (order solve -> absorb). last_n_nodes is
        # therefore the solve size (<= W + 1); the persistent bound on
        # ``self.active`` is what M1 checks.
        g = self.graph
        w = self._cap_window()
        idx = {k: i for i, k in enumerate(g._kf)}
        loop_window = self._loop_window()
        bearing = g.p.get("loop_observation_model", "se3") == "bearing"
        while len(self.active) > w:
            chosen = None
            for E in sorted(self.active):
                age = len(g._kf) - 1 - idx.get(E, len(g._kf) - 1)
                if age < loop_window:
                    continue
                if bearing and self._has_incident_loop(E):
                    continue
                if not self._blanket_active(E):
                    continue
                chosen = E
                break
            if chosen is not None:
                self.absorb(chosen)
                continue
            # No absorbable keyframe: a bearing loop endpoint is blocking.
            # Retire the oldest retained loop (drop its constraint) so its
            # endpoint can be absorbed next iteration.
            if bearing and self._retire_oldest_bearing_loop():
                continue
            # Nothing left to free (e.g. measurement-free keyframes): the
            # bound cannot be met for this step; count it and stop.
            self.n_over_cap += 1
            break

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
        # canonical_cap is opt-in; canonical_raw keeps every keyframe free and
        # therefore silently ignores it. canonical_shared is rejected below.
        cap = bool(g.p.get("canonical_cap", False)) and self.mode != "canonical_raw"
        if cap and self.mode == "canonical_shared":
            # Shared lambda scale keys span adjacent segments; absorbing one
            # endpoint would eliminate a key still referenced elsewhere.
            raise ValueError("canonical_cap is unsupported with canonical_shared")
        if cap and self._loop_window() <= 0:
            # Condition B (never absorb inside the loop detection window)
            # requires a positive window; loop_window<=0 means "scan all KFs".
            raise ValueError("canonical_cap requires loop_window > 0")
        if cap:
            if self.active is None:
                self.active = set(g._kf)
            live = set(self.active) | {b} | set(trans)
            outside = [e.key for e in all_raw if e.a not in live or e.b not in live]
            outside += [f for f in all_priors
                        if any(kind == "pose" and k not in live
                               for kind, k in f.keys)]
            if outside:
                raise RuntimeError(
                    "canonical cap: retained factor outside the active set")
            nodes = live
        else:
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
        if cap:
            # Only keyframes (never transients) are stored as absolute poses;
            # transients keep their anchor-relative reconstruction.
            for k in sorted(set(self.active) | {b}):
                if k in act.pose_ids:
                    g._est[k] = act.get_pose(k)
        else:
            for k in list(g._kf) + [b]:
                g._est[k] = act.get_pose(k)
        # Keep the existing output reconstruction fixed across the A/B variants.
        for k in trans:
            g._rel[k] = (a, np.linalg.inv(g._T[a]) @ g._T[k])
            g._spoke.pop(k, None)
        g._kf.append(b)
        g._seg = []
        if cap:
            self.active.add(b)
            self._absorb_overflow()
        g._pending_global = False  # every solve already includes all loop factors
        g._last_global_kf = len(g._kf)
        if g.p.get("loop_scale_mode", "fixed") == "mad":
            g._commit_loop_history(harvested)
        if shared:
            self.previous_lambda = lam
        self.revision += 1
