"""Online keyframe pose graph with a fixed-lag window and NL-Reg.

This module owns **all** the pose-graph logic so the offline evaluator
(``eval/rustuna_tune_loop.py``) and the streaming sample
(``sample/visual_odometry.py``) share one implementation:

* keyframe promotion by a motion criterion;
* per-frame odometry propagation (chain) and additive keyframe spokes;
* Schur marginalization of transient frames into relative priors;
* deferred (option 1) marginalization of exited keyframes into multi-node
  priors;
* loop closure with the temporal-consistency gate and cycle handling;
* direction-dependent Tikhonov regularization (NL-Reg).

The graph is matcher-agnostic and **index-based**: the caller passes
``match_fn(i, j) -> dict`` where the dict has keys ``ok`` (bool), ``R``
(3x3), ``t`` (3,), ``inlier_ratio`` and ``n_matches``. Indices are opaque
increasing frame ids (stride-frame ids for the evaluator, frame counters for
the sample); the graph stores no images, only poses/keyframes/priors, so the
caller owns whatever features it needs to answer the matches.

Pose convention matches ``Trajectory.add_relative_pose``: the relative
measurement ``(R, t)`` is a world-to-camera point transform
(``x_curr = R @ x_prev + t``, as returned by ``cv2.recoverPose``).
"""

from __future__ import annotations

import numpy as np

from .cycle_consistency import rotation_angle_deg
from .loop_closure import confirmed_loop_hits, edge_key
from .scale_kf import ScaleKF
from .se3_window import SlidingWindowOptimizer

# Default parameters for the online sequential graph. Adopted from the
# 200-trial optuna study "online_bounded_kf" (best trial 174, mean+worst
# ATE 0.5245 on desk/desk2/room; a near-tie with trials 119/121/122 at
# 0.5247). Some values sit on the search-grid boundary (kf_max_gap=8,
# max_keyframes=12) and may improve further with a wider grid.
DEFAULT_PARAMS = {
    "odom_ref": "kf", "kf_mode": "motion", "keyframe_decim": 15,
    "kf_trans_thresh": 4.0, "kf_rot_thresh": 30.0, "kf_max_gap": 8,
    "kf_local_map_k": 2, "kf_edge_min_inlier": 0.0, "trans_gate_deg": 0.0,
    "loop_window": 60, "loop_min_gap": 20, "loop_min_inlier": 0.5,
    "loop_temporal_k": 2, "loop_sigma_scale": 1.0,
    "loop_robust": "none", "loop_robust_phi": 1.0,
    "loop_robust_min_weight": 0.0,
    "loop_gm_mu": 11.34, "loop_dir_sigma": 0.4,
    "loop_robust_gnc_gamma": 1.4,
    # Rotation-only odometry-cycle gate. Zero keeps it disabled; non-zero
    # values reject loop measurements whose rotation disagrees with the
    # independently accumulated odometry chain by more than this many degrees.
    "cycle_threshold_deg": 0.0,
    "scale_prior_sigma": 0.6, "step_scale_t": 0.1, "loop_iterations": 10,
    "tsvd_ratio": 0.0, "scale_kf": False,
    "scale_kf_q": 1e-3, "scale_kf_r": 0.1, "scale_kf_sigma": 0.5,
    "graph_mode": "kf_prior", "max_keyframes": 12, "seq_tsvd_ratio": 0.0,
    "nl_reg": True, "nl_reg_c": 10.0, "nl_reg_tau": 30.0, "nl_reg_length": 0.5,
    # Bounding: hard cap on held (exited-but-referenced) keyframes; None uses
    # 2 * max_keyframes. Loop closures additionally trigger one global reduced
    # pass over all keyframes (Tier 2) when global_opt_on_loop is set.
    "held_cap": 4, "global_opt_on_loop": True, "global_opt_period": 10,
}


def _compose(T_prev, R, t):
    """Compose a camera-to-world pose with a relative (R, t) measurement."""
    R_new = T_prev[:3, :3] @ np.asarray(R, float).T
    t = np.asarray(t, float).reshape(3)
    C_new = T_prev[:3, 3] - R_new @ t
    T = np.eye(4)
    T[:3, :3] = R_new
    T[:3, 3] = C_new
    return T


def _rot_deg(R):
    co = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(co, -1.0, 1.0))))


def _trans_consistent(dT, t, gate_deg):
    """True when the chain translation and the measured one agree in direction."""
    if gate_deg <= 0.0:
        return True
    d = np.asarray(dT[:3, 3], float).reshape(3)
    t = np.asarray(t, float).reshape(3)
    n = np.linalg.norm(d) * np.linalg.norm(t)
    if n < 1e-12:
        return True
    cos = float(np.clip(float(d @ t) / n, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos))) <= gate_deg


class OnlinePoseGraph:
    """Online sequential pose graph.

    Parameters are read from a plain dict (defaults come from
    ``DEFAULT_PARAMS``). The graph keeps at most ``max_keyframes`` active
    keyframes; transient frames are marginalized into relative priors when a
    new keyframe is promoted, and exited keyframes are deferred (option 1)
    until their Markov blanket is active, then marginalized into multi-node
    priors.
    """

    def __init__(self, params, cam, match_fn):
        self.p = params
        self.cam = cam
        self.match = match_fn
        self.kf_min_gap = 4
        self.kf_max_gap = int(params.get("kf_max_gap", 16))
        self.max_kf = params.get("max_keyframes")
        self._T = {}            # odom chain camera-to-world
        self._kf = []
        self._est = {}          # optimized keyframe poses
        self._last = None
        self._prev = {}         # frame id -> previous frame id (chain)
        self._step = 1          # index step between consecutive frames
        self._seg = []          # transient ids since the last keyframe
        self._odom = {}         # frame id -> (R, t) relative to the previous frame
        self._spoke = {}        # frame id -> (ref_kf, R, t, inlier, n_matches)
        self._priors = []       # pairwise transient priors (a, b, G, Omega)
        self._node_priors = []  # multi-node priors (ids, H, b, x0)
        self._hits = []         # per-keyframe loop hits (temporal confirmation)
        self._added = set()     # edge_key set (odometry + accepted closures)
        self._loops = []        # accepted loop edges (a, b, R, t, sig)
        # Kalman filter on the motion-normalised keyframe-spoke scale
        # (unit-norm translation -> baseline). Sequential: each keyframe's
        # spoke is recentred with the k_hat from the previous keyframes.
        self._scale_kf = None
        if params.get("scale_kf", False):
            self._scale_kf = ScaleKF(params.get("scale_kf_q", 1e-3),
                                     params.get("scale_kf_r", 0.1))
        self._pending_global = False  # a loop arrived; Tier-2 pass is owed
        self.n_loop = 0
        self.n_cycle_rejected = 0
        self.n_robust_downweighted = 0
        self.n_robust_rot_downweighted = 0
        self.n_robust_dir_downweighted = 0
        self.last_n_nodes = 0
        self._held_cap = params.get("held_cap")
        if self._held_cap is None and self.max_kf:
            self._held_cap = 2 * int(self.max_kf)
        self._eliminated = set()  # keyframes Schur-marginalized out
        self._rel = {}            # closed transient id -> (anchor_kf, rel transform)
        self.n_dropped_held = 0   # unique held keyframes dropped by the cap
        self._dropped_held = set()
        self._last_global_kf = 0  # keyframe count at the last Tier-2 pass
        self.last_stats = {"ok": True, "inlier_ratio": 1.0, "n_matches": 0}

    # -- public -------------------------------------------------------------

    @property
    def n_kf(self):
        return len(self._kf)

    @property
    def keyframes(self):
        """All keyframe ids seen so far (including eliminated ones)."""
        return list(self._kf)

    @property
    def match_keyframes(self):
        """Keyframes the loop matcher may query (recent ``loop_window`` only).

        Callers that cache per-frame features can prune to this set; it stays
        bounded even though ``keyframes`` grows with the sequence.
        """
        w = int(self.p.get("loop_window", 80))
        return list(self._kf) if w <= 0 else self._kf[-w:]

    def add_frame(self, idx, odom=None):
        """Process one frame ``idx`` and return its camera-to-world 4x4 pose.

        ``odom`` optionally supplies the chain measurement ``(R, t)`` from the
        previous frame; when omitted it is obtained from ``match_fn``. The
        evaluator injects its precomputed odometry this way.
        """
        idx = int(idx)
        if self._last is None:
            self._T[idx] = np.eye(4)
            self._kf = [idx]
            self._est[idx] = np.eye(4)
            self._last = idx
            self.last_stats = {"ok": True, "inlier_ratio": 1.0, "n_matches": 0}
            return np.eye(4)

        prev = self._last
        if odom is None:
            res = self.match(prev, idx)
            ok = bool(res and res.get("ok"))
            R = np.asarray(res["R"], float) if ok else None
            t = np.asarray(res["t"], float).reshape(3) if ok else None
            inl = float(res.get("inlier_ratio", 0.0)) if res else 0.0
            nm = int(res.get("n_matches", 0)) if res else 0
        else:
            ok = odom[0] is not None
            R = np.asarray(odom[0], float) if ok else None
            t = np.asarray(odom[1], float).reshape(3) if ok else None
            inl = float(odom[2]) if len(odom) > 2 else 1.0
            nm = int(odom[3]) if len(odom) > 3 else -1
        self.last_stats = {"ok": bool(ok), "inlier_ratio": inl, "n_matches": nm}

        self._prev[idx] = prev
        self._step = idx - prev
        T = self._T[prev]
        if ok:
            self._odom[idx] = (R, t)
            T = _compose(T, R, t)
        self._T[idx] = T

        # Additive keyframe spokes: constrain the frame to the last K keyframes
        # (K=1 keeps the previous single-hub behaviour).
        gate_deg = float(self.p.get("trans_gate_deg", 0.0))
        min_inl = float(self.p.get("kf_edge_min_inlier", 0.0))
        for ref in self._kf[-max(1, int(self.p.get("kf_local_map_k", 1))):]:
            if ref == prev:
                continue
            sres = self.match(ref, idx)
            if not (sres and sres.get("ok")):
                continue
            if float(sres.get("inlier_ratio", 0.0)) < min_inl:
                continue
            dT = np.linalg.inv(self._T[idx]) @ self._T[ref]
            if not _trans_consistent(dT, sres.get("t"), gate_deg):
                continue
            self._spoke[idx] = (int(ref), np.asarray(sres["R"], float),
                                np.asarray(sres["t"], float).reshape(3),
                                float(sres.get("inlier_ratio", 0.0)),
                                int(sres.get("n_matches", -1)))

        self._last = idx
        self._seg.append(idx)
        if self._promote(idx):
            self._close_segment(idx)
        return self._pose_of(idx)

    def pose(self, frame_id=None):
        return self._pose_of(self._last if frame_id is None else frame_id)

    # -- internals ----------------------------------------------------------

    def _promote(self, fid):
        last_kf = self._kf[-1]
        dT = np.linalg.inv(self._T[fid]) @ self._T[last_kf]
        trans = float(np.linalg.norm(dT[:3, 3]))
        rot = _rot_deg(dT[:3, :3])
        gap = fid - last_kf
        if gap >= self.kf_max_gap:
            return True
        return (gap >= self.kf_min_gap
                and (trans >= self.p.get("kf_trans_thresh", 8.0)
                     or rot >= self.p.get("kf_rot_thresh", 10.0)))

    def _close_segment(self, new_kf):
        """Optimize [prev_kf, transients, new_kf], marginalize the transients.

        Mirrors the eval driver's option-1 scheme: exited keyframes referenced
        by a prior are kept (as free nodes) until their Markov blanket is fully
        active, then Schur-marginalized into a multi-node prior and dropped.
        """
        a = self._kf[-1]
        b = new_kf
        trans = [i for i in self._seg if i != b and i != a]
        # Active (free) keyframes: the last ``max_keyframes - 1`` existing
        # keyframes plus the new one, matching the legacy driver's window
        # (which includes b).
        if self.max_kf:
            n = max(1, int(self.max_kf))
            free_kf = (self._kf[-(n - 1):] if n > 1 else []) + [b]
        else:
            free_kf = list(self._kf) + [b]
        free_set = set(free_kf)
        # Held keyframes (bounded): exited keyframes that an existing prior
        # still links to a free node. Only one hop from the priors is used --
        # transitive propagation along KF-KF adjacency, which would pull in the
        # whole trajectory and defeat max_keyframes, is deliberately avoided.
        # Eliminated keyframes are never reintroduced.
        trans_set = set(trans)
        if a not in free_set:
            free_set.add(a)
        held = set()
        for (x, y, *_r) in self._priors:
            if x in free_set and y not in free_set and y not in self._eliminated:
                held.add(y)
            if y in free_set and x not in free_set and x not in self._eliminated:
                held.add(x)
        for (ids, *_r) in self._node_priors:
            if any(nd in free_set for nd in ids):
                for nd in ids:
                    if (nd not in free_set and nd not in trans_set
                            and nd not in self._eliminated):
                        held.add(nd)
        # Hard cap: node set is bounded by construction. The oldest held
        # keyframes beyond the cap are dropped (their pending priors are then
        # skipped); counted so the loss is observable, not silent.
        if self._held_cap is not None and len(held) > int(self._held_cap):
            keep = int(self._held_cap)
            excess = set(sorted(held)[:len(held) - keep])
            self.n_dropped_held += len(excess - self._dropped_held)
            self._dropped_held |= excess
            held = set(sorted(held)[len(held) - keep:]) if keep > 0 else set()
        node_set = set(free_kf) | held | trans_set | {a, b}
        nodes = sorted(node_set)
        self.last_n_nodes = len(node_set)

        act = self._new_window()
        for nd in nodes:
            act.add_node(nd, self._est.get(nd, self._T[nd]))
        for (x, y, G, Om) in self._priors:
            if x in node_set and y in node_set:
                act.add_edge(x, y, G, omega=Om)
        for (ids, H, bb, x0) in self._node_priors:
            if all(nd in node_set for nd in ids):
                act.add_prior_factor(ids, H, bb, x0)
        # Chain edges (consecutive frames) and spokes (frame -> keyframe).
        for v in trans + [b]:
            if v in self._odom:
                R, t = self._odom[v]
                p = self._prev[v]
                act.add_edge(p, v, self._meas(R, t), scale_free=True)
                self._added.add(edge_key(p, v))
        kf_spoke = None  # (act edge index, odometry baseline) for the new KF
        for v in trans + [b]:
            if v in self._spoke:
                ref, R, t, _inl, _n = self._spoke[v]
                if ref in node_set and ref != self._prev.get(v):
                    kwargs, m = {}, None
                    if v == b and self._scale_kf is not None:
                        # Unit-norm spoke translation -> baseline m: recentre
                        # its scale prior at log(k_hat * m) (Kalman filter).
                        m = float(np.linalg.norm(
                            self._T[b][:3, 3] - self._T[ref][:3, 3]))
                        sig = float(self.p.get("scale_kf_sigma", 0.5))
                        if (m > 1e-9 and sig > 0.0
                                and self._scale_kf.k is not None):
                            kwargs["scale_prior_mean"] = float(
                                np.log(max(self._scale_kf.k * m, 1e-6)))
                            kwargs["scale_prior_sigma"] = sig
                    act.add_edge(ref, v, self._meas(R, t),
                                 scale_free=True, **kwargs)
                    if v == b and m is not None and m > 1e-9:
                        kf_spoke = (len(act.edges) - 1, m)
                    self._added.add(edge_key(ref, v))
        self._add_loops(b)
        # Re-inject every accepted loop edge whose endpoints are both active,
        # exactly like the legacy driver's ``loop_edges``: a loop measured once
        # keeps constraining later windows instead of being used a single time.
        present = set()
        for v in trans + [b]:
            if v in self._odom:
                present.add(edge_key(self._prev[v], v))
            if v in self._spoke:
                present.add(edge_key(self._spoke[v][0], v))
        loop_act_ei = {}  # edge_key -> index in act.edges (for the local naive)
        for (la, lb, lR, lt, lsig) in self._loops:
            k = edge_key(la, lb)
            if la in node_set and lb in node_set and k not in present:
                act.add_edge(la, lb, self._meas(lR, lt), scale_free=True,
                             robust=True, **lsig)
                present.add(k)
                loop_act_ei[k] = len(act.edges) - 1
        nl_reg = bool(self.p.get("nl_reg", True))
        if nl_reg:
            act.add_nl_regularization(self.p.get("nl_reg_c", 10.0),
                                      self.p.get("nl_reg_tau", 10.0),
                                      self.p.get("nl_reg_length", 1.0))
        act.optimize()
        self.n_robust_downweighted += act.n_robust_downweighted
        self.n_robust_rot_downweighted += act.n_robust_rot_downweighted
        self.n_robust_dir_downweighted += act.n_robust_dir_downweighted
        # Kalman-filter update from the optimised spoke scale (z = s / m). The
        # filtered k_hat recentres the next keyframe's spoke scale prior.
        if (kf_spoke is not None
                and act.scale_col[kf_spoke[0]] is not None):
            self._scale_kf.update(
                float(act.edge_scale[kf_spoke[0]]) / kf_spoke[1])
        # NL-Reg is a solver-only regularizer: remove it before marginalizing so
        # it is neither counted as a blanket edge nor baked into the prior.
        if nl_reg:
            act.pop_last_prior_factor()
        n_pri = len(self._priors)  # factors assembled into `act` above
        if trans:
            aa, bb2, G, Om = act.marginalize_relative(trans, [a, b],
                                                      fix_scales=True)
            self._priors.append((aa, bb2, G, Om))
        # Only keyframes keep absolute optimized poses; transients are stored as
        # relative transforms from their anchor keyframe so a later global
        # update (Tier 2) moves them consistently.
        for nd in nodes:
            if nd in self._kf or nd == b:
                self._est[nd] = act.get_pose(nd)
        # Marginalize a held keyframe once its blanket is fully active. The
        # marginal is computed over E's own incident subgraph only (not the
        # whole window), and replaces exactly the factors incident to E, so
        # neither boundary factors (e.g. prev--prevprev) nor separator-internal
        # factors are double counted or lost.
        new_npri = []
        remove_pri_ids, remove_loop_ids, remove_npri_ids = set(), set(), set()
        for E in sorted(held):
            if E in self._eliminated or E not in act.pose_ids:
                continue
            # Only incident factors whose endpoints are ALL in the current
            # window enter the marginal (they were in `act`), so only those may
            # be removed. Incident factors reaching outside the window are not
            # represented by H_r; they are kept so Tier 2 (which re-adds their
            # far endpoints, possibly as fixed landmarks) can still use them.
            inc_pri = [pr for pr in self._priors[:n_pri]
                       if E in (pr[0], pr[1])
                       and pr[0] in node_set and pr[1] in node_set]
            inc_loop = [lp for lp in self._loops
                        if E in (lp[0], lp[1])
                        and lp[0] in node_set and lp[1] in node_set]
            inc_npri = [npr for npr in self._node_priors
                        if E in npr[0]
                        and all(nd in node_set for nd in npr[0])]
            if not (inc_pri or inc_loop or inc_npri):
                continue
            nbr = set()
            for (x, y, *_r) in inc_pri:
                nbr.add(x if y == E else y)
            for (la, lb, *_r) in inc_loop:
                nbr.add(la if lb == E else lb)
            for (ids, *_r) in inc_npri:
                nbr |= set(ids)
            nbr.discard(E)
            nbr = {x for x in nbr if x in node_set and x not in trans
                   and x not in self._eliminated}
            if not nbr or not nbr <= free_set:
                continue
            sub = self._new_window()
            for nd in sorted(set(nbr) | {E}):
                sub.add_node(nd, act.get_pose(nd))
            for (x, y, G, Om) in inc_pri:
                sub.add_edge(x, y, G, omega=Om)
            for (la, lb, R, t, sig) in inc_loop:
                # Carry the scale optimized in `act` so the local linearization
                # is consistent with the window solution. The edge index was
                # recorded when the loop was re-injected, so it is unambiguous.
                ei = loop_act_ei.get(edge_key(la, lb))
                scale = float(act.get_scale(ei)) if ei is not None else 1.0
                sub.add_edge(la, lb, self._meas(R, t), scale=scale,
                             robust=True, **sig)
            for (ids, H, bb, x0) in inc_npri:
                sub.add_prior_factor(ids, H, bb, x0)
            keep_ids, H_r, b_r = sub.marginalize_general([E], sorted(nbr))
            new_npri.append((tuple(keep_ids), H_r, b_r,
                             [act.get_pose(k) for k in keep_ids]))
            self._eliminated.add(E)
            remove_pri_ids |= {id(pr) for pr in inc_pri}
            remove_loop_ids |= {id(lp) for lp in inc_loop}
            remove_npri_ids |= {id(npr) for npr in inc_npri}
        if remove_pri_ids:
            self._priors = [pr for pr in self._priors
                            if id(pr) not in remove_pri_ids]
        if remove_loop_ids:
            self._loops = [lp for lp in self._loops
                           if id(lp) not in remove_loop_ids]
        if remove_npri_ids:
            self._node_priors = [npr for npr in self._node_priors
                                 if id(npr) not in remove_npri_ids]
        self._node_priors.extend(new_npri)
        # Prune priors/loops fully consumed by elimination so the per-window
        # scan stays proportional to the bounded active set, not to N.
        if self._eliminated:
            self._priors = [pr for pr in self._priors
                            if not (pr[0] in self._eliminated
                                    and pr[1] in self._eliminated)]
            self._loops = [lp for lp in self._loops
                           if not (lp[0] in self._eliminated
                                   and lp[1] in self._eliminated)]
        # Drop transients (keeping their anchor-relative transform).
        for i in trans:
            self._rel[i] = (a, np.linalg.inv(self._T[a]) @ self._T[i])
            self._T.pop(i, None)
            self._spoke.pop(i, None)
        self._kf.append(b)
        self._seg = []  # transients of the next segment start after b
        # Tier 2: a loop closure owes one (heavier, occasional) global pass
        # over all keyframes + accumulated priors + loop edges.
        if self._pending_global:
            period = int(self.p.get("global_opt_period", 0) or 0)
            if period <= 0 or len(self._kf) - self._last_global_kf >= period:
                self._pending_global = False
                self._last_global_kf = len(self._kf)
                self._global_reduce_optimize()

    def _add_loops(self, b):
        # Loop closure with the temporal-consistency gate: match the new
        # keyframe against older keyframes in the window, then keep only hits
        # confirmed by the previous ``loop_temporal_k - 1`` keyframes.
        window = int(self.p.get("loop_window", 80))
        min_gap = int(self.p.get("loop_min_gap", 30))
        min_inl = float(self.p.get("loop_min_inlier", 0.4))
        cycle_deg = float(self.p.get("cycle_threshold_deg", 0.0))
        need = max(1, int(self.p.get("loop_temporal_k", 1)))
        hits = []
        for a in self._kf[-window:]:
            if a == b or b - a < min_gap:
                continue
            # Detect against every keyframe in the window, not only the active
            # ones: a far loop endpoint may have left the bounded node set, in
            # which case Tier 2 applies the edge to the global reduced graph.
            r = self.match(a, b)
            if r and r.get("ok") and r.get("inlier_ratio", 0.0) >= min_inl:
                hits.append((int(a), np.asarray(r["R"], float),
                             np.asarray(r["t"], float).reshape(3),
                             float(r.get("inlier_ratio", 0.0)),
                             int(r.get("n_matches", -1))))
        bi = len(self._kf)  # index of the new keyframe
        if need > 1:
            # Only the temporal-consistency gate needs the per-keyframe hit
            # history; skip storing it (it would grow O(N)) when disabled.
            while len(self._hits) < bi:
                self._hits.append([])
            self._hits.append(hits)
        gaps = [self._kf[i + 1] - self._kf[i] for i in range(len(self._kf) - 1)]
        gaps.append(b - self._kf[-1])
        kf_step = max(gaps) if gaps else self._step
        margin = max(int(1.5 * kf_step), kf_step)
        accepted = confirmed_loop_hits(b, hits, bi, self._hits, need, margin,
                                       self._added)
        if cycle_deg > 0.0:
            cycle_accepted = []
            for hit in accepted:
                a, R, _t, _inl, _n = hit
                # Compare against the odometry-only chain, before any graph
                # optimization can make a bad loop appear self-consistent.
                # Rotation is scale-free, so this remains meaningful for mono.
                R_chain = (np.linalg.inv(self._T[b]) @ self._T[a])[:3, :3]
                if rotation_angle_deg(R, R_chain) > cycle_deg:
                    self.n_cycle_rejected += 1
                else:
                    cycle_accepted.append(hit)
            accepted = cycle_accepted
        sig_scale = float(self.p.get("loop_sigma_scale", 0.0))
        for (a, R, t, _inl, _n) in accepted:
            sig = {}
            if sig_scale > 0:
                sig = {"sigma_t": self.p.get("step_scale_t", 0.1) * sig_scale,
                       "sigma_r": self.p.get("step_scale_t", 0.1) * sig_scale}
            self._loops.append((int(a), int(b), np.asarray(R, float),
                                np.asarray(t, float).reshape(3), sig))
            self._added.add(edge_key(a, b))
            self.n_loop += 1
            self._pending_global = True

    def _new_window(self):
        return SlidingWindowOptimizer(
            window_size=None,
            max_iterations=self.p.get("loop_iterations", 10),
            huber=1.0,
            step_scale_t=self.p.get("step_scale_t", 0.1),
            step_scale_r=self.p.get("step_scale_t", 0.1),
            optimize_scale=True,
            scale_prior_sigma=self.p.get("scale_prior_sigma", 2.0),
            tsvd_ratio=(0.0 if self.p.get("nl_reg", True)
                        else self.p.get("seq_tsvd_ratio", 0.0)),
            loop_robust=self.p.get("loop_robust", "none"),
            loop_robust_phi=self.p.get("loop_robust_phi", 1.0),
            loop_robust_min_weight=self.p.get("loop_robust_min_weight", 0.0),
            loop_gm_mu=self.p.get("loop_gm_mu", 11.34),
            loop_dir_sigma=self.p.get("loop_dir_sigma", 0.4),
            loop_robust_gnc_gamma=self.p.get("loop_robust_gnc_gamma", 1.4))

    def _global_reduce_optimize(self):
        """Tier 2: occasional global pass over the reduced keyframe graph.

        Nodes are all keyframes, factors are the accumulated pairwise/multi-node
        priors and every accepted loop edge. This distributes a new loop
        constraint across the whole trajectory without running a full-graph
        solve on every keyframe. NL-Reg is intentionally skipped here to keep
        the occasional pass affordable.
        """
        if not self.p.get("global_opt_on_loop", True) or len(self._kf) < 2:
            return
        kfs = [k for k in self._kf if k in self._est]
        if len(kfs) < 2:
            return
        act = self._new_window()
        for k in kfs:
            # Eliminated keyframes are already summarized by their marginal
            # priors, so they are re-added as *fixed* landmarks only: this lets
            # loop edges measured against them be applied without optimizing
            # (and double counting) them again.
            act.add_node(k, self._est[k], fixed=(k in self._eliminated))
        for (x, y, G, Om) in self._priors:
            if x in act.pose_ids and y in act.pose_ids:
                act.add_edge(x, y, G, omega=Om)
        for (ids, H, bb, x0) in self._node_priors:
            if all(nd in act.pose_ids for nd in ids):
                act.add_prior_factor(ids, H, bb, x0)
        for (a, b, R, t, sig) in self._loops:
            if a in act.pose_ids and b in act.pose_ids:
                act.add_edge(a, b, self._meas(R, t), scale_free=True,
                             robust=True, **sig)
        act.optimize()
        self.n_robust_downweighted += act.n_robust_downweighted
        self.n_robust_rot_downweighted += act.n_robust_rot_downweighted
        self.n_robust_dir_downweighted += act.n_robust_dir_downweighted
        for k in kfs:
            if k not in self._eliminated and k in act.pose_ids:
                self._est[k] = act.get_pose(k)

    def _pose_of(self, fid):
        if fid in self._est:
            return self._est[fid].copy()
        # Transient: propagate from its anchor keyframe, rescaling the segment
        # translation (scale-free KF edges). Closed transients use the stored
        # anchor-relative transform so later global updates stay consistent.
        if fid in self._rel:
            p, rel = self._rel[fid]
            rel = rel.copy()
        else:
            p = max(k for k in self._kf if k <= fid)
            rel = np.linalg.inv(self._T[p]) @ self._T[fid]
        rel[:3, 3] *= self._seg_alpha(p)
        return self._est[p] @ rel

    def _seg_alpha(self, p):
        i = self._kf.index(p)
        if i + 1 >= len(self._kf):
            return 1.0
        q = self._kf[i + 1]
        if p not in self._est or q not in self._est:
            return 1.0
        d_opt = float(np.linalg.norm(self._est[q][:3, 3] - self._est[p][:3, 3]))
        d_odom = float(np.linalg.norm(self._T[q][:3, 3] - self._T[p][:3, 3]))
        return d_opt / d_odom if d_odom > 1e-9 else 1.0

    @staticmethod
    def _meas(R, t):
        M = np.eye(4)
        M[:3, :3] = np.asarray(R, float)
        M[:3, 3] = np.asarray(t, float).reshape(3)
        return M
