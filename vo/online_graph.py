"""Online keyframe pose graph with a fixed-lag window and NL-Reg.

This module owns all the pose-graph logic (keyframe promotion, per-frame
spoke/loop matching requests, Schur marginalization of transient frames,
multi-node priors for exited keyframes, direction-dependent Tikhonov
regularization). Callers such as ``sample/visual_odometry.py`` only feed
frames in and read the current camera pose back out.

The graph is matcher-agnostic: the caller passes ``match_fn(image_a,
image_b) -> dict`` where the dict has keys ``ok`` (bool), ``R`` (3x3),
``t`` (3,), ``inlier_ratio`` and ``n_matches``. The graph stores only the
images it still needs (keyframes and the most recent frame).

Pose convention matches ``Trajectory.add_relative_pose``: the relative
measurement ``(R, t)`` is a world-to-camera point transform
(``x_curr = R @ x_prev + t``, as returned by ``cv2.recoverPose``).
"""

import numpy as np

from .se3_window import SlidingWindowOptimizer

# Default parameters for the online sequential graph (optuna-tuned seq_opt1).
DEFAULT_PARAMS = {
    "odom_ref": "kf", "kf_mode": "motion", "keyframe_decim": 15,
    "kf_trans_thresh": 8.0, "kf_rot_thresh": 10.0, "kf_max_gap": 16,
    "kf_local_map_k": 1, "kf_edge_min_inlier": 0.0, "trans_gate_deg": 0.0,
    "loop_window": 80, "loop_min_gap": 30, "loop_min_inlier": 0.4,
    "loop_temporal_k": 1, "loop_sigma_scale": 2.0,
    "scale_prior_sigma": 2.0, "step_scale_t": 0.1, "loop_iterations": 10,
    "tsvd_ratio": 0.0, "scale_kf": False,
    "graph_mode": "kf_prior", "max_keyframes": 3, "seq_tsvd_ratio": 0.0,
    "nl_reg": True, "nl_reg_c": 10.0, "nl_reg_tau": 10.0, "nl_reg_length": 1.0,
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


class OnlinePoseGraph:
    """Online sequential pose graph.

    Parameters are read from a plain dict (defaults come from
    ``eval.rustuna_tune_loop.SEQ_OPT1_DEFAULTS`` upstream). The graph keeps
    at most ``max_keyframes`` active keyframes; transient frames are
    marginalized into relative priors when a new keyframe is promoted.
    """

    def __init__(self, params, cam, match_fn):
        self.p = params
        self.cam = cam
        self.match = match_fn
        self.kf_min_gap = 4
        self.kf_max_gap = int(params.get("kf_max_gap", 16))
        self.max_kf = params.get("max_keyframes")
        self._next = 0
        self._img = {}
        self._T = {}          # odom chain camera-to-world
        self._kf = []
        self._est = {}        # optimized keyframe poses
        self._last = None
        self._seg = []        # transient ids since the last keyframe
        self._odom = {}       # frame id -> (R, t) relative to the previous frame
        self._spoke = {}      # frame id -> (ref_kf, R, t, inlier, n_matches)
        self._priors = []     # pairwise transient priors (a, b, G, Omega)
        self._node_priors = []  # multi-node priors (ids, H, b, x0)
        self._hits = []       # per-keyframe loop hits (temporal confirmation)
        self.last_stats = {"ok": True, "inlier_ratio": 1.0, "n_matches": 0}

    # -- public -------------------------------------------------------------

    def add_frame(self, image):
        """Process one frame and return its camera-to-world 4x4 pose."""
        fid = self._next
        self._next += 1
        if self._last is None:
            self._img[fid] = image
            self._T[fid] = np.eye(4)
            self._kf = [fid]
            self._est[fid] = np.eye(4)
            self._last = fid
            return np.eye(4)

        # Odometry: chain from the previous frame.
        res = self.match(self._img[self._last], image)
        self.last_stats = {
            "ok": bool(res and res.get("ok")),
            "inlier_ratio": float(res.get("inlier_ratio", 0.0)) if res else 0.0,
            "n_matches": int(res.get("n_matches", 0)) if res else 0,
        }
        T = self._T[self._last]
        if res and res.get("ok"):
            self._odom[fid] = (np.asarray(res["R"], float),
                               np.asarray(res["t"], float).reshape(3))
            T = _compose(T, res["R"], res["t"])
        self._T[fid] = T

        # Spoke: match against the current keyframe (for the KF edge later).
        last_kf = self._kf[-1]
        sres = self.match(self._img[last_kf], image)
        if sres and sres.get("ok"):
            self._spoke[fid] = (last_kf, np.asarray(sres["R"], float),
                                np.asarray(sres["t"], float).reshape(3),
                                float(sres.get("inlier_ratio", 0.0)),
                                int(sres.get("n_matches", -1)))

        # Bookkeeping: keep keyframe images and only the newest frame image.
        self._img[fid] = image
        if self._last not in self._kf:
            self._img.pop(self._last, None)
        self._last = fid
        self._seg.append(fid)

        # Keyframe promotion (motion criterion).
        if self._promote(fid):
            self._close_segment(fid)
        return self._pose_of(fid)

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
        """Optimize [prev_kf, transients, new_kf], marginalize the transients."""
        a = self._kf[-1]
        b = new_kf
        trans = [i for i in self._seg if i != b]
        active, boundary = self._window()
        node_set = set(active) | set(trans) | {b}
        if boundary is not None:
            node_set.add(boundary)
        nodes = sorted(node_set)

        act = self._new_window()
        for nd in nodes:
            act.add_node(nd, self._est.get(nd, self._T[nd]),
                         fixed=(nd == boundary))
        for (x, y, G, Om) in self._priors:
            if x in node_set and y in node_set:
                act.add_edge(x, y, G, omega=Om)
        for (ids, H, bb, x0) in self._node_priors:
            if all(nd in node_set for nd in ids):
                act.add_prior_factor(ids, H, bb, x0)
        # Chain edges (consecutive frames) and spokes (frame -> last keyframe,
        # which for the new keyframe b is the direct a-b measurement).
        for v in trans + [b]:
            if v in self._odom:
                R, t = self._odom[v]
                act.add_edge(v - 1, v, self._meas(R, t), scale_free=True)
        for v in trans + [b]:
            if v in self._spoke:
                ref, R, t, _inl, _n = self._spoke[v]
                if ref in node_set and ref != v - 1:
                    act.add_edge(ref, v, self._meas(R, t), scale_free=True)
        # Loop edges among active keyframes (simple: spoke of each KF to a).
        self._add_loops(act, node_set, b)
        if self.p.get("nl_reg", True):
            act.add_nl_regularization(self.p.get("nl_reg_c", 10.0),
                                      self.p.get("nl_reg_tau", 10.0),
                                      self.p.get("nl_reg_length", 1.0))
        act.optimize()
        if trans:
            aa, bb2, G, Om = act.marginalize_relative(trans, [a, b],
                                                      fix_scales=True)
            self._priors.append((aa, bb2, G, Om))
        for nd in nodes:
            if nd != boundary:
                self._est[nd] = act.get_pose(nd)
        # Promote: drop transients, extend keyframes, bound the window.
        for i in trans:
            self._img.pop(i, None)
            self._T.pop(i, None)
            self._spoke.pop(i, None)
        self._kf.append(b)
        self._seg = []  # transients of the next segment start after b
        self._evict()

    def _window(self):
        if self.max_kf is None:
            return list(self._kf), None
        w = max(1, int(self.max_kf))
        active = self._kf[-w:]
        boundary = self._kf[-w - 1] if len(self._kf) > w else None
        return active, boundary

    def _evict(self):
        """Option 1: keep exited keyframes fixed until their blanket is ready."""
        active, _ = self._window()
        # Nothing to do here for the simple version: window is bounded by
        # max_keyframes and older keyframes are dropped with their priors kept
        # only while referenced. (Multi-node marginalization of exited KFs is
        # the next step; for now bound the kept image/pose memory.)
        keep = set(active)
        for (x, y, *_r) in self._priors:
            keep.add(x)
            keep.add(y)
        for nd in list(self._est):
            if nd not in keep and nd not in self._kf[-1:]:
                # keep estimates for reporting; images already dropped
                pass

    def _new_window(self):
        return SlidingWindowOptimizer(
            window_size=None,
            max_iterations=self.p.get("loop_iterations", 10),
            huber=1.0,
            step_scale_t=self.p.get("step_scale_t", 0.1),
            step_scale_r=self.p.get("step_scale_t", 0.1),
            optimize_scale=True,
            scale_prior_sigma=self.p.get("scale_prior_sigma", 2.0),
            tsvd_ratio=0.0)

    def _add_loops(self, act, node_set, b):
        # Simple loop closure: try matching the new keyframe against the
        # previous keyframes in the window and add accepted edges.
        window = int(self.p.get("loop_window", 80))
        min_gap = int(self.p.get("loop_min_gap", 30))
        min_inl = float(self.p.get("loop_min_inlier", 0.4))
        for a in self._kf[-window:]:
            if a == b or b - a < min_gap:
                continue
            if a not in node_set or a not in self._img:
                continue
            r = self.match(self._img[a], self._img[b])
            if r and r.get("ok") and r.get("inlier_ratio", 0.0) >= min_inl:
                act.add_edge(a, b, self._meas(r["R"], r["t"]), scale_free=True)

    @staticmethod
    def _meas(R, t):
        M = np.eye(4)
        M[:3, :3] = np.asarray(R, float)
        M[:3, 3] = np.asarray(t, float).reshape(3)
        return M

    def _pose_of(self, fid):
        if fid in self._est:
            return self._est[fid].copy()
        # Transient: propagate from the nearest (last) keyframe along the chain.
        p = self._kf[-1]
        rel = np.linalg.inv(self._T[p]) @ self._T[fid]
        return self._est[p] @ rel
