"""Relative canonical priors and smooth joint pose/log-scale regularization.

Experimental dense solver. Regularization belongs to a solve, never to the
factor graph exported by marginalize_joint(). Only fixed Huber loss is supported.
"""
import numpy as np
import scipy.linalg as sla

from .joint_prior import JointWindowOptimizer
from .se3 import se3_ad, se3_exp, se3_inv, se3_log, se3_right_jacobian_inv
from .se3_window import SlidingWindowOptimizer


class RelativeJointWindowOptimizer(JointWindowOptimizer):
    def __init__(self, regularization="soft", alpha=1., tau=1.,
                 sigma_r=1., sigma_t=1., sigma_l=1., **kwargs):
        super().__init__(**kwargs)
        if regularization not in ("soft", "ridge", "none"):
            raise ValueError("unknown relative regularization")
        values = np.array([alpha, tau, sigma_r, sigma_t, sigma_l])
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("relative regularization parameters must be positive")
        if self.tsvd_ratio or self.loop_robust != "none":
            raise ValueError("relative solver requires TSVD off and loop_robust=none")
        self.regularization = regularization
        self.alpha, self.tau = float(alpha), float(tau)
        self.sigmas = (float(sigma_r), float(sigma_t), float(sigma_l))
        self.relative_diagnostics = {}
        # Memoize each joint-prior delta for the current linearization point.
        # ``_delta_rev`` is bumped whenever the poses or scales change, so a
        # delta is computed once per (factor, linearization point) and reused by
        # the residual, Jacobian and cost evaluations that share that point.
        self._delta_rev = 0
        self._delta_cache = {}
        self._delta_cache_active = False

    @staticmethod
    def _prior_anchors(fac):
        poses = [k for kind, k in fac.keys if kind == "pose"]
        groups = fac.pose_components or (tuple(poses),)
        return {p: min(group) for group in groups for p in group}

    def add_joint_prior(self, prior):
        poses = [k for kind, k in prior.keys if kind == "pose"]
        if poses:
            if not prior.relative_pose:
                raise ValueError("relative solver does not support absolute/partial pose priors")
            anchors = self._prior_anchors(prior)
            roots = sorted(set(anchors.values()))
            N = np.zeros((prior.A.shape[1], 6 * len(roots)))
            offset = 0
            for (kind, key), x0 in zip(prior.keys, prior.x0):
                width = 6 if kind == "pose" else 1
                if kind == "pose":
                    j = roots.index(anchors[key]) * 6
                    N[offset:offset+6, j:j+6] = se3_ad(np.linalg.inv(x0))
                offset += width
            error = np.linalg.norm(prior.A @ N)
            scale = max(1., np.linalg.norm(prior.A) * np.linalg.norm(N))
            if error > 1e-8 * scale:
                raise ValueError("relative prior has inconsistent SE3 nullspace")
        super().add_joint_prior(prior)

    def _joint_delta(self, fac):
        if not fac.keys:
            return np.zeros(0)
        active = self._delta_cache_active
        if active:
            cached = self._delta_cache.get(id(fac))
            if cached is not None and cached[0] == self._delta_rev:
                return cached[1]
        anchors = self._prior_anchors(fac)
        refs = {k: x0 for (kind, k), x0 in zip(fac.keys, fac.x0) if kind == "pose"}
        pieces = []
        for (kind, key), x0 in zip(fac.keys, fac.x0):
            if kind == "pose":
                a = anchors[key]
                aligned = refs[a] @ se3_inv(self.T[a]) @ self.T[key]
                pieces.append(se3_log(se3_inv(x0) @ aligned))
            else:
                pieces.append(np.array([self.log_scale(key) - x0]))
        out = np.concatenate(pieces) if pieces else np.zeros(0)
        if active:
            self._delta_cache[id(fac)] = (self._delta_rev, out)
        return out

    def _jacobian_coo(self, eps=1e-6):
        rows, cols, vals, shape = SlidingWindowOptimizer._jacobian_coo(self, eps)
        rows, cols, vals = list(rows), list(cols), list(vals)
        base = shape[0] - sum(f.A.shape[0] for f in self.joint_priors)
        columns = self._key_columns()
        for fac in self.joint_priors:
            delta = self._joint_delta(fac)
            anchors = self._prior_anchors(fac)
            offset = 0
            for key in fac.keys:
                width = 6 if key[0] == "pose" else 1
                block = fac.A[:, offset:offset+width]
                contributions = [(key, block)]
                if key[0] == "pose":
                    block = block @ se3_right_jacobian_inv(delta[offset:offset+6])
                    a = anchors[key[1]]
                    anchor_block = -block @ se3_ad(se3_inv(self.T[key[1]]) @ self.T[a])
                    contributions = [(key, block), (("pose", a), anchor_block)]
                offset += width
                for target, jac in contributions:
                    if target not in columns:
                        continue
                    rr, cc = np.nonzero(jac)
                    rows.extend(base + rr)
                    cols.extend(np.asarray(columns[target])[cc])
                    vals.extend(jac[rr, cc])
            base += fac.A.shape[0]
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int), np.asarray(vals), shape

    def _jacobian(self, eps=1e-6):
        rows, cols, vals, shape = self._jacobian_coo(eps)
        J = np.zeros(shape)
        # A prior anchor contributes through every relative pose block.
        np.add.at(J, (rows, cols), vals)
        return J

    def _residuals(self):
        r = super()._residuals()
        row = 6 * len(self.edges)
        if self.optimize_scale:
            for e, c in enumerate(self.scale_col):
                if c is not None and self.edge_scale_sigma[e] > 0:
                    r[row] = ((np.log(self.edge_scale[e]) - self.scale_prior_mean[e])
                              / self.edge_scale_sigma[e])
                    row += 1
        return r

    def add_nl_regularization(self, *args, **kwargs):
        raise ValueError("legacy NL-Reg cannot be combined with relative regularization")

    def _apply_scale_step(self, dx):
        # No clipping/floor: an unrepresentable trial is rejected by the solver.
        with np.errstate(over="raise", under="raise", invalid="raise"):
            for e, c in enumerate(self.scale_col):
                if c is not None:
                    self.edge_scale[e] = float(np.exp(np.log(self.edge_scale[e]) + dx[c]))
            for key in self.scalar_value:
                self.scalar_value[key] += float(dx[self.scalar_col[key]])

    def _validate_relative(self):
        if self.fixed_ids or self.prior_factors:
            raise ValueError("relative solver requires no fixed poses or absolute pose factors")
        if sorted(self.scalar_col.values()) != list(range(self.n_scale)):
            raise ValueError("relative solver requires named keys for every variable scale; use add_named_edge")

    def marginalize_joint(self, *args, **kwargs):
        self._validate_relative()
        return super().marginalize_joint(*args, **kwargs)

    def _true_cost(self):
        r = self._residuals()
        n = 6 * len(self.edges)
        norms = np.linalg.norm(r[:n].reshape(-1, 6), axis=1)
        edge = np.where(norms <= self.huber, .5*norms**2,
                        self.huber*(norms - .5*self.huber)).sum()
        return float(edge + .5 * (r[n:] @ r[n:]))

    def optimize(self, verbose=False):
        self.relative_diagnostics = {"status": "started", "rejected_trials": 0,
                                     "last_failure": None}
        try:
            return self._optimize_relative(verbose)
        except Exception as exc:
            # Never leave the delta memoization armed after a failed solve.
            self._delta_cache_active = False
            self._delta_cache.clear()
            self.relative_diagnostics.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise

    def _optimize_relative(self, verbose=False):
        self._validate_relative()
        components, absolute = self._pose_components()
        if absolute:
            raise ValueError("absolute pose constraints are unsupported")
        roots = {min(c) for c in components}
        keys = [k for k in self.variable_keys() if k[0] != "pose" or k[1] not in roots]
        columns = self._key_columns()
        cols = np.array([c for k in keys for c in columns[k]], dtype=int)
        sr, st, sl = self.sigmas
        D = np.concatenate([np.array([sr]*3 + [st]*3) if k[0] == "pose"
                            else np.array([sl]) for k in keys]) if keys else np.zeros(0)
        reference = {k: self.T[k[1]].copy() if k[0] == "pose" else self.log_scale(k[1])
                     for k in keys}
        self.last_steps = 0
        if not keys:
            self.last_cost = self._true_cost()
            self.relative_diagnostics = {"status": "empty", "regularization_cost": 0.}
            return self.last_cost
        # Memoize each prior delta within this solve: the poses/scales only
        # change inside the trial loop below, which bumps the revision.
        self._delta_cache.clear()
        self._delta_rev += 1
        self._delta_cache_active = True
        J = self._jacobian()[:, cols] * D
        H0 = J.T @ (self._huber_weights()[:, None] * J)
        if self.regularization == "soft":
            try:
                L = sla.cholesky(H0 + self.tau*np.eye(len(D)), lower=True)
            except np.linalg.LinAlgError:
                # H0 is numerically indefinite at a huge dynamic range, so the
                # Wiener Cholesky is undefined. Project H0 onto the PSD cone
                # (symmetrize, clip roundoff-only negatives to 0) and use the
                # symmetric inverse square root; it satisfies
                # B^T B = alpha*tau*(P(H0)+tau I)^-1 for the projected P(H0),
                # the same information as the triangular factor, with no
                # threshold on the positive spectrum.
                w, V = np.linalg.eigh(0.5*(H0 + H0.T))
                w = np.maximum(w, 0.)
                B = np.sqrt(self.alpha*self.tau) * (V * (1./np.sqrt(w + self.tau))) @ V.T
            else:
                B = np.sqrt(self.alpha*self.tau) * sla.solve_triangular(L, np.eye(len(D)), lower=True)
        elif self.regularization == "ridge":
            B = np.sqrt(self.alpha)*np.eye(len(D))
        else:
            B = np.zeros((0, len(D)))
        if self.scale_regularizer_exclude and B.shape[0] == len(D):
            mask = np.concatenate([
                np.ones(6) if k[0] == "pose"
                else np.array([0. if k[1] in self.scale_regularizer_exclude else 1.])
                for k in keys])
            # Zero only the excluded columns: eta of an excluded scale no longer
            # enters any regularizer row, and its regularizer Jacobian column
            # vanishes, while included keys keep their exact regularization.
            B = B @ np.diag(mask)

        def regularizer(jacobian=False):
            eta, blocks = [], []
            for key in keys:
                if key[0] == "pose":
                    d = se3_log(se3_inv(reference[key]) @ self.T[key[1]])
                    eta.extend(d)
                    if jacobian:
                        blocks.append(se3_right_jacobian_inv(d))
                else:
                    eta.append(self.log_scale(key[1]) - reference[key])
                    if jacobian:
                        blocks.append(np.ones((1, 1)))
            residual = B @ (np.asarray(eta) / D)
            if not jacobian:
                return residual
            jac = sla.block_diag(*blocks) * D[None, :] / D[:, None]
            return residual, B @ jac

        lam = self.lambda_init
        history = []
        status = "iteration_limit"
        for _ in range(self.max_iterations):
            r, w = self._residuals(), self._huber_weights()
            J = self._jacobian()[:, cols] * D
            rr, Jr = regularizer(True)
            cost = self._true_cost() + .5*(rr @ rr)
            H = J.T @ (w[:, None]*J) + Jr.T @ Jr
            g = J.T @ (w*r) + Jr.T @ rr
            cost_resolution = 32*np.finfo(float).eps*max(1., abs(cost))
            # A raw gradient grows with observation curvature. Measure the
            # stationarity error in local curvature units for roundoff only.
            scaled_gradient = np.max(np.abs(g) / np.maximum(1., np.sqrt(np.diag(H))))
            if np.linalg.norm(g, ord=np.inf) < 1e-8:
                status = "gradient"
                break
            poses = {p: self.T[p].copy() for p in self.pose_ids}
            scales = self._capture_scales()
            accepted = False
            for _attempt in range(12):
                failure = "nondecreasing or nonfinite trial"
                roundoff = False
                try:
                    step = sla.solve(H + lam*np.eye(len(D)), -g, assume_a="pos")
                    predicted = float(-g @ step - .5*step @ H @ step)
                    # Only a first-trial failure near stationarity is eligible.
                    # Certify it below using an undamped least-squares step;
                    # inherited or increased LM damping must not hide failure.
                    roundoff_candidate = (_attempt == 0 and scaled_gradient <= np.sqrt(cost_resolution)
                            and 0 <= predicted <= cost_resolution)
                    dx = np.zeros(6*len(self.pose_ids) + self.n_scale)
                    dx[cols] = D*step
                    for i, p in enumerate(self.pose_ids):
                        self.T[p] = poses[p] @ se3_exp(dx[6*i:6*i+6])
                    self._apply_scale_step(dx[6*len(self.pose_ids):])
                    self._delta_rev += 1  # trial linearization point changed
                    trial_reg = regularizer()
                    trial = self._true_cost() + .5*(trial_reg @ trial_reg)
                    if np.isfinite(trial) and trial < cost:
                        history.append(float(trial))
                        self.last_steps += 1
                        lam = max(lam/3., 1e-12)
                        accepted = True
                        break
                    if (roundoff_candidate and np.isfinite(trial)
                            and abs(trial-cost) <= cost_resolution):
                        model_J = np.vstack((np.sqrt(w)[:, None]*J, Jr))
                        model_r = np.concatenate((np.sqrt(w)*r, rr))
                        undamped, _, rank, _ = sla.lstsq(model_J, -model_r)
                        model_change = model_J @ undamped
                        reduction = float(.5*(model_change @ model_change))
                        # A truncated weak direction is not evidence of
                        # convergence. Conservatively decline this certificate
                        # for any unresolved rank deficiency, even an exact null.
                        roundoff = (rank == len(D) and np.isfinite(undamped).all()
                                    and np.linalg.norm(undamped, ord=np.inf) <= np.sqrt(cost_resolution)
                                    and reduction <= cost_resolution)
                except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError) as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                self.T.update({p: T.copy() for p, T in poses.items()})
                # _restore_scales assigns mutable state; give each trial a fresh copy.
                self._restore_scales((list(scales[0]), dict(scales[1])))
                self._delta_rev += 1  # rejected trial restored the linearization point
                if roundoff:
                    status = "roundoff"
                    break
                self.relative_diagnostics["rejected_trials"] += 1
                self.relative_diagnostics["last_failure"] = failure
                lam *= 10.
            if not accepted:
                if status == "roundoff":
                    break
                # No lambda yields a decreasing step: stop and return the last
                # accepted iterate (the initial estimate if none was accepted)
                # as a best-effort result, matching the legacy sliding-window
                # solver. The state is finite; status "no_progress" surfaces it.
                status = "no_progress"
                break
        rr = regularizer()
        reg_cost = float(.5*(rr @ rr))
        self.last_cost = self._true_cost() + reg_cost
        self.relative_diagnostics.update({"status": status, "regularization_cost": reg_cost,
                                     "data_cost": self._true_cost(), "cost_history": history,
                                     "initial_eigenvalues": sla.eigvalsh(H0).tolist(),
                                     "roots": sorted(roots), "steps": self.last_steps,
                                     "final_lambda": float(lam)})
        self._delta_cache_active = False
        return self.last_cost
