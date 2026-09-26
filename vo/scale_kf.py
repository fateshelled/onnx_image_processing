"""Linear scalar Kalman filter for the motion-normalised edge scale coefficient.

Model (random-walk on the coefficient k):
    k_{t+1} = k_t + w,   w ~ N(0, q)      (process)
    z_t     = k_t + v,   v ~ N(0, r)      (measurement z = s_e / m_e)

With constant q, r this is equivalent to an EMA with alpha = K (steady state),
but it additionally yields the posterior variance P, which the pose graph can
use as the per-edge scale-prior strength.

Used by eval/rustuna_tune_loop.py eval_seq when scale_kf is enabled.
"""


class ScaleKF:
    def __init__(self, process_var: float = 1e-3, meas_var: float = 0.05,
                 k0: float | None = None, p0: float = 1.0) -> None:
        self.q = float(process_var)
        self.r = float(meas_var)
        self.k = None if k0 is None else float(k0)
        self.p = float(p0)

    def update(self, z: float, meas_var: float | None = None) -> tuple:
        """Predict + update with measurement z.

        Returns (k, P, innov) where innov is the normalized innovation
        (z - k_pred) / sqrt(P_pred + r); ~N(0,1) when the random-walk model
        fits, larger when k does not follow the motion model.
        """
        if self.k is None:              # initialise on first measurement
            self.k = float(z)
            self.p = float(self.r) if meas_var is None else float(meas_var)
            return self.k, self.p, 0.0
        self.p += self.q               # predict (state unchanged)
        r = self.r if meas_var is None else float(meas_var)
        innov = (float(z) - self.k) / (self.p + r) ** 0.5
        gain = self.p / (self.p + r)
        self.k = self.k + gain * (float(z) - self.k)
        self.p = (1.0 - gain) * self.p
        return self.k, self.p, innov
