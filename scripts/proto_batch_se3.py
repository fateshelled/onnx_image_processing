"""Prototype: batched (vectorised) se3 residual/Jacobian vs the per-edge loop.

Feasibility + timing only (not wired into the optimizer). Compares
_edge_residual + se3_left/right_jacobian_inv computed edge-by-edge (current)
against numpy-batched versions.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vo.se3 import (se3_ad, se3_exp, se3_left_jacobian_inv,
                    se3_right_jacobian_inv, skew)  # noqa: E402
from vo.se3_window import _edge_residual  # noqa: E402

_BERNOULLI = [1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0,
              -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0, 0.0,
              7.0 / 6.0, 0.0, -3617.0 / 510.0]


def batch_skew(v):  # (N,3) -> (N,3,3)
    N = v.shape[0]
    K = np.zeros((N, 3, 3))
    K[:, 0, 1] = -v[:, 2]; K[:, 0, 2] = v[:, 1]
    K[:, 1, 0] = v[:, 2]; K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]; K[:, 2, 1] = v[:, 0]
    return K


def batch_so3_log(R):  # (N,3,3) -> (N,3)
    tr = np.trace(R, axis1=1, axis2=2)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    w = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0],
                  R[:, 1, 0] - R[:, 0, 1]], axis=1)
    s = 2.0 * np.sin(th)
    out = np.zeros_like(w)
    small = th < 1e-9
    general = ~small
    out[small] = 0.5 * w[small]
    out[general] = (w[general] / s[general, None]) * th[general, None]
    return out


def batch_se3_log(T):  # (N,4,4) -> (N,6)
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    phi = batch_so3_log(R)
    th = np.linalg.norm(phi, axis=1)
    K = batch_skew(phi)
    K2 = K @ K
    V = (np.eye(3)[None] + ((1 - np.cos(th)) / np.where(th < 1e-9, 1.0, th) ** 2)[:, None, None] * K
         + ((th - np.sin(th)) / np.where(th < 1e-9, 1.0, th) ** 3)[:, None, None] * K2)
    small = th < 1e-9
    V[small] = np.eye(3)
    tau = np.linalg.solve(V, t[..., None])[..., 0]
    return np.concatenate([phi, tau], axis=1)


def batch_ad_e(e):  # (N,6) -> (N,6,6)
    N = e.shape[0]
    A = np.zeros((N, 6, 6))
    sw = batch_skew(e[:, :3])
    st = batch_skew(e[:, 3:])
    A[:, :3, :3] = sw
    A[:, 3:, 3:] = sw
    A[:, 3:, :3] = st
    return A


def batch_left_jac_inv(e):  # (N,6) -> (N,6,6)
    A = batch_ad_e(e)
    S = np.zeros((e.shape[0], 6, 6))
    Ak = np.broadcast_to(np.eye(6), (e.shape[0], 6, 6)).copy()
    fact = 1.0
    for k in range(17):
        if k > 0:
            Ak = Ak @ A
            fact *= k
        S += (_BERNOULLI[k] / fact) * Ak
    return S


rng = np.random.default_rng(0)
for N in (200, 2000):
    Ti = np.stack([se3_exp(rng.normal(size=6) * 0.1) for _ in range(N)])
    Tj = np.stack([se3_exp(rng.normal(size=6) * 0.1) for _ in range(N)])
    M = np.stack([se3_exp(rng.normal(size=6) * 0.1) for _ in range(N)])
    Ms = M.copy()

    def loop():
        out_e = np.empty((N, 6)); out_J = np.empty((N, 6, 6))
        for k in range(N):
            e = _edge_residual(Ti[k], Tj[k], M[k])
            Ji = se3_right_jacobian_inv(e)
            Jj = -se3_left_jacobian_inv(e) @ se3_ad(np.linalg.inv(Ms[k]))
            out_e[k] = e; out_J[k] = Ji + Jj
        return out_e, out_J

    def batch():
        P = np.linalg.inv(Tj) @ Ti
        Q = np.linalg.inv(M) @ P
        e = batch_se3_log(Q)
        Ji = batch_left_jac_inv(-e)
        InvM = np.linalg.inv(Ms)
        n = e.shape[0]
        ad = np.zeros((n, 6, 6))
        ad[:, :3, :3] = InvM[:, :3, :3]
        ad[:, 3:, 3:] = InvM[:, :3, :3]
        ad[:, 3:, :3] = batch_skew(InvM[:, :3, 3]) @ InvM[:, :3, :3]
        Jj = -batch_left_jac_inv(e) @ ad
        return e, Ji + Jj

    e1, J1 = loop(); e2, J2 = batch()
    err = max(np.abs(e1 - e2).max(), np.abs(J1 - J2).max())
    t0 = time.perf_counter()
    for _ in range(5):
        loop()
    tl = (time.perf_counter() - t0) / 5 * 1000
    t0 = time.perf_counter()
    for _ in range(5):
        batch()
    tb = (time.perf_counter() - t0) / 5 * 1000
    print(f"N={N}: loop={tl:.1f}ms  batch={tb:.1f}ms  speedup={tl/tb:.1f}x  maxdiff={err:.2e}")
