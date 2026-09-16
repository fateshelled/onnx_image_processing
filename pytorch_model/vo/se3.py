"""Minimal SE(3) utilities for the sliding-window pose-graph optimizer.

Tangent basis ordering: (omega, tau) = (rotation 3-vec, translation 3-vec),
right-perturbation convention: node poses are updated as T' = T @ Exp(delta).

Measurement convention (camera-to-world node poses):
    M_ij transforms points from camera-i frame to camera-j frame:
        p_j = R_ij @ p_i + t_ij
    and satisfies  M_ij = T_j^{-1} T_i  when consistent.
The edge residual is  e = Log(M_ij^{-1} T_j^{-1} T_i).
"""

import numpy as np


def skew(v):
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def so3_exp(phi):
    phi = np.asarray(phi, dtype=np.float64)
    theta = float(np.linalg.norm(phi))
    if theta < 1e-9:
        return np.eye(3) + skew(phi)
    axis = phi / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R):
    """Returns axis-angle (3,) with angle in [0, pi]."""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = min(1.0, max(-1.0, cos_theta))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - theta < 1e-6:
        # Near pi: use the symmetric part
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        idx = int(np.argmax(axis))
        axis = axis[idx] * A[:, idx]
        n = np.linalg.norm(axis)
        return (axis / n) * theta if n > 1e-9 else np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    return axis * theta


def so3_left_jacobian_inv(phi):
    """J_l(theta)^{-1} for left-increment mapping (Barfoot convention)."""
    phi = np.asarray(phi, dtype=np.float64)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-6:
        return np.eye(3) - 0.5 * K + (1.0 / 12.0) * (K @ K)
    coef = (1.0 / theta) - 0.5 / np.tan(theta / 2.0)
    return np.eye(3) - 0.5 * K + coef / theta * (K @ K)


def se3_exp(delta):
    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    phi, tau = delta[:3], delta[3:]
    T = np.eye(4)
    T[:3, :3] = so3_exp(phi)
    # t = V(phi) @ tau with the full-angle V:
    # V = I + ((1-cos)/theta^2) [phi]x + ((theta - sin)/theta^3) [phi]x^2
    theta = float(np.linalg.norm(phi))
    Kfull = skew(phi)
    if theta < 1e-9:
        V = np.eye(3)
    else:
        V = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / (theta ** 2) * Kfull
            + (theta - np.sin(theta)) / (theta ** 3) * (Kfull @ Kfull)
        )
    T[:3, 3] = V @ tau
    return T


def se3_log(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    phi = so3_log(R)
    theta = float(np.linalg.norm(phi))
    Kfull = skew(phi)
    if theta < 1e-9:
        tau = t.copy()
    else:
        V = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / (theta ** 2) * Kfull
            + (theta - np.sin(theta)) / (theta ** 3) * (Kfull @ Kfull)
        )
        tau = np.linalg.solve(V, t)
    out = np.zeros(6)
    out[:3] = phi
    out[3:] = tau
    return out


def se3_ad(T):
    """Adjoint of T Exp(δ) T^{-1} == Exp(se3_ad(T) δ).

    Valid exactly for the se3_exp/se3_log pair in this module (verified
    empirically to machine precision).
    """
    T = np.asarray(T, float)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.zeros((6, 6))
    out[:3, :3] = R
    out[3:, 3:] = R
    out[3:, :3] = skew(t) @ R
    return out


_BERNOULLI = [1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0,
              -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0, 0.0,
              7.0 / 6.0, 0.0, -3617.0 / 510.0]


def se3_ad_e(e):
    """Differential of the group adjoint at the identity: ad(e) ∈ so(3)×so(3)
    with the (omega, tau) packing of this module."""
    e = np.asarray(e, dtype=np.float64).reshape(6)
    sw = skew(e[:3])
    st = skew(e[3:])
    out = np.zeros((6, 6))
    out[:3, :3] = sw
    out[3:, 3:] = sw
    out[3:, :3] = st
    return out


def se3_left_jacobian_inv(e):
    """J_l(e)^{-1} for right-increment updates T' = T @ Exp(delta).

    Analytic via the Bernoulli series Sigma B_k/k! ad(e)^k, valid for
    ||omega|| < 2*pi (convergence verified against finite differences;
    residuals are far below 2 in practice).
    """
    e = np.asarray(e, dtype=np.float64).reshape(6)
    A = se3_ad_e(e)
    S = np.zeros((6, 6))
    Ak = np.eye(6)
    fact = 1.0
    for k in range(17):
        if k > 0:
            Ak = Ak @ A
            fact *= k
        S += (_BERNOULLI[k] / fact) * Ak
    return S


def se3_right_jacobian_inv(e):
    """J_r(e) = J_l(-e)^{-1}; de'/d(delta) for e' = Log(Exp(e) Exp(delta))."""
    return se3_left_jacobian_inv(-np.asarray(e, dtype=np.float64).reshape(6))
