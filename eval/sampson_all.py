"""Sampson distance helper (diagnostic) for fundamental/essential matrix evaluation.

Correct Sampson first-order geometric error for an epipolar constraint
x2^T F x1 = 0.

For a single correspondence (x1, x2 homogeneous, F 3x3):

    l2 = F x1          # epipolar line of x1 seen in image 2
    l1 = F^T x2        # epipolar line of x2 seen in image 1
    num  = x2^T F x1   # scalar residual (== x1^T F^T x2)
    den  = l2[0]^2 + l2[1]^2 + l1[0]^2 + l1[1]^2
    d    = sqrt(num^2 / den)

Note the transpose direction: the scalar is x2^T F x1, NOT x1^T F x2
(which is x2^T F^T x1 and differs in general).  The numerator must be
squared (d = sqrt(num**2 / den)), not num / den.
"""

import numpy as np


def sampson_all(x1, x2, F):
    """Sampson distance per correspondence.

    Args:
        x1: (N, 3) homogeneous points in image 1.
        x2: (N, 3) homogeneous points in image 2.
        F:  (3, 3) fundamental matrix.

    Returns:
        (N,) Sampson distance (pixels), sqrt(num^2 / den).
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)

    Fx1 = (F @ x1.T).T      # (N, 3), column i = F x1_i
    Ft_x2 = (F.T @ x2.T).T  # (N, 3), column i = F^T x2_i

    num = np.sum(x2 * Fx1, axis=1)  # x2_i^T F x1_i  (= x1_i^T F^T x2_i)
    den = (
        Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2
        + Ft_x2[:, 0] ** 2 + Ft_x2[:, 1] ** 2
    )
    den = np.maximum(den, 1e-12)
    return np.sqrt(num ** 2 / den)


def build_F_from_pose(R, t, K):
    """Fundamental matrix from relative pose (R, t) and intrinsics K.

    F = K^{-T} E K^{-1},  E = [t]_x R.
    """
    t = np.asarray(t, dtype=np.float64).reshape(3)
    R = np.asarray(R, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    Tx = np.array(
        [[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float64
    )
    E = Tx @ R
    Kinv = np.linalg.inv(K)
    return Kinv.T @ E @ Kinv


def project(P, X):
    """Project 3D points X (N,3) with camera matrix P (3,3) -> homogeneous (N,3)."""
    X = np.asarray(X, dtype=np.float64)
    x = (P @ np.vstack([X.T, np.ones(X.shape[0])])).T
    x = x / x[:, 2:3]
    return x
