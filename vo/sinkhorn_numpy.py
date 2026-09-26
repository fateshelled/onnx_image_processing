"""
NumPy re-implementation of SinkhornMatcher (matching/sinkhorn.py).

Faithful port of the torch/ONNX matcher for inference-time use without
torch: same L2 cost, same dustbin padding, same fixed
log-space iterations, same output layout P[K+1, K+1].

Used together with a per-frame keypoint/descriptor cache (single-image
ONNX export) so loop-closure candidate pairs only pay the matching cost,
not the per-pair detection cost.
"""

import numpy as np


def logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


class NumpySinkhornMatcher:
    """NumPy twin of ``SinkhornMatcher`` (pytorch_model/matching/sinkhorn.py).

    Args mirror the torch class. Verify against an exported model with
    :func:`match_probs` before relying on equality.
    """

    def __init__(
        self,
        iterations: int = 20,
        epsilon: float = 0.05,
        unused_score: float = 1.0,
        distance_type: str = "l2",
    ) -> None:
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.iterations = iterations
        self.epsilon = epsilon
        self.unused_score = unused_score
        self.distance_type = distance_type.lower()
        if self.distance_type not in ("l1", "l2"):
            raise ValueError(
                f"distance_type must be 'l1' or 'l2', got {distance_type}"
            )

    def cost_matrix(self, desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        """Pairwise cost [N, M] (single batch)."""
        if self.distance_type == "l2":
            norm1 = np.sum(desc1 ** 2, axis=-1, keepdims=True)  # [N, 1]
            norm2 = np.sum(desc2 ** 2, axis=-1, keepdims=True).T  # [1, M]
            cost = norm1 + norm2 - 2.0 * (desc1 @ desc2.T)
            return np.maximum(cost, 0.0)
        diff = desc1[:, None, :] - desc2[None, :, :]  # [N, M, D]
        return np.abs(diff).sum(axis=-1)

    def match_probs(self, desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        """Matching probability matrix [N+1, M+1] (dustbin row/col appended)."""
        N, D = desc1.shape
        M = desc2.shape[0]

        cost = self.cost_matrix(desc1, desc2)  # [N, M]
        log_scores_core = -cost / self.epsilon

        dustbin_score = -self.unused_score / self.epsilon

        log_scores = np.full((N + 1, M + 1), dustbin_score, dtype=desc1.dtype)
        log_scores[:N, :M] = log_scores_core

        log_mu = np.zeros(N + 1, dtype=desc1.dtype)
        log_mu[N] = np.log(float(M))
        log_nu = np.zeros(M + 1, dtype=desc1.dtype)
        log_nu[M] = np.log(float(N))

        u = np.zeros_like(log_mu)
        v = np.zeros_like(log_nu)
        for _ in range(self.iterations):
            u = log_mu - logsumexp(log_scores + v[None, :], axis=1)
            v = log_nu - logsumexp(log_scores + u[:, None], axis=0)

        log_P = log_scores + u[:, None] + v[None, :]
        return np.exp(log_P)
