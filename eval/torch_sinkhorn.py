"""Torch twin of ``NumpySinkhornMatcher`` (same math, faster logsumexp).

``NumpySinkhornMatcher`` spends most of its time in the elementwise
``logsumexp`` over the [N+1, M+1] score matrix. ``torch.logsumexp`` is
multi-threaded and roughly 6x faster on the cached 512x512 descriptors,
with float32-equivalent results (verified in scripts/bench_sinkhorn_torch.py).

Used only by the tuner (eval/rustuna_tune_loop.py --matcher torch); the
production pipeline keeps using the NumPy matcher.
"""

import os

import numpy as np
import torch


class TorchSinkhornMatcher:
    """Drop-in replacement exposing ``match_probs`` returning a numpy array."""

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
        threads = os.environ.get("TORCH_THREADS")
        if threads:
            torch.set_num_threads(int(threads))

    def cost_matrix(self, desc1: torch.Tensor, desc2: torch.Tensor) -> torch.Tensor:
        if self.distance_type == "l2":
            norm1 = (desc1 * desc1).sum(-1, keepdim=True)
            norm2 = (desc2 * desc2).sum(-1, keepdim=True).T
            cost = norm1 + norm2 - 2.0 * (desc1 @ desc2.T)
            return torch.clamp(cost, min=0.0)
        diff = desc1[:, None, :] - desc2[None, :, :]
        return diff.abs().sum(-1)

    def match_probs(self, desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        t1 = torch.as_tensor(desc1, dtype=torch.float32)
        t2 = torch.as_tensor(desc2, dtype=torch.float32)
        N = t1.shape[0]
        M = t2.shape[0]
        cost = self.cost_matrix(t1, t2)
        log_scores = torch.full((N + 1, M + 1),
                                -self.unused_score / self.epsilon)
        log_scores[:N, :M] = -cost / self.epsilon
        log_mu = torch.zeros(N + 1)
        log_mu[N] = float(np.log(float(M)))
        log_nu = torch.zeros(M + 1)
        log_nu[M] = float(np.log(float(N)))
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(self.iterations):
            u = log_mu - torch.logsumexp(log_scores + v[None, :], dim=1)
            v = log_nu - torch.logsumexp(log_scores + u[:, None], dim=0)
        return torch.exp(log_scores + u[:, None] + v[None, :]).numpy()
