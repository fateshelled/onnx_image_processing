"""Benchmark a torch-based NumpySinkhornMatcher equivalent.

Compares numpy logsumexp vs torch.logsumexp on real cached descriptors and
checks numerical equivalence.

Usage: .venv/bin/python scripts/bench_sinkhorn_torch.py [seq]
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

if os.environ.get("TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vo.sinkhorn_numpy import NumpySinkhornMatcher  # noqa: E402

seq = sys.argv[1] if len(sys.argv) > 1 else "room"
z = np.load(REPO_ROOT / f"eval/results/tune_cache_loop/{seq}.npz",
            allow_pickle=True)
feat = list(zip(z["feat_idx"], z["feat_kp"], z["feat_desc"]))
desc = {int(i): d for i, k, d in feat}
idx = sorted(desc)
a, b = idx[len(idx) // 3], idx[len(idx) // 3 + 50]
d1, d2 = desc[a][0], desc[b][0]
print(f"seq={seq} desc {d1.shape} {d2.shape}")
print("torch threads:", torch.get_num_threads())


def match_probs_torch(d1, d2, iterations=20, epsilon=0.05, unused_score=1.0):
    t1 = torch.as_tensor(d1, dtype=torch.float32)
    t2 = torch.as_tensor(d2, dtype=torch.float32)
    N = t1.shape[0]
    M = t2.shape[0]
    norm1 = (t1 * t1).sum(-1, keepdim=True)
    norm2 = (t2 * t2).sum(-1, keepdim=True).T
    cost = torch.clamp(norm1 + norm2 - 2.0 * (t1 @ t2.T), min=0.0)
    log_scores = torch.full((N + 1, M + 1), -unused_score / epsilon)
    log_scores[:N, :M] = -cost / epsilon
    log_mu = torch.zeros(N + 1)
    log_mu[N] = float(np.log(float(M)))
    log_nu = torch.zeros(M + 1)
    log_nu[M] = float(np.log(float(N)))
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(log_scores + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(log_scores + u[:, None], dim=0)
    return torch.exp(log_scores + u[:, None] + v[None, :])


m = NumpySinkhornMatcher(iterations=20, epsilon=0.05, unused_score=1.0,
                         distance_type="l2")
P_ref = m.match_probs(d1, d2)
P_t = match_probs_torch(d1, d2).numpy()
print("max abs diff:", float(np.max(np.abs(P_ref - P_t))))
print("allclose(1e-5):", bool(np.allclose(P_ref, P_t, atol=1e-5)))

d1t, d2t = d1, d2
for name, fn in (("numpy", lambda: m.match_probs(d1t, d2t)),
                 ("torch", lambda: match_probs_torch(d1t, d2t))):
    fn()
    t = time.perf_counter()
    for _ in range(20):
        fn()
    dt = (time.perf_counter() - t) / 20
    print(f"{name}: {dt*1000:.1f} ms/call")
