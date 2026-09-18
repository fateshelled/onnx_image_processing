"""Benchmark NumpySinkhornMatcher.logsumexp vs scipy.special.logsumexp.

Verifies numerical equivalence and measures match_probs speedup on real
cached descriptors.

Usage: .venv/bin/python scripts/bench_sinkhorn.py [seq]
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp as sp_logsumexp

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
print(f"seq={seq} frames {a},{b} desc {d1.shape} {d2.shape} dtype={d1.dtype}")


def match_probs_scipy(self, desc1, desc2):
    N, D = desc1.shape
    M = desc2.shape[0]
    cost = self.cost_matrix(desc1, desc2)
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
        u = log_mu - sp_logsumexp(log_scores + v[None, :], axis=1)
        v = log_nu - sp_logsumexp(log_scores + u[:, None], axis=0)
    return np.exp(log_scores + u[:, None] + v[None, :])


m = NumpySinkhornMatcher(iterations=20, epsilon=0.05, unused_score=1.0,
                         distance_type="l2")

P_ref = m.match_probs(d1, d2)
P_sp = match_probs_scipy(m, d1, d2)
print("max abs diff:", float(np.max(np.abs(P_ref - P_sp))))
print("allclose(1e-5):", bool(np.allclose(P_ref, P_sp, atol=1e-5)))

for name, fn in (("current", lambda x, y: m.match_probs(x, y)),
                 ("scipy", lambda x, y: match_probs_scipy(m, x, y))):
    fn(d1, d2)
    t = time.perf_counter()
    for _ in range(20):
        fn(d1, d2)
    dt = (time.perf_counter() - t) / 20
    print(f"{name}: {dt*1000:.1f} ms/call")
