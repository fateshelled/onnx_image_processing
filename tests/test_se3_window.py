#!/usr/bin/env python3
"""Tests for se3 utils and the sliding-window pose-graph optimizer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from vo.se3 import se3_ad, se3_exp, se3_log
from vo.se3_window import SlidingWindowOptimizer, _edge_residual


def rand_T(rng, max_t=1.0, max_rot=0.5):
    T = np.eye(4)
    phi = rng.uniform(-max_rot, max_rot, 3)
    t = rng.uniform(-max_t, max_t, 3)
    T[:3, :3] = se3_exp(np.concatenate([phi, np.zeros(3)]))[:3, :3]
    T[:3, 3] = t
    return T


class TestLoopRobustWeights:
    def _optimizer_with_edge(self, **kwargs):
        opt = SlidingWindowOptimizer(window_size=None, **kwargs)
        opt.add_node(0, np.eye(4))
        opt.add_node(1, np.eye(4))
        opt.add_edge(0, 1, np.eye(4), robust=True)
        return opt

    def test_default_mode_preserves_legacy_weight(self):
        opt = self._optimizer_with_edge(loop_robust="none")
        assert opt._loop_robust_weight(0, 100.0) == 1.0
        assert np.allclose(opt._huber_weights(), np.ones(6))

    def test_dcs_formula_and_floor(self):
        opt = self._optimizer_with_edge(loop_robust="dcs",
                                        loop_robust_phi=2.0,
                                        loop_robust_min_weight=0.1)
        assert opt._loop_robust_weight(0, 0.0) == 1.0
        assert opt._loop_robust_weight(0, np.sqrt(2.0)) == 1.0
        assert opt._loop_robust_weight(0, np.sqrt(6.0)) == pytest.approx(0.5)
        assert opt._loop_robust_weight(0, 1e6) == pytest.approx(0.1)

    def test_only_flagged_edge_is_downweighted(self):
        opt = SlidingWindowOptimizer(window_size=None, loop_robust="dcs",
                                     loop_robust_phi=1.0, huber=1e9)
        for i in range(3):
            opt.add_node(i, np.eye(4))
        bad = se3_exp(np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0]))
        opt.add_edge(0, 1, bad, robust=False)
        opt.add_edge(1, 2, bad, robust=True)
        weights = opt._huber_weights()
        assert np.allclose(weights[:6], 1.0)
        assert np.all(weights[6:12] < 0.5)
        assert opt.n_robust_downweighted == 1

    def test_drop_oldest_keeps_robust_flags_aligned(self):
        opt = SlidingWindowOptimizer(window_size=3, loop_robust="dcs")
        for i in range(3):
            opt.add_node(i, np.eye(4))
        opt.add_edge(0, 1, np.eye(4), robust=True)
        opt.add_edge(1, 2, np.eye(4), robust=False)
        opt.add_node(3, np.eye(4))
        assert len(opt.edges) == len(opt.edge_robust) == 1
        assert opt.edge_robust == [False]

    def test_gm_formula_and_block_separation(self):
        opt = SlidingWindowOptimizer(window_size=None, loop_robust="gm",
                                     loop_gm_mu=4.0, loop_dir_sigma=0.4,
                                     huber=1e9)
        opt.add_node(0, np.eye(4))
        T1 = np.eye(4)
        T1[:3, 3] = [1.0, 0.0, 0.0]
        opt.add_node(1, T1)
        M = np.linalg.inv(T1)
        M[:3, 3] = [0.0, 1.0, 0.0]
        opt.add_edge(0, 1, M, robust=True)
        e = _edge_residual(opt.T[0], opt.T[1], opt._scaled_M(0))
        wr, wt = opt._loop_block_weights(0, e)
        assert wr == pytest.approx(1.0)
        assert wt < 0.5
        # Pin the actual implementation at x=sqrt(mu): GM weight = 0.25.
        assert (4.0 / (4.0 + 4.0)) ** 2 == pytest.approx(0.25)
        opt._loop_block_norms = lambda _k, _e: (0.0, 2.0)
        assert opt._loop_block_weights(0, e)[1] == pytest.approx(0.25)

    def test_block_mode_is_per_block_huber_only(self):
        opt = self._optimizer_with_edge(loop_robust="block", huber=1.0)
        opt._loop_block_norms = lambda _k, _e: (0.5, 2.0)
        wr, wt = opt._loop_block_weights(0, np.zeros(6))
        assert wr == pytest.approx(1.0)
        assert wt == pytest.approx(0.5)

    def test_direction_weight_is_scale_invariant(self):
        opt = SlidingWindowOptimizer(window_size=None, loop_robust="gm")
        opt.add_node(0, np.eye(4))
        T1 = np.eye(4)
        T1[:3, 3] = [1.0, 1.0, 0.0]
        opt.add_node(1, T1)
        M = np.eye(4)
        M[:3, 3] = [-1.0, 0.0, 0.0]
        opt.add_edge(0, 1, M, robust=True)
        vals = []
        for scale in (0.2, 1.0, 5.0):
            opt.edge_scale[0] = scale
            e = _edge_residual(opt.T[0], opt.T[1], opt._scaled_M(0))
            vals.append(opt._loop_block_weights(0, e)[1])
        assert np.allclose(vals, vals[0], atol=1e-12)

    def test_gnc_schedule_reaches_final_mu(self):
        opt = SlidingWindowOptimizer(window_size=None, loop_robust="gnc_gm",
                                     loop_gm_mu=11.34)
        opt.add_node(0, np.eye(4))
        opt.add_node(1, np.eye(4))
        bad = se3_exp(np.array([0.5, 0.0, 0.0, 1.0, 0.0, 0.0]))
        opt.add_edge(0, 1, bad, robust=True)
        schedule = opt._gnc_schedule()
        assert all(a >= b for a, b in zip(schedule, schedule[1:]))
        assert schedule[-1] == pytest.approx(11.34)
        assert len(schedule) <= 8
        # Exercise dispatcher and final-mu restoration, not only the helper.
        opt.add_node(2, np.eye(4))
        opt.add_edge(1, 2, np.eye(4))
        opt.optimize()
        assert opt._gnc_mu == pytest.approx(opt.loop_gm_mu)

    def test_mad_scale_leave_one_out_and_clamps(self):
        hist = {"rot": {(0, 2): 0.1, (1, 3): 1.0, (2, 4): 2.0,
                        (3, 5): 3.0},
                "dir": {}}
        opt = SlidingWindowOptimizer(
            window_size=None, loop_robust="gm", loop_scale_mode="mad",
            loop_robust_hist=hist, loop_mad_min_samples=2,
            loop_scale_floor=0.01, loop_scale_max_rot=10.0)
        opt.add_node(0, np.eye(4))
        opt.add_node(2, np.eye(4))
        opt.add_edge(0, 2, np.eye(4), robust=True)
        # Target edge's 0.1 is excluded. Including it would give median 1.5.
        assert opt._mad_scale("rot", 0) == pytest.approx(1.4826 * 2.0)

    def test_mad_scale_fallback_and_zero_floor(self):
        opt = SlidingWindowOptimizer(
            window_size=None, loop_scale_mode="mad", loop_rot_sigma=0.12,
            loop_robust_hist={"rot": {(1, 2): 0.0, (2, 3): 0.0}, "dir": {}},
            loop_mad_min_samples=2, loop_scale_floor=0.01)
        opt.add_node(0, np.eye(4))
        opt.add_node(1, np.eye(4))
        opt.add_edge(0, 1, np.eye(4), robust=True)
        assert opt._mad_scale("rot", 0) == pytest.approx(0.01)

        fallback = SlidingWindowOptimizer(
            window_size=None, loop_scale_mode="mad", loop_rot_sigma=0.12,
            loop_robust_hist={"rot": {(2, 3): 0.5}, "dir": {}},
            loop_mad_min_samples=2)
        fallback.add_node(0, np.eye(4))
        fallback.add_node(1, np.eye(4))
        fallback.add_edge(0, 1, np.eye(4), robust=True)
        assert fallback._mad_scale("rot", 0) == pytest.approx(0.12)


class TestPerEdgeScalePriorSigma:
    def test_per_edge_sigma_and_zero_sigma_safe(self):
        rng = np.random.default_rng(7)
        T = [rand_T(rng) for _ in range(4)]
        opt = SlidingWindowOptimizer(window_size=None, max_iterations=5,
                                     optimize_scale=True, scale_prior_sigma=2.0)
        for i, Ti in enumerate(T):
            opt.add_node(i, Ti)
        # edge 0-1 becomes the gauge; edge 1-2 has sigma 0 (no prior row);
        # edge 2-3 uses a tight per-edge sigma.
        opt.add_edge(0, 1, rand_T(rng), scale_free=True)
        opt.add_edge(1, 2, rand_T(rng), scale_free=True,
                     scale_prior_mean=float(np.log(3.0)), scale_prior_sigma=0.0)
        opt.add_edge(2, 3, rand_T(rng), scale_free=True,
                     scale_prior_mean=float(np.log(3.0)), scale_prior_sigma=0.1)
        assert opt.edge_scale_sigma == [2.0, 0.0, 0.1]
        # Only the sigma>0 non-gauge edge contributes a scale-prior row.
        assert opt._n_prior() == 1
        opt.optimize()  # must not raise ZeroDivisionError
        assert all(np.isfinite(s) for s in opt.edge_scale)

    def test_drop_oldest_keeps_lists_in_sync(self):
        rng = np.random.default_rng(8)
        T = [rand_T(rng) for _ in range(4)]
        opt = SlidingWindowOptimizer(window_size=3, optimize_scale=True,
                                     scale_prior_sigma=2.0)
        for i in range(3):
            opt.add_node(i, T[i])
        opt.add_edge(0, 1, rand_T(rng), scale_free=True)
        opt.add_edge(1, 2, rand_T(rng), scale_free=True)
        opt.add_node(3, T[3])  # drops node 0 and the edge (0,1)
        opt.add_edge(2, 3, rand_T(rng), scale_free=True)
        assert len(opt.edge_scale) == len(opt.edge_scale_sigma) \
            == len(opt.scale_prior_mean) == len(opt.edges)


class TestSe3:
    def test_exp_log_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            delta = rng.uniform(-0.8, 0.8, 6)
            T = se3_exp(delta)
            back = se3_log(T)
            assert np.allclose(back, delta, atol=1e-6)

    def test_log_exp_roundtrip(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            T = rand_T(rng, max_t=2.0, max_rot=2.0)
            delta = se3_log(T)
            assert np.allclose(se3_exp(delta), T, atol=1e-6)

    def test_rotation_matrix_valid(self):
        T = se3_exp(np.array([0.5, 0.1, -0.2, 0.3, 0.4, 0.5]))
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)

    def test_adjoint_definition(self):
        rng = np.random.default_rng(2)
        T = rand_T(rng, max_rot=1.0)
        delta = rng.uniform(-0.3, 0.3, 6)
        lhs = se3_exp(se3_ad(T) @ delta)
        rhs = T @ se3_exp(delta) @ np.linalg.inv(T)
        assert np.allclose(lhs, rhs, atol=1e-9)


class TestWindowOptimizer:
    def test_exact_chain_is_fixed_point(self):
        rng = np.random.default_rng(3)
        T_gen = [np.eye(4)]
        meas = []
        for _ in range(6):
            M = rand_T(rng, max_t=0.4, max_rot=0.2)
            meas.append(M)
            # T_{k+1}^{-1} T_k = M  =>  T_{k+1} = T_k @ M^{-1}
            T_gen.append(T_gen[-1] @ np.linalg.inv(M))
        opt = SlidingWindowOptimizer(window_size=7)
        opt.add_node(0, T_gen[0])
        for k in range(6):
            opt.add_node(k + 1, T_gen[k + 1])
            opt.add_edge(k, k + 1, meas[k])
        before = [opt.get_pose(n).copy() for n in range(7)]
        cost = opt.optimize()
        after = [opt.get_pose(n).copy() for n in range(7)]
        assert cost == pytest.approx(0.0, abs=1e-6)
        for b, a in zip(before, after):
            assert np.allclose(b, a, atol=1e-7)

    def test_noisy_chain_with_loop_edge_improves(self):
        """Noisy chain: pure chaining drifts; one clean (star) edge between
        the first and last node lets the solver pull the chain back."""
        rng = np.random.default_rng(4)
        corr = se3_exp(np.array([0.02, -0.01, 0.015, 0.015, -0.02, 0.02]))
        # Ground truth chain and noisy measurements of the same relative steps
        T_true = [np.eye(4)]
        G_true, meas = [], []
        for _ in range(6):
            Gt = rand_T(rng, max_t=0.4, max_rot=0.2)
            G_true.append(Gt)
            meas.append(Gt @ corr)
            T_true.append(T_true[-1] @ np.linalg.inv(Gt))
        T_init = [np.eye(4)]
        for k in range(6):
            T_init.append(T_init[-1] @ np.linalg.inv(meas[k]))
        opt = SlidingWindowOptimizer(window_size=7, step_scale_t=0.1, step_scale_r=0.1)
        opt.add_node(0, T_init[0])
        for k in range(6):
            opt.add_node(k + 1, T_init[k + 1])
            opt.add_edge(k, k + 1, meas[k])
        opt.add_edge(0, 6, np.linalg.inv(T_true[6]) @ T_true[0], sigma_t=0.2, sigma_r=0.2)
        cost_init = opt._cost()
        cost = opt.optimize()
        assert cost <= cost_init + 1e-12
        # Node 6 position should end closer to ground truth than the noisy chain
        e_init = np.linalg.norm(T_init[6][:3, 3] - T_true[6][:3, 3])
        e_after = np.linalg.norm(opt.get_pose(6)[:3, 3] - T_true[6][:3, 3])
        assert e_after < e_init

    def test_window_slides(self):
        rng = np.random.default_rng(5)
        opt = SlidingWindowOptimizer(window_size=4)
        opt.add_node(0, np.eye(4))
        T = np.eye(4)
        for k in range(8):
            M = rand_T(rng, 0.3, 0.2)
            T = T @ np.linalg.inv(M)
            opt.add_node(k + 1, T)
            opt.add_edge(k, k + 1, M)
        assert len(opt.pose_ids) == 4
        assert opt.pose_ids[-1] == 8

    def test_bad_node_ids(self):
        opt = SlidingWindowOptimizer(window_size=4)
        opt.add_node(0, np.eye(4))
        with pytest.raises(ValueError):
            opt.add_node(0, np.eye(4))
        opt.add_node(5, np.eye(4))
        with pytest.raises(ValueError):
            opt.add_node(3, np.eye(4))


def _random_graph(seed=0, n_nodes=40):
    rng = np.random.default_rng(seed)
    T = [np.eye(4)]
    meas = []
    for _ in range(n_nodes - 1):
        M = rand_T(rng, 0.5, 0.3)
        meas.append(M)
        T.append(T[-1] @ np.linalg.inv(M))
    return rng, T, meas


def _run_graph(dense_max_cols, tsvd_ratio=0.0):
    """Build a noisy chain + one loop edge with free scales and optimize."""
    rng, T, meas = _random_graph()
    n = len(T)
    opt = SlidingWindowOptimizer(
        window_size=None, max_iterations=40, step_scale_t=0.1, step_scale_r=0.1,
        optimize_scale=True, scale_prior_sigma=0.5, tsvd_ratio=tsvd_ratio,
        dense_max_cols=dense_max_cols)
    for i, Ti in enumerate(T):
        opt.add_node(i, Ti)
    for k, M in enumerate(meas):
        opt.add_edge(k, k + 1, M, scale_free=True)
    # Noisy global loop edge 0 -> n-1 (true relative pose is inv(T_{n-1}) T_0).
    M_loop = np.linalg.inv(T[-1]) @ T[0]
    M_loop[:3, 3] = M_loop[:3, 3] * 0.9 + rng.normal(0, 0.05, 3)
    opt.add_edge(0, n - 1, M_loop, scale_free=True)
    cost = opt.optimize()
    poses = [opt.get_pose(i) for i in range(n)]
    return cost, poses, list(opt.edge_scale), opt.tsvd_kept


class TestSparseSolver:
    def test_jacobian_coo_matches_dense(self):
        rng, T, meas = _random_graph(seed=1)
        opt = SlidingWindowOptimizer(window_size=None, max_iterations=1,
                                     optimize_scale=True, scale_prior_sigma=0.3)
        for i, Ti in enumerate(T):
            opt.add_node(i, Ti)
        for k, M in enumerate(meas):
            opt.add_edge(k, k + 1, M, scale_free=True)
        rows, cols, vals, shape = opt._jacobian_coo()
        Jcoo = np.zeros(shape)
        Jcoo[rows, cols] = vals
        assert shape == opt._jacobian().shape
        assert np.allclose(Jcoo, opt._jacobian(), atol=1e-12)

    @pytest.mark.parametrize("tsvd_ratio", [0.0, 1e-3])
    def test_sparse_matches_dense(self, tsvd_ratio):
        cost_d, poses_d, scale_d, _ = _run_graph(10 ** 9, tsvd_ratio)
        cost_s, poses_s, scale_s, kept_s = _run_graph(0, tsvd_ratio)
        assert cost_s == pytest.approx(cost_d, rel=1e-8, abs=1e-10)
        for pd, ps in zip(poses_d, poses_s):
            assert np.allclose(pd, ps, atol=1e-8)
        assert np.allclose(scale_d, scale_s, atol=1e-8)
        if tsvd_ratio > 0.0:
            assert kept_s > 0


class TestMarginalization:
    def test_relative_prior_reproduces_full_solution(self):
        rng = np.random.default_rng(7)
        true = [np.eye(4)]
        meas = []
        for _ in range(4):
            Gt = rand_T(rng, 0.4, 0.25)
            true.append(true[-1] @ np.linalg.inv(Gt))
            meas.append(Gt @ rand_T(rng, 0.02, 0.01))
        T = [np.eye(4)]
        for k in range(4):
            T.append(T[-1] @ np.linalg.inv(meas[k]))

        full = SlidingWindowOptimizer(window_size=None, max_iterations=60,
                                     step_scale_t=0.1, step_scale_r=0.1)
        for i, Ti in enumerate(T):
            full.add_node(i, Ti)
        for k in range(4):
            full.add_edge(k, k + 1, meas[k])
        full.add_edge(0, 2, np.linalg.inv(T[2]) @ T[0])  # spoke
        cost_full = full.optimize()

        a, b, G, Omega = full.marginalize_relative([1, 2, 3], [0, 4])
        assert a == 0 and b == 4
        assert Omega.shape == (6, 6)
        assert np.allclose(Omega, Omega.T, atol=1e-9)

        red = SlidingWindowOptimizer(window_size=None, max_iterations=60,
                                     step_scale_t=0.1, step_scale_r=0.1)
        red.add_node(0, T[0])
        red.add_node(4, T[4])
        red.add_edge(0, 4, G, omega=Omega)
        red.optimize()

        # The marginalized relative prior should place node 4 where the full
        # batch graph does (nonlinearity is small after convergence).
        assert np.allclose(red.get_pose(4), full.get_pose(4), atol=5e-3)
        assert cost_full < 1e-1

class TestMultiNodePrior:
    def test_general_marginalization_with_loop(self):
        rng = np.random.default_rng(21)
        meas = []
        for _ in range(4):
            meas.append(rand_T(rng, 0.4, 0.25) @ rand_T(rng, 0.02, 0.01))
        T = [np.eye(4)]
        for k in range(4):
            T.append(T[-1] @ np.linalg.inv(meas[k]))

        full = SlidingWindowOptimizer(window_size=None, max_iterations=60,
                                     step_scale_t=0.1, step_scale_r=0.1)
        for i, Ti in enumerate(T):
            full.add_node(i, Ti)
        for k in range(4):
            full.add_edge(k, k + 1, meas[k])
        M_loop = np.linalg.inv(T[3]) @ T[0]  # loop 0 -> 3
        M_loop[:3, 3] *= 0.9
        full.add_edge(0, 3, M_loop)
        full.optimize()

        # Node 0's Markov blanket here is {1 (chain), 3 (loop)} -> 2 nodes, but
        # the primitive is general (any number of retained nodes).
        keep, H_r, b_r = full.marginalize_general([0], [1, 3])
        assert keep == [1, 3]
        assert H_r.shape == (12, 12)

        red = SlidingWindowOptimizer(window_size=None, max_iterations=60,
                                     step_scale_t=0.1, step_scale_r=0.1)
        for i in (1, 2, 3, 4):
            red.add_node(i, full.get_pose(i))
        for k in (1, 2, 3):
            red.add_edge(k, k + 1, meas[k])
        red.add_prior_factor([1, 3], H_r, b_r,
                             [full.get_pose(1), full.get_pose(3)])
        red.optimize()
        for nd in (2, 3, 4):
            assert np.allclose(red.get_pose(nd), full.get_pose(nd), atol=5e-3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
