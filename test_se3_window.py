#!/usr/bin/env python3
"""Tests for se3 utils and the sliding-window pose-graph optimizer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from pytorch_model.vo.se3 import se3_ad, se3_exp, se3_log
from pytorch_model.vo.se3_window import SlidingWindowOptimizer, _edge_residual


def rand_T(rng, max_t=1.0, max_rot=0.5):
    T = np.eye(4)
    phi = rng.uniform(-max_rot, max_rot, 3)
    t = rng.uniform(-max_t, max_t, 3)
    T[:3, :3] = se3_exp(np.concatenate([phi, np.zeros(3)]))[:3, :3]
    T[:3, 3] = t
    return T


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
