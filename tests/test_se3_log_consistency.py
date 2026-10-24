import numpy as np
import pytest
from scipy.special import bernoulli, factorial

from vo.se3 import (se3_ad_e, se3_exp, se3_inv, se3_left_jacobian_inv,
                    se3_log, se3_right_jacobian_inv)
from vo.se3_window import SlidingWindowOptimizer


def _full_bernoulli(e):
    a = se3_ad_e(e)
    coeff = bernoulli(40) / factorial(np.arange(41))
    out = np.zeros((6, 6))
    power = np.eye(6)
    for k, c in enumerate(coeff):
        if k > 0:
            power = power @ a
        out += c * power
    return out


def test_se3_inv_matches_numpy_inverse_on_valid_poses():
    """The analytic SE(3) inverse must equal np.linalg.inv for valid poses."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        T = se3_exp(rng.normal(size=6) * rng.uniform(0.01, 1.0))
        np.testing.assert_allclose(se3_inv(T), np.linalg.inv(T),
                                   rtol=0, atol=1e-11)


def test_se3_left_jacobian_inv_matches_full_bernoulli_series():
    """The adaptive truncation must equal the fixed 41-term series."""
    rng = np.random.default_rng(7)
    axis = np.array([.3, -.5, .8])
    axis /= np.linalg.norm(axis)
    samples = [np.zeros(6),
               np.array([1e-9, 0.0, 0.0, 0.1, -0.2, 0.3]),
               np.array([0.3, -0.5, 0.8, 1.0, -2.0, 3.0])]
    for angle in (0.5, 1.8, 2.9, np.pi - 1e-6, np.pi):
        samples.append(np.r_[axis * angle, [50.0, -30.0, 20.0]])
    for _ in range(200):
        wd = rng.normal(size=3)
        wd /= np.linalg.norm(wd)
        samples.append(np.r_[wd * rng.uniform(0.0, np.pi),
                             rng.normal(size=3) * rng.uniform(0.0, 100.0)])
    for e in samples:
        np.testing.assert_allclose(se3_left_jacobian_inv(e), _full_bernoulli(e),
                                   rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("gap", [5e-7, 2e-6, 1e-4])
@pytest.mark.parametrize("sign", [-1, 1])
def test_log_near_pi_roundtrip_and_jacobian(gap, sign):
    axis = np.array([.3, -.5, .8])
    axis *= sign / np.linalg.norm(axis)
    xi = np.r_[axis * (np.pi-gap), [.4, -.2, .7]]
    T = se3_exp(xi)
    np.testing.assert_allclose(se3_log(T), xi, atol=1e-9)
    eps = 1e-8
    numeric = np.column_stack([
        (se3_log(T @ se3_exp(eps*d))-se3_log(T @ se3_exp(-eps*d)))/(2*eps)
        for d in np.eye(6)])
    np.testing.assert_allclose(se3_right_jacobian_inv(xi), numeric, atol=2e-7)


def test_scale_derivative_with_nondiagonal_information():
    obj = SlidingWindowOptimizer()
    obj.add_node(0, np.eye(4))
    obj.add_node(1, se3_exp(np.array([.1, .2, -.1, .3, -.2, .5])))
    W = np.diag(np.arange(1., 7.))
    W[0, 4] = .7
    obj.add_edge(0, 1, se3_exp(np.array([.05, 0., .1, .5, .2, -.1])),
                 scale_free=True, omega=W.T@W)
    eps = 1e-6
    obj.edge_scale[0] = np.exp(eps)
    rp = obj._residuals()[:6]
    obj.edge_scale[0] = np.exp(-eps)
    rm = obj._residuals()[:6]
    obj.edge_scale[0] = 1.
    np.testing.assert_allclose(obj._scale_jacobian(0), (rp-rm)/(2*eps), atol=1e-7)
