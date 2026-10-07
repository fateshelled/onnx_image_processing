import numpy as np
import pytest

from vo.se3 import se3_exp, se3_log, se3_right_jacobian_inv
from vo.se3_window import SlidingWindowOptimizer


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
