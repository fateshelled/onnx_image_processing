"""so3_log must match scipy's principal-branch rotation vector exactly.

Pins the Shepperd quaternion branches (trace>0, largest diagonal, sign fix,
near-zero) against ``scipy.spatial.transform.Rotation`` so the numpy-only
replacement cannot silently drift in sign or branch selection.
"""
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vo.se3 import so3_log


def _scipy(R):
    return Rotation.from_matrix(R).as_rotvec()


def test_identity_and_small_angles():
    np.testing.assert_array_equal(so3_log(np.eye(3)), np.zeros(3))
    rng = np.random.default_rng(0)
    for mag in (1e-9, 1e-6, 1e-3, 0.1):
        for _ in range(20):
            R = Rotation.from_rotvec(rng.normal(size=3) * mag).as_matrix()
            np.testing.assert_allclose(so3_log(R), _scipy(R), atol=1e-12)


@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1),
                                  (-1, 0, 0), (0.3, -0.5, 0.8)])
@pytest.mark.parametrize("angle", [0.0, 0.2, 1.0, 2.0, math.pi - 1e-6, math.pi])
def test_branches_match_scipy(axis, angle):
    axis = np.asarray(axis, float)
    R = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
    np.testing.assert_allclose(so3_log(R), _scipy(R), atol=1e-10)


def test_random_rotations_match_scipy():
    rng = np.random.default_rng(1)
    for _ in range(5000):
        R = Rotation.random(random_state=rng).as_matrix()
        np.testing.assert_allclose(so3_log(R), _scipy(R), atol=1e-12)


def test_near_pi_matches_scipy_and_roundtrips():
    rng = np.random.default_rng(2)
    for _ in range(2000):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = math.pi - rng.uniform(0.0, 1e-3)
        R = Rotation.from_rotvec(axis * angle).as_matrix()
        v = so3_log(R)
        np.testing.assert_allclose(v, _scipy(R), atol=1e-9)
        assert np.linalg.norm(v) <= math.pi + 1e-9
        np.testing.assert_allclose(Rotation.from_rotvec(v).as_matrix(), R,
                                   atol=1e-9)
