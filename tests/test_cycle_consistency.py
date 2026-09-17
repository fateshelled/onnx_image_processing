"""Synthetic tests for rotation-only cycle consistency (loop-edge verification)."""

import numpy as np

from vo.se3 import se3_exp
from vo.cycle_consistency import (
    adjacency,
    best_cycle_residual_deg,
    chain_residual_deg,
    chain_rotation,
    cumulative_rotations,
    cycle_verified,
    relative_rotation,
    rotation_angle_deg,
)

IDS = [0, 16, 32, 48]
DELTAS = [
    np.array([0.10, -0.05, 0.02, 0.00, 0.03, 0.00]),
    np.array([0.12, 0.02, -0.01, 0.00, -0.02, 0.01]),
    np.array([0.08, 0.04, 0.03, 0.02, 0.00, -0.01]),
]


def _poses():
    T = {IDS[0]: np.eye(4)}
    for k in range(1, len(IDS)):
        T[IDS[k]] = T[IDS[k - 1]] @ se3_exp(DELTAS[k - 1])
    return T


def _rotations(T):
    rot = {}
    for i in IDS:
        for j in IDS:
            if i < j:
                rot[(i, j)] = (np.linalg.inv(T[j]) @ T[i])[:3, :3]
    return rot


def test_rotation_angle_deg():
    assert rotation_angle_deg(np.eye(3), np.eye(3)) < 1e-6
    Rz = se3_exp(np.array([0.0, 0.0, np.radians(90.0), 0, 0, 0]))[:3, :3]
    assert abs(rotation_angle_deg(np.eye(3), Rz) - 90.0) < 1e-6


def test_relative_rotation_direction():
    T = _poses()
    R_ab = (np.linalg.inv(T[16]) @ T[0])[:3, :3]
    rot = {(0, 16): R_ab}
    assert np.allclose(relative_rotation(rot, 0, 16), R_ab)
    assert np.allclose(relative_rotation(rot, 16, 0), R_ab.T)
    assert relative_rotation(rot, 0, 32) is None


def test_clean_triangle_is_consistent():
    T = _poses()
    rot = _rotations(T)
    adj = adjacency([(0, 16), (16, 32), (0, 32)])
    res = best_cycle_residual_deg(0, 32, rot[(0, 32)], rot, adj)
    assert res is not None and res < 1e-6
    assert cycle_verified(0, 32, rot[(0, 32)], rot, adj, 10.0) is True


def test_wrong_rotation_is_rejected():
    T = _poses()
    rot = _rotations(T)
    adj = adjacency([(0, 16), (16, 32), (0, 32)])
    R_wrong = rot[(0, 32)] @ se3_exp(
        np.array([0.0, 0.0, np.radians(30.0), 0, 0, 0]))[:3, :3]
    res = best_cycle_residual_deg(0, 32, R_wrong, rot, adj)
    assert 29.0 < res < 31.0
    assert cycle_verified(0, 32, R_wrong, rot, adj, 10.0) is False


def test_no_alternative_path_is_unverifiable():
    T = _poses()
    rot = _rotations(T)
    adj = adjacency([(0, 16)])  # 32 has no second edge
    assert best_cycle_residual_deg(0, 32, rot[(0, 32)], rot, adj) is None
    assert cycle_verified(0, 32, rot[(0, 32)], rot, adj, 10.0) is None


def test_skip_excludes_intermediate():
    T = _poses()
    rot = _rotations(T)
    adj = adjacency([(0, 16), (16, 32), (0, 32)])
    assert best_cycle_residual_deg(0, 32, rot[(0, 32)], rot, adj,
                                   skip=(16,)) is None


def test_cumulative_chain_matches_direct_edges():
    T = _poses()
    rot = _rotations(T)
    cum = cumulative_rotations(IDS, rot)
    for (i, j) in [(0, 16), (16, 32), (0, 32), (0, 48)]:
        assert rotation_angle_deg(chain_rotation(cum, i, j), rot[(i, j)]) < 1e-9
        assert chain_residual_deg(i, j, rot[(i, j)], cum) < 1e-9


def test_chain_residual_detects_wrong_edge():
    T = _poses()
    rot = _rotations(T)
    cum = cumulative_rotations(IDS, rot)
    R_wrong = rot[(0, 32)] @ se3_exp(
        np.array([0.0, 0.0, np.radians(30.0), 0, 0, 0]))[:3, :3]
    assert abs(chain_residual_deg(0, 32, R_wrong, cum) - 30.0) < 1e-6


def test_cumulative_missing_edge_inherits():
    T = _poses()
    rot = _rotations(T)
    pruned = {k: v for k, v in rot.items() if k != (16, 32)}
    cum = cumulative_rotations(IDS, pruned)
    assert np.allclose(cum[32], cum[16])


if __name__ == "__main__":
    test_rotation_angle_deg()
    test_relative_rotation_direction()
    test_clean_triangle_is_consistent()
    test_wrong_rotation_is_rejected()
    test_no_alternative_path_is_unverifiable()
    test_skip_excludes_intermediate()
    test_cumulative_chain_matches_direct_edges()
    test_chain_residual_detects_wrong_edge()
    test_cumulative_missing_edge_inherits()
    print("cycle consistency tests passed")
