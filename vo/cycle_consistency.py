"""Rotation-only cycle consistency for loop-closure edge verification.

Monocular scale drifts, so a metric cycle check is unreliable; the rotation
part of SE(3) is scale-free. Given a candidate closure edge ``(a, c)``, this
module checks whether an alternative path agrees on rotation, either a 2-hop
path ``a -> b -> c`` through other accepted closure edges
(``R_ac ~= R_bc @ R_ab``) or the odometry chain between the same nodes
(``chain_residual_deg``).

The alternative path deliberately excludes odometry edges: the whole point is
to compare independent matched measurements, not to re-derive the drift the
loop is meant to fix.

Conventions match ``se3_window``: for a transform from ``src`` to ``dst``,
``M_src_dst = T_dst^{-1} T_src``, so rotations compose as ``R_ac = R_bc @ R_ab``
along ``a -> b -> c``.
"""

from __future__ import annotations

import numpy as np

from .loop_closure import edge_key


def adjacency(edge_pairs) -> dict:
    """Undirected adjacency map from an iterable of ``(i, j)`` pairs."""
    adj: dict[int, set] = {}
    for (i, j) in edge_pairs:
        i, j = int(i), int(j)
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    return adj


def relative_rotation(rotations: dict, src: int, dst: int):
    """Rotation of the ``src -> dst`` transform from canonical stored data.

    ``rotations`` maps ``edge_key(i, j)`` to the rotation of the ``i -> j``
    transform where ``i == min(i, j)`` (the harness always adds ``a < b``
    edges). Returns ``None`` when the edge is absent.
    """
    i, j = edge_key(src, dst)
    R = rotations.get((i, j))
    if R is None:
        return None
    R = np.asarray(R, float)
    return R if src == i else R.T


def rotation_angle_deg(R1, R2) -> float:
    """Geodesic angle between two rotation matrices, in degrees."""
    R = np.asarray(R1, float) @ np.asarray(R2, float).T
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def cumulative_rotations(node_ids, edge_rotations: dict) -> dict:
    """Cumulative rotation along an ordered node chain.

    ``edge_rotations`` maps ``edge_key(i, j)`` to the rotation of the
    ``i -> j`` transform (``i == min(i, j)``). Nodes must be given in
    increasing order. Missing edges (skipped frames) inherit the previous
    cumulative rotation, so ``R_ab = cum[b] @ cum[a].T`` for ``a < b``.
    """
    ids = list(node_ids)
    cum: dict = {}
    if not ids:
        return cum
    cum[ids[0]] = np.eye(3)
    for k in range(1, len(ids)):
        prev, cur = ids[k - 1], ids[k]
        R_e = edge_rotations.get(edge_key(prev, cur))
        cum[cur] = (np.asarray(R_e, float) @ cum[prev]
                    if R_e is not None else cum[prev])
    return cum


def chain_rotation(cum: dict, src: int, dst: int):
    """Rotation of the ``src -> dst`` transform from cumulative rotations."""
    return np.asarray(cum[dst], float) @ np.asarray(cum[src], float).T


def chain_residual_deg(a: int, b: int, R_ab, cum: dict) -> float:
    """Rotation disagreement between an edge and an alternative chain path.

    With ``cum`` built from the odometry chain this is the rotation part of
    the cycle formed by the loop edge and the odometry path between the same
    nodes. Scale-free, and independent of any other loop decision.
    """
    return rotation_angle_deg(np.asarray(R_ab, float), chain_rotation(cum, a, b))


def best_cycle_residual_deg(a: int, c: int, R_ac, rotations: dict, adj: dict,
                            skip=()):
    """Smallest rotation disagreement over 2-hop paths ``a -> b -> c``.

    Returns ``None`` when no alternative path exists (the edge cannot be
    verified), otherwise the smallest residual in degrees.
    """
    R_ac = np.asarray(R_ac, float)
    common = adj.get(int(a), set()) & adj.get(int(c), set())
    best = None
    for b in common:
        if b == a or b == c or b in skip:
            continue
        R_ab = relative_rotation(rotations, a, b)
        R_bc = relative_rotation(rotations, b, c)
        if R_ab is None or R_bc is None:
            continue
        res = rotation_angle_deg(R_ac, R_bc @ R_ab)
        if best is None or res < best:
            best = res
    return best


def cycle_verified(a: int, c: int, R_ac, rotations: dict, adj: dict,
                   threshold_deg: float, skip=()):
    """Tri-state verification: ``True``/``False``, or ``None`` if unverifiable.

    ``None`` means no alternative path exists yet (for example the very first
    closure edge); callers should fall back to other gates rather than reject.
    """
    res = best_cycle_residual_deg(a, c, R_ac, rotations, adj, skip=skip)
    if res is None:
        return None
    return res <= threshold_deg
