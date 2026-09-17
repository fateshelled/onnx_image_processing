"""Integrated keyframe matching + loop-closure edge selection.

The VO harness uses one appearance matcher for two purposes:

* **loop closure** -- a new keyframe matches an older keyframe (index gap
  >= ``loop_min_gap``) and the measured relative pose is added as a loop
  edge;
* **local refine** -- when no loop is confirmed, the new keyframe matches
  its immediate predecessor, a shorter baseline than the chained odometry
  edges, and the measured relative pose is added as a refinement edge.

Both share the same matcher. This module holds the numpy-only decision
logic (temporal-consistency gate, duplicate-edge guard, loop/local branch)
so it can be unit-tested without ONNX or image data.

A *hit* is the tuple ``(a, R, t, inlier_ratio, n_matches)``: the older
keyframe index plus the relative pose measured against the current
keyframe ``b``.
"""

from __future__ import annotations

Hit = tuple


def edge_key(i: int, j: int) -> tuple:
    """Undirected, order-independent edge identity."""
    i, j = int(i), int(j)
    return (i, j) if i <= j else (j, i)


def temporal_confirmed(a: int, kk: int, hits_per_kf: list, need: int,
                       margin: float) -> bool:
    """True when the previous ``need - 1`` keyframes also saw loop spot ``a``.

    ``hits_per_kf[m]`` is the list of hits collected by keyframe ``kf[m]``.
    A previous keyframe confirms when it reported a hit whose old-keyframe
    index is within ``margin`` of ``a`` (the loop spot wanders by up to
    roughly one keyframe step between consecutive keyframes). ``need == 1``
    disables the gate.
    """
    if need <= 1:
        return True
    return all(
        kk - d >= 0
        and any(abs(int(a2) - int(a)) <= margin for (a2, *_) in hits_per_kf[kk - d])
        for d in range(1, need)
    )


def confirmed_loop_hits(b: int, hits: list, kk: int, hits_per_kf: list,
                        need: int, margin: float, added_edges: set) -> list:
    """Loop hits for keyframe ``b`` that pass the temporal gate and dedup.

    ``added_edges`` holds :func:`edge_key` tuples already present in the
    graph (odometry plus accepted closures), so an edge is never added
    twice.
    """
    out = []
    for hit in hits:
        a = int(hit[0])
        if edge_key(a, b) in added_edges:
            continue
        if not temporal_confirmed(a, kk, hits_per_kf, need, margin):
            continue
        out.append(hit)
    return out


def local_candidate(kf: list, bi: int, added_edges: set):
    """Nearest previous keyframe for a refinement edge, or ``None``.

    Uses the immediately preceding keyframe. Returns ``None`` at the start
    of the sequence or when that edge already exists (for example when
    ``keyframe_decim == 1`` makes it an odometry edge).
    """
    if bi <= 0:
        return None
    a = int(kf[bi - 1])
    b = int(kf[bi])
    if edge_key(a, b) in added_edges:
        return None
    return a
