"""Information conservation, mixed coordinates and canonical ownership."""
import numpy as np
import pytest

from vo.joint_prior import JointPrior, JointWindowOptimizer, project_rows
from vo.online_graph import OnlinePoseGraph
from vo.se3 import se3_exp


def window():
    return JointWindowOptimizer(optimize_scale=True, scale_prior_sigma=.6,
                                step_scale_t=.1, step_scale_r=.1, huber=1e6,
                                max_iterations=20)


def test_square_root_elimination_matches_full_least_squares():
    rng = np.random.default_rng(7)
    J = rng.normal(size=(45, 17))
    J[:, 3] = J[:, 1]  # a genuine eliminated null direction
    r = rng.normal(size=45)
    A, b = project_rows(J, r, list(range(7)), list(range(7, 17)))
    full = np.linalg.lstsq(J, -r, rcond=1e-10)[0]
    reduced = np.linalg.lstsq(A, -b, rcond=1e-10)[0]
    np.testing.assert_allclose(reduced, full[7:], atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(A @ reduced + b),
                               np.linalg.norm(J @ full + r), atol=1e-10)
    expected = J[:, 7:].T @ (np.eye(45) - J[:, :7] @
                              np.linalg.pinv(J[:, :7])) @ J[:, 7:]
    np.testing.assert_allclose(A.T @ A, expected, atol=1e-10)


def test_keep_scales_then_eliminate_equals_immediate_elimination():
    w = window()
    for i in range(4):
        w.add_node(i, se3_exp(np.array([0, 0, .02*i, .7*i, .03*i, 0])))
    for i, j in [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]:
        M = np.linalg.inv(w.T[j]) @ w.T[i]
        M = M @ se3_exp(np.array([.01, -.02, .03, .04, -.02, .01]))
        w.add_named_edge(("edge", i, j), i, j, M)
    poses = {("pose", 1), ("pose", 2)}
    scales = {("scale", key) for key in w.scalar_col}
    direct = w.marginalize_joint(poses | scales)
    retained = w.marginalize_joint(poses)
    reduced = window()
    for k in (0, 3):
        reduced.add_node(k, w.T[k])
    for key in w.scalar_col:
        reduced.add_scalar(key, w.log_scale(key))
    reduced.add_joint_prior(retained)
    two_step = reduced.marginalize_joint(scales)
    np.testing.assert_allclose(two_step.A.T @ two_step.A,
                               direct.A.T @ direct.A, atol=1e-8)
    np.testing.assert_allclose(two_step.A.T @ two_step.r0,
                               direct.A.T @ direct.r0, atol=1e-8)


def test_mixed_prior_jacobian_and_prior_only_solve():
    rng = np.random.default_rng(9)
    w = window()
    w.add_node(0, se3_exp(np.array([.12, -.08, .1, .3, -.2, .1])))
    w.add_scalar(("lambda", 0), .5)
    A = rng.normal(size=(10, 7))
    prior = JointPrior((("pose", 0), ("scale", ("lambda", 0))),
                       (np.eye(4), 0.), A, np.zeros(10), frozenset({"p"}))
    w.add_joint_prior(prior)
    J = w._jacobian()
    numeric = np.zeros_like(J)
    T = w.T[0].copy()
    for k in range(7):
        delta = np.zeros(6)
        if k < 6:
            delta[k] = 1e-6
            w.T[0] = T @ se3_exp(delta)
        else:
            w.scalar_value[("lambda", 0)] = .5 + 1e-6
        plus = w._residuals()
        if k < 6:
            w.T[0] = T @ se3_exp(-delta)
        else:
            w.scalar_value[("lambda", 0)] = .5 - 1e-6
        minus = w._residuals()
        numeric[:, k] = (plus - minus) / 2e-6
        w.T[0] = T.copy()
        w.scalar_value[("lambda", 0)] = .5
    np.testing.assert_allclose(J, numeric, atol=1e-7)
    assert w.optimize() < 1e-10
    np.testing.assert_allclose(w.T[0], np.eye(4), atol=1e-6)
    assert abs(w.log_scale(("lambda", 0))) < 1e-6


def test_unrelated_factor_is_not_baked_into_segment_summary():
    owned = window()
    for k in range(3):
        T = np.eye(4)
        T[0, 3] = k
        owned.add_node(k, T)
    for i, j in [(0, 1), (1, 2)]:
        owned.add_named_edge(("e", i, j), i, j,
                             np.linalg.inv(owned.T[j]) @ owned.T[i])
    f = owned.marginalize_joint({("pose", 1)} |
                                {("scale", k) for k in owned.scalar_col})
    reduced = window()
    reduced.add_node(0, owned.T[0])
    reduced.add_node(2, owned.T[2])
    reduced.add_joint_prior(f)
    J0 = reduced._jacobian()
    reduced.add_named_edge(("loop", 0, 2), 0, 2,
                           np.linalg.inv(owned.T[2]) @ owned.T[0])
    # The old prior stays unchanged; the loop contributes independently once.
    np.testing.assert_allclose(reduced._jacobian()[7:, :12], J0)


def test_named_and_standalone_scales_dense_sparse_and_rejected_step(monkeypatch):
    w = window()
    w.add_node(0, np.eye(4))
    w.add_node(1, se3_exp(np.array([.1, -.1, .2, 1.2, .1, 0])))
    M = np.eye(4)
    M[0, 3] = -1.
    edge, lam = ("odom", 0, 1), ("lambda", 0)
    w.add_named_edge(edge, 0, 1, M, scale=1.1)
    w.add_scalar(lam, .2)
    w.add_joint_prior(JointPrior((("scale", edge), ("scale", lam)),
                                 (0., 0.), np.array([[2., -2.]]),
                                 np.array([.1]), frozenset({"scale_link"})))
    J = w._jacobian()
    for key in (edge, lam):
        state = w._capture_scales()
        du = np.zeros(w.n_scale)
        du[w.scalar_col[key]] = 1e-6
        w._apply_scale_step(du)
        plus = w._residuals()
        w._restore_scales(state)
        # _restore_scales transfers mutable state; capture afresh before minus.
        state = w._capture_scales()
        w._apply_scale_step(-du)
        minus = w._residuals()
        w._restore_scales(state)
        np.testing.assert_allclose(J[:, 12+w.scalar_col[key]],
                                   (plus-minus)/2e-6, atol=1e-7)
    r, weights = w._residuals(), w._huber_weights()
    H, g = w._sparse_normal(r, weights)
    np.testing.assert_allclose(H.toarray(), J.T @ (weights[:, None]*J), atol=1e-10)
    np.testing.assert_allclose(g, J.T @ (weights*r), atol=1e-10)
    scales = {key: w.log_scale(key) for key in (edge, lam)}
    poses = {k: T.copy() for k, T in w.T.items()}
    calls = [0]
    original = w._weighted_cost
    def reject_trial(weights):
        calls[0] += 1
        return original(weights) if calls[0] == 1 else float("inf")
    monkeypatch.setattr(w, "_weighted_cost", reject_trial)
    w.optimize()
    assert calls[0] > 1
    assert w.last_steps == 0
    for key, value in scales.items():
        assert w.log_scale(key) == value
    for key, value in poses.items():
        np.testing.assert_array_equal(w.T[key], value)


@pytest.mark.parametrize("mode", ["canonical_raw", "canonical_conditional",
                                  "canonical_marginal", "canonical_retained",
                                  "canonical_shared"])
def test_canonical_history_keeps_all_loops_and_separator_kfs(mode):
    rng = np.random.default_rng(31)
    gt = [np.eye(4)]
    for _ in range(12):
        gt.append(gt[-1] @ se3_exp(rng.normal(size=6) * .06))

    def match(a, b):
        M = np.linalg.inv(gt[b]) @ gt[a]
        return dict(ok=True, R=M[:3, :3], t=M[:3, 3], inlier_ratio=1., n_matches=100)

    p = dict(marginalization_backend=mode, kf_max_gap=4, kf_local_map_k=2,
             kf_trans_thresh=100, kf_rot_thresh=180, max_keyframes=1,
             held_cap=0, loop_min_gap=4, loop_temporal_k=1,
             scale_prior_sigma=.6, loop_iterations=3, nl_reg=False)
    g = OnlinePoseGraph(p, None, match)
    for i in range(len(gt)):
        g.add_frame(i)
    state = g._canonical
    state.assert_ownership()
    assert g.n_dropped_held == 0
    assert not g._eliminated
    assert set(g.keyframes) <= set(state.poses)
    assert g.n_loop > 0
    assert len([k for k in state.raw if k[0] == "loop"]) == g.n_loop
    assert all(np.isfinite(g.pose(i)).all() for i in range(len(gt)))
    # A spoke to an earlier KF survives even when max_keyframes=1.
    assert ("spoke", 0, 5) in state.sources
    if mode == "canonical_shared":
        assert len([k for k in state.scales if k[0] == "segment"]) == 3
        assert ("scale_transition", 4, 8, 5) in state.sources


@pytest.mark.parametrize("mode", ["canonical_marginal", "canonical_conditional"])
def test_fully_eliminated_disconnected_component_is_a_constant(mode):
    g = OnlinePoseGraph(dict(marginalization_backend=mode, kf_max_gap=4,
                             loop_enable=False, nl_reg=False),
                        None, lambda a, b: {"ok": False})
    g.add_frame(0)
    for i in range(1, 9):
        odom = (np.eye(3), np.array([-1., 0, 0])) if i == 2 else (None, None)
        g.add_frame(i, odom=odom)
    assert g._canonical.priors[0].keys == ()
    g._canonical.assert_ownership()
    assert np.isfinite(g.pose(8)).all()


def test_canonical_cap_bounds_active_keyframes_and_absorbs():
    n = 40
    g = OnlinePoseGraph(dict(
        marginalization_backend="canonical_marginal", kf_max_gap=1,
        loop_enable=False, nl_reg=False, canonical_cap=True,
        canonical_max_keyframes=2, loop_window=2, scale_prior_sigma=.6),
        None, lambda a, b: {"ok": False})
    g.add_frame(0)
    for i in range(1, n):
        g.add_frame(i, odom=(np.eye(3), np.array([-1., 0, 0])))
    state = g._canonical
    state.assert_ownership()
    assert state.n_absorbed > 0
    assert len(state.active) <= 2
    assert state.n_over_cap == 0
    assert g.last_n_nodes <= 3
    assert all(np.isfinite(g.pose(i)).all() for i in range(n))


def test_canonical_cap_off_matches_unbounded_history():
    def build(cap):
        g = OnlinePoseGraph(dict(
            marginalization_backend="canonical_marginal", kf_max_gap=1,
            loop_enable=False, nl_reg=False, canonical_cap=cap,
            canonical_max_keyframes=2, loop_window=2, scale_prior_sigma=.6),
            None, lambda a, b: {"ok": False})
        g.add_frame(0)
        for i in range(1, 25):
            g.add_frame(i, odom=(np.eye(3), np.array([-1., 0, 0])))
        return np.array([g.pose(i)[:3, 3] for i in range(25)])

    off = build(False)
    on = build(True)
    # Absorbing a chain of isotropic priors is exact for this consistent
    # (loop-free) case, so bounded and unbounded positions must agree.
    np.testing.assert_allclose(on, off, atol=1e-6)


def test_canonical_cap_does_not_bake_transients_into_est():
    n = 30
    g = OnlinePoseGraph(dict(
        marginalization_backend="canonical_marginal", kf_max_gap=4,
        loop_enable=False, nl_reg=False, canonical_cap=True,
        canonical_max_keyframes=1, loop_window=1, scale_prior_sigma=.6),
        None, lambda a, b: {"ok": False})
    g.add_frame(0)
    for i in range(1, n):
        g.add_frame(i, odom=(np.eye(3), np.array([-1., 0, 0])))
    trans = [i for i in range(n) if i not in g._kf]
    assert trans  # transients exist under kf_max_gap=4
    assert all(t not in g._est for t in trans)
    assert all(np.isfinite(g.pose(i)).all() for i in range(n))


def test_canonical_cap_absorbs_se3_loop_endpoint():
    def match(a, b):
        if a == 0 and b == 12:
            return dict(ok=True, R=np.eye(3), t=np.array([-2., 0, 0]),
                        inlier_ratio=1., n_matches=100)
        return {"ok": False}

    g = OnlinePoseGraph(dict(
        marginalization_backend="canonical_marginal", kf_max_gap=4,
        loop_min_gap=8, loop_temporal_k=1, scale_prior_sigma=.2, nl_reg=False,
        canonical_cap=True, canonical_max_keyframes=1, loop_window=4),
        None, match)
    g.add_frame(0)
    for i in range(1, 40):
        g.add_frame(i, odom=(np.eye(3), np.array([-1., 0, 0])))
    state = g._canonical
    state.assert_ownership()
    assert 0 in state.absorbed
    assert ("loop", 0, 12) not in state.raw
    assert len(state.active) <= 4
    assert all(np.isfinite(g.pose(i)).all() for i in range(40))


def test_canonical_cap_absorb_bridge_multineighbor():
    # KF 0 gets three active neighbours (segment prior -> 4, loops -> 8, 12);
    # absorbing it exercises the multi-neighbour nullspace gate on every later
    # solve that consumes the resulting bridge prior.
    def match(a, b):
        if a == 0 and b in (8, 12):
            return dict(ok=True, R=np.eye(3), t=np.array([-2., 0, 0]),
                        inlier_ratio=1., n_matches=100)
        return {"ok": False}

    g = OnlinePoseGraph(dict(
        marginalization_backend="canonical_marginal", kf_max_gap=4,
        loop_min_gap=8, loop_temporal_k=1, scale_prior_sigma=.2, nl_reg=False,
        canonical_cap=True, canonical_max_keyframes=1, loop_window=4),
        None, match)
    g.add_frame(0)
    for i in range(1, 40):
        g.add_frame(i, odom=(np.eye(3), np.array([-1., 0, 0])))
    state = g._canonical
    state.assert_ownership()
    assert 0 in state.absorbed
    assert len(state.active) <= 4
    assert all(np.isfinite(g.pose(i)).all() for i in range(40))


def test_canonical_cap_measurement_free_keyframes_do_not_fail_stop():
    # A run of failed odometry leaves keyframes without incident factors; they
    # must not be absorbed (or crash the graph) -- they are counted as over-cap.
    g = OnlinePoseGraph(dict(
        marginalization_backend="canonical_marginal", kf_max_gap=1,
        loop_enable=False, nl_reg=False, canonical_cap=True,
        canonical_max_keyframes=1, loop_window=1, scale_prior_sigma=.6),
        None, lambda a, b: {"ok": False})
    g.add_frame(0)
    for i in range(1, 20):
        odom = ((None, None) if 5 <= i <= 8
                else (np.eye(3), np.array([-1., 0, 0])))
        g.add_frame(i, odom=odom)
    state = g._canonical
    state.assert_ownership()
    assert state.n_over_cap > 0
    assert all(np.isfinite(g.pose(i)).all() for i in range(20))


def test_canonical_cap_rejects_shared_scale_mode():
    g = OnlinePoseGraph(dict(
        marginalization_backend="canonical_shared", kf_max_gap=1,
        loop_enable=False, nl_reg=False, canonical_cap=True,
        canonical_max_keyframes=1, loop_window=1), None, lambda a, b: {"ok": False})
    g.add_frame(0)
    with pytest.raises(ValueError, match="canonical_shared"):
        g.add_frame(1, odom=(np.eye(3), np.array([-1., 0, 0])))


def test_shared_scale_splits_at_tracking_failure_inside_and_between_segments():
    g = OnlinePoseGraph(dict(marginalization_backend="canonical_shared",
                             kf_max_gap=4, loop_enable=False, nl_reg=False),
                        None, lambda a, b: {"ok": False})
    g.add_frame(0)
    for i in range(1, 13):
        odom = (None, None) if i in (2, 8) else (np.eye(3), np.array([-1., 0, 0]))
        g.add_frame(i, odom=odom)
    sources = g._canonical.sources
    assert ("scale_anchor", 0, 4, 1) in sources
    assert ("scale_anchor", 0, 4, 3) in sources
    assert ("scale_transition", 4, 8, 5) in sources
    assert ("scale_anchor", 8, 12, 9) in sources
    assert ("scale_transition", 8, 12, 9) not in sources


def test_failed_summary_cannot_publish_or_retry(monkeypatch):
    g = OnlinePoseGraph(dict(marginalization_backend="canonical_marginal",
                             kf_max_gap=1, loop_enable=False, nl_reg=False),
                        None, lambda a, b: {"ok": False})
    g.add_frame(0)
    def fail(*args, **kwargs):
        raise ValueError("simulated failure")
    monkeypatch.setattr(JointWindowOptimizer, "marginalize_joint", fail)
    with pytest.raises(ValueError, match="simulated"):
        g.add_frame(1, odom=(np.eye(3), np.array([-1., 0, 0])))
    with pytest.raises(RuntimeError, match="recreate"):
        g.add_frame(2)
    with pytest.raises(RuntimeError, match="snapshot"):
        g.pose(0)


def test_zero_transient_segment_is_not_lost():
    g = OnlinePoseGraph(dict(marginalization_backend="canonical_marginal",
                             kf_max_gap=1, loop_enable=False, nl_reg=False),
                        None, lambda a, b: {"ok": False})
    g.add_frame(0)
    g.add_frame(1, odom=(np.eye(3), np.array([-1., 0, 0])))
    g.add_frame(2, odom=(np.eye(3), np.array([-1., 0, 0])))
    assert len(g._canonical.priors) == 2
    assert g._canonical.sources == {("odom", 0, 1), ("odom", 1, 2)}


def test_delayed_loop_survives_next_canonical_solve():
    # Closed historical KFs remain variables and their loop remains a factor
    # on the next segment, instead of returning only a corrected initial value.
    def match(a, b):
        if a == 0 and b >= 8 and b % 4 == 0:
            return dict(ok=True, R=np.eye(3), t=np.array([-2., 0, 0]),
                        inlier_ratio=1., n_matches=100)
        return {"ok": False}
    g = OnlinePoseGraph(dict(marginalization_backend="canonical_marginal",
                             kf_max_gap=4, loop_min_gap=8, loop_temporal_k=1,
                             scale_prior_sigma=.2, nl_reg=False), None, match)
    g.add_frame(0)
    for i in range(1, 13):
        g.add_frame(i, odom=(np.eye(3), np.array([-1., 0, 0])))
    assert ("loop", 0, 8) in g._canonical.raw
    act = g._canonical._window(list(g._canonical.raw.values()),
                                g._canonical.priors, set(g.keyframes))
    assert ("loop", 0, 8) in act.scalar_edge
    assert not act.fixed_ids
    g._canonical.assert_ownership()
