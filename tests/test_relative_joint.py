import copy

import numpy as np
import pytest

from vo.joint_prior import JointPrior, JointWindowOptimizer
from vo.relative_joint import RelativeJointWindowOptimizer
from vo.se3 import se3_exp


def graph(cls=RelativeJointWindowOptimizer, **kw):
    w = cls(optimize_scale=True, scale_prior_sigma=.8, max_iterations=40, **kw)
    for i in range(3):
        w.add_node(i, se3_exp(np.array([.03*i, -.02*i, .01*i, i*1.2, .1*i, 0.])))
    for i in range(2):
        w.add_named_edge((i, i+1), i, i+1,
                         se3_exp(np.array([-.02, .01, 0., -1., 0., 0.])))
    return w


def numerical_jacobian(w):
    J = np.zeros_like(w._jacobian())
    poses = copy.deepcopy(w.T)
    scales = w._capture_scales()
    for c in range(J.shape[1]):
        residuals = []
        for sign in (1, -1):
            w.T = copy.deepcopy(poses)
            w._restore_scales(copy.deepcopy(scales))
            if c < 6*len(w.pose_ids):
                p = w.pose_ids[c//6]
                dx = np.eye(6)[c % 6] * sign*1e-6
                w.T[p] = poses[p] @ se3_exp(dx)
            else:
                dx = np.zeros(w.n_scale)
                dx[c-6*len(w.pose_ids)] = sign*1e-6
                w._apply_scale_step(dx)
            residuals.append(w._residuals())
        J[:, c] = (residuals[0]-residuals[1])/2e-6
    w.T = poses
    w._restore_scales(scales)
    return J


def test_relative_prior_local_equivalence_and_far_jacobian():
    source = graph(JointWindowOptimizer)
    prior = source.marginalize_joint({("pose", 1)})
    old, new = JointWindowOptimizer(optimize_scale=True), RelativeJointWindowOptimizer(optimize_scale=True)
    for w in (old, new):
        for p in (0, 2):
            w.add_node(p, source.T[p])
        for k in source.scalar_col:
            w.add_scalar(k, source.log_scale(k))
        w.add_joint_prior(prior)
    np.testing.assert_allclose(old._residuals(), new._residuals(), atol=1e-12)
    np.testing.assert_allclose(old._jacobian(), new._jacobian(), atol=1e-9)
    new.T[2] = new.T[2] @ se3_exp(np.array([.3, -.1, .2, .4, -.2, .1]))
    np.testing.assert_allclose(new._jacobian(), numerical_jacobian(new), atol=2e-7)
    before = new._residuals().copy()
    G = se3_exp(np.array([1.3, -.7, .4, 20., -13., 8.]))
    new.T = {p: G @ T for p, T in new.T.items()}
    np.testing.assert_allclose(before, new._residuals(), atol=1e-10)


@pytest.mark.parametrize("mode", ["soft", "ridge", "none"])
def test_solver_equivariance_and_monotone_cost(mode):
    a = graph(regularization=mode)
    b = copy.deepcopy(a)
    G = se3_exp(np.array([.6, -.4, .3, 8., -4., 3.]))
    b.T = {p: G @ T for p, T in b.T.items()}
    ca, cb = a.optimize(), b.optimize()
    np.testing.assert_allclose(ca, cb, atol=1e-9)
    for p in a.pose_ids:
        np.testing.assert_allclose(G @ a.T[p], b.T[p], atol=1e-7)
    assert np.all(np.diff(a.relative_diagnostics["cost_history"]) < 0.)
    assert not a.prior_factors


def test_disconnected_prior_remarginalization_anchor_removed():
    w = graph()
    w.add_node(3, se3_exp(np.array([.1, 0., 0., 3., .1, 0.])))
    w.add_named_edge((2, 3), 2, 3, se3_exp(np.array([0., 0., 0., -1., 0., 0.])))
    for p in (10, 11):
        w.add_node(p, se3_exp(np.array([0., 0., 0., p, 0., 0.])))
    w.add_named_edge((10, 11), 10, 11, se3_exp(np.array([0., 0., 0., -1., 0., 0.])))
    w.optimize()
    prior = w.marginalize_joint({("pose", 0), ("pose", 10)})
    assert prior.pose_components == ((1, 2, 3), (11,))
    reduced = RelativeJointWindowOptimizer(optimize_scale=True)
    for (kind, key), x0 in zip(prior.keys, prior.x0):
        if kind == "pose":
            reduced.add_node(key, x0)
        else:
            reduced.add_scalar(key, x0)
    reduced.add_joint_prior(prior)
    np.testing.assert_allclose(reduced._jacobian(), numerical_jacobian(reduced), atol=2e-7)
    assert not reduced.prior_factors
    reduced.optimize()
    assert reduced.relative_diagnostics["roots"] == [1, 11]
    reduced.T[2] = reduced.T[2] @ se3_exp(np.array([.2, -.1, .1, .2, .1, -.1]))
    again = reduced.marginalize_joint({("pose", 1)})
    final = RelativeJointWindowOptimizer(optimize_scale=True)
    for (kind, key), x0 in zip(again.keys, again.x0):
        if kind == "pose":
            final.add_node(key, x0)
        else:
            final.add_scalar(key, x0)
    final.add_joint_prior(again)
    np.testing.assert_allclose(final._jacobian(), numerical_jacobian(final), atol=2e-7)
    before = final._residuals().copy()
    for p in final.T:
        final.T[p] = se3_exp(np.array([.4, .2, -.1, 4., 2., 1.])) @ final.T[p]
    np.testing.assert_allclose(final._residuals(), before, atol=1e-9)


def test_unsupported_inputs_and_inconsistent_prior():
    for kw in ({"tsvd_ratio": .01}, {"loop_robust": "gnc_gm"}, {"alpha": 0.}):
        with pytest.raises(ValueError):
            RelativeJointWindowOptimizer(**kw)
    for relative in (False, True):
        w = RelativeJointWindowOptimizer()
        w.add_node(0, np.eye(4))
        with pytest.raises(ValueError):
            w.add_joint_prior(JointPrior((("pose", 0),), (np.eye(4),),
                                        np.eye(6), np.zeros(6), frozenset(), relative))
    w = graph()
    with pytest.raises(ValueError):
        w.add_nl_regularization()
    w.fixed_ids.add(0)
    with pytest.raises(ValueError):
        w.optimize()


def test_no_scale_clamp_and_trial_rollback():
    w = graph()
    w._apply_scale_step(np.array([-35., 0.]))
    assert w.edge_scale[0] < 1e-12
    n = 6*len(w.edges)
    assert w._residuals()[n] == pytest.approx(-35./.8)
    original = w._capture_scales()
    w._restore_scales(original)
    with pytest.raises(FloatingPointError):
        w._apply_scale_step(np.array([1000., 0.]))


def test_scalar_linear_solution_and_iteration_independence():
    solutions = []
    for iterations in (10, 40, 100):
        w = RelativeJointWindowOptimizer(alpha=2., tau=3., sigma_l=.7, max_iterations=iterations)
        w.add_scalar("s", 0.)
        w.add_joint_prior(JointPrior((("scale", "s"),), (0.,), np.array([[2.]]),
                                    np.array([-4.]), frozenset()))
        w.optimize()
        lam = 4*.7**2
        rho = 2*3/(lam+3)
        expected = 8/(4+rho/.7**2)
        assert w.log_scale("s") == pytest.approx(expected, abs=1e-8)
        solutions.append(w.log_scale("s"))
    np.testing.assert_allclose(solutions, solutions[0], atol=1e-10)


def test_online_wiring():
    from vo.online_graph import OnlinePoseGraph
    g = OnlinePoseGraph(dict(marginalization_backend="canonical_marginal", nl_reg=False,
                            canonical_regularization="relative_joint_soft"), None, lambda a, b: {})
    assert isinstance(g._new_window(), RelativeJointWindowOptimizer)
    g.p["nl_reg"] = True
    with pytest.raises(ValueError):
        g._new_window()


def test_shipped_soft_bearing_defaults_run_canonical_marginalization():
    from vo.online_graph import DEFAULT_PARAMS, OnlinePoseGraph
    params = dict(DEFAULT_PARAMS)
    g = OnlinePoseGraph(params, None, lambda a, b: {"ok": False})
    window = g._new_window()
    assert isinstance(window, RelativeJointWindowOptimizer)
    assert window.regularization == "soft"
    assert window.alpha == 10. and window.tau == 1.
    assert window.sigmas == (1., 1., 1.)
    assert window.max_iterations == 20
    assert g._canonical.mode == "canonical_marginal"
    # Pass the real default graph through several promotions, transient
    # marginalizations and replacement priors, rather than only checking keys.
    for frame in range(25):
        odometry = None if frame == 0 else (np.eye(3), np.array([-1., 0., 0.]), 1.)
        g.add_frame(frame, odom=odometry)
    assert g._canonical.revision >= 3
    assert g._canonical.priors
    g._canonical.assert_ownership()
    assert all(np.isfinite(g.pose(frame)).all() for frame in range(25))


def test_shipped_defaults_allow_explicit_legacy_nl_reg():
    from vo.online_graph import DEFAULT_PARAMS, OnlinePoseGraph
    from vo.se3_window import SlidingWindowOptimizer
    params = {**DEFAULT_PARAMS, "marginalization_backend": "legacy",
              "canonical_regularization": "legacy", "nl_reg": True,
              "loop_iterations": 10, "loop_observation_model": "se3"}
    g = OnlinePoseGraph(params, None, lambda a, b: {"ok": False})
    assert type(g._new_window()) is SlidingWindowOptimizer
    assert g._canonical is None


def test_unnamed_scale_is_rejected_before_solve_and_marginalization():
    w = graph()
    w.add_edge(0, 2, np.eye(4), scale_free=True)
    for operation in (w.optimize, lambda: w.marginalize_joint(set())):
        with pytest.raises(ValueError, match="named keys"):
            operation()


def test_partial_trial_overflow_restores_all_state(monkeypatch):
    w = graph()
    w.add_scalar("independent", .2)
    initial_T = copy.deepcopy(w.T)
    initial_scales = copy.deepcopy(w._capture_scales())
    real_apply = w._apply_scale_step
    calls = []

    def fail_once(dx):
        calls.append(1)
        if len(calls) == 1:
            w.edge_scale[0] *= 2.
            w.scalar_value["independent"] = 19.
            raise FloatingPointError("injected second-scale overflow")
        if len(calls) == 2:
            assert w._capture_scales() == initial_scales
        real_apply(dx)

    real_restore = w._restore_scales
    restores = []

    def check_restore(state):
        real_restore(state)
        if not restores:
            for p in w.T:
                np.testing.assert_array_equal(w.T[p], initial_T[p])
            assert w._capture_scales() == initial_scales
        restores.append(1)

    monkeypatch.setattr(w, "_apply_scale_step", fail_once)
    monkeypatch.setattr(w, "_restore_scales", check_restore)
    w.optimize()
    assert len(calls) >= 2 and restores
    assert w.relative_diagnostics["rejected_trials"] >= 1
    assert "overflow" in w.relative_diagnostics["last_failure"]
    assert w.scalar_value["independent"] == pytest.approx(.2)


def test_regularization_information_not_marginalized():
    a, b = graph(alpha=.1), graph(alpha=100.)
    # Identical state: regularization can change the linearization state, but
    # cannot contribute information rows of its own.
    for w in (a, b):
        w.max_iterations = 0
        w.optimize()
    pa = a.marginalize_joint({("pose", 1)})
    pb = b.marginalize_joint({("pose", 1)})
    np.testing.assert_allclose(pa.A.T @ pa.A, pb.A.T @ pb.A, atol=1e-12)
    np.testing.assert_allclose(pa.A.T @ pa.r0, pb.A.T @ pb.r0, atol=1e-12)


@pytest.mark.parametrize("weak", [0., 1e-8, .01, .1, 1., 10.])
def test_coupled_translation_scale_linear_response(weak):
    class LinearPair(RelativeJointWindowOptimizer):
        # Exact linear test model embedded in the same pose/scalar layout.
        A = np.array([[1., -1.], [weak, weak]]) / np.sqrt(2.)
        b = np.array([1., .4])

        def _pose_components(self):
            return [{0, 1}], set()

        def _residuals(self):
            x = np.array([self.T[1][0, 3]-self.T[0][0, 3], self.log_scale("s")])
            return self.A @ x - self.b

        def _jacobian(self):
            J = np.zeros((2, 13))
            J[:, 3], J[:, 9], J[:, 12] = -self.A[:, 0], self.A[:, 0], self.A[:, 1]
            return J

        def _huber_weights(self):
            return np.ones(2)

        def _true_cost(self):
            r = self._residuals()
            return .5*(r @ r)

    w = LinearPair(alpha=1.2, tau=.8, sigma_t=.3, sigma_l=.7, max_iterations=40)
    w.add_node(0, np.eye(4)); w.add_node(1, np.eye(4)); w.add_scalar("s")
    w.optimize()
    D = np.diag([.3, .7])
    H = D @ w.A.T @ w.A @ D
    R = 1.2*.8*np.linalg.inv(H+.8*np.eye(2))
    expected = D @ np.linalg.solve(H+R, D @ w.A.T @ w.b)
    np.testing.assert_allclose([w.T[1][0, 3], w.log_scale("s")], expected, atol=1e-8)


def test_distance_unit_change():
    a = graph(sigma_t=.4, sigma_r=.2, sigma_l=.6)
    b = copy.deepcopy(a)
    factor = 100.
    b.sigmas = (.2, .4*factor, .6)
    for p in b.T:
        b.T[p][:3, 3] *= factor
    for k, (i, j, M, st, sr) in enumerate(b.edges):
        M[:3, 3] *= factor
        b.edges[k] = (i, j, M, st*factor, sr)
    np.testing.assert_allclose(a._true_cost(), b._true_cost(), atol=1e-9)
    np.testing.assert_allclose(a.optimize(), b.optimize(), atol=1e-9)
    for p in a.T:
        np.testing.assert_allclose(a.T[p][:3, 3]*factor, b.T[p][:3, 3], atol=1e-6)
        np.testing.assert_allclose(a.T[p][:3, :3], b.T[p][:3, :3], atol=1e-8)


@pytest.mark.parametrize("curvature_scale", [1., 100.])
def test_roundoff_convergence_and_large_gradient_failure(monkeypatch, curvature_scale):
    w = RelativeJointWindowOptimizer()
    w.add_scalar("s")
    w.add_joint_prior(JointPrior((("scale", "s"),), (0.,),
                                np.array([[curvature_scale], [0.]]), np.array([-2e-8, 10.]), frozenset()))
    w.optimize()
    assert w.relative_diagnostics["status"] == "roundoff"
    assert w.last_steps == 0
    bad = RelativeJointWindowOptimizer(regularization="none")
    bad.add_scalar("s")
    bad.add_joint_prior(JointPrior((("scale", "s"),), (0.,), np.ones((1, 1)),
                                  np.array([-1.]), frozenset()))
    monkeypatch.setattr(bad, "_true_cost", lambda: 1.)
    bad.optimize()
    assert bad.relative_diagnostics["status"] == "no_progress"
    assert bad.log_scale("s") == 0.


@pytest.mark.parametrize("curvature,correction,lambda_init,converged", [
    (1., 2e-7, 1e-3, True),
    (100., 2e-9, 1e-3, True),
    (1e-3, .02, 1., False),
    (.1, 5e-5, 1e8, False),
])
def test_roundoff_requires_small_undamped_correction(
        monkeypatch, curvature, correction, lambda_init, converged):
    # An irreducible residual sets the cost resolution. The second row has an
    # exact known minimizer, including weak curvature and huge LM damping cases.
    w = RelativeJointWindowOptimizer(regularization="none",
                                    lambda_init=lambda_init)
    w.add_scalar("s")
    w.add_joint_prior(JointPrior((("scale", "s"),), (0.,),
                                np.array([[0.], [curvature]]),
                                np.array([10., -curvature*correction]), frozenset()))
    # Model a cost evaluation unable to resolve these tiny differences.
    monkeypatch.setattr(w, "_true_cost", lambda: 50.)
    if converged:
        w.optimize()
        assert w.relative_diagnostics["status"] == "roundoff"
        assert w.last_steps == 0
    else:
        w.optimize()
        assert w.relative_diagnostics["status"] == "no_progress"
    assert w.log_scale("s") == 0.


@pytest.mark.parametrize("strong", [0., 1e8])
def test_roundoff_does_not_certify_rank_truncated_model(monkeypatch, strong):
    w = RelativeJointWindowOptimizer(regularization="none", lambda_init=1.)
    w.add_scalar("strong")
    w.add_scalar("weak")
    weak, residual = (1., -2e-7) if strong == 0. else (1e-8, -2.)
    w.add_joint_prior(JointPrior((("scale", "strong"), ("scale", "weak")),
                                (0., 0.), np.array([[strong, 0.], [0., weak], [0., 0.]]),
                                np.array([0., residual, 10.]), frozenset()))
    # Exact nullspace: decline the certificate conservatively. Mixed singular
    # values: never discard a large improvable weak-direction residual as noise.
    monkeypatch.setattr(w, "_true_cost", lambda: 50. + .5*residual**2)
    w.optimize()
    assert w.relative_diagnostics["status"] == "no_progress"
    assert w.log_scale("strong") == w.log_scale("weak") == 0.


def test_stall_returns_last_accepted(monkeypatch):
    # An exhausted LM schedule returns a finite best-effort state with status
    # "no_progress" instead of raising (matching the legacy sliding-window
    # solver's no-progress semantics).
    w = RelativeJointWindowOptimizer(regularization="none")
    w.add_scalar("s")
    w.add_joint_prior(JointPrior((("scale", "s"),), (0.,), np.ones((1, 1)),
                                np.array([-1.]), frozenset()))
    monkeypatch.setattr(w, "_true_cost", lambda: 1.)
    w.optimize()
    assert w.relative_diagnostics["status"] == "no_progress"
    assert w.last_steps == 0
    assert w.relative_diagnostics["final_lambda"] > w.lambda_init
    assert w.log_scale("s") == 0.
