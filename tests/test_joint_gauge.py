import numpy as np
import pytest

from vo.joint_prior import JointPrior, JointWindowOptimizer
from vo.se3 import se3_exp


def window(**kwargs):
    return JointWindowOptimizer(pose_gauge_policy="physical", **kwargs)


def test_disconnected_relative_components_and_nl_regularization():
    w = window()
    for k in range(4):
        w.add_node(k, se3_exp(np.array([.1*k, 0, 0, k, .1, 0])))
    for a,b in ((0,1),(2,3)):
        w.add_edge(a,b,np.linalg.inv(w.T[b])@w.T[a])
    w.add_nl_regularization()
    anchors = w._gauge_anchor_columns(list(w.pose_ids),4)
    assert set(range(6)) | set(range(12,18)) <= anchors
    assert not set(range(6,12)) & anchors
    assert not set(range(18,24)) & anchors
    w.pop_last_prior_factor()
    assert not w._regularization_priors


def test_absolute_and_partial_prior_do_not_freeze_observed_pose():
    for axes in (list(range(6)), [0,1,2], [3,4,5]):
        w = window(max_iterations=40)
        w.add_node(0, se3_exp(np.array([.1,-.2,.1,.5,-.4,.2])))
        A = np.eye(6)[axes]
        w.add_joint_prior(JointPrior((("pose",0),),(np.eye(4),),A,
                                    np.zeros(len(axes)),frozenset()))
        # The solver's finite iteration budget need not reach machine zero;
        # a frozen pose would retain a cost of order 1e-2 to 1e-1 here.
        initial = w._cost()
        assert w.optimize() < min(1e-8, initial * 1e-6)


def test_fixed_component_does_not_anchor_another_whole_pose():
    w = window()
    w.add_node(0,np.eye(4),fixed=True)
    w.add_node(1,se3_exp(np.array([0,0,0,2.,0,0])))
    M = se3_exp(np.array([0,0,0,-1.,0,0]))
    w.add_edge(0,1,M)
    assert not w._gauge_anchor_columns([1],1)
    w.optimize()
    np.testing.assert_allclose(w.T[1][:3,3],[1,0,0],atol=1e-6)


def test_relative_summary_preserves_physical_anchor_far_from_origin():
    w = window()
    for k in range(3):
        w.add_node(k,se3_exp(np.array([0,0,0,k,0,0])))
    for a,b in ((0,1),(1,2)):
        w.add_edge(a,b,np.linalg.inv(w.T[b])@w.T[a])
    prior = w.marginalize_joint({("pose",1)})
    assert prior.relative_pose
    reduced = window()
    reduced.add_node(0,se3_exp(np.array([0,0,2.5,0,1,0])))
    reduced.add_node(2,se3_exp(np.array([.3,0,-1,2,0,0])))
    reduced.add_joint_prior(prior)
    assert set(range(6)) <= reduced._gauge_anchor_columns([0,2],2)
    before = reduced.T[0].copy()
    reduced.optimize()
    np.testing.assert_array_equal(reduced.T[0],before)


def test_absolute_provenance_survives_marginalization():
    w = window()
    w.add_node(0,np.eye(4)); w.add_node(1,np.eye(4))
    w.add_edge(0,1,np.eye(4))
    w.add_joint_prior(JointPrior((("pose",0),),(np.eye(4),),np.eye(6),np.zeros(6),frozenset()))
    assert not w.marginalize_joint({("pose",0)}).relative_pose


@pytest.mark.parametrize("regularize", [False, True])
def test_disconnected_summary_keeps_component_anchors(regularize):
    w = window()
    for k in range(6):
        w.add_node(k,se3_exp(np.array([0,0,0,k,0,0])))
    for a,b in ((0,1),(1,2),(3,4),(4,5)):
        w.add_edge(a,b,np.linalg.inv(w.T[b])@w.T[a])
    prior=w.marginalize_joint({("pose",1),("pose",4)})
    assert prior.pose_components == ((0,2),(3,5))
    reduced=window()
    for k in (0,2,3,5):
        reduced.add_node(k,se3_exp(np.array([.1*k,.2,-.2*k,k,1,2])))
    reduced.add_joint_prior(prior)
    if regularize:
        reduced.add_nl_regularization()
    anchors=reduced._gauge_anchor_columns([0,2,3,5],4)
    assert set(range(6)) | set(range(12,18)) <= anchors


@pytest.mark.parametrize("sigma", [0., .6])
@pytest.mark.parametrize("regularize", [False, True])
def test_scale_nulls_after_component_pose_gauges(sigma, regularize):
    w=window(optimize_scale=True,scale_prior_sigma=sigma,max_iterations=40)
    for k in range(4):
        w.add_node(k,se3_exp(np.array([0,0,0,k,0,0])))
    for a,b in ((0,1),(2,3)):
        w.add_named_edge((a,b),a,b,np.linalg.inv(w.T[b])@w.T[a],scale=1.)
    if regularize:
        w.add_nl_regularization()
    anchors=w._gauge_anchor_columns([0,1,2,3],4)
    assert set(range(6)) | set(range(12,18)) <= anchors
    if sigma > 0:
        assert not {24,25} & anchors
        w.edge_scale=[1.2,1.2]
        before=w._cost()
        assert w.optimize() < before
        assert max(w.edge_scale) < 1.2
    elif not regularize:
        assert len(anchors) >= 14  # two additional unobservable scale modes


def test_policy_default_validation_and_graph_wiring():
    from vo.online_graph import OnlinePoseGraph
    assert JointWindowOptimizer().pose_gauge_policy == "rank"
    with pytest.raises(ValueError, match="pose_gauge_policy"):
        JointWindowOptimizer(pose_gauge_policy="typo")
    for policy in ("rank", "physical"):
        g = OnlinePoseGraph(dict(marginalization_backend="canonical_marginal",
                                 canonical_pose_gauge=policy), None, lambda a, b: {"ok": False})
        assert g._new_window().pose_gauge_policy == policy
