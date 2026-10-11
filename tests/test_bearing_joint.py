import numpy as np
import pytest
from vo.bearing_joint import BearingJointWindowOptimizer
from vo.relative_joint import RelativeJointWindowOptimizer
from vo.se3 import se3_exp


def build(cls):
    obj=cls(regularization='ridge',alpha=10,max_iterations=40,optimize_scale=True)
    obj.add_node(0,np.eye(4));b=np.eye(4);b[0,3]=1.2;obj.add_node(2,b)
    m=np.eye(4);m[0,3]=-1
    obj.add_named_edge(('odom',0,2),0,2,m)
    return obj


def test_nonloop_model_is_identical():
    a,b=build(RelativeJointWindowOptimizer),build(BearingJointWindowOptimizer)
    np.testing.assert_array_equal(a._residuals(),b._residuals())
    np.testing.assert_array_equal(a._jacobian(),b._jacobian())
    assert a._true_cost()==b._true_cost()
    a.optimize();b.optimize()
    np.testing.assert_array_equal(a.T[2],b.T[2])
    p=b.marginalize_joint({('scale',('odom',0,2))},{('odom',0,2)})
    assert p.sources==frozenset({('odom',0,2)})


def test_loop_has_no_scale_and_full_jacobian_matches():
    obj=build(BearingJointWindowOptimizer);m=np.eye(4);m[0,3]=-1
    previous=obj.n_scale
    obj.add_named_edge(('loop',0,2),0,2,m)
    assert obj.n_scale==previous and ('loop',0,2) not in obj.scalar_col
    obj.T[2]=obj.T[2]@se3_exp(np.array([.03,.02,-.01,0,.1,0]))
    j=obj._jacobian();r=obj._residuals();assert j.shape[0]==len(r)==len(obj._huber_weights())
    numeric=[];cost_gradient=[]
    import copy
    for column in range(j.shape[1]):
        values=[];costs=[]
        for sign in (1,-1):
            a=copy.deepcopy(obj);dx=np.zeros(j.shape[1]);dx[column]=sign*1e-6
            for i,k in enumerate(a.pose_ids):a.T[k]=a.T[k]@se3_exp(dx[6*i:6*i+6])
            a._apply_scale_step(dx[6*len(a.pose_ids):])
            values.append(a._residuals());costs.append(a._true_cost())
        numeric.append((values[0]-values[1])/2e-6);cost_gradient.append((costs[0]-costs[1])/2e-6)
    np.testing.assert_allclose(j,np.array(numeric).T,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(j.T@(obj._huber_weights()*r),cost_gradient,rtol=1e-6,atol=1e-7)
    with pytest.raises(ValueError):obj.marginalize_joint(set())


def test_bearing_baseline_floor_propagates_to_observation():
    obj=BearingJointWindowOptimizer(regularization='ridge',alpha=10,
                                    optimize_scale=True,bearing_baseline_floor=.05)
    obj.add_node(0,np.eye(4));b=np.eye(4);b[0,3]=1.2;obj.add_node(2,b)
    m=np.eye(4);m[0,3]=-1;obj.add_named_edge(('loop',0,2),0,2,m)
    assert obj.bearing_loops[('loop',0,2)][2].baseline_floor==.05


def test_harvest_reports_nan_direction_below_baseline_floor():
    obj=BearingJointWindowOptimizer(regularization='ridge',alpha=10,optimize_scale=True)
    obj.add_node(0,np.eye(4));b=np.eye(4);b[0,3]=1e-6;obj.add_node(2,b)
    m=np.eye(4);m[0,3]=-1;obj.add_named_edge(('loop',0,2),0,2,m)
    samples=[s for s in obj.harvest_robust_residuals() if s[0]==0 and s[1]==2]
    assert len(samples)==1
    assert np.isfinite(samples[0][2]) and np.isnan(samples[0][3])


def test_loop_connects_components_for_gauge():
    obj=BearingJointWindowOptimizer(regularization='ridge')
    obj.add_node(0,np.eye(4));b=np.eye(4);b[0,3]=1;obj.add_node(4,b)
    m=np.eye(4);m[0,3]=-1;obj.add_named_edge(('loop',0,4),0,4,m)
    assert obj._pose_components()==([{0,4}],set())
    with pytest.raises(ValueError):obj.add_named_edge(('loop',0,4),0,4,m)


def radial_optimizer():
    return BearingJointWindowOptimizer(regularization='ridge',alpha=10,max_iterations=40,
                                       optimize_scale=True,bearing_radial=True,
                                       bearing_sigma_radial=.3,bearing_scale_prior_sigma=.6)


def test_loop_radial_registers_scale_and_matches_numeric_jacobian():
    obj=radial_optimizer();obj.add_node(0,np.eye(4));b=np.eye(4);b[0,3]=1.2;obj.add_node(2,b)
    m=np.eye(4);m[0,3]=-1;obj.add_named_edge(('odom',0,2),0,2,m)
    previous=obj.n_scale
    obj.add_named_edge(('loop',0,2),0,2,m)
    key=('loop',0,2)
    assert obj.n_scale==previous+1 and key in obj.scalar_col
    assert key in obj.scale_regularizer_exclude
    obj.T[2]=obj.T[2]@se3_exp(np.array([.03,.02,-.01,0,.1,0]))
    obj.scalar_value[key]=np.log(.9)
    j=obj._jacobian();r=obj._residuals();assert j.shape[0]==len(r)==len(obj._huber_weights())
    import copy
    numeric=[];cost_gradient=[]
    for column in range(j.shape[1]):
        values=[];costs=[]
        for sign in (1,-1):
            a=copy.deepcopy(obj);dx=np.zeros(j.shape[1]);dx[column]=sign*1e-6
            for i,k in enumerate(a.pose_ids):a.T[k]=a.T[k]@se3_exp(dx[6*i:6*i+6])
            a._apply_scale_step(dx[6*len(a.pose_ids):])
            values.append(a._residuals());costs.append(a._true_cost())
        numeric.append((values[0]-values[1])/2e-6);cost_gradient.append((costs[0]-costs[1])/2e-6)
    np.testing.assert_allclose(j,np.array(numeric).T,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(j.T@(obj._huber_weights()*r),cost_gradient,rtol=1e-6,atol=1e-7)


def test_excluded_loop_scale_is_not_regularized():
    def make(exclude):
        obj=radial_optimizer();obj.add_node(0,np.eye(4));b=np.eye(4);b[0,3]=1.2;obj.add_node(2,b)
        m=np.eye(4);m[0,3]=-1;obj.add_named_edge(('odom',0,2),0,2,m)
        obj.add_named_edge(('loop',0,2),0,2,m)
        if not exclude:obj.scale_regularizer_exclude=set()
        obj.scalar_value[('loop',0,2)]=np.log(.2)
        return obj
    free,held=make(True),make(False)
    free.optimize();held.optimize()
    start=abs(np.log(.2))
    moved_free=abs(free.log_scale(('loop',0,2))-np.log(.2))
    moved_held=abs(held.log_scale(('loop',0,2))-np.log(.2))
    assert moved_free>moved_held
