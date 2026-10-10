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


def test_loop_connects_components_for_gauge():
    obj=BearingJointWindowOptimizer(regularization='ridge')
    obj.add_node(0,np.eye(4));b=np.eye(4);b[0,3]=1;obj.add_node(4,b)
    m=np.eye(4);m[0,3]=-1;obj.add_named_edge(('loop',0,4),0,4,m)
    assert obj._pose_components()==([{0,4}],set())
    with pytest.raises(ValueError):obj.add_named_edge(('loop',0,4),0,4,m)
