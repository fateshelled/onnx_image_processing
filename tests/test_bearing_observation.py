import numpy as np
import pytest
from vo.bearing_observation import BearingObservation
from vo.se3 import se3_exp


def test_jacobian_gradient_and_gauges():
    a=se3_exp(np.array([.1,.2,-.1,.2,-.3,.1]))
    b=se3_exp(np.array([-.2,.05,.1,1.,.3,-.2]))
    obs=BearingObservation(np.eye(3),[.1,.2,1.],q_direction=.3)
    r,ja,jb=obs.linearize(a,b);j=np.hstack((ja,jb));num=np.empty_like(j)
    gradient=j.T@(obs.cost_weights(r)[1]*r);ng=[]
    for c in range(12):
        values=[];costs=[]
        for sign in (1,-1):
            x,y=a.copy(),b.copy();dx=np.zeros(6);dx[c%6]=sign*1e-6
            if c<6:x=x@se3_exp(dx)
            else:y=y@se3_exp(dx)
            rr,_,_=obs.linearize(x,y);values.append(rr);costs.append(obs.cost_weights(rr)[0])
        num[:,c]=(values[0]-values[1])/2e-6;ng.append((costs[0]-costs[1])/2e-6)
    np.testing.assert_allclose(j,num,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(gradient,ng,rtol=1e-6,atol=1e-7)
    common=se3_exp(np.array([.4,-.1,.3,3.,2.,1.]))
    np.testing.assert_allclose(obs.linearize(common@a,common@b)[0],r,atol=1e-12)
    x,y=a.copy(),b.copy();x[:3,3]*=7;y[:3,3]*=7
    np.testing.assert_allclose(obs.linearize(x,y)[0],r,atol=1e-12)


def test_confidence_block_separation_sign_and_domain():
    a=np.eye(4);b=np.eye(4);b[0,3]=1
    r= BearingObservation(np.eye(3),[-1,0,0]).linearize(a,b)[0]
    assert np.linalg.norm(r)==0
    reverse=BearingObservation(np.eye(3),[1,0,0]).linearize(a,b)[0]
    assert np.linalg.norm(reverse[3:])==5
    changed=reverse.copy();changed[:3]=[1,2,3]
    c1,w1=BearingObservation(np.eye(3),[1,0,0],q_direction=1).cost_weights(changed)
    c0,w0=BearingObservation(np.eye(3),[1,0,0],q_direction=0).cost_weights(changed)
    np.testing.assert_array_equal(w0[:3],w1[:3]);assert np.all(w0[3:]==0) and c0<c1
    for q in (0,1):
        with pytest.raises(ValueError):BearingObservation(np.eye(3),[1,0,0],q_direction=q).linearize(a,a)
