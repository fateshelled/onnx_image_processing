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


def _step(c,sign):
    dx=np.zeros(6);dx[c]=sign*1e-6;return se3_exp(dx)


def test_radial_residual_sign_sigma_and_jacobian():
    a=se3_exp(np.array([.1,.2,-.1,.2,-.3,.1]))
    b=se3_exp(np.array([-.2,.05,.1,1.,.3,-.2]))
    v=b[:3,:3].T@(a[:3,3]-b[:3,3]);length=float(np.linalg.norm(v))
    obs=BearingObservation(np.eye(3),[.1,.2,1.],sigma_radial=.3)
    r,ja,jb,ds=obs.linearize_radial(a,b,np.log(length))
    np.testing.assert_allclose(r,0.,atol=1e-12)
    j=np.hstack((ja,jb,ds.reshape(-1,1)))
    num=np.empty_like(j)
    for c in range(13):
        values=[]
        for sign in (1,-1):
            x,y=a.copy(),b.copy();dlog=0.
            if c<6:x=x@_step(c,sign)
            elif c<12:y=y@_step(c-6,sign)
            else:dlog=sign*1e-6
            values.append(obs.linearize_radial(x,y,np.log(length)+dlog)[0])
        num[:,c]=(values[0]-values[1])/2e-6
    np.testing.assert_allclose(j,num,rtol=1e-6,atol=1e-7)
    short=obs.linearize_radial(a,b,np.log(length*.5))[0]
    assert short[0]>0  # estimated baseline below prediction -> positive residual
    with pytest.raises(ValueError):BearingObservation(np.eye(3),[1,0,0]).linearize_radial(a,b,0.)
    with pytest.raises(ValueError):BearingObservation(np.eye(3),[1,0,0],sigma_radial=-1.)


def test_direction_kernel_cost_gradient_and_blocks():
    a=se3_exp(np.array([.1,.2,-.1,.2,-.3,.1]))
    b=se3_exp(np.array([-.2,.05,.1,1.,.3,-.2]))
    for kernel,delta in (("gm",1.),("huber",.3),("quadratic",1.)):
        obs=BearingObservation(np.eye(3),[.1,.2,1.],sigma_d=.2,
                               direction_kernel=kernel,direction_delta=delta)
        r,ja,jb=obs.linearize(a,b);j=np.hstack((ja,jb))
        gradient=j.T@(obs.cost_weights(r)[1]*r)
        ng=[]
        for c in range(12):
            values=[]
            for sign in (1,-1):
                x,y=a.copy(),b.copy();dx=np.zeros(6);dx[c%6]=sign*1e-6
                if c<6:x=x@se3_exp(dx)
                else:y=y@se3_exp(dx)
                rr,_,_=obs.linearize(x,y);values.append(obs.cost_weights(rr)[0])
            ng.append((values[0]-values[1])/2e-6)
        np.testing.assert_allclose(gradient,ng,rtol=1e-6,atol=1e-7)
    r=np.array([0.,0.,0.,3.,4.,0.])
    cost,w=BearingObservation(np.eye(3),[1,0,0],direction_kernel="quadratic").cost_weights(r)
    np.testing.assert_allclose(w,1.);np.testing.assert_allclose(cost,.5*25.)
    cost,w=BearingObservation(np.eye(3),[1,0,0],direction_kernel="huber",
                              direction_delta=.3).cost_weights(r)
    assert w[3]==pytest.approx(.3/5.);np.testing.assert_allclose(w[:3],1.);assert w[4]==w[5]==w[3]
    cost,w=BearingObservation(np.eye(3),[1,0,0]).cost_weights(r)  # default gm
    assert cost==pytest.approx(.5*25./26.)
    assert w[3]==pytest.approx(1./26.**2)
    large_cost,_=BearingObservation(np.eye(3),[1,0,0]).cost_weights(
        np.array([0.,0.,0.,300.,400.,0.]))
    assert large_cost<.5
    rotation=np.array([3.,4.,0.,0.,0.,0.])
    cost,w=BearingObservation(np.eye(3),[1,0,0]).cost_weights(rotation)
    assert cost==pytest.approx(np.sqrt(26.)-1.)
    np.testing.assert_allclose(w[:3],1./np.sqrt(26.))
    with pytest.raises(ValueError):BearingObservation(np.eye(3),[1,0,0],direction_kernel="nope")
