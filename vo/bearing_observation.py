"""Experimental SO(3)/signed-bearing observation, without a distance variable."""
from dataclasses import dataclass
import numpy as np
from .se3 import so3_log, so3_left_jacobian_inv, skew


@dataclass(frozen=True)
class BearingObservation:
    rotation: np.ndarray
    direction: np.ndarray
    sigma_r: float = .1
    sigma_d: float = .4
    q_direction: float = 1.

    def __post_init__(self):
        R=np.array(self.rotation,dtype=float,copy=True)
        u=np.array(self.direction,dtype=float,copy=True).reshape(3)
        if R.shape!=(3,3) or not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(R),1):
            raise ValueError('rotation must be SO(3)')
        if not np.isfinite(u).all() or np.linalg.norm(u)==0:
            raise ValueError('signed direction must be finite and nonzero')
        if not np.isfinite([self.sigma_r,self.sigma_d,self.q_direction]).all() or self.sigma_r<=0 or self.sigma_d<=0 or not 0<=self.q_direction<=1:
            raise ValueError('invalid observation scale/confidence')
        u/=np.linalg.norm(u);R.flags.writeable=False;u.flags.writeable=False
        object.__setattr__(self,'rotation',R);object.__setattr__(self,'direction',u)

    def linearize(self,Ti,Tj):
        Q=Tj[:3,:3].T@Ti[:3,:3]
        v=Tj[:3,:3].T@(Ti[:3,3]-Tj[:3,3]);length=float(np.linalg.norm(v))
        if length==0 or not np.isfinite(length):
            raise ValueError('bearing undefined at zero/nonfinite baseline')
        n=v/length;e=so3_log(self.rotation.T@Q)
        # Jr^-1(e)=Jl^-1(-e), matching right SE3 increments.
        Jr=so3_left_jacobian_inv(-e)/self.sigma_r
        G=(np.eye(3)-np.outer(n,n))/length/self.sigma_d
        a=np.zeros((6,6));b=np.zeros((6,6))
        a[:3,:3]=Jr;b[:3,:3]=-Jr@Q.T
        a[3:,3:]=G@Q;b[3:,:3]=G@skew(v);b[3:,3:]=-G
        r=np.r_[e/self.sigma_r,(n-self.direction)/self.sigma_d]
        if not np.isfinite(r).all() or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('nonfinite bearing factor')
        return r,a,b

    def cost_weights(self,r):
        norms2=np.sum(np.asarray(r).reshape(2,3)**2,axis=1)
        q=np.array([1.,self.q_direction])
        cost=float(np.sum(q*norms2/(np.sqrt(1+norms2)+1)))
        return cost,np.repeat(q/np.sqrt(1+norms2),3)
