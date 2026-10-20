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
    sigma_radial: float | None = None
    direction_kernel: str = "gm"
    direction_delta: float = 1.

    def __post_init__(self):
        R=np.array(self.rotation,dtype=float,copy=True)
        u=np.array(self.direction,dtype=float,copy=True).reshape(3)
        if R.shape!=(3,3) or not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(R),1):
            raise ValueError('rotation must be SO(3)')
        if not np.isfinite(u).all() or np.linalg.norm(u)==0:
            raise ValueError('signed direction must be finite and nonzero')
        if not np.isfinite([self.sigma_r,self.sigma_d,self.q_direction]).all() or self.sigma_r<=0 or self.sigma_d<=0 or not 0<=self.q_direction<=1:
            raise ValueError('invalid observation scale/confidence')
        if self.sigma_radial is not None and (not np.isfinite(self.sigma_radial) or self.sigma_radial<=0):
            raise ValueError('invalid radial scale')
        if self.direction_kernel not in ("gm","huber","quadratic"):
            raise ValueError('unknown direction kernel')
        if self.direction_kernel=="huber" and (not np.isfinite(self.direction_delta) or self.direction_delta<=0):
            raise ValueError('invalid direction huber delta')
        u/=np.linalg.norm(u);R.flags.writeable=False;u.flags.writeable=False
        object.__setattr__(self,'rotation',R);object.__setattr__(self,'direction',u)

    def _geometry(self,Ti,Tj):
        Q=Tj[:3,:3].T@Ti[:3,:3]
        v=Tj[:3,:3].T@(Ti[:3,3]-Tj[:3,3]);length=float(np.linalg.norm(v))
        if length==0 or not np.isfinite(length):
            raise ValueError('bearing undefined at zero/nonfinite baseline')
        return Q,v,length,v/length

    def linearize(self,Ti,Tj):
        Q,v,length,n=self._geometry(Ti,Tj);e=so3_log(self.rotation.T@Q)
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

    def linearize_radial(self,Ti,Tj,log_scale):
        """Baseline-length residual coupling |v| to exp(log_scale).

        The scale state is a log-scale (additive increments), so the residual
        derivative w.r.t. the variable is -exp(log_scale)/sigma_radial.
        """
        if self.sigma_radial is None:
            raise ValueError('radial residual requires sigma_radial')
        Q,v,length,n=self._geometry(Ti,Tj)
        s=float(np.exp(float(log_scale)))
        G=n/self.sigma_radial
        a=np.zeros((1,6));b=np.zeros((1,6))
        a[0,3:]=G@Q;b[0,:3]=G@skew(v);b[0,3:]=-G
        r=np.array([(length-s)/self.sigma_radial]);d_scale=np.array([-s/self.sigma_radial])
        if not np.isfinite(r).all() or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(d_scale).all():
            raise ValueError('nonfinite bearing radial factor')
        return r,a,b,d_scale

    def _block_cost_weight(self,norm2,kernel,q):
        if kernel=="pseudo_huber":
            return q*norm2/(np.sqrt(1.+norm2)+1.), q/np.sqrt(1.+norm2)
        if kernel=="gm":
            denom=1.+norm2
            return .5*q*norm2/denom, q/(denom*denom)
        if kernel=="quadratic":
            return .5*q*norm2, q
        if kernel=="huber":
            norm=np.sqrt(norm2)
            if norm<=self.direction_delta:
                return .5*q*norm2, q
            return q*(self.direction_delta*norm-.5*self.direction_delta**2), q*self.direction_delta/norm
        raise ValueError('unknown kernel')

    def cost_weights(self,r):
        norms2=np.sum(np.asarray(r).reshape(2,3)**2,axis=1)
        q=(1.,self.q_direction)
        # Keep the calibrated rotation loss while exposing mathematically
        # correct GM semantics for the configurable direction kernel.
        kernels=("pseudo_huber",self.direction_kernel)
        cost=0.;weights=np.empty(6)
        for b,(n2,kernel) in enumerate(zip(norms2,kernels)):
            c,w=self._block_cost_weight(float(n2),kernel,q[b])
            cost+=c;weights[3*b:3*b+3]=w
        return float(cost),weights
