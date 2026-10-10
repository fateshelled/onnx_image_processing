"""Opt-in canonical solver replacing raw loop edges with bearing observations."""
import numpy as np
from .bearing_observation import BearingObservation
from .relative_joint import RelativeJointWindowOptimizer


class BearingJointWindowOptimizer(RelativeJointWindowOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bearing_loops = {}

    @classmethod
    def from_empty(cls, optimizer):
        if type(optimizer) is not RelativeJointWindowOptimizer or optimizer.pose_ids or optimizer.edges or optimizer.joint_priors:
            raise ValueError('adapter requires an empty relative optimizer')
        result=cls.__new__(cls)
        result.__dict__=optimizer.__dict__.copy()
        result.bearing_loops={}
        return result

    def add_named_edge(self, key, i, j, M, **kwargs):
        if key[0]!='loop':
            return super().add_named_edge(key,i,j,M,**kwargs)
        if key in self.bearing_loops or key in self.scalar_col:
            raise ValueError('duplicate loop observation')
        if i not in self.T or j not in self.T:
            raise KeyError('loop endpoints must be live')
        if kwargs.get('scale',1.)!=1.:
            raise ValueError('bearing loops have no distance scale state')
        sigma=kwargs.get('sigma_r')
        self.bearing_loops[key]=(i,j,BearingObservation(M[:3,:3],M[:3,3],
            sigma_r=self.step_scale_r if sigma is None else sigma,sigma_d=.4,q_direction=1.))

    def _pose_components(self):
        groups,absolute=super()._pose_components()
        groups=[set(c) for c in groups]
        for i,j,_ in self.bearing_loops.values():
            touched=[c for c in groups if i in c or j in c]
            merged=set().union(*touched)
            groups=[c for c in groups if not(c&merged)]+[merged]
        return sorted(groups,key=min),absolute

    def _residuals(self):
        rows=[super()._residuals()]
        rows.extend(f.linearize(self.T[i],self.T[j])[0] for i,j,f in self.bearing_loops.values())
        return np.concatenate(rows)

    def _jacobian_coo(self,eps=1e-6):
        # The legacy row builder obtains nrow through virtual _residuals().
        # Hide only the appended rows while building its prior row offsets.
        loops=self.bearing_loops
        self.bearing_loops={}
        try:
            rr,cc,vv,shape=super()._jacobian_coo(eps)
        finally:
            self.bearing_loops=loops
        rows=list(rr);cols=list(cc);values=list(vv);offset=shape[0];columns=self._key_columns()
        for i,j,f in self.bearing_loops.values():
            _,a,b=f.linearize(self.T[i],self.T[j])
            for key,block in ((i,a),(j,b)):
                r,c=np.nonzero(block)
                rows.extend(offset+r);cols.extend(np.asarray(columns['pose',key])[c]);values.extend(block[r,c])
            offset+=6
        return np.asarray(rows,dtype=int),np.asarray(cols,dtype=int),np.asarray(values,float),(offset,shape[1])

    def _huber_weights(self):
        rows=[super()._huber_weights()]
        rows.extend(f.cost_weights(f.linearize(self.T[i],self.T[j])[0])[1] for i,j,f in self.bearing_loops.values())
        return np.concatenate(rows)

    def _true_cost(self):
        r=super()._residuals();n=6*len(self.edges)
        norms=np.linalg.norm(r[:n].reshape(-1,6),axis=1)
        cost=np.where(norms<=self.huber,.5*norms**2,self.huber*(norms-.5*self.huber)).sum()+.5*(r[n:]@r[n:])
        return float(cost+sum(f.cost_weights(f.linearize(self.T[i],self.T[j])[0])[0] for i,j,f in self.bearing_loops.values()))

    def marginalize_joint(self,*args,**kwargs):
        if self.bearing_loops:
            raise ValueError('loop observations must remain raw, outside segment summaries')
        return super().marginalize_joint(*args,**kwargs)

    def harvest_robust_residuals(self):
        result=super().harvest_robust_residuals()
        for i,j,f in self.bearing_loops.values():
            r,_,_=f.linearize(self.T[i],self.T[j])
            result.append((i,j,float(np.linalg.norm(r[:3])*f.sigma_r),
                           float(2*np.arcsin(np.clip(np.linalg.norm(r[3:])*f.sigma_d/2,0,1)))))
        return result
