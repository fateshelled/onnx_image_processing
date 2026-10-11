"""Opt-in canonical solver replacing raw loop edges with bearing observations."""
import numpy as np
from .bearing_observation import DEFAULT_BASELINE_FLOOR, BearingObservation
from .relative_joint import RelativeJointWindowOptimizer


class BearingJointWindowOptimizer(RelativeJointWindowOptimizer):
    def __init__(self, bearing_radial=False, bearing_sigma_d=None,
                 bearing_sigma_radial=None, bearing_scale_prior_sigma=None,
                 bearing_scale_prior_mean=0.0, bearing_dir_kernel="gm",
                 bearing_dir_delta=1.0, bearing_baseline_floor=None, **kwargs):
        super().__init__(**kwargs)
        self.bearing_loops = {}
        self.bearing_radial = bool(bearing_radial)
        self.bearing_sigma_d = (self.loop_dir_sigma if bearing_sigma_d is None
                                else float(bearing_sigma_d))
        self.bearing_sigma_radial = (self.step_scale_t if bearing_sigma_radial is None
                                     else float(bearing_sigma_radial))
        self.bearing_scale_prior_sigma = (self.scale_prior_sigma
                                          if bearing_scale_prior_sigma is None
                                          else float(bearing_scale_prior_sigma))
        self.bearing_scale_prior_mean = float(bearing_scale_prior_mean)
        self.bearing_dir_kernel = str(bearing_dir_kernel)
        self.bearing_dir_delta = float(bearing_dir_delta)
        self.bearing_baseline_floor = (DEFAULT_BASELINE_FLOOR
                                       if bearing_baseline_floor is None
                                       else float(bearing_baseline_floor))
        if self.bearing_sigma_d <= 0:
            raise ValueError('bearing direction sigma must be positive')
        if self.bearing_dir_kernel not in ("gm","huber","quadratic"):
            raise ValueError('unknown bearing direction kernel')
        if self.bearing_radial and (self.bearing_sigma_radial <= 0
                                    or self.bearing_scale_prior_sigma <= 0):
            raise ValueError('bearing radial/scale-prior sigma must be positive')

    @classmethod
    def from_empty(cls, optimizer):
        if type(optimizer) is not RelativeJointWindowOptimizer or optimizer.pose_ids or optimizer.edges or optimizer.joint_priors:
            raise ValueError('adapter requires an empty relative optimizer')
        result=cls.__new__(cls)
        result.__dict__=optimizer.__dict__.copy()
        result.bearing_loops={}
        result.bearing_radial=False
        result.bearing_sigma_d=optimizer.loop_dir_sigma
        result.bearing_sigma_radial=optimizer.step_scale_t
        result.bearing_scale_prior_sigma=optimizer.scale_prior_sigma
        result.bearing_scale_prior_mean=0.0
        result.bearing_dir_kernel="gm"
        result.bearing_dir_delta=1.0
        result.bearing_baseline_floor=DEFAULT_BASELINE_FLOOR
        result.scale_regularizer_exclude=set()
        return result

    def add_named_edge(self, key, i, j, M, **kwargs):
        if key[0]!='loop':
            return super().add_named_edge(key,i,j,M,**kwargs)
        if key in self.bearing_loops or key in self.scalar_col:
            raise ValueError('duplicate loop observation')
        if i not in self.T or j not in self.T:
            raise KeyError('loop endpoints must be live')
        scale=float(kwargs.get('scale',1.))
        if not self.bearing_radial and scale!=1.:
            raise ValueError('bearing loops have no distance scale state')
        if self.bearing_radial and scale<=0.:
            raise ValueError('bearing loop scale must be positive')
        sigma=kwargs.get('sigma_r')
        radial_sigma=self.bearing_sigma_radial if self.bearing_radial else None
        self.bearing_loops[key]=(i,j,BearingObservation(M[:3,:3],M[:3,3],
            sigma_r=self.step_scale_r if sigma is None else sigma,
            sigma_d=self.bearing_sigma_d,q_direction=1.,sigma_radial=radial_sigma,
            direction_kernel=self.bearing_dir_kernel,
            direction_delta=self.bearing_dir_delta,
            baseline_floor=self.bearing_baseline_floor),key)
        if self.bearing_radial:
            self.add_scalar(key,np.log(scale))
            self.scale_regularizer_exclude.add(key)

    def _pose_components(self):
        groups,absolute=super()._pose_components()
        groups=[set(c) for c in groups]
        for i,j,_f,_s in self.bearing_loops.values():
            touched=[c for c in groups if i in c or j in c]
            merged=set().union(*touched)
            groups=[c for c in groups if not(c&merged)]+[merged]
        return sorted(groups,key=min),absolute

    def _residuals(self):
        rows=[super()._residuals()]
        rows.extend(f.linearize(self.T[i],self.T[j])[0]
                    for i,j,f,_s in self.bearing_loops.values())
        if self.bearing_radial:
            for key in self.bearing_loops:
                i,j,f,scale_key=self.bearing_loops[key]
                a=self.log_scale(scale_key)
                rows.append(f.linearize_radial(self.T[i],self.T[j],a)[0])
                rows.append(np.array([(a-self.bearing_scale_prior_mean)
                                      /self.bearing_scale_prior_sigma]))
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
        for i,j,f,_s in self.bearing_loops.values():
            _,a,b=f.linearize(self.T[i],self.T[j])
            for key,block in ((i,a),(j,b)):
                r,c=np.nonzero(block)
                rows.extend(offset+r);cols.extend(np.asarray(columns['pose',key])[c]);values.extend(block[r,c])
            offset+=6
        if self.bearing_radial:
            for key in self.bearing_loops:
                i,j,f,scale_key=self.bearing_loops[key]
                a=self.log_scale(scale_key)
                _,ra,rb,dscale=f.linearize_radial(self.T[i],self.T[j],a)
                for target,block in ((i,ra),(j,rb)):
                    r,c=np.nonzero(block)
                    rows.extend(offset+r);cols.extend(np.asarray(columns['pose',target])[c]);values.extend(block[r,c])
                sc=columns['scale',scale_key][0]
                if dscale[0]!=0.:
                    rows.append(offset);cols.append(sc);values.append(dscale[0])
                offset+=1
                rows.append(offset);cols.append(sc)
                values.append(1./self.bearing_scale_prior_sigma)
                offset+=1
        return np.asarray(rows,dtype=int),np.asarray(cols,dtype=int),np.asarray(values,float),(offset,shape[1])

    def _huber_weights(self):
        rows=[super()._huber_weights()]
        rows.extend(f.cost_weights(f.linearize(self.T[i],self.T[j])[0])[1]
                    for i,j,f,_s in self.bearing_loops.values())
        if self.bearing_radial:
            for key in self.bearing_loops:
                i,j,f,scale_key=self.bearing_loops[key]
                rr=f.linearize_radial(self.T[i],self.T[j],self.log_scale(scale_key))[0]
                rows.append(np.array([1./np.sqrt(1.+rr[0]**2)]))
                rows.append(np.ones(1))
        return np.concatenate(rows)

    def _true_cost(self):
        r=super()._residuals();n=6*len(self.edges)
        norms=np.linalg.norm(r[:n].reshape(-1,6),axis=1)
        cost=np.where(norms<=self.huber,.5*norms**2,self.huber*(norms-.5*self.huber)).sum()+.5*(r[n:]@r[n:])
        cost+=sum(f.cost_weights(f.linearize(self.T[i],self.T[j])[0])[0]
                  for i,j,f,_s in self.bearing_loops.values())
        if self.bearing_radial:
            for key in self.bearing_loops:
                i,j,f,scale_key=self.bearing_loops[key]
                radial=float(f.linearize_radial(self.T[i],self.T[j],self.log_scale(scale_key))[0][0])
                cost+=radial*radial/(np.sqrt(1.+radial*radial)+1.)
                prior=(self.log_scale(scale_key)-self.bearing_scale_prior_mean)/self.bearing_scale_prior_sigma
                cost+=.5*float(prior*prior)
        return float(cost)

    def marginalize_joint(self,*args,**kwargs):
        if self.bearing_loops:
            raise ValueError('loop observations must remain raw, outside segment summaries')
        return super().marginalize_joint(*args,**kwargs)

    def harvest_robust_residuals(self):
        result=super().harvest_robust_residuals()
        for i,j,f,_s in self.bearing_loops.values():
            r,_,_=f.linearize(self.T[i],self.T[j])
            rot=float(np.linalg.norm(r[:3])*f.sigma_r)
            Tij=np.linalg.inv(self.T[j])@self.T[i]
            baseline=float(np.linalg.norm(Tij[:3,3]))
            if baseline>=f.baseline_floor:
                dirn=float(2*np.arcsin(np.clip(np.linalg.norm(r[3:6])*f.sigma_d/2,0,1)))
            else:
                # Direction is dropped below the baseline floor; its residual is
                # not a real observation (0 would bias the MAD loop scale).
                dirn=float('nan')
            result.append((i,j,rot,dirn))
        return result
