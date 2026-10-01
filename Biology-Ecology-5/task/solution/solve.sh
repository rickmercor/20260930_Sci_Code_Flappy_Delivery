#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

def _arr(x, ndim=None, shape=None):
    try:
        if np.iscomplexobj(x): raise ValueError('real data required')
        a=np.asarray(x,dtype=float)
    except (ValueError,TypeError,OverflowError) as e: raise ValueError('numeric data required') from e
    if (ndim is not None and a.ndim!=ndim) or (shape is not None and a.shape!=shape) or not np.all(np.isfinite(a)): raise ValueError('invalid shape or finite data')
    return a

def _solve(a,b):
    try: z=np.linalg.solve(a,b)
    except np.linalg.LinAlgError as e: raise ValueError('singular response') from e
    return _arr(z)

def uptake_tensors(U: 'np.ndarray', half: 'np.ndarray', R: 'np.ndarray') -> 'np.ndarray':
    U=_arr(U,2);half=_arr(half,2,U.shape);R=_arr(R,1,(U.shape[1],))
    if min(U.shape)<1 or np.any(U<0) or np.any(half<=0) or np.any(R<=0):raise ValueError('invalid uptake domain')
    den=half+R
    return _arr(np.stack([U*R/den,U*half/den**2,-2*U*half/den**3,6*U*half/den**4]))

import numpy as np



def cavity_state_jet(tensors: 'np.ndarray', B: 'np.ndarray', leakage: float, N: 'np.ndarray', R: 'np.ndarray', v: 'np.ndarray', w: 'np.ndarray') -> 'np.ndarray':
    t=_arr(tensors,3);s,m=t.shape[1:]
    if t.shape[0]!=4 or not 1<=s<=m:raise ValueError('invalid tensor shape')
    B=_arr(B,2,(m,m));N=_arr(N,1,(s,));R=_arr(R,1,(m,));v=_arr(v,1,(m,));w=_arr(w,1,(m,));ell=float(_arr(leakage,0))
    if np.any(B<0) or not np.allclose(B.sum(0),1,atol=1e-12,rtol=0) or not 0<=ell<1 or np.any(N<=0) or np.any(R<=0):raise ValueError('invalid ecology domain')
    L=np.eye(m)-ell*B;C=(1-ell)*t[1];E=L@t[0].T;Q=np.eye(m)+L@np.diag(N@t[1]);A=np.block([[np.zeros((s,s)),C],[E,Q]])
    a=_solve(A,np.r_[np.zeros(s),v]);b=_solve(A,np.r_[np.zeros(s),w]);Na,Ra=a[:s],a[s:];Nb,Rb=b[:s],b[s:]
    top=(1-ell)*(t[2]@(Ra*Rb))
    bottom=L@((Na@t[1])*Rb+(Nb@t[1])*Ra+(N@t[2])*Ra*Rb)
    ab=_solve(A,-np.r_[top,bottom])
    return _arr(np.stack([np.r_[N,R],a,b,ab]))

import numpy as np

def _mul(a,b):
    return np.stack([a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+a[0]*b[2],a[3]*b[0]+a[1]*b[2]+a[2]*b[1]+a[0]*b[3]])

def _compose(t,r):
    # t contains orders 0,1,2 of a univariate resource-dependent quantity.
    return np.stack([t[0],t[1]*r[1],t[1]*r[2],t[1]*r[3]+t[2]*r[1]*r[2]])

def susceptibility_jet(tensors: 'np.ndarray', B: 'np.ndarray', leakage: float, state: 'np.ndarray') -> 'np.ndarray':
    t=_arr(tensors,3);s,m=t.shape[1:]
    if t.shape[0]!=4 or not 1<=s<=m:raise ValueError('invalid tensors')
    B=_arr(B,2,(m,m));state=_arr(state,2,(4,s+m));ell=float(_arr(leakage,0))
    if not 0<=ell<1 or np.any(B<0) or not np.allclose(B.sum(0),1,atol=1e-12,rtol=0):raise ValueError('invalid conversion')
    N=state[:,:s];R=state[:,s:];L=np.eye(m)-ell*B
    flux=_compose(t,R);grad=_compose(t[1:],R)
    C=(1-ell)*grad;E=np.stack([L@x.T for x in flux])
    loss=_mul(N[:,:,None],grad).sum(axis=1)
    Q=np.stack([L@np.diag(x) for x in loss]);Q[0]+=np.eye(m)
    A=np.stack([np.block([[np.zeros((s,s)),C[k]],[E[k],Q[k]]]) for k in range(4)])
    Z0=_solve(A[0],np.eye(s+m));Za=-Z0@A[1]@Z0;Zb=-Z0@A[2]@Z0
    Zab=-Z0@(A[3]@Z0+A[1]@Zb+A[2]@Za)
    return _arr(np.stack([Z0[s:,s:],Za[s:,s:],Zb[s:,s:],Zab[s:,s:]]))

import numpy as np

def _contract(a,b,c):
    # Scalar a^T b c jets via the two commuting derivative labels.
    masks=(0,1,2,3);out=np.zeros(4)
    for k in masks:
        for i in masks:
            if i&~k:continue
            for j in masks:
                if j&~(k^i):continue
                h=k^i^j
                out[k]+=a[i]@b[j]@c[h]
    return out

def selection_mixed_jet(focal: 'np.ndarray', B: 'np.ndarray', leakage: float, resources: 'np.ndarray', response: 'np.ndarray', N0: float, mp: float, mm: float) -> 'np.ndarray':
    t=_arr(focal,3);m=t.shape[-1]
    if t.shape!=(4,2,m) or m<1:raise ValueError('two focal lineages required')
    B=_arr(B,2,(m,m));r=_arr(resources,2,(4,m));H=_arr(response,3,(4,m,m));ell=float(_arr(leakage,0));N0=float(_arr(N0,0));mp=float(_arr(mp,0));mm=float(_arr(mm,0))
    if not 0<=ell<1 or N0<=0 or min(mp,mm)<0 or np.any(B<0) or not np.allclose(B.sum(0),1,atol=1e-12,rtol=0):raise ValueError('invalid focal domain')
    L=np.eye(m)-ell*B;q=_compose(t,r);grad=_compose(t[1:],r)
    dc=(1-ell)*(grad[:,1]-grad[:,0]);ep=np.stack([L@x for x in q[:,0]]);de=np.stack([L@x for x in q[:,1]-q[:,0]])
    g=(1-ell)*(q[:,1]-q[:,0]).sum(axis=1);g[0]-=mm-mp
    sj=g-N0*_contract(dc,H,ep);eta=N0*_contract(dc,H,de)
    return _arr(np.stack([sj,eta],axis=1))

import numpy as np

def _diffusion_inputs(selection,N0,D):
    selection=_arr(selection,2,(4,2));N0=float(_arr(N0,0));D=float(_arr(D,0))
    if N0<=0 or D<=0:raise ValueError('positive abundance and noise required')
    n=N0/(2*D)
    if not np.isfinite(n):raise ValueError('nonfinite effective size')
    return selection,n

def _order(order):
    if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or not 16<=order<=256:raise ValueError('order must be an integer in [16,256]')
    return np.polynomial.legendre.leggauss(order)

def _potential(sel,n,x):
    return n*(sel[:,1,None]*x*x-2*sel[:,0,None]*x)

def _exp(a):
    e=np.exp(a[0]);return np.stack([e,e*a[1],e*a[2],e*(a[3]+a[1]*a[2])])

def _log(a):
    if np.any(a[0]<=0): raise ValueError('positive baseline needed')
    return np.stack([np.log(a[0]),a[1]/a[0],a[2]/a[0],a[3]/a[0]-a[1]*a[2]/a[0]**2])

def log_scale_integrals(selection: 'np.ndarray', N0: float, D: float, intervals: 'np.ndarray', order: int = 64) -> 'np.ndarray':
    sel,n=_diffusion_inputs(selection,N0,D);iv=_arr(intervals,2)
    if iv.shape[1]!=2 or iv.shape[0]<1 or np.any(iv[:,0]<0) or np.any(iv[:,1]>1) or np.any(iv[:,0]>=iv[:,1]):raise ValueError('intervals must obey 0<=lo<hi<=1')
    nodes,weights=_order(order);out=[]
    for lo,hi in iv:
        x=lo+(hi-lo)*(nodes+1)/2;V=_potential(sel,n,x);_arr(V)
        shift=V[0].max();V[0]-=shift
        integral=(_exp(V)*weights).sum(axis=1)*(hi-lo)/2
        lg=_log(integral);lg[0]+=shift;out.append(lg)
    return _arr(np.stack(out,axis=1))

import numpy as np



def hitting_moment_jet(selection: 'np.ndarray', N0: float, D: float, f0: float, order: int = 64) -> 'np.ndarray':
    sel,n=_diffusion_inputs(selection,N0,D);f0=float(_arr(f0,0));nodes,weights=_order(order)
    if not 0<f0<1:raise ValueError('initial frequency must be interior')
    base=log_scale_integrals(sel,N0,D,np.array([[0,f0],[f0,1],[0,1]]),order)
    logS,logT,logZ=base.T
    p=_exp(logS-logZ);terms=[];quadweights=[]
    for lo,hi,left in [(0.,f0,True),(f0,1.,False)]:
        y=lo+(hi-lo)*(nodes+1)/2
        intervals=np.array([[0.,z] for z in y]+[[z,1.] for z in y])
        logs=log_scale_integrals(sel,N0,D,intervals,order);Sy=logs[:,:order];Ty=logs[:,order:]
        V=_potential(sel,n,y)
        log_integrand=(2*Sy+logT[:,None] if left else Sy+Ty+logS[:,None])-2*logZ[:,None]-V
        log_integrand[0]+=np.log(2*n)-np.log(y)-np.log1p(-y)
        terms.append(log_integrand);quadweights.append(weights*(hi-lo)/2)
    log_terms=np.concatenate(terms,axis=1);qw=np.concatenate(quadweights);shift=log_terms[0].max();log_terms[0]-=shift
    u=(_exp(log_terms)*qw).sum(axis=1)*np.exp(shift)
    return _arr(np.stack([p,u],axis=1))

import numpy as np



def conditional_time_statistics(moments: 'np.ndarray') -> 'np.ndarray':
    a=_arr(moments,2,(4,2))
    if not 0<a[0,0]<=1 or a[0,1]<=0:raise ValueError('positive valid hitting probability and moment required')
    logt=_log(a[:,1])-_log(a[:,0]);t=_exp(logt)
    return _arr(np.r_[t,logt[3]])

import numpy as np



def mixed_fixation_time(config: dict, order: int = 64) -> float:
    keys=('U','half','B','leakage','N','R','v','w','N0','mp','mm','D','f0')
    if not isinstance(config,dict) or any(k not in config for k in keys):raise ValueError('missing configuration')
    c=config;N=_arr(c['N'],1);s=len(N);U=_arr(c['U'],2)
    if U.shape[0]!=s+2:raise ValueError('U must contain residents then parent and mutant')
    t=uptake_tensors(U,c['half'],c['R'])
    state=cavity_state_jet(t[:,:s],c['B'],c['leakage'],N,c['R'],c['v'],c['w'])
    H=susceptibility_jet(t[:,:s],c['B'],c['leakage'],state)
    sel=selection_mixed_jet(t[:,s:],c['B'],c['leakage'],state[:,s:],H,c['N0'],c['mp'],c['mm'])
    mom=hitting_moment_jet(sel,c['N0'],c['D'],c['f0'],order)
    return float(conditional_time_statistics(mom)[4])
SCICODE_GOLD_EOF
