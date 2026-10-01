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
 
def construct_event_steps(points: "np.ndarray", times: "np.ndarray", initial_heading: "np.ndarray", centre: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    p=np.asarray(points,float); t=np.asarray(times,float)
    h=np.asarray(initial_heading,float); c=np.asarray(centre,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or t.shape!=(len(p),) or h.shape!=(2,) or c.shape!=(2,) or not all(np.all(np.isfinite(a)) for a in (p,t,h,c)):
        raise ValueError("invalid event input shapes or nonfinite values")
    v=np.diff(p,axis=0); lengths=np.linalg.norm(v,axis=1); dt=np.diff(t)
    if np.any(lengths<=0) or np.linalg.norm(h)==0 or np.any(dt<=0):
        raise ValueError("segments and heading must be nonzero; times increase")
    prev=np.vstack((h,v[:-1]))
    phi=np.arctan2(prev[:,0]*v[:,1]-prev[:,1]*v[:,0],np.sum(prev*v,axis=1))
    phi=(phi+np.pi)%(2*np.pi)-np.pi
    return dt,lengths/dt,phi,np.linalg.norm(p[1:]-c,axis=1),np.mod(t[1:],24.)

import numpy as np
from scipy.special import i0e, i1e
 
def carved_turn_log_density(angles: "np.ndarray", kappa: "np.ndarray", width: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    phi=np.asarray(angles,float); k=np.asarray(kappa,float); w=np.asarray(width,float)
    if phi.ndim!=1 or k.shape!=(2,) or w.shape!=(2,) or not all(np.all(np.isfinite(a)) for a in (phi,k,w)) or np.any(k<=0) or np.any(k>50) or np.any(w<1e-5) or np.any(phi< -np.pi) or np.any(phi>=np.pi):
        raise ValueError("invalid carved density parameters")
    a=i0e(k); z=i0e(k+w); D=a-z
    # The scaled Bessel difference removes the large exponential factor.
    logZ=np.log(2*np.pi)+k+np.log(D)
    dlogZ=1+((i1e(k)-a)-(i1e(k+w)-z))/D
    u=1-np.cos(phi[:,None]); factor=-np.expm1(-w[None,:]*u)
    with np.errstate(divide='ignore'):
        logf=k[None,:]*np.cos(phi[:,None])+np.log(factor)-logZ[None,:]
    score=np.cos(phi[:,None])-dlogZ[None,:]
    return logf,score

import numpy as np
from scipy.special import gammaln
 
def gamma_step_log_density(durations: "np.ndarray", speeds: "np.ndarray", time_shape: "np.ndarray", time_rate: "np.ndarray", speed_shape: "np.ndarray", speed_rate: "np.ndarray") -> "np.ndarray":
    dt=np.asarray(durations,float); s=np.asarray(speeds,float)
    arrays=[np.asarray(a,float) for a in (time_shape,time_rate,speed_shape,speed_rate)]
    if dt.ndim!=1 or s.shape!=dt.shape or not all(np.all(np.isfinite(a)) and np.all(a>0) for a in [dt,s]+arrays) or any(a.shape!=(2,) for a in arrays):
        raise ValueError("gamma observations and parameters must be positive")
    at,bt,as_,bs=arrays
    return (at*np.log(bt)-gammaln(at)+(at-1)*np.log(dt[:,None])-bt*dt[:,None]
            +as_*np.log(bs)-gammaln(as_)+(as_-1)*np.log(s[:,None])-bs*s[:,None])

import numpy as np
from scipy.special import expit
 
def event_transition_matrices(distances: "np.ndarray", hours: "np.ndarray", coefficients: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    d=np.asarray(distances,float); h=np.asarray(hours,float); beta=np.asarray(coefficients,float)
    if d.ndim!=1 or h.shape!=d.shape or beta.shape!=(2,4) or not all(np.all(np.isfinite(a)) for a in (d,h,beta)) or np.any(d<0):
        raise ValueError("invalid event covariates or coefficients")
    cs=np.cos(np.pi*h/12); sn=np.sin(np.pi*h/12)
    u=beta[:,0]+d[:,None]*beta[:,1]+cs[:,None]*beta[:,2]+sn[:,None]*beta[:,3]
    p=expit(u); G=np.empty((len(d),2,2));G[:,0,0]=1-p[:,0];G[:,0,1]=p[:,0];G[:,1,0]=p[:,1];G[:,1,1]=1-p[:,1]
    D=np.zeros_like(G); q=p[:,0]*(1-p[:,0])*cs;D[:,0,0]=-q;D[:,0,1]=q
    return G,D

import numpy as np
 
def joint_event_emissions(angle_log: "np.ndarray", gamma_log: "np.ndarray") -> "np.ndarray":
    a=np.asarray(angle_log,float); b=np.asarray(gamma_log,float)
    if a.ndim!=2 or a.shape[1]!=2 or a.shape[0]<2 or b.shape!=a.shape or np.any(np.isnan(a)) or not np.all(np.isfinite(b)):
        raise ValueError("emissions require matching n by two arrays")
    return a+b

import numpy as np
 
def scaled_event_forward(log_emissions: "np.ndarray", transitions: "np.ndarray", initial: "np.ndarray") -> "tuple[float, np.ndarray]":
    L=np.asarray(log_emissions,float);G=np.asarray(transitions,float);delta=np.asarray(initial,float)
    if L.ndim!=2 or L.shape[1]!=2 or len(L)<2 or G.shape!=(len(L),2,2) or delta.shape!=(2,) or np.any(np.isnan(L)) or np.any(np.isposinf(L)) or not np.all(np.any(np.isfinite(L),axis=1)) or not np.all(np.isfinite(G)) or np.any(G<0) or not np.allclose(G.sum(axis=2),1,atol=1e-10) or np.any(delta<=0) or not np.allclose(delta.sum(),1):
        raise ValueError("invalid HMM inputs")
    out=np.empty_like(L); ll=0.;f=delta
    for t in range(len(L)):
        if t: f=f@G[t]
        m=np.max(L[t]);raw=f*np.exp(L[t]-m);scale=raw.sum()
        if scale<=0:raise ValueError("zero event likelihood")
        f=raw/scale;out[t]=f;ll+=m+np.log(scale)
    return float(ll),out

import numpy as np
 
def mixed_event_curvature(log_emissions: "np.ndarray", concentration_score: "np.ndarray", transitions: "np.ndarray", transition_derivative: "np.ndarray", initial: "np.ndarray") -> float:
    L=np.asarray(log_emissions,float);S=np.asarray(concentration_score,float)
    G=np.asarray(transitions,float);D=np.asarray(transition_derivative,float);delta=np.asarray(initial,float)
    if L.ndim!=2 or L.shape[1]!=2 or len(L)<2 or S.shape!=L.shape or G.shape!=(len(L),2,2) or D.shape!=G.shape or not np.all(np.isfinite(S)) or not np.all(np.isfinite(D)) or delta.shape!=(2,) or not np.all(delta>0):
        raise ValueError("invalid mixed curvature inputs")
    # Only the first state's emission depends on concentration kappa[0].
    S=S.copy(); S[:,1]=0.
    f=delta.copy(); fk=np.zeros(2);fb=np.zeros(2);fkb=np.zeros(2);ans=0.
    for t in range(len(L)):
        if t:
            oldf,oldk,oldb,oldkb=f,fk,fb,fkb
            f=oldf@G[t];fk=oldk@G[t]
            fb=oldb@G[t]+oldf@D[t]
            fkb=oldkb@G[t]+oldk@D[t]
        m=np.max(L[t]);e=np.exp(L[t]-m);ek=e*S[t]
        r=f*e;rk=fk*e+f*ek;rb=fb*e;rkb=fkb*e+fb*ek
        c=r.sum();ck=rk.sum();cb=rb.sum();ckb=rkb.sum()
        if c<=0:raise ValueError("zero event likelihood")
        ans+=ckb/c-ck*cb/c**2
        fn=r/c;fkn=(rk-fn*ck)/c;fbn=(rb-fn*cb)/c
        fkbn=(rkb-fbn*ck-fkn*cb-fn*ckb)/c
        f,fk,fb,fkb=fn,fkn,fbn,fkbn
    return float(ans)

import numpy as np
 
def evaluate_event_curvature(points: "np.ndarray", times: "np.ndarray", heading: "np.ndarray", centre: "np.ndarray", kappa: "np.ndarray", width: "np.ndarray", time_shape: "np.ndarray", time_rate: "np.ndarray", speed_shape: "np.ndarray", speed_rate: "np.ndarray", coefficients: "np.ndarray", initial: "np.ndarray") -> float:
    dt,s,phi,d,h=construct_event_steps(points,times,heading,centre)
    angle,score=carved_turn_log_density(phi,kappa,width)
    gamma=gamma_step_log_density(dt,s,time_shape,time_rate,speed_shape,speed_rate)
    G,D=event_transition_matrices(d,h,coefficients)
    joint=joint_event_emissions(angle,gamma)
    scaled_event_forward(joint,G,initial)
    return mixed_event_curvature(joint,score,G,D,initial)
SCICODE_GOLD_EOF
