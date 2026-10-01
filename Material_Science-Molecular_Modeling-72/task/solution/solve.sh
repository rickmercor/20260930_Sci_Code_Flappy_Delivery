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

def simulate_biased_paths(initial: "np.ndarray", bias: "np.ndarray", noise: "np.ndarray", dt: float, friction: float, mass: float, thermal: float, barrier: float, tilt: float) -> "np.ndarray":
    e,n=noise.shape
    out=np.empty((e,n,6))
    q,p=initial[:,0].copy(),initial[:,1].copy()
    a=np.exp(-friction*dt)
    sigma=np.sqrt(thermal*mass*(-np.expm1(-2*friction*dt)))
    for t in range(n):
        mid=q+dt*p/(2*mass)
        center=bias[:,1]+bias[:,2]*np.sin(bias[:,3]*(t+0.5)*dt+bias[:,4])
        bg=bias[:,0]*(mid-center)
        grad=4*barrier*mid*(mid**2-1)+tilt+bg
        newp=a*(p-0.5*dt*grad)+sigma*noise[:,t]-0.5*dt*grad
        newq=mid+dt*newp/(2*mass)
        out[:,t,:]=np.column_stack([q,newq,mid,bg,noise[:,t],newp])
        q,p=newq,newp
    return out

import numpy as np

def _perturbation_gradient(q, width=0.4, tilt=0.18):
    return -q/width**2*np.exp(-0.5*(q/width)**2)+tilt

def compute_path_actions(records: "np.ndarray", alpha: float, width: float, htilt: float, dt: float, friction: float, mass: float, thermal: float) -> "np.ndarray":
    q=records[:,:,2]; bg=records[:,:,3]; eta=records[:,:,4]
    coeff=0.5*dt*(1+np.exp(-friction*dt))/np.sqrt(thermal*mass*(-np.expm1(-2*friction*dt)))
    delta=coeff*(alpha*_perturbation_gradient(q,width,htilt)-bg)
    delta_prime=coeff*_perturbation_gradient(q,width,htilt)
    ell=-eta*delta-0.5*delta**2
    dell=-(eta+delta)*delta_prime
    return np.stack([ell,dell],axis=-1)

import numpy as np

def accumulate_path_counts(records: "np.ndarray", action: "np.ndarray", cuts: "np.ndarray", lag: int) -> "np.ndarray":
    k=len(cuts)+1
    C=np.zeros((k,k));dC=C.copy()
    logs=np.concatenate([np.zeros((len(records),1,2)),np.cumsum(action,axis=1)],axis=1)
    windows=logs[:,lag:]-logs[:,:-lag]
    start=np.searchsorted(cuts,records[:,:-lag+1 if lag>1 else None,0],side='right')
    end=np.searchsorted(cuts,records[:,lag-1:,1],side='right')
    weight=np.exp(windows[:,:,0])
    np.add.at(C,(start.ravel(),end.ravel()),weight.ravel())
    np.add.at(dC,(start.ravel(),end.ravel()),(weight*windows[:,:,1]).ravel())
    return np.stack([C,dC])

import numpy as np
from scipy.integrate import quad

def _perturbation_value(q, width=0.4, tilt=0.18):
    return np.exp(-0.5*(q/width)**2)+tilt*q

def compute_equilibrium_masses(cuts: "np.ndarray", alpha: float, width: float, htilt: float, thermal: float, barrier: float, tilt: float) -> "np.ndarray":
    edges=np.r_[-np.inf,cuts,np.inf]
    def _weight(q):
        return np.exp(-(barrier*(q*q-1)**2+tilt*q+alpha*_perturbation_value(q,width,htilt))/thermal)
    z=np.array([quad(_weight,a,b,epsabs=2e-12,epsrel=2e-12)[0] for a,b in zip(edges[:-1],edges[1:])])
    dz=np.array([quad(lambda q: -_perturbation_value(q,width,htilt)*_weight(q)/thermal,a,b,epsabs=2e-12,epsrel=2e-12)[0] for a,b in zip(edges[:-1],edges[1:])])
    pi=z/z.sum();dpi=dz/z.sum()-pi*dz.sum()/z.sum()
    return np.stack([pi,dpi])

import numpy as np
from scipy.optimize import root

def fit_stationary_flux(C: "np.ndarray", pi: "np.ndarray") -> "np.ndarray":
    scale=C.sum();C=C/scale
    S=C+C.T
    def _fun(z):
        lam=np.exp(z); den=lam[:,None]+lam[None,:]
        return np.sum(S/den,axis=1)/pi-1
    def _jac(z):
        lam=np.exp(z);den=lam[:,None]+lam[None,:]
        R=S/den**2
        return -(np.diag(R.sum(axis=1))+R)*lam[None,:]/pi[:,None]
    solved=root(_fun,np.log(C.sum(axis=1)/pi),jac=_jac,tol=1e-11)
    lam=np.exp(solved.x)
    X=S/(lam[:,None]+lam[None,:])
    if np.max(np.abs(X.sum(axis=1)-pi))>2e-11:raise RuntimeError('Bad constraint')
    return X

import numpy as np

def differentiate_stationary_flux(C: "np.ndarray", dC: "np.ndarray", pi: "np.ndarray", dpi: "np.ndarray", X: "np.ndarray") -> "np.ndarray":
    # The same fixed normalization is applied to C and its derivative.
    scale=C.sum(); C=C/scale;dC=dC/scale
    lam=np.diag(C)/np.diag(X)
    den=lam[:,None]+lam[None,:]
    S=C+C.T; dS=dC+dC.T
    R=S/den**2
    H=np.diag(R.sum(axis=1))+R
    rhs=np.sum(dS/den,axis=1)-dpi
    dlam=np.linalg.solve(H,rhs)
    return dS/den-R*(dlam[:,None]+dlam[None,:])

import numpy as np

def assemble_transition_response(X: "np.ndarray", dX: "np.ndarray", pi: "np.ndarray", dpi: "np.ndarray") -> "np.ndarray":
    P=X/pi[:,None]
    dP=dX/pi[:,None]-P*(dpi/pi)[:,None]
    return np.stack([P,dP])

import numpy as np

def compute_relaxation_response(op: "np.ndarray", pi: "np.ndarray", lagtime: float) -> "np.ndarray":
    P,dP=op
    sp=np.sqrt(pi)
    S=sp[:,None]*P/sp[None,:]
    vals,vecs=np.linalg.eigh((S+S.T)/2)
    lam=vals[-2];u=vecs[:,-2]
    dlam=u@(sp[:,None]*dP/sp[None,:])@u
    return np.array([lam,-lagtime/np.log(lam),-dlam/(lam*np.log(lam))])

import numpy as np

def infer_kinetic_sensitivity(initial: "np.ndarray", bias: "np.ndarray", noise: "np.ndarray", cuts: "np.ndarray", alpha: float, dt: float, friction: float, mass: float, thermal: float, barrier: float, tilt: float, width: float, htilt: float, lag: int) -> float:
    records = simulate_biased_paths(initial, bias, noise, dt, friction, mass, thermal, barrier, tilt)
    action = compute_path_actions(records, alpha, width, htilt, dt, friction, mass, thermal)
    C, dC = accumulate_path_counts(records, action, cuts, lag)
    pi, dpi = compute_equilibrium_masses(cuts, alpha, width, htilt, thermal, barrier, tilt)
    X = fit_stationary_flux(C, pi)
    dX = differentiate_stationary_flux(C, dC, pi, dpi, X)
    op = assemble_transition_response(X, dX, pi, dpi)
    result = compute_relaxation_response(op, pi, lag * dt)
    return float(result[2])
SCICODE_GOLD_EOF
