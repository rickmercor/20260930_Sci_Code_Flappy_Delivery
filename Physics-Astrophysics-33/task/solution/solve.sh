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
from scipy.optimize import brentq

def stationary_density(b: np.ndarray, u: np.ndarray, mean_density: float) -> np.ndarray:
    b,u,m=np.asarray(b,dtype=float),np.asarray(u,dtype=float),np.asarray(mean_density,dtype=float)
    if b.ndim!=1 or not b.size or u.shape!=b.shape or m.ndim:
        raise ValueError('invalid shapes')
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(u)) or not np.isfinite(m):
        raise ValueError('nonfinite inputs')
    if np.any(b<=0) or np.any(u<0) or m<=0:
        raise ValueError('invalid positive state')
    # Integrating the zero flux gives b/sqrt(r)+u=C. The mean is strictly
    # decreasing in C on C>max(u), hence this root is unique.
    umax=float(np.max(u))
    lo=np.nextafter(umax,np.inf)
    hi=umax+float(np.max(b)/np.sqrt(m))+1.
    with np.errstate(over='ignore', divide='ignore'):
        c=brentq(lambda s: np.mean((b/(s-u))**2)-m,lo,hi,xtol=2e-14,rtol=1e-14)
        out=np.r_[(b/(c-u))**2,c]
    if not np.all(np.isfinite(out)) or np.any(out<=0):
        raise ValueError('invalid equilibrium')
    return out

import numpy as np

def stationary_pressure(rho: np.ndarray, mean_pressure: float, gamma: float) -> np.ndarray:
    r,m,g=np.asarray(rho,dtype=float),np.asarray(mean_pressure,dtype=float),np.asarray(gamma,dtype=float)
    if r.ndim!=1 or not r.size or m.ndim or g.ndim:
        raise ValueError('invalid shapes')
    if not np.all(np.isfinite(r)) or not np.isfinite(m) or not np.isfinite(g):
        raise ValueError('nonfinite input')
    if np.any(r<=0) or m<=0 or g<=1: raise ValueError('invalid thermodynamic state')
    logs=g*np.log(r)
    weights=np.exp(logs-np.max(logs))
    out=float(m)*weights/np.mean(weights)
    if not np.all(np.isfinite(out)) or np.any(out<=0): raise ValueError('invalid pressure')
    return out

import numpy as np

def transport_tangent(background: np.ndarray, rho: np.ndarray, pressure: np.ndarray,
                              gamma: float, chi: float) -> np.ndarray:
    bg=np.asarray(background,dtype=float)
    r,p=np.asarray(rho,dtype=float),np.asarray(pressure,dtype=float)
    g,ch=np.asarray(gamma,dtype=float),np.asarray(chi,dtype=float)
    if bg.ndim!=2 or bg.shape[1]!=8 or not len(bg) or r.shape!=(len(bg),) or p.shape!=r.shape or g.ndim or ch.ndim:
        raise ValueError('invalid shapes')
    if not all(np.all(np.isfinite(v)) for v in [bg,r,p,g,ch]): raise ValueError('nonfinite input')
    if np.any(bg[:,[0,3]]<=0) or np.any(bg[:,[1,2]]<0) or np.any(r<=0) or np.any(p<=0) or g<=1 or ch<=0:
        raise ValueError('invalid physical state')
    va=bg[:,0]/np.sqrt(r)
    eta=bg[:,2]**2*bg[:,3]/(4*va*ch)
    a=2*bg[:,4:6]+3*bg[:,6:8]/va[:,None]
    h=2*g*(bg[:,4:6]+bg[:,6:8]/va[:,None])
    c=g*p[:,None]*bg[:,6:8]/(va*r)[:,None]
    out=np.column_stack((eta,a,h,c))
    if not np.all(np.isfinite(out)): raise ValueError('nonfinite output')
    return out

import numpy as np

def flux_generator(coefficients: np.ndarray, shape: tuple, periods: tuple) -> np.ndarray:
    cf=np.asarray(coefficients,dtype=float); dims=np.asarray(shape,dtype=float); lengths=np.asarray(periods,dtype=float)
    if dims.shape!=(2,) or lengths.shape!=(2,) or not np.all(np.isfinite(dims)) or not np.all(np.isfinite(lengths)):
        raise ValueError('invalid grid shape')
    if np.any(dims<3) or np.any(dims!=np.floor(dims)) or np.any(dims%2!=1) or np.any(lengths<=0):
        raise ValueError('invalid grid parameters')
    nx,ny=map(int,dims); n=nx*ny
    if cf.shape!=(n,7) or not np.all(np.isfinite(cf)) or np.any(cf[:,0]<0):
        raise ValueError('invalid coefficients')
    derivatives=[]
    for count,length in zip((nx,ny),lengths):
        k=2*np.pi*np.fft.fftfreq(count,d=length/count)
        derivatives.append(np.fft.ifft(1j*k[:,None]*np.fft.fft(np.eye(count),axis=0),axis=0).real)
    dx=np.kron(derivatives[0],np.eye(ny)); dy=np.kron(np.eye(nx),derivatives[1])
    rr=np.zeros((n,n)); pp=np.zeros((n,n)); pr=np.zeros((n,n))
    for axis,d in enumerate((dx,dy)):
        rr+=d@(cf[:,0,None]*(d-np.diag(cf[:,1+axis])))
        pp+=d@(cf[:,0,None]*(d-np.diag(cf[:,3+axis])))
        pr-=d@np.diag(cf[:,0]*cf[:,5+axis])
    out=np.block([[rr,np.zeros((n,n))],[pr,pp]])
    return out

import numpy as np
from scipy.linalg import expm

def pressure_response(generator: np.ndarray, time: float) -> np.ndarray:
    l=np.asarray(generator,dtype=float); t=np.asarray(time,dtype=float)
    if l.ndim!=2 or l.shape[0]!=l.shape[1] or len(l)<4 or len(l)%2 or t.ndim:
        raise ValueError('invalid generator shape')
    if not np.all(np.isfinite(l)) or not np.isfinite(t) or t<0:
        raise ValueError('invalid generator or time')
    n=len(l)//2
    p=np.eye(n)-np.ones((n,n))/n
    out=p@expm(float(t)*l)[n:,:n]@p
    if not np.all(np.isfinite(out)): raise ValueError('nonfinite response')
    return out

import numpy as np

def optimal_response(response: np.ndarray) -> float:
    r=np.asarray(response,dtype=float)
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or not len(r) or not np.all(np.isfinite(r)):
        raise ValueError('invalid response matrix')
    out=float(np.linalg.svd(r,compute_uv=False)[0])
    return out

import numpy as np

def pressure_susceptibility(params: dict | None = None) -> float:
    p=dict(nx=21,ny=23,period_x=2*np.pi,period_y=2*np.pi,time=1200.,
           field_x=.8,field_y=.65,field_mix=.4,flow_scale=1.,z_scale=.04,
           lpar_scale=.8,rho_mean=1.,p_mean=1.,chi=1.,gamma=5/3)
    if params is not None:
        if not isinstance(params,dict) or set(params)-set(p): raise ValueError('invalid parameters')
        p.update(params)
    for key,val in p.items():
        a=np.asarray(val,dtype=float)
        if a.ndim or not np.isfinite(a): raise ValueError('nonfinite or nonscalar parameter')
        p[key]=float(a)
    for key in ['nx','ny']:
        if p[key]!=int(p[key]) or p[key]<3 or int(p[key])%2!=1: raise ValueError('invalid grid count')
    if any(p[k]<=0 for k in ['period_x','period_y','lpar_scale','rho_mean','p_mean','chi']) or p['gamma']<=1:
        raise ValueError('invalid positive parameter')
    if any(p[k]<0 for k in ['time','flow_scale','z_scale']): raise ValueError('invalid nonnegative parameter')
    nx,ny=int(p['nx']),int(p['ny'])
    x,y=np.meshgrid(2*np.pi*np.arange(nx)/nx,2*np.pi*np.arange(ny)/ny,indexing='ij')
    sx,sy=2*np.pi/p['period_x'],2*np.pi/p['period_y']
    b=np.exp(p['field_x']*np.cos(x)+p['field_y']*np.sin(y)+p['field_mix']*np.cos(x+2*y+.3))
    u=p['flow_scale']*(.9+.3*np.sin(x+.4)+.2*np.cos(2*y)+.15*np.sin(x-y+.2))
    z=p['z_scale']*np.exp(.15*np.sin(2*x-y-.3))
    ell=p['lpar_scale']*(1+.12*np.cos(x+y+.2))
    kx=sx*(-p['field_x']*np.sin(x)-p['field_mix']*np.sin(x+2*y+.3))
    ky=sy*(p['field_y']*np.cos(y)-2*p['field_mix']*np.sin(x+2*y+.3))
    ux=sx*p['flow_scale']*(.3*np.cos(x+.4)+.15*np.cos(x-y+.2))
    uy=sy*p['flow_scale']*(-.4*np.sin(2*y)-.15*np.cos(x-y+.2))
    bg=np.column_stack([v.ravel() for v in (b,u,z,ell,kx,ky,ux,uy)])
    eq=stationary_density(bg[:,0],bg[:,1],p['rho_mean'])
    rho=eq[:-1]
    press=stationary_pressure(rho,p['p_mean'],p['gamma'])
    tangent=transport_tangent(bg,rho,press,p['gamma'],p['chi'])
    generator=flux_generator(tangent,(nx,ny),(p['period_x'],p['period_y']))
    response=pressure_response(generator,p['time'])*p['rho_mean']/p['p_mean']
    out=optimal_response(response)
    if not np.isfinite(out): raise ValueError('nonfinite susceptibility')
    return out
SCICODE_GOLD_EOF
