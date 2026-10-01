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
from scipy.fft import dst as _dst

def _grid():
    n=1024; dr=0.02; r=(np.arange(n)+1)*dr; k=np.pi*(np.arange(n)+1)/((n+1)*dr)
    return r,k,dr
def _tz(y,x):
    y=np.asarray(y,dtype=np.float64); x=np.asarray(x,dtype=np.float64)
    return float(np.sum((y[1:]+y[:-1])*(x[1:]-x[:-1]))*0.5)
def _ft3(f):
    r,k,dr=_grid(); return 4.0*np.pi*_dst(r*f,type=1)*dr/(2.0*k)
def _ift3(F):
    r,k,dr=_grid(); return _dst(k*F,type=1)*(k[0]/(2.0*np.pi**2))/(2.0*r)
def _oz(bphi, rho, closure, n_iter, mix=0.15):
    b=np.clip(np.asarray(bphi,dtype=np.float64),-60.0,60.0); e=np.exp(-b); gam=np.zeros(b.size)
    def _cf(g): return (1.0+g)*(e-1.0) if closure=="PY" else np.exp(-b+g)-g-1.0
    for _ in range(int(n_iter)):
        c=_cf(gam); ck=_ft3(c); hk=ck/(1.0-rho*ck); gam=(1.0-mix)*gam+mix*_ift3(hk-ck)
    return 1.0+gam+_cf(gam)
def _hs_ck(k, sigma, rho):
    eta=np.pi*rho*sigma**3/6.0; q=k*sigma
    a=(1.0+2.0*eta)**2/(1.0-eta)**4; b=-6.0*eta*(1.0+0.5*eta)**2/(1.0-eta)**4; c=0.5*eta*a
    sq=np.sin(q); cq=np.cos(q)
    t=a*q**3*(sq-q*cq)+b*q**2*(2.0*q*sq+(2.0-q*q)*cq-2.0)+c*((4.0*q**3-24.0*q)*sq-(q**4-12.0*q*q+24.0)*cq+24.0)
    return -4.0*np.pi*sigma**3*t/q**6
def _hs_rdf(sigma, rho):
    r,k,dr=_grid(); ck=_hs_ck(k,sigma,rho); hk=ck/(1.0-rho*ck)
    return 1.0+_ift3(hk)
def _pos(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0.0: raise ValueError("bad "+name)
    return v
def _arr(x,name):
    a=np.asarray(x,dtype=np.float64).ravel()
    if a.size!=1024 or not np.all(np.isfinite(a)): raise ValueError("bad "+name)
    return a

def stf_target_rdf(rho: float, beta_eps: float, rc: float, n_iter: int) -> "np.ndarray":
    p=_pos(rho,"rho"); be=_pos(beta_eps,"beta_eps"); c=_pos(rc,"rc")
    ni=float(n_iter)
    if not np.isfinite(ni) or ni<=0 or abs(ni-round(ni))>0: raise ValueError("bad n_iter")
    r,_,_=_grid()
    u=4.0*be*((1.0/r)**12-(1.0/r)**6); uc=4.0*be*((1.0/c)**12-(1.0/c)**6)
    bphi=np.where(r<c,u-uc,0.0)
    return np.maximum(_oz(bphi,p,"HNC",int(round(ni))),0.0)

import numpy as np
from scipy.fft import dst as _dst

def _grid():
    n=1024; dr=0.02; r=(np.arange(n)+1)*dr; k=np.pi*(np.arange(n)+1)/((n+1)*dr)
    return r,k,dr
def _tz(y,x):
    y=np.asarray(y,dtype=np.float64); x=np.asarray(x,dtype=np.float64)
    return float(np.sum((y[1:]+y[:-1])*(x[1:]-x[:-1]))*0.5)
def _ft3(f):
    r,k,dr=_grid(); return 4.0*np.pi*_dst(r*f,type=1)*dr/(2.0*k)
def _ift3(F):
    r,k,dr=_grid(); return _dst(k*F,type=1)*(k[0]/(2.0*np.pi**2))/(2.0*r)
def _oz(bphi, rho, closure, n_iter, mix=0.15):
    b=np.clip(np.asarray(bphi,dtype=np.float64),-60.0,60.0); e=np.exp(-b); gam=np.zeros(b.size)
    def _cf(g): return (1.0+g)*(e-1.0) if closure=="PY" else np.exp(-b+g)-g-1.0
    for _ in range(int(n_iter)):
        c=_cf(gam); ck=_ft3(c); hk=ck/(1.0-rho*ck); gam=(1.0-mix)*gam+mix*_ift3(hk-ck)
    return 1.0+gam+_cf(gam)
def _hs_ck(k, sigma, rho):
    eta=np.pi*rho*sigma**3/6.0; q=k*sigma
    a=(1.0+2.0*eta)**2/(1.0-eta)**4; b=-6.0*eta*(1.0+0.5*eta)**2/(1.0-eta)**4; c=0.5*eta*a
    sq=np.sin(q); cq=np.cos(q)
    t=a*q**3*(sq-q*cq)+b*q**2*(2.0*q*sq+(2.0-q*q)*cq-2.0)+c*((4.0*q**3-24.0*q)*sq-(q**4-12.0*q*q+24.0)*cq+24.0)
    return -4.0*np.pi*sigma**3*t/q**6
def _hs_rdf(sigma, rho):
    r,k,dr=_grid(); ck=_hs_ck(k,sigma,rho); hk=ck/(1.0-rho*ck)
    return 1.0+_ift3(hk)
def _pos(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0.0: raise ValueError("bad "+name)
    return v
def _arr(x,name):
    a=np.asarray(x,dtype=np.float64).ravel()
    if a.size!=1024 or not np.all(np.isfinite(a)): raise ValueError("bad "+name)
    return a

def stf_ibi_potential(g_target: "np.ndarray", rho: float, gamma: float, n_ibi: int) -> "np.ndarray":
    g=_arr(g_target,"g_target"); p=_pos(rho,"rho"); gm=_pos(gamma,"gamma")
    ni=float(n_ibi)
    if not np.isfinite(ni) or ni<=0 or abs(ni-round(ni))>0: raise ValueError("bad n_ibi")
    gt=np.maximum(g,1e-10); core=g<1e-6
    bphi=np.where(core,60.0,-np.log(gt)); gam=np.zeros(g.size)
    for _ in range(int(round(ni))):
        e=np.exp(-np.clip(bphi,-60.0,60.0))
        for _ in range(80):
            c=(1.0+gam)*(e-1.0); ck=_ft3(c); hk=ck/(1.0-p*ck); gam=0.85*gam+0.15*_ift3(hk-ck)
        gn=np.maximum(1.0+gam+(1.0+gam)*(e-1.0),1e-10)
        bphi=bphi+gm*np.log(gn/gt); bphi=np.where(core,60.0,bphi)
    return bphi

import numpy as np
from scipy.fft import dst as _dst

def _grid():
    n=1024; dr=0.02; r=(np.arange(n)+1)*dr; k=np.pi*(np.arange(n)+1)/((n+1)*dr)
    return r,k,dr
def _tz(y,x):
    y=np.asarray(y,dtype=np.float64); x=np.asarray(x,dtype=np.float64)
    return float(np.sum((y[1:]+y[:-1])*(x[1:]-x[:-1]))*0.5)
def _ft3(f):
    r,k,dr=_grid(); return 4.0*np.pi*_dst(r*f,type=1)*dr/(2.0*k)
def _ift3(F):
    r,k,dr=_grid(); return _dst(k*F,type=1)*(k[0]/(2.0*np.pi**2))/(2.0*r)
def _oz(bphi, rho, closure, n_iter, mix=0.15):
    b=np.clip(np.asarray(bphi,dtype=np.float64),-60.0,60.0); e=np.exp(-b); gam=np.zeros(b.size)
    def _cf(g): return (1.0+g)*(e-1.0) if closure=="PY" else np.exp(-b+g)-g-1.0
    for _ in range(int(n_iter)):
        c=_cf(gam); ck=_ft3(c); hk=ck/(1.0-rho*ck); gam=(1.0-mix)*gam+mix*_ift3(hk-ck)
    return 1.0+gam+_cf(gam)
def _hs_ck(k, sigma, rho):
    eta=np.pi*rho*sigma**3/6.0; q=k*sigma
    a=(1.0+2.0*eta)**2/(1.0-eta)**4; b=-6.0*eta*(1.0+0.5*eta)**2/(1.0-eta)**4; c=0.5*eta*a
    sq=np.sin(q); cq=np.cos(q)
    t=a*q**3*(sq-q*cq)+b*q**2*(2.0*q*sq+(2.0-q*q)*cq-2.0)+c*((4.0*q**3-24.0*q)*sq-(q**4-12.0*q*q+24.0)*cq+24.0)
    return -4.0*np.pi*sigma**3*t/q**6
def _hs_rdf(sigma, rho):
    r,k,dr=_grid(); ck=_hs_ck(k,sigma,rho); hk=ck/(1.0-rho*ck)
    return 1.0+_ift3(hk)
def _pos(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0.0: raise ValueError("bad "+name)
    return v
def _arr(x,name):
    a=np.asarray(x,dtype=np.float64).ravel()
    if a.size!=1024 or not np.all(np.isfinite(a)): raise ValueError("bad "+name)
    return a

def stf_wca_split(bphi: "np.ndarray", g_target: "np.ndarray") -> "np.ndarray":
    if np.asarray(bphi).size!=1024 or np.asarray(g_target).size!=1024: raise ValueError("arrays must have length 1024")
    b=_arr(bphi,"bphi"); g=_arr(g_target,"g_target"); r,_,_=_grid()
    core=g<1e-6; iR=int(np.argmin(np.where(core,np.inf,b))); Rm=r[iR]; bR=b[iR]
    bref=np.where(r<Rm,b-bR,0.0); btail=np.where(r<Rm,bR,b)
    return np.vstack([bref,btail]).astype(np.float64)

import numpy as np
from scipy.fft import dst as _dst

def _grid():
    n=1024; dr=0.02; r=(np.arange(n)+1)*dr; k=np.pi*(np.arange(n)+1)/((n+1)*dr)
    return r,k,dr
def _tz(y,x):
    y=np.asarray(y,dtype=np.float64); x=np.asarray(x,dtype=np.float64)
    return float(np.sum((y[1:]+y[:-1])*(x[1:]-x[:-1]))*0.5)
def _ft3(f):
    r,k,dr=_grid(); return 4.0*np.pi*_dst(r*f,type=1)*dr/(2.0*k)
def _ift3(F):
    r,k,dr=_grid(); return _dst(k*F,type=1)*(k[0]/(2.0*np.pi**2))/(2.0*r)
def _oz(bphi, rho, closure, n_iter, mix=0.15):
    b=np.clip(np.asarray(bphi,dtype=np.float64),-60.0,60.0); e=np.exp(-b); gam=np.zeros(b.size)
    def _cf(g): return (1.0+g)*(e-1.0) if closure=="PY" else np.exp(-b+g)-g-1.0
    for _ in range(int(n_iter)):
        c=_cf(gam); ck=_ft3(c); hk=ck/(1.0-rho*ck); gam=(1.0-mix)*gam+mix*_ift3(hk-ck)
    return 1.0+gam+_cf(gam)
def _hs_ck(k, sigma, rho):
    eta=np.pi*rho*sigma**3/6.0; q=k*sigma
    a=(1.0+2.0*eta)**2/(1.0-eta)**4; b=-6.0*eta*(1.0+0.5*eta)**2/(1.0-eta)**4; c=0.5*eta*a
    sq=np.sin(q); cq=np.cos(q)
    t=a*q**3*(sq-q*cq)+b*q**2*(2.0*q*sq+(2.0-q*q)*cq-2.0)+c*((4.0*q**3-24.0*q)*sq-(q**4-12.0*q*q+24.0)*cq+24.0)
    return -4.0*np.pi*sigma**3*t/q**6
def _hs_rdf(sigma, rho):
    r,k,dr=_grid(); ck=_hs_ck(k,sigma,rho); hk=ck/(1.0-rho*ck)
    return 1.0+_ift3(hk)
def _pos(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0.0: raise ValueError("bad "+name)
    return v
def _arr(x,name):
    a=np.asarray(x,dtype=np.float64).ravel()
    if a.size!=1024 or not np.all(np.isfinite(a)): raise ValueError("bad "+name)
    return a

def stf_reference_rdf(bphi_ref: "np.ndarray", rho: float, n_iter: int) -> "np.ndarray":
    b=_arr(bphi_ref,"bphi_ref"); p=_pos(rho,"rho"); ni=float(n_iter)
    if not np.isfinite(ni) or ni<=0 or abs(ni-round(ni))>0: raise ValueError("bad n_iter")
    return _oz(b,p,"PY",int(round(ni)))

import numpy as np
from scipy.fft import dst as _dst

def _grid():
    n=1024; dr=0.02; r=(np.arange(n)+1)*dr; k=np.pi*(np.arange(n)+1)/((n+1)*dr)
    return r,k,dr
def _tz(y,x):
    y=np.asarray(y,dtype=np.float64); x=np.asarray(x,dtype=np.float64)
    return float(np.sum((y[1:]+y[:-1])*(x[1:]-x[:-1]))*0.5)
def _ft3(f):
    r,k,dr=_grid(); return 4.0*np.pi*_dst(r*f,type=1)*dr/(2.0*k)
def _ift3(F):
    r,k,dr=_grid(); return _dst(k*F,type=1)*(k[0]/(2.0*np.pi**2))/(2.0*r)
def _oz(bphi, rho, closure, n_iter, mix=0.15):
    b=np.clip(np.asarray(bphi,dtype=np.float64),-60.0,60.0); e=np.exp(-b); gam=np.zeros(b.size)
    def _cf(g): return (1.0+g)*(e-1.0) if closure=="PY" else np.exp(-b+g)-g-1.0
    for _ in range(int(n_iter)):
        c=_cf(gam); ck=_ft3(c); hk=ck/(1.0-rho*ck); gam=(1.0-mix)*gam+mix*_ift3(hk-ck)
    return 1.0+gam+_cf(gam)
def _hs_ck(k, sigma, rho):
    eta=np.pi*rho*sigma**3/6.0; q=k*sigma
    a=(1.0+2.0*eta)**2/(1.0-eta)**4; b=-6.0*eta*(1.0+0.5*eta)**2/(1.0-eta)**4; c=0.5*eta*a
    sq=np.sin(q); cq=np.cos(q)
    t=a*q**3*(sq-q*cq)+b*q**2*(2.0*q*sq+(2.0-q*q)*cq-2.0)+c*((4.0*q**3-24.0*q)*sq-(q**4-12.0*q*q+24.0)*cq+24.0)
    return -4.0*np.pi*sigma**3*t/q**6
def _hs_rdf(sigma, rho):
    r,k,dr=_grid(); ck=_hs_ck(k,sigma,rho); hk=ck/(1.0-rho*ck)
    return 1.0+_ift3(hk)
def _pos(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0.0: raise ValueError("bad "+name)
    return v
def _arr(x,name):
    a=np.asarray(x,dtype=np.float64).ravel()
    if a.size!=1024 or not np.all(np.isfinite(a)): raise ValueError("bad "+name)
    return a

def stf_hs_diameter(g_ref: "np.ndarray", g_target: "np.ndarray", bphi_ref: "np.ndarray", rho: float) -> "np.ndarray":
    if not np.isfinite(float(rho)) or float(rho)<=0.0: raise ValueError("bad rho")
    gr=_arr(g_ref,"g_ref"); gt=_arr(g_target,"g_target"); br=_arr(bphi_ref,"bphi_ref"); p=_pos(rho,"rho")
    r,_,_=_grid(); n=r.size
    pk=[i for i in range(2,n-2) if gt[i]>gt[i-1] and gt[i]>gt[i+1] and gt[i]>1.02]
    if len(pk)>1:
        rp=float(np.mean(np.diff(r[pk]))); rc=float(r[pk[-1]]+rp)
    else:
        rc=float(2.0*r[pk[0]])
    m=r<=rc
    def _D(s):
        gh=_hs_rdf(s,p)
        return _tz((gr[m]-gh[m])**2,r[m])
    grid=np.linspace(0.86,1.04,19); vals=[_D(s) for s in grid]; i=int(np.argmin(vals))
    a=grid[max(i-1,0)]; b=grid[min(i+1,18)]; phi=(np.sqrt(5.0)-1.0)/2.0
    x1=b-phi*(b-a); x2=a+phi*(b-a); f1=_D(x1); f2=_D(x2)
    for _ in range(12):
        if f1<f2: b,x2,f2=x2,x1,f1; x1=b-phi*(b-a); f1=_D(x1)
        else:     a,x1,f1=x1,x2,f2; x2=a+phi*(b-a); f2=_D(x2)
    s_opt=float((a+b)/2.0)
    f=1.0-np.exp(-np.clip(br,-60.0,60.0))
    s_bh=_tz(np.concatenate(([1.0],f)),np.concatenate(([0.0],r)))
    return np.array([s_opt,rc,s_bh],dtype=np.float64)

import numpy as np
from scipy.fft import dst as _dst

def _grid():
    n=1024; dr=0.02; r=(np.arange(n)+1)*dr; k=np.pi*(np.arange(n)+1)/((n+1)*dr)
    return r,k,dr
def _tz(y,x):
    y=np.asarray(y,dtype=np.float64); x=np.asarray(x,dtype=np.float64)
    return float(np.sum((y[1:]+y[:-1])*(x[1:]-x[:-1]))*0.5)
def _ft3(f):
    r,k,dr=_grid(); return 4.0*np.pi*_dst(r*f,type=1)*dr/(2.0*k)
def _ift3(F):
    r,k,dr=_grid(); return _dst(k*F,type=1)*(k[0]/(2.0*np.pi**2))/(2.0*r)
def _oz(bphi, rho, closure, n_iter, mix=0.15):
    b=np.clip(np.asarray(bphi,dtype=np.float64),-60.0,60.0); e=np.exp(-b); gam=np.zeros(b.size)
    def _cf(g): return (1.0+g)*(e-1.0) if closure=="PY" else np.exp(-b+g)-g-1.0
    for _ in range(int(n_iter)):
        c=_cf(gam); ck=_ft3(c); hk=ck/(1.0-rho*ck); gam=(1.0-mix)*gam+mix*_ift3(hk-ck)
    return 1.0+gam+_cf(gam)
def _hs_ck(k, sigma, rho):
    eta=np.pi*rho*sigma**3/6.0; q=k*sigma
    a=(1.0+2.0*eta)**2/(1.0-eta)**4; b=-6.0*eta*(1.0+0.5*eta)**2/(1.0-eta)**4; c=0.5*eta*a
    sq=np.sin(q); cq=np.cos(q)
    t=a*q**3*(sq-q*cq)+b*q**2*(2.0*q*sq+(2.0-q*q)*cq-2.0)+c*((4.0*q**3-24.0*q)*sq-(q**4-12.0*q*q+24.0)*cq+24.0)
    return -4.0*np.pi*sigma**3*t/q**6
def _hs_rdf(sigma, rho):
    r,k,dr=_grid(); ck=_hs_ck(k,sigma,rho); hk=ck/(1.0-rho*ck)
    return 1.0+_ift3(hk)
def _pos(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0.0: raise ValueError("bad "+name)
    return v
def _arr(x,name):
    a=np.asarray(x,dtype=np.float64).ravel()
    if a.size!=1024 or not np.all(np.isfinite(a)): raise ValueError("bad "+name)
    return a

def stf_integrated_strength(g_target: "np.ndarray", bphi_ref: "np.ndarray", bphi_tail: "np.ndarray") -> "np.ndarray":
    if np.asarray(g_target).size!=1024: raise ValueError("g_target must have length 1024")
    g=_arr(g_target,"g_target"); br=_arr(bphi_ref,"bphi_ref"); bt=_arr(bphi_tail,"bphi_tail")
    r,_,_=_grid()
    a_ti=4.0*np.pi*_tz(r*r*g*bt,r)
    ef=np.exp(-np.clip(br+bt,-60.0,60.0)); er=np.exp(-np.clip(br,-60.0,60.0))
    b2=-2.0*np.pi*_tz((ef-1.0)*r*r,r); b2r=-2.0*np.pi*_tz((er-1.0)*r*r,r)
    return np.array([a_ti,2.0*(b2-b2r)],dtype=np.float64)

import numpy as np
from scipy.fft import dst as _dst

def _grid():
    n=1024; dr=0.02; r=(np.arange(n)+1)*dr; k=np.pi*(np.arange(n)+1)/((n+1)*dr)
    return r,k,dr
def _tz(y,x):
    y=np.asarray(y,dtype=np.float64); x=np.asarray(x,dtype=np.float64)
    return float(np.sum((y[1:]+y[:-1])*(x[1:]-x[:-1]))*0.5)
def _ft3(f):
    r,k,dr=_grid(); return 4.0*np.pi*_dst(r*f,type=1)*dr/(2.0*k)
def _ift3(F):
    r,k,dr=_grid(); return _dst(k*F,type=1)*(k[0]/(2.0*np.pi**2))/(2.0*r)
def _oz(bphi, rho, closure, n_iter, mix=0.15):
    b=np.clip(np.asarray(bphi,dtype=np.float64),-60.0,60.0); e=np.exp(-b); gam=np.zeros(b.size)
    def _cf(g): return (1.0+g)*(e-1.0) if closure=="PY" else np.exp(-b+g)-g-1.0
    for _ in range(int(n_iter)):
        c=_cf(gam); ck=_ft3(c); hk=ck/(1.0-rho*ck); gam=(1.0-mix)*gam+mix*_ift3(hk-ck)
    return 1.0+gam+_cf(gam)
def _hs_ck(k, sigma, rho):
    eta=np.pi*rho*sigma**3/6.0; q=k*sigma
    a=(1.0+2.0*eta)**2/(1.0-eta)**4; b=-6.0*eta*(1.0+0.5*eta)**2/(1.0-eta)**4; c=0.5*eta*a
    sq=np.sin(q); cq=np.cos(q)
    t=a*q**3*(sq-q*cq)+b*q**2*(2.0*q*sq+(2.0-q*q)*cq-2.0)+c*((4.0*q**3-24.0*q)*sq-(q**4-12.0*q*q+24.0)*cq+24.0)
    return -4.0*np.pi*sigma**3*t/q**6
def _hs_rdf(sigma, rho):
    r,k,dr=_grid(); ck=_hs_ck(k,sigma,rho); hk=ck/(1.0-rho*ck)
    return 1.0+_ift3(hk)
def _pos(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0.0: raise ValueError("bad "+name)
    return v
def _arr(x,name):
    a=np.asarray(x,dtype=np.float64).ravel()
    if a.size!=1024 or not np.all(np.isfinite(a)): raise ValueError("bad "+name)
    return a

def stf_free_energy(rho: float, sigma: float, a_strength: float) -> "np.ndarray":
    p=_pos(rho,"rho"); s=_pos(sigma,"sigma"); a=float(a_strength)
    if not np.isfinite(a): raise ValueError("bad a_strength")
    eta=np.pi*p*s**3/6.0
    if eta>=1.0: raise ValueError("packing fraction out of range")
    f_cs=eta*(4.0-3.0*eta)/(1.0-eta)**2
    z_cs=(1.0+eta+eta**2-eta**3)/(1.0-eta)**3
    return np.array([f_cs+0.5*p*a,f_cs,z_cs],dtype=np.float64)

import numpy as np
from scipy.fft import dst as _dst

def _grid():
    n=1024; dr=0.02; r=(np.arange(n)+1)*dr; k=np.pi*(np.arange(n)+1)/((n+1)*dr)
    return r,k,dr
def _tz(y,x):
    y=np.asarray(y,dtype=np.float64); x=np.asarray(x,dtype=np.float64)
    return float(np.sum((y[1:]+y[:-1])*(x[1:]-x[:-1]))*0.5)
def _ft3(f):
    r,k,dr=_grid(); return 4.0*np.pi*_dst(r*f,type=1)*dr/(2.0*k)
def _ift3(F):
    r,k,dr=_grid(); return _dst(k*F,type=1)*(k[0]/(2.0*np.pi**2))/(2.0*r)
def _oz(bphi, rho, closure, n_iter, mix=0.15):
    b=np.clip(np.asarray(bphi,dtype=np.float64),-60.0,60.0); e=np.exp(-b); gam=np.zeros(b.size)
    def _cf(g): return (1.0+g)*(e-1.0) if closure=="PY" else np.exp(-b+g)-g-1.0
    for _ in range(int(n_iter)):
        c=_cf(gam); ck=_ft3(c); hk=ck/(1.0-rho*ck); gam=(1.0-mix)*gam+mix*_ift3(hk-ck)
    return 1.0+gam+_cf(gam)
def _hs_ck(k, sigma, rho):
    eta=np.pi*rho*sigma**3/6.0; q=k*sigma
    a=(1.0+2.0*eta)**2/(1.0-eta)**4; b=-6.0*eta*(1.0+0.5*eta)**2/(1.0-eta)**4; c=0.5*eta*a
    sq=np.sin(q); cq=np.cos(q)
    t=a*q**3*(sq-q*cq)+b*q**2*(2.0*q*sq+(2.0-q*q)*cq-2.0)+c*((4.0*q**3-24.0*q)*sq-(q**4-12.0*q*q+24.0)*cq+24.0)
    return -4.0*np.pi*sigma**3*t/q**6
def _hs_rdf(sigma, rho):
    r,k,dr=_grid(); ck=_hs_ck(k,sigma,rho); hk=ck/(1.0-rho*ck)
    return 1.0+_ift3(hk)
def _pos(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0.0: raise ValueError("bad "+name)
    return v
def _arr(x,name):
    a=np.asarray(x,dtype=np.float64).ravel()
    if a.size!=1024 or not np.all(np.isfinite(a)): raise ValueError("bad "+name)
    return a

def stf_audit(rho: float, beta_eps: float, rc: float, gamma: float, n_ibi: int) -> "np.ndarray":
    for v,nm in ((rho,"rho"),(beta_eps,"beta_eps"),(rc,"rc"),(gamma,"gamma")):
        if not np.isfinite(float(v)) or float(v)<=0.0: raise ValueError("bad "+nm)
    r,_,_=_grid()
    g=stf_target_rdf(rho,beta_eps,rc,400)
    b=stf_ibi_potential(g,rho,gamma,n_ibi)
    sp=stf_wca_split(b,g)
    Rm=float(r[int(np.argmax(sp[0]==0.0))])
    gr=stf_reference_rdf(sp[0],rho,300)
    d=stf_hs_diameter(gr,g,sp[0],rho)
    A=stf_integrated_strength(g,sp[0],sp[1])
    f_ti=stf_free_energy(rho,d[0],A[0]); f_b2=stf_free_energy(rho,d[0],A[1])
    return np.array([Rm,d[1],d[0],d[2],A[0],A[1],f_ti[1],f_ti[0],f_b2[0]],dtype=np.float64)
SCICODE_GOLD_EOF
