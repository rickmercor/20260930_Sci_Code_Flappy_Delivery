#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import itertools
import numpy as np

def _padd(p,q,scale=1.0):
    r=dict(p)
    for m,v in q.items(): r[m]=r.get(m,0.0)+scale*v
    return {m:v for m,v in r.items() if abs(v)>1e-15}

def _pmul(p,q):
    r={}
    for m,v in p.items():
        for n,w in q.items():
            e=tuple(m[i]+n[i] for i in range(3)); r[e]=r.get(e,0.0)+v*w
    return {m:v for m,v in r.items() if abs(v)>1e-15}

def triangle_normalized_baikov(s: float) -> "np.ndarray":
    s=float(s)
    if not np.isfinite(s) or s<=0: raise ValueError('s must be positive and finite')
    basis=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
    C=np.array([[0.,0.,s],[0.,0.,0.],[s,0.,0.]])
    z=[{(1,0,0):1.},{(0,1,0):1.},{(0,0,1):1.}]
    A=[[{} for _ in range(5)] for _ in range(5)]
    A[0]=[{},z[0],z[1],z[2],{(0,0,0):1.}]
    for i in range(3):
        A[i+1][0]=z[i]
        for j in range(3): A[i+1][j+1]={(0,0,0):float(C[i,j])} if C[i,j]!=0 else {}
        A[i+1][4]={(0,0,0):1.}
    A[4]=[{(0,0,0):1.},{(0,0,0):1.},{(0,0,0):1.},{(0,0,0):1.},{}]
    det={}
    for perm in itertools.permutations(range(5)):
        inv=sum(perm[i]>perm[j] for i in range(5) for j in range(i+1,5))
        term={(0,0,0):-1. if inv%2 else 1.}
        for i,j in enumerate(perm):
            term=_pmul(term,A[i][j])
            if not term: break
        det=_padd(det,term)
    Bhat={m:(4.0/s)*(v/8.0) for m,v in det.items()}
    return np.array([Bhat.get(m,0.0) for m in basis])

import numpy as np

def triangle_source_nonprincipal_syzygy(s: float) -> "np.ndarray":
    s=float(s)
    if not np.isfinite(s) or s<=0: raise ValueError('s must be positive and finite')
    m1=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    m2=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
    i1={m:i for i,m in enumerate(m1)}; i2={m:i for i,m in enumerate(m2)}
    c=np.zeros(43,dtype=float); c[0]=-2.0*s*s
    bars=[{}, {}, {}]
    bars[0]={(0,0,0):s*s,(1,0,0):-s,(0,0,1):s,(0,1,0):8*s,(1,1,0):-4,(0,2,0):4,(0,1,1):-8}
    bars[1]={(0,0,0):2*s*s,(1,0,0):s,(0,1,0):6*s,(0,0,1):s,(1,1,0):-4,(1,0,1):-4,(0,2,0):4,(0,1,1):-4}
    bars[2]={(0,0,0):s*s,(1,0,0):s,(0,1,0):8*s,(0,0,1):-s,(1,1,0):-8,(0,2,0):4,(0,1,1):-4}
    for e,p in enumerate(bars):
        for m,v in p.items(): c[1+10*e+i2[m]]=v
    # source tilde term: only the middle propagator block is nonzero
    tilde1={(0,0,0):-8*s,(1,0,0):8,(0,1,0):-8,(0,0,1):8}
    for m,v in tilde1.items(): c[35+i1[m]]=v
    return c

import numpy as np

def _dict(c,basis): return {m:float(v) for m,v in zip(basis,c) if abs(v)>1e-15}
def _mul(p,q):
    r={}
    for m,v in p.items():
        for n,w in q.items():
            e=tuple(m[i]+n[i] for i in range(3)); r[e]=r.get(e,0.0)+v*w
    return r

def _deriv(p,j):
    r={}
    for m,v in p.items():
        if m[j]:
            e=list(m); e[j]-=1; e=tuple(e); r[e]=r.get(e,0.0)+v*m[j]
    return r

def triangle_syzygy_matrix(b: "np.ndarray") -> "np.ndarray":
    b=np.asarray(b,dtype=float)
    if b.shape!=(10,) or not np.all(np.isfinite(b)): raise ValueError('b must be finite shape-(10,)')
    m1=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    m2=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
    m4=[]
    for d in range(5):
        for a in range(d,-1,-1):
            for bb in range(d-a,-1,-1): m4.append((a,bb,d-a-bb))
    B=_dict(b,m2); dB=[_deriv(B,e) for e in range(3)]; cols=[B]
    for e in range(3):
        ze={tuple(1 if i==e else 0 for i in range(3)):1.}; base=_mul(ze,dB[e])
        for m in m2: cols.append(_mul({m:1.},base))
    for e in range(3):
        ze={tuple(1 if i==e else 0 for i in range(3)):1.}; base=_mul(ze,B)
        for m in m1: cols.append(_mul({m:1.},base))
    ridx={m:i for i,m in enumerate(m4)}; M=np.zeros((35,43))
    for j,p in enumerate(cols):
        for m,v in p.items(): M[ridx[m],j]=v
    return M

import numpy as np

def triangle_minimum_norm_lift(s: float) -> "np.ndarray":
    s=float(s)
    b=triangle_normalized_baikov(s)
    source=triangle_source_nonprincipal_syzygy(s)
    M=triangle_syzygy_matrix(b)
    if np.max(np.abs(M @ source)) > 1e-10:
        raise ValueError('source representative does not satisfy the assembled syzygy system')
    A=np.vstack([M,np.eye(1,43,0,dtype=float)])
    rhs=np.r_[np.zeros(35),source[0]]
    return np.linalg.lstsq(A,rhs,rcond=1e-12)[0]

import numpy as np

def _poly(c,basis): return {m:float(v) for m,v in zip(basis,c) if abs(v)>1e-15}
def _d(p,j):
    r={}
    for m,v in p.items():
        if m[j]:
            e=list(m); e[j]-=1; e=tuple(e); r[e]=r.get(e,0.0)+v*m[j]
    return r

def _zmul(p,j):
    r={}
    for m,v in p.items():
        e=list(m); e[j]+=1; r[tuple(e)]=r.get(tuple(e),0.0)+v
    return r

def _acc(dst,src,scale=1.0):
    for m,v in src.items(): dst[m]=dst.get(m,0.0)+scale*v

def triangle_surface_polynomial(s: float, gamma: float) -> "np.ndarray":
    gamma=float(gamma)
    if not np.isfinite(gamma) or gamma==0: raise ValueError('gamma must be finite and nonzero')
    m1=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    m2=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
    c=triangle_minimum_norm_lift(float(s)); S={(0,0,0):float(c[0])}
    for e in range(3):
        bar=_poly(c[1+10*e:1+10*(e+1)],m2); til=_poly(c[31+4*e:31+4*(e+1)],m1)
        _acc(S,_zmul(til,e),1.0); _acc(S,_zmul(_d(bar,e),e),-1.0/gamma)
    return np.array([S.get(m,0.0) for m in m2])

import numpy as np

def _eval_surface(c,z):
    basis=np.array([(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)],dtype=int)
    z=np.asarray(z,dtype=float)
    return float(sum(float(v)*np.prod(z**m) for v,m in zip(c,basis)))

def triangle_surface_path_table(s_values: "np.ndarray", gamma: float, probes: "np.ndarray") -> "np.ndarray":
    sv=np.asarray(s_values,dtype=float); probes=np.asarray(probes,dtype=float)
    if sv.ndim!=1 or sv.size<1 or np.any(~np.isfinite(sv)) or np.any(sv<=0): raise ValueError('invalid s_values')
    if probes.ndim!=2 or probes.shape[1]!=3 or np.any(~np.isfinite(probes)): raise ValueError('invalid probes')
    rows=[]
    for s in sv:
        c=triangle_surface_polynomial(float(s),gamma)
        rows.append([_eval_surface(c,float(s)*u)/(float(s)*float(s)) for u in probes])
    return np.asarray(rows,dtype=float)

import numpy as np

def cumulative_triangle_surface_path(s_values: "np.ndarray", gamma: float, probes: "np.ndarray") -> float:
    table=triangle_surface_path_table(s_values,gamma,probes)
    if table.shape[0]<2: return 0.0
    return float(np.linalg.norm(np.diff(table,axis=0),axis=1).sum())
SCICODE_GOLD_EOF
