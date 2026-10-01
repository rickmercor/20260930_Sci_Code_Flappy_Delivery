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

def intrinsic_orbitals(F: "np.ndarray", M: "np.ndarray", nocc: int) -> "np.ndarray":
    """Deterministic reference implementation."""
    try:
        if np.iscomplexobj(F) or np.iscomplexobj(M): raise ValueError('real inputs required')
        F=np.asarray(F,dtype=float); M=np.asarray(M,dtype=float)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError('real arrays required') from exc
    if F.ndim!=2 or F.shape[0]!=F.shape[1] or F.shape[0]<2 or M.ndim!=2:
        raise ValueError('shape')
    n=F.shape[0]
    if M.shape[0]!=n or not 1<=M.shape[1]<=n or not np.isfinite(F).all() or not np.isfinite(M).all(): raise ValueError('shape or finiteness')
    if isinstance(nocc,(bool,np.bool_)) or not isinstance(nocc,(int,np.integer)) or not 1<=nocc< n or nocc>M.shape[1]: raise ValueError('nocc')
    if np.max(np.abs(F-F.T))>1e-10: raise ValueError('symmetry')
    e,W=np.linalg.eigh((F+F.T)/2)
    if e[nocc]-e[nocc-1]<=1e-10: raise ValueError('reference gap')
    C=W[:,:nocc]; S=M.T@M
    if np.linalg.eigvalsh(S).min()<=1e-12: raise ValueError('minimal rank')
    X=M@np.linalg.solve(S,M.T@C)
    d,V=np.linalg.eigh(X.T@X)
    if d.min()<=1e-12: raise ValueError('projected occupied rank')
    Ct=X@((V/np.sqrt(d))@V.T)
    O=C@C.T; Ot=Ct@Ct.T; I=np.eye(n)
    R=(O@Ot+(I-O)@(I-Ot))@M
    d,V=np.linalg.eigh(R.T@R)
    if d.min()<=1e-12: raise ValueError('polarized rank')
    return R@((V/np.sqrt(d))@V.T)

import numpy as np

def thermal_reference(F: "np.ndarray", beta: float, ne: float) -> "tuple[np.ndarray, float]":
    """Deterministic reference implementation."""
    try:
        if np.iscomplexobj(F) or np.iscomplexobj(beta) or np.iscomplexobj(ne) or np.ndim(beta)!=0 or np.ndim(ne)!=0: raise ValueError('real scalar inputs')
        F=np.asarray(F,dtype=float); beta=float(beta); ne=float(ne)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError('numeric inputs') from exc
    if F.ndim!=2 or F.shape[0]!=F.shape[1] or F.shape[0]<1 or not np.isfinite(F).all(): raise ValueError('F')
    n=len(F)
    if np.max(np.abs(F-F.T))>1e-10 or not np.isfinite([beta,ne]).all() or not 0<beta<=100 or not 1e-6<=ne<=2*n-1e-6: raise ValueError('parameters')
    e,V=np.linalg.eigh((F+F.T)/2)
    lo=e.min()-50/beta; hi=e.max()+50/beta
    for _ in range(180):
        mu=(lo+hi)/2
        f=np.exp(-np.logaddexp(0,beta*(e-mu)))
        if 2*f.sum()<ne: lo=mu
        else: hi=mu
    mu=float((lo+hi)/2)
    f=np.exp(-np.logaddexp(0,beta*(e-mu)))
    return (V*f)@V.T,mu

import numpy as np

def bath_projectors(D: "np.ndarray", A: "np.ndarray", cutoff: float = 1e-10) -> "tuple[np.ndarray, np.ndarray]":
    """Deterministic reference implementation."""
    try:
        if np.iscomplexobj(D) or np.iscomplexobj(A) or np.iscomplexobj(cutoff) or np.ndim(cutoff)!=0: raise ValueError('real inputs')
        D=np.asarray(D,dtype=float); A=np.asarray(A,dtype=float); cutoff=float(cutoff)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError('numeric inputs') from exc
    if D.ndim!=2 or D.shape[0]!=D.shape[1] or len(D)<2 or A.shape!=(len(D),2): raise ValueError('shape')
    if not np.isfinite(D).all() or not np.isfinite(A).all() or not np.isfinite(cutoff) or not 0<cutoff<1: raise ValueError('finite inputs')
    if np.max(np.abs(D-D.T))>1e-10 or np.max(np.abs(A.T@A-np.eye(2)))>1e-10: raise ValueError('symmetry or metric')
    D=(D+D.T)/2
    eig=np.linalg.eigvalsh(D)
    if eig.min()<-1e-10 or eig.max()>1+1e-10: raise ValueError('occupations')
    R=np.eye(len(D))-A@A.T
    matrices=[R@D@A,np.column_stack([R@D@A[:,1],R@D@D@A[:,1]])]
    out=[]
    for X in matrices:
        B,s,_=np.linalg.svd(X,full_matrices=False)
        B=B[:,s>cutoff]
        out.append(B@B.T)
    return out[0],out[1]

import numpy as np

def embedded_operators(h: "np.ndarray", U: "np.ndarray", A: "np.ndarray", P: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Deterministic reference implementation."""
    try:
        if any(np.iscomplexobj(x) for x in [h,U,A,P]): raise ValueError('real inputs')
        h=np.asarray(h,dtype=float); U=np.asarray(U,dtype=float); A=np.asarray(A,dtype=float); P=np.asarray(P,dtype=float)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError('numeric inputs') from exc
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or len(h)<2: raise ValueError('h shape')
    n=len(h)
    if U.shape!=(n,) or A.shape!=(n,2) or P.shape!=(n,n) or not all(np.isfinite(x).all() for x in [h,U,A,P]): raise ValueError('shape or finiteness')
    if np.max(np.abs(h-h.T))>1e-9 or np.max(np.abs(P-P.T))>1e-9 or np.max(np.abs(A.T@A-np.eye(2)))>1e-9 or np.max(np.abs(P@P-P))>1e-9 or np.max(np.abs(P@A))>1e-9: raise ValueError('matrix conditions')
    h=(h+h.T)/2; P=(P+P.T)/2
    r=int(np.count_nonzero(np.linalg.eigvalsh(P)>.5))
    if r>2: raise ValueError('rank')
    R=P.copy(); cols=[]
    for _ in range(r):
        diagonal=np.diag(R); j=int(np.flatnonzero(diagonal>=diagonal.max()-1e-12)[0])
        b=R[:,j]/np.linalg.norm(R[:,j]); cols.append(b)
        R-=np.outer(b,b); R=(R+R.T)/2
    E=np.column_stack([A]+cols); k=E.shape[1]; dim=1<<(2*k)
    ht=E.T@h@E
    H=np.zeros((dim,dim)); nt=np.zeros(dim); na=np.zeros(dim)
    # Second quantization of each original-site density, separately by spin.
    densities=[]
    for spin in range(2):
        site=np.zeros((n,dim,dim))
        for state in range(dim):
            if spin==0:
                nt[state]=state.bit_count()
                na[state]=sum((state>>p)&1 for p in [0,1,k,k+1])
            for q in range(k):
                qb=q+spin*k
                if not (state>>qb)&1: continue
                t=state^(1<<qb); sq=(-1)**((state&((1<<qb)-1)).bit_count())
                for p in range(k):
                    pb=p+spin*k
                    if (t>>pb)&1: continue
                    dest=t|(1<<pb); sign=sq*(-1)**((t&((1<<pb)-1)).bit_count())
                    H[dest,state]+=ht[p,q]*sign
                    site[:,dest,state]+=E[:,p]*E[:,q]*sign
        densities.append(site)
    for i in range(n): H+=U[i]*(densities[0][i]@densities[1][i])
    return (H+H.T)/2,np.diag(na),np.diag(nt)

import numpy as np

def thermal_response(H: "np.ndarray", NA: "np.ndarray", NT: "np.ndarray", beta: float, mu: "np.ndarray") -> "tuple[np.ndarray, float, np.ndarray]":
    """Deterministic reference implementation."""
    try:
        if any(np.iscomplexobj(x) for x in [H,NA,NT,mu,beta]) or np.ndim(beta)!=0: raise ValueError('real inputs')
        H=np.asarray(H,dtype=float); NA=np.asarray(NA,dtype=float); NT=np.asarray(NT,dtype=float); mu=np.asarray(mu,dtype=float); beta=float(beta)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError('numeric inputs') from exc
    if H.ndim!=2 or H.shape[0]!=H.shape[1] or len(H)<1 or NA.shape!=H.shape or NT.shape!=H.shape or mu.shape!=(2,): raise ValueError('shape')
    if not all(np.isfinite(x).all() for x in [H,NA,NT,mu]) or not np.isfinite(beta) or not 0<beta<=100: raise ValueError('finite inputs')
    if any(np.max(np.abs(x-x.T))>1e-9 for x in [H,NA,NT]): raise ValueError('symmetry')
    H=(H+H.T)/2; NA=(NA+NA.T)/2; NT=(NT+NT.T)/2
    e,V=np.linalg.eigh(H-mu[0]*NT-mu[1]*NA)
    w=np.exp(-beta*(e-e.min())); w/=w.sum()
    ops=[V.T@NT@V,V.T@NA@V]
    means=np.array([w@np.diag(O) for O in ops])
    variance=float(w@np.sum(ops[1]**2,axis=1)-means[1]**2)
    gap=np.abs(e[:,None]-e[None,:]); x=beta*gap
    ratio=np.ones_like(x)
    np.divide(-np.expm1(-x),x,out=ratio,where=x!=0)
    L=beta*np.maximum(w[:,None],w[None,:])*ratio
    J=np.empty((2,2))
    for i in range(2):
        for j in range(2): J[i,j]=np.sum(L*ops[i]*ops[j])-beta*means[i]*means[j]
    return means,variance,(J+J.T)/2

import numpy as np

def match_numbers(H: "np.ndarray", NA: "np.ndarray", NT: "np.ndarray", beta: float, targets: "np.ndarray") -> "tuple[np.ndarray, float, np.ndarray]":
    """Deterministic reference implementation."""
    try:
        if np.iscomplexobj(targets): raise ValueError('real targets')
        targets=np.asarray(targets,dtype=float)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError('targets') from exc
    if targets.shape!=(2,) or not np.isfinite(targets).all(): raise ValueError('targets')
    mu=np.zeros(2)
    means,var,J=thermal_response(H,NA,NT,beta,mu)
    operators=[np.asarray(NT,dtype=float),np.asarray(NA,dtype=float)]
    centered=[]
    for op,target in zip(operators,targets):
        op=(op+op.T)/2; e=np.linalg.eigvalsh(op)
        if not e[0]<target<e[-1]: raise ValueError('target outside open spectrum')
        centered.append(op-np.trace(op)/len(op)*np.eye(len(op)))
    gram=np.array([[np.sum(a*b) for b in centered] for a in centered])
    if np.linalg.eigvalsh(gram)[0]<=1e-12: raise ValueError('dependent fields')
    for iteration in range(100):
        r=means-targets
        if np.max(np.abs(r))<=2e-11: break
        d,V=np.linalg.eigh(J)
        step=V@((V.T@r)/np.maximum(d,1e-10))
        size=np.max(np.abs(step))
        if size>10: step*=10/size
        accepted=False
        for backtrack in range(30):
            trial=mu-step*(.5**backtrack)
            tm,tv,tj=thermal_response(H,NA,NT,beta,trial)
            if np.linalg.norm(tm-targets)<np.linalg.norm(r):
                mu=trial; means=tm; var=tv; J=tj; accepted=True; break
        if not accepted: break
    if np.max(np.abs(means-targets))>1e-9 or not np.isfinite(mu).all(): raise ValueError('joint fit did not converge')
    return mu,float(var),means

import numpy as np

def compare_baths(h: "np.ndarray", U: "np.ndarray", A: "np.ndarray", D: "np.ndarray", beta: float) -> tuple[float, float, float]:
    """Deterministic reference implementation."""
    Pev,Pme=bath_projectors(D,A,1e-10)
    A=np.asarray(A,dtype=float); D=np.asarray(D,dtype=float)
    out=[]
    for P in [Pev,Pme]:
        if np.count_nonzero(np.linalg.eigvalsh(P)>.5)!=2: raise ValueError('two bath orbitals required')
        H,NA,NT=embedded_operators(h,U,A,P)
        targets=np.array([2*np.trace((A@A.T+P)@D),2*np.trace(A.T@D@A)])
        _,var,_=match_numbers(H,NA,NT,beta,targets)
        out.append(float(var))
    return out[0],out[1],float(out[1]-out[0])

import numpy as np

def charge_variance_difference(F: "np.ndarray", M: "np.ndarray", U: "np.ndarray", beta: float = 2.3, ne: float = 4.0, nocc: int = 2) -> float:
    """Deterministic reference implementation."""
    Q=intrinsic_orbitals(F,M,nocc)
    F=np.asarray(F,dtype=float)
    if len(F)<4 or Q.shape[1]<2: raise ValueError('at least four primary and two minimal orbitals')
    D,_=thermal_reference(F,beta,ne)
    return float(compare_baths((F+F.T)/2,U,Q[:,:2],D,beta)[2])
SCICODE_GOLD_EOF
