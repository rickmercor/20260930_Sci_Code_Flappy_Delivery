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

def prepare_least_squares_problem(A, b, x0):
    """Reference implementation for least-squares input preparation."""
    try:
        A64=np.asarray(A,dtype=np.float64)
        b64=np.asarray(b,dtype=np.float64)
        x064=np.asarray(x0,dtype=np.float64)
    except (TypeError,ValueError) as exc:
        raise ValueError("A, b, and x0 must be real numeric arrays") from exc
    if A64.ndim!=2 or A64.shape[0]<1 or A64.shape[1]<1:
        raise ValueError("A must be a nonempty 2D array")
    n,d=A64.shape
    if b64.ndim!=1 or b64.shape!=(n,):
        raise ValueError("b must have shape (n,)")
    if x064.ndim!=1 or x064.shape!=(d,):
        raise ValueError("x0 must have shape (d,)")
    if not np.all(np.isfinite(A64)) or not np.all(np.isfinite(b64)) or not np.all(np.isfinite(x064)):
        raise ValueError("all entries must be finite")
    return A64.copy(),b64.copy(),x064.copy()

import numpy as np

def normalized_coordinate_sketch(A, index):
    """Reference implementation for the normalized coordinate sketch."""
    try: A64=np.asarray(A,dtype=np.float64)
    except (TypeError,ValueError) as exc: raise ValueError("A must be real numeric") from exc
    if A64.ndim!=2 or A64.shape[0]<1 or A64.shape[1]<1 or not np.all(np.isfinite(A64)):
        raise ValueError("A must be a nonempty finite 2D array")
    if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)):
        raise ValueError("index must be an integer")
    j=int(index)-1
    if j<0 or j>=A64.shape[1]: raise ValueError("index out of range")
    norm=float(np.linalg.norm(A64[:,j]))
    if not np.isfinite(norm) or norm<=0.0: raise ValueError("selected column must have positive norm")
    S=np.zeros(A64.shape[1],dtype=np.float64); S[j]=1.0/norm
    return S

import numpy as np

def rcgls_sketch_seed(A,r,S):
    """Reference implementation of the RCGLS sketched-gradient seed."""
    try:
        A64=np.asarray(A,dtype=np.float64); r64=np.asarray(r,dtype=np.float64); S64=np.asarray(S,dtype=np.float64)
    except (TypeError,ValueError) as exc: raise ValueError("inputs must be real numeric") from exc
    if A64.ndim!=2 or A64.shape[0]<1 or A64.shape[1]<1: raise ValueError("A must be nonempty 2D")
    n,d=A64.shape
    if r64.ndim!=1 or r64.shape!=(n,): raise ValueError("r shape mismatch")
    if S64.ndim==1:
        if S64.shape!=(d,): raise ValueError("S shape mismatch")
        Smat=S64.reshape(d,1)
    elif S64.ndim==2:
        if S64.shape[0]!=d or S64.shape[1]<1: raise ValueError("S shape mismatch")
        Smat=S64
    else: raise ValueError("S must be vector or matrix")
    if not np.all(np.isfinite(A64)) or not np.all(np.isfinite(r64)) or not np.all(np.isfinite(Smat)): raise ValueError("nonfinite input")
    g=A64.T@r64
    seed=Smat@(Smat.T@g)
    image=A64@seed
    if not np.all(np.isfinite(seed)) or not np.all(np.isfinite(image)): raise ValueError("nonfinite seed")
    if float(image@image)<=0.0: raise ValueError("zero search image")
    return np.asarray(seed,dtype=np.float64),np.asarray(image,dtype=np.float64)

import numpy as np

def rcgls_conjugate_direction(A,r_next,S_next,p_prev,v_prev):
    """Reference implementation of a coupled RCGLS conjugate/update transition."""
    try:
        A64=np.asarray(A,dtype=np.float64); r64=np.asarray(r_next,dtype=np.float64)
        p64=np.asarray(p_prev,dtype=np.float64); v64=np.asarray(v_prev,dtype=np.float64)
        S64=np.asarray(S_next,dtype=np.float64)
    except (TypeError,ValueError) as exc: raise ValueError("inputs must be real numeric") from exc
    if A64.ndim!=2 or A64.shape[0]<1 or A64.shape[1]<1: raise ValueError("A must be nonempty 2D")
    n,d=A64.shape
    if r64.ndim!=1 or r64.shape!=(n,) or p64.ndim!=1 or p64.shape!=(d,) or v64.ndim!=1 or v64.shape!=(n,): raise ValueError("shape mismatch")
    if S64.ndim==1:
        if S64.shape!=(d,): raise ValueError("S shape mismatch")
        Smat=S64.reshape(d,1)
    elif S64.ndim==2:
        if S64.shape[0]!=d or S64.shape[1]<1: raise ValueError("S shape mismatch")
        Smat=S64
    else: raise ValueError("S must be vector or matrix")
    if not np.all(np.isfinite(A64)) or not np.all(np.isfinite(r64)) or not np.all(np.isfinite(p64)) or not np.all(np.isfinite(v64)) or not np.all(np.isfinite(Smat)): raise ValueError("nonfinite input")
    Ap=A64@p64
    if not np.allclose(Ap,v64,rtol=1e-12,atol=1e-12): raise ValueError("v_prev inconsistent with A @ p_prev")
    prev_den=float(v64@v64)
    if not np.isfinite(prev_den) or prev_den<=0.0: raise ValueError("zero previous image")
    g=A64.T@r64
    base=Smat@(Smat.T@g)
    Abase=A64@base
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(Abase)) or float(Abase@Abase)<=0.0: raise ValueError("invalid sketch seed")
    tau=-float(Abase@v64)/prev_den
    p_next=base+tau*p64
    v_next=Abase+tau*v64
    next_den=float(v_next@v_next)
    if not np.isfinite(tau) or not np.all(np.isfinite(p_next)) or not np.all(np.isfinite(v_next)) or next_den<=0.0: raise ValueError("invalid corrected direction")
    dnext=Smat.T@g
    mu_next=float(dnext@dnext)/next_den
    if not np.isfinite(mu_next): raise ValueError("nonfinite mu")
    r_after=r64-mu_next*v_next
    residual_sq=float(r_after@r_after)
    conjugacy_defect=float(v_next@v64)
    line_search_defect=float(r_after@v_next)
    if not np.all(np.isfinite(r_after)) or not np.isfinite(residual_sq) or not np.isfinite(conjugacy_defect) or not np.isfinite(line_search_defect): raise ValueError("nonfinite updated state")
    return (np.asarray(p_next,dtype=np.float64),np.asarray(v_next,dtype=np.float64),float(tau),
            float(mu_next),np.asarray(r_after,dtype=np.float64),float(residual_sq),
            float(conjugacy_defect),float(line_search_defect))

import numpy as np

def rcgls_residual_history(A,b,x0,indices,scale_factors):
    """Reference implementation of canonical and equivalently rescaled fixed-stream RCGLS."""
    A64,b64,x_can=prepare_least_squares_problem(A,b,x0)
    x_scaled=x_can.copy()
    if not isinstance(indices,(tuple,list)) or len(indices)<1: raise ValueError("indices must be nonempty tuple/list")
    d=A64.shape[1]; checked=[]
    for index in indices:
        if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)): raise ValueError("indices must be integers")
        j=int(index)
        if j<1 or j>d: raise ValueError("coordinate out of range")
        checked.append(j)
    try:
        scales=np.asarray(scale_factors,dtype=np.float64)
    except (TypeError,ValueError) as exc:
        raise ValueError("scale_factors must be real numeric") from exc
    if scales.ndim!=1 or scales.shape!=(len(checked),): raise ValueError("scale_factors length mismatch")
    if not np.all(np.isfinite(scales)) or np.any(scales==0.0): raise ValueError("scale_factors must be finite and nonzero")

    r_can=b64-A64@x_can
    r_scaled=r_can.copy()
    S0=normalized_coordinate_sketch(A64,checked[0]).reshape(d,1)
    Sc0=scales[0]*S0
    p_can,v_can=rcgls_sketch_seed(A64,r_can,S0)
    p_scaled,v_scaled=rcgls_sketch_seed(A64,r_scaled,Sc0)
    den_can=float(v_can@v_can); den_scaled=float(v_scaled@v_scaled)
    if den_can<=0.0 or den_scaled<=0.0 or not np.isfinite(den_can+den_scaled): raise ValueError("RCGLS breakdown")
    g_can=A64.T@r_can; g_scaled=A64.T@r_scaled
    d_can=S0.T@g_can; d_scaled=Sc0.T@g_scaled
    mu_can=float(d_can@d_can)/den_can; mu_scaled=float(d_scaled@d_scaled)/den_scaled
    if not np.isfinite(mu_can) or not np.isfinite(mu_scaled): raise ValueError("nonfinite mu")
    x_can=x_can+mu_can*p_can; r_can=r_can-mu_can*v_can
    x_scaled=x_scaled+mu_scaled*p_scaled; r_scaled=r_scaled-mu_scaled*v_scaled

    residual=[float(r_can@r_can)]; mus=[mu_can]; taus=[]; smus=[mu_scaled]; staus=[]
    idef=[float(np.linalg.norm(x_can-x_scaled))]; rdef=[float(np.linalg.norm(r_can-r_scaled))]
    if not np.all(np.isfinite([residual[0],idef[0],rdef[0]])): raise ValueError("nonfinite first update")

    for k in range(1,len(checked)):
        S=normalized_coordinate_sketch(A64,checked[k]).reshape(d,1)
        Ss=scales[k]*S
        p_can,v_can,tau,mu_can,r_can,rs,_,_=rcgls_conjugate_direction(A64,r_can,S,p_can,v_can)
        p_scaled,v_scaled,stau,mu_scaled,r_scaled,srs,_,_=rcgls_conjugate_direction(A64,r_scaled,Ss,p_scaled,v_scaled)
        x_can=x_can+mu_can*p_can
        x_scaled=x_scaled+mu_scaled*p_scaled
        residual.append(rs); mus.append(mu_can); taus.append(tau); smus.append(mu_scaled); staus.append(stau)
        idef.append(float(np.linalg.norm(x_can-x_scaled))); rdef.append(float(np.linalg.norm(r_can-r_scaled)))
        if not np.isclose(rs,srs,rtol=5e-12,atol=5e-12): raise ValueError("equivalent sketch scaling changed residual history")
    arrays=(np.asarray(residual,dtype=np.float64),np.asarray(mus,dtype=np.float64),np.asarray(taus,dtype=np.float64),
            np.asarray(smus,dtype=np.float64),np.asarray(staus,dtype=np.float64),np.asarray(idef,dtype=np.float64),np.asarray(rdef,dtype=np.float64))
    if any(not np.all(np.isfinite(a)) for a in arrays): raise ValueError("nonfinite history")
    return arrays

import numpy as np

def least_squares_orthonormal_reference(A,b,ell,field="real"):
    """Reference implementation of the coupled LS/orthonormal benchmark."""
    try: A64=np.asarray(A,dtype=np.float64); b64=np.asarray(b,dtype=np.float64)
    except (TypeError,ValueError) as exc: raise ValueError("A and b must be real numeric") from exc
    if A64.ndim!=2 or A64.shape[0]<1 or A64.shape[1]<1: raise ValueError("A must be nonempty 2D")
    n=A64.shape[0]
    if b64.ndim!=1 or b64.shape!=(n,): raise ValueError("b shape mismatch")
    if not np.all(np.isfinite(A64)) or not np.all(np.isfinite(b64)): raise ValueError("nonfinite data")
    try: x,_,rank,_=np.linalg.lstsq(A64,b64,rcond=None)
    except np.linalg.LinAlgError as exc: raise ValueError("least-squares solve failed") from exc
    r=b64-A64@x; opt=float(r@r); rank=int(rank)
    if not np.isfinite(opt) or opt<=0.0: raise ValueError("optimal residual must be positive finite")
    if isinstance(ell,(bool,np.bool_)) or not isinstance(ell,(int,np.integer)): raise ValueError("ell must be integer")
    ell=int(ell)
    if field not in {"real","complex"}: raise ValueError("field must be real or complex")
    alpha=1 if field=="real" else 0
    if ell<1 or ell>n or not (rank < ell-alpha): raise ValueError("dimensions outside theorem domain")
    rho=1.0+((n-ell)/(n-rank))*(rank/(ell-rank-alpha))
    if not np.isfinite(rho): raise ValueError("nonfinite benchmark")
    return float(opt),rank,float(rho)

import numpy as np

def rcgls_benchmark_crossing(residual_sq,optimal_residual_sq,benchmark):
    """Reference implementation for normalized first-crossing certification."""
    try: arr=np.asarray(residual_sq,dtype=np.float64); opt=float(optimal_residual_sq); bench=float(benchmark)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError("invalid inputs") from exc
    if arr.ndim!=1 or arr.size<1 or not np.all(np.isfinite(arr)) or np.any(arr<0): raise ValueError("invalid residual history")
    if not np.isfinite(opt) or opt<=0.0: raise ValueError("optimal residual must be positive finite")
    if not np.isfinite(bench) or bench<=0.0: raise ValueError("benchmark must be positive finite")
    inflation=arr/opt
    hits=np.flatnonzero(inflation<=bench)
    if hits.size==0: raise ValueError("benchmark is not reached")
    j=int(hits[0]); val=float(inflation[j]); margin=float(val/bench)
    return np.asarray(inflation,dtype=np.float64),j+1,val,margin

import numpy as np

def rcgls_method_certified_field_sensitivity(A,b,x0,indices,ell):
    """Reference orchestrator for scale-certified method field sensitivity."""
    A64,b64,x064=prepare_least_squares_problem(A,b,x0)
    if not isinstance(indices,(tuple,list)) or len(indices)<1:
        raise ValueError("indices must be nonempty tuple/list")
    pattern=np.asarray([2.0,0.5,4.0,0.25,8.0,0.125],dtype=np.float64)
    scales=np.resize(pattern,len(indices))
    residual_sq,_,_,_,_,iterate_defects,residual_defects=rcgls_residual_history(A64,b64,x064,indices,scales)
    if float(np.max(iterate_defects))>5e-10 or float(np.max(residual_defects))>5e-10:
        raise ValueError("equivalent sketch scaling changed RCGLS trajectory")
    opt,_,bench_real=least_squares_orthonormal_reference(A64,b64,ell,"real")
    opt_c,_,bench_complex=least_squares_orthonormal_reference(A64,b64,ell,"complex")
    if not np.isclose(opt,opt_c,rtol=0.0,atol=0.0):
        raise ValueError("inconsistent unsketched reference")
    _,_,_,margin_real=rcgls_benchmark_crossing(residual_sq,opt,bench_real)
    _,_,_,margin_complex=rcgls_benchmark_crossing(residual_sq,opt,bench_complex)
    value=float(margin_real/margin_complex)
    if not np.isfinite(value): raise ValueError("nonfinite field sensitivity")
    return value
SCICODE_GOLD_EOF
