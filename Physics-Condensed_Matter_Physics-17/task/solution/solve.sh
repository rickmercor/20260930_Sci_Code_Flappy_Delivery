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

def build_periodic_chain_tangent(positions: np.ndarray, length: float, force_index: int) -> np.ndarray:
    import numpy as np
    from numbers import Real, Integral
    if np.iscomplexobj(positions):
        raise ValueError("positions must be real")
    x=np.asarray(positions,dtype=float)
    if isinstance(length,(bool,np.bool_)) or not isinstance(length,Real) or not np.isfinite(length) or length<=0:
        raise ValueError("length must be positive and finite")
    length=float(length)
    if x.ndim!=1 or x.size<2 or not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>=length) or np.any(np.diff(x)<=0):
        raise ValueError("positions must be finite, ordered and distinct in [0,length)")
    if isinstance(force_index,(bool,np.bool_)) or not isinstance(force_index,Integral) or not 0<=force_index<x.size:
        raise ValueError("force_index is out of range")
    n=x.size; k=2*np.pi/length
    raw=x[:,None]-x[None,:]
    delta=raw-length*np.floor(raw/length+0.5)
    d=np.abs(delta)
    phase=k*(x[:,None]+x[None,:])
    unit=np.zeros(n); unit[int(force_index)]=1.0
    dd=np.sign(delta)*(unit[:,None]-unit[None,:])
    dd=np.triu(dd,1); dd=dd+dd.T
    dphase=k*(unit[:,None]+unit[None,:])
    sfactor=np.exp(-d/0.85); hfactor=np.exp(-d/1.25)
    s=0.16*sfactor*(1.0+0.05*np.cos(phase))*(d<=2.25)
    h=-0.90*hfactor*(1.0+0.10*np.cos(phase))*(d<=3.25)
    ds=0.16*sfactor*(-dd/0.85*(1.0+0.05*np.cos(phase))-0.05*np.sin(phase)*dphase)*(d<=2.25)
    dh=-0.90*hfactor*(-dd/1.25*(1.0+0.10*np.cos(phase))-0.10*np.sin(phase)*dphase)*(d<=3.25)
    np.fill_diagonal(s,1.0+0.03*np.cos(k*x))
    np.fill_diagonal(h,-0.25+0.35*np.cos(k*x)+0.08*np.sin(2*k*x)+0.025*(-1.0)**np.arange(n))
    np.fill_diagonal(ds,-0.03*k*np.sin(k*x)*unit)
    np.fill_diagonal(dh,(-0.35*k*np.sin(k*x)+0.16*k*np.cos(2*k*x))*unit)
    return np.stack((d,s,h,ds,dh))

import numpy as np

def build_colored_signed_probes(distances: np.ndarray, exclusion: float, signs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np
    from numbers import Real
    if np.iscomplexobj(distances) or np.iscomplexobj(signs):
        raise ValueError("inputs must be real")
    d=np.asarray(distances,dtype=float);s=np.asarray(signs,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or d.shape[0]<1 or not np.all(np.isfinite(d)) or np.any(d<0):
        raise ValueError("distances must be finite nonnegative square matrix")
    if not np.allclose(d,d.T,rtol=0,atol=1e-12) or not np.allclose(np.diag(d),0,rtol=0,atol=1e-12):
        raise ValueError("distances must be symmetric with zero diagonal")
    if isinstance(exclusion,(bool,np.bool_)) or not isinstance(exclusion,Real) or not np.isfinite(exclusion) or exclusion<0:
        raise ValueError("exclusion must be finite and nonnegative")
    n=d.shape[0]
    if s.shape!=(n,) or not np.all((s==1)|(s==-1)):
        raise ValueError("signs must have unit magnitude")
    c=np.zeros(n,dtype=int)
    for i in range(n):
        used={int(c[j]) for j in range(i) if d[i,j]<exclusion}
        label=0
        while label in used:label+=1
        c[i]=label
    p=np.zeros((n,int(c.max())+1))
    p[np.arange(n),c]=s
    return c,p

import numpy as np

def canonical_block_krylov_basis(operator: np.ndarray, block: np.ndarray, depth: int, rank_tolerance: float=1e-12) -> np.ndarray:
    import numpy as np

    a = np.asarray(operator, dtype=float)
    b = np.asarray(block, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1:
        raise ValueError("operator must be a nonempty square matrix")
    if b.ndim != 2 or b.shape[0] != a.shape[0] or b.shape[1] < 1:
        raise ValueError("block must have shape (n,m) with m >= 1")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("operator and block must be finite")
    if not np.allclose(a, a.T, rtol=0.0, atol=1e-12):
        raise ValueError("operator must be symmetric")
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)) or int(depth) < 1:
        raise ValueError("depth must be a positive integer")
    if not np.isfinite(rank_tolerance) or float(rank_tolerance) <= 0.0:
        raise ValueError("rank_tolerance must be positive and finite")
    depth = int(depth)
    threshold = float(rank_tolerance) * max(1.0, float(np.linalg.norm(b, ord="fro")))
    accepted = []
    current = b.copy()
    for level in range(depth):
        for column in range(current.shape[1]):
            vector = current[:, column].copy()
            for _pass in range(2):
                for q in accepted:
                    vector -= q * float(np.dot(q, vector))
            norm = float(np.linalg.norm(vector))
            if norm > threshold:
                vector /= norm
                pivot = int(np.argmax(np.abs(vector)))
                if vector[pivot] < 0.0:
                    vector = -vector
                accepted.append(vector)
        if level + 1 < depth:
            current = a @ current
    if not accepted:
        raise ValueError("the block Krylov matrix has zero numerical rank")
    return np.column_stack(accepted)

import numpy as np

def projected_inverse_sqrt_action(overlap: np.ndarray, probes: np.ndarray, basis: np.ndarray) -> np.ndarray:
    import numpy as np

    if any(np.iscomplexobj(a) for a in (overlap, probes, basis)):
        raise ValueError("inputs must be real")
    s = np.asarray(overlap, dtype=float)
    o = np.asarray(probes, dtype=float)
    q = np.asarray(basis, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] < 1:
        raise ValueError("overlap must be a nonempty square matrix")
    n = s.shape[0]
    if o.ndim != 2 or o.shape[0] != n or o.shape[1] < 1:
        raise ValueError("probes must have shape (n,m)")
    if q.ndim != 2 or q.shape[0] != n or not 1 <= q.shape[1] <= n:
        raise ValueError("basis must have shape (n,r), 1<=r<=n")
    if not all(np.all(np.isfinite(a)) for a in (s, o, q)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(s, s.T, rtol=0.0, atol=1e-12):
        raise ValueError("overlap must be symmetric")
    if not np.allclose(q.T @ q, np.eye(q.shape[1]), rtol=1e-10, atol=1e-12):
        raise ValueError("basis columns must be orthonormal")
    scale = float(np.max(np.abs(s)))
    if scale == 0.0:
        raise ValueError("overlap must be positive definite")
    scaled = s / scale
    if float(np.linalg.eigvalsh(scaled)[0]) <= 0.0:
        raise ValueError("overlap must be positive definite on the full space")
    projected = q.T @ scaled @ q
    projected = 0.5 * projected + 0.5 * projected.T
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("projected overlap must be positive definite")
    coefficients = eigenvectors.T @ (q.T @ o)
    coefficients = coefficients / np.sqrt(eigenvalues[:, None])
    result = (q @ (eigenvectors @ coefficients)) / np.sqrt(scale)
    if not np.all(np.isfinite(result)):
        raise ValueError("the action is not representable in binary64")
    return result

import numpy as np

def extract_sparse_operator(action: np.ndarray, distances: np.ndarray, colors: np.ndarray, signs: np.ndarray, keep_radius: float, magnitude_threshold: float=0.0) -> np.ndarray:
    import numpy as np
    from numbers import Real
    if any(np.iscomplexobj(v) for v in (action,distances,colors,signs)):
        raise ValueError("all arrays must be real")
    z=np.asarray(action,dtype=float);d=np.asarray(distances,dtype=float)
    craw=np.asarray(colors);s=np.asarray(signs,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or d.shape[0]<1 or not np.all(np.isfinite(d)) or np.any(d<0):
        raise ValueError("distances must be finite nonnegative square matrix")
    n=d.shape[0]
    if not np.allclose(d,d.T,rtol=0,atol=1e-12) or not np.allclose(np.diag(d),0,rtol=0,atol=1e-12):
        raise ValueError("distances must be symmetric with zero diagonal")
    if craw.shape!=(n,) or not np.issubdtype(craw.dtype,np.number) or not np.all(np.isfinite(craw)) or np.any(craw<0) or np.any(craw!=np.floor(craw)):
        raise ValueError("colors must be nonnegative integer-valued labels")
    c=craw.astype(int);m=int(c.max())+1
    if not np.array_equal(np.unique(c),np.arange(m)):
        raise ValueError("colors must be contiguous")
    if z.shape!=(n,m) or not np.all(np.isfinite(z)):
        raise ValueError("action shape or values invalid")
    if s.shape!=(n,) or not np.all((s==1)|(s==-1)):
        raise ValueError("signs must have magnitude one")
    for value in (keep_radius,magnitude_threshold):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<0:
            raise ValueError("cutoffs must be finite nonnegative scalars")
    one_sided=z[:,c]*s[None,:]
    t=0.5*(one_sided+one_sided.T)
    keep=(d<=keep_radius)&((np.abs(t)>=magnitude_threshold)|np.eye(n,dtype=bool))
    return np.where(keep,t,0.0)

import numpy as np

def solve_fixed_population_density(overlap: np.ndarray, overlap_map: np.ndarray, transformed_hamiltonian: np.ndarray, probes: np.ndarray, distances: np.ndarray, colors: np.ndarray, signs: np.ndarray, beta: float, electron_count: float, depth: int, keep_radius: float) -> tuple[float, np.ndarray, np.ndarray]:
    import numpy as np
    from numbers import Real, Integral
    import math
    if any(np.iscomplexobj(a) for a in (overlap,overlap_map,transformed_hamiltonian,probes)):
        raise ValueError("matrices must be real")
    s=np.asarray(overlap,dtype=float);m=np.asarray(overlap_map,dtype=float)
    k=np.asarray(transformed_hamiltonian,dtype=float);r=np.asarray(probes,dtype=float)
    if s.ndim!=2 or s.shape[0]!=s.shape[1] or s.shape[0]<1:
        raise ValueError("overlap must be square")
    n=s.shape[0]
    for a in (s,m,k):
        if a.shape!=(n,n) or not np.all(np.isfinite(a)) or not np.allclose(a,a.T,rtol=0,atol=1e-12):
            raise ValueError("matrices must be finite symmetric and shape-compatible")
    if np.linalg.eigvalsh(s)[0]<=0:
        raise ValueError("overlap must be positive definite")
    if r.ndim!=2 or r.shape[0]!=n or r.shape[1]<1 or not np.all(np.isfinite(r)):
        raise ValueError("probes must be a finite block")
    # Validate the extraction contract before interpreting the labels.
    extract_sparse_operator(np.zeros_like(r),distances,colors,signs,keep_radius,0.0)
    c=np.asarray(colors,dtype=int);sig=np.asarray(signs,dtype=float)
    expected=np.zeros_like(r);expected[np.arange(n),c]=sig
    if not np.array_equal(r,expected):
        raise ValueError("probes must match colors and supplied signs without normalization")
    if isinstance(beta,(bool,np.bool_)) or not isinstance(beta,Real) or not np.isfinite(beta) or beta<=0:
        raise ValueError("beta must be positive finite scalar")
    if isinstance(electron_count,(bool,np.bool_)) or not isinstance(electron_count,Real) or not np.isfinite(electron_count):
        raise ValueError("electron_count must be finite real scalar")
    if isinstance(depth,(bool,np.bool_)) or not isinstance(depth,Integral) or depth<1:
        raise ValueError("depth must be positive integer")
    b=m@r
    q=canonical_block_krylov_basis(k,b,depth)
    t=q.T@k@q;t=0.5*(t+t.T)
    e,u=np.linalg.eigh(t)
    left=m@q@u;right=u.T@(q.T@b)
    weights=np.empty(e.size)
    for j in range(e.size):
        rank_one=left[:,j,None]*right[j,None,:]
        component=extract_sparse_operator(rank_one,distances,c,sig,keep_radius,0.0)
        weights[j]=np.sum(s*component.T)
    total=math.fsum(float(w) for w in weights)
    if not np.all(np.isfinite(weights)) or np.any(weights<0) or total<=0 or not 0<electron_count<total:
        raise ValueError("spectral weights and electron count do not define a monotone root")
    pad=max(2.0,50.0/float(beta));low=float(e[0]-pad);high=float(e[-1]+pad)
    if not np.isfinite(low) or not np.isfinite(high) or low>=high:
        raise ValueError("chemical-potential bracket must be finite and ordered")

    def _residual_sign(mu):
        # Split off exactly occupied reference levels, retaining hole tails.
        # fsum avoids losing a small noninteger filling in that baseline.
        below=e<mu
        baseline=math.fsum([float(w) for w in weights[below]]+[-float(electron_count)])
        positive=[];negative=[]
        if baseline>0:positive.append(math.log(baseline))
        elif baseline<0:negative.append(math.log(-baseline))
        for energy,weight,is_below in zip(e,weights,below):
            if weight==0:continue
            t=abs(float(beta)*(float(energy)-float(mu)))
            log_tail=math.log(float(weight))-t-math.log1p(math.exp(-t))
            if is_below:negative.append(log_tail)
            else:positive.append(log_tail)
        # Compare positive and negative contributions without materializing
        # exponentially small carrier populations.
        logs=[]
        for terms in (positive,negative):
            if not terms:logs.append(-math.inf);continue
            peak=max(terms)
            if peak==-math.inf:logs.append(peak);continue
            logs.append(peak+math.log(math.fsum(math.exp(v-peak) for v in terms)))
        return (logs[0]>logs[1])-(logs[0]<logs[1])

    if not _residual_sign(low)<0 or not _residual_sign(high)>0:
        raise ValueError("chemical-potential interval does not strictly bracket the population")
    for _ in range(64):
        mu=low+(high-low)/2
        if _residual_sign(mu)<0:low=mu
        else:high=mu
    mu=low+(high-low)/2
    scaled=float(beta)*(e-mu);p=np.exp(-np.abs(scaled))
    occupation=np.where(scaled>=0,p/(1+p),1/(1+p))
    x=(left*occupation[None,:])@right
    rho=extract_sparse_operator(x,distances,c,sig,keep_radius,0.0)
    return float(mu),rho,x

import numpy as np

def compute_frozen_potential_force(overlap_map: np.ndarray, hamiltonian: np.ndarray, density_action: np.ndarray, density: np.ndarray, overlap_tangent: np.ndarray, hamiltonian_tangent: np.ndarray, distances: np.ndarray, colors: np.ndarray, signs: np.ndarray, keep_radius: float) -> np.ndarray:
    import numpy as np
    if any(np.iscomplexobj(a) for a in (overlap_map,hamiltonian,density_action,density,overlap_tangent,hamiltonian_tangent)):
        raise ValueError("all matrices must be real")
    m=np.asarray(overlap_map,dtype=float);h=np.asarray(hamiltonian,dtype=float)
    x=np.asarray(density_action,dtype=float);rho=np.asarray(density,dtype=float)
    ds=np.asarray(overlap_tangent,dtype=float);dh=np.asarray(hamiltonian_tangent,dtype=float)
    if m.ndim!=2 or m.shape[0]!=m.shape[1] or m.shape[0]<1:
        raise ValueError("overlap map must be a square matrix")
    n=m.shape[0]
    for a in (m,h,rho,ds,dh):
        if a.shape!=(n,n) or not np.all(np.isfinite(a)) or not np.allclose(a,a.T,rtol=0,atol=1e-12):
            raise ValueError("matrices must be finite, symmetric and shape-compatible")
    if x.ndim!=2 or x.shape[0]!=n or x.shape[1]<1 or not np.all(np.isfinite(x)):
        raise ValueError("density action must be a finite block")
    xe=m@(m@(h@x))
    rhoe=extract_sparse_operator(xe,distances,colors,signs,keep_radius,0.0)
    band=-float(np.sum(rho*dh.T))
    pulay=float(np.sum(rhoe*ds.T))
    return np.array([band,pulay,band+pulay],dtype=float)

import numpy as np

def run_chromatic_charge_force(positions: np.ndarray | None=None, beta: float=32.0, electron_count: float=8.0, mixing: tuple[float, ...]=(0.6, 0.35, 0.75), hamiltonian_threshold: float=0.135, density_depth: int=2, force_index: int=0) -> float:
    import numpy as np
    from numbers import Real, Integral
    if positions is None:
        x=np.array([0.,.72,1.83,3.05,4.12,5.55,6.44,7.90,9.15,10.02,11.48,12.73,14.05,15.22,16.87,18.31,20.12,22.01])
    else:
        if np.iscomplexobj(positions):raise ValueError("positions must be real")
        x=np.asarray(positions,dtype=float)
    if x.shape!=(18,):raise ValueError("exactly 18 positions are required")
    if isinstance(beta,(bool,np.bool_)) or not isinstance(beta,Real) or not np.isfinite(beta) or beta<=0:
        raise ValueError("beta must be finite and positive")
    if isinstance(electron_count,(bool,np.bool_)) or not isinstance(electron_count,Real) or not np.isfinite(electron_count) or not 0<electron_count<18:
        raise ValueError("electron_count must be in (0,18)")
    if np.iscomplexobj(mixing):raise ValueError("mixing must be real")
    alphas=np.asarray(mixing,dtype=float)
    if alphas.ndim!=1 or not np.all(np.isfinite(alphas)) or np.any(alphas<0) or np.any(alphas>1):
        raise ValueError("mixing must be a vector of factors in [0,1]")
    if isinstance(hamiltonian_threshold,(bool,np.bool_)) or not isinstance(hamiltonian_threshold,Real) or not np.isfinite(hamiltonian_threshold) or hamiltonian_threshold<0:
        raise ValueError("Hamiltonian threshold must be finite and nonnegative")
    if isinstance(density_depth,(bool,np.bool_)) or not isinstance(density_depth,Integral) or density_depth<1:
        raise ValueError("density_depth must be positive integer")
    d,s,h0,ds,dh0=build_periodic_chain_tangent(x,24.0,force_index)
    signs_s=np.array([1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,1,-1,1,-1])
    signs_r=np.array([-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1])
    c_s,o=build_colored_signed_probes(d,4.9,signs_s)
    c_r,r=build_colored_signed_probes(d,7.3,signs_r)
    q_s=canonical_block_krylov_basis(s,o,3)
    y=projected_inverse_sqrt_action(s,o,q_s)
    m=extract_sparse_operator(y,d,c_s,signs_s,2.4,0.006)
    gamma=1.1*np.eye(18)+0.15*np.exp(-d/3.0)
    idx=np.arange(18)
    charge=0.12*np.cos(2*np.pi*idx/18)+0.04*np.sin(4*np.pi*idx/18)
    charge=charge-charge.mean()
    for iteration in range(alphas.size+1):
        potential=gamma@charge
        h=h0+0.5*s*(potential[:,None]+potential[None,:])
        kh_action=m@(h@y)
        kh=extract_sparse_operator(kh_action,d,c_s,signs_s,3.4,hamiltonian_threshold)
        mu,rho,xrho=solve_fixed_population_density(s,m,kh,r,d,c_r,signs_r,beta,electron_count,density_depth,3.6)
        if iteration<alphas.size:
            output_charge=np.sum(rho*s.T,axis=1)-float(electron_count)/18
            charge=(1-alphas[iteration])*charge+alphas[iteration]*output_charge
    h_tangent=dh0+0.5*ds*(potential[:,None]+potential[None,:])
    terms=compute_frozen_potential_force(m,h,xrho,rho,ds,h_tangent,d,c_r,signs_r,3.6)
    return float(round(float(terms[2]),10))
SCICODE_GOLD_EOF
