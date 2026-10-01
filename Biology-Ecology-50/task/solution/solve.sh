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

def movement_kernel(coords: "np.ndarray", lam: float, barrier_x: float, delta: float, p0: float) -> "np.ndarray":
    pts=np.asarray(coords,dtype=float)
    dist=np.linalg.norm(pts[:,None,:]-pts[None,:,:],axis=2)
    sides=pts[:,0]<barrier_x
    weights=np.exp(-lam*dist)*np.where(sides[:,None]!=sides[None,:],delta,1.)
    np.fill_diagonal(weights,0.)
    M=(1.-p0)*weights/weights.sum(axis=1,keepdims=True)
    np.fill_diagonal(M,p0)
    return M

import numpy as np
import math

def adult_transition(M: "np.ndarray", t0: int, t1: int) -> "np.ndarray":
    if t1 < t0:
        raise ValueError("Adult movement must be forward in time")
    return np.linalg.matrix_power(M, 1 + t1 - t0)

import numpy as np
import math

def reverse_origin(M: "np.ndarray", current: int, t0: int, t1: int) -> "np.ndarray":
    P = adult_transition(M, t0, t1)
    col = P[:, current]
    return col / col.sum()

import numpy as np
import math

def larval_denominator(nf: float, beta: float, te: int, tl: int, mu_e: float, mu_l: float) -> float:
    return nf * beta * (1-mu_e)**te * sum((1-mu_l)**a for a in range(tl))

import numpy as np
import math

def mother_larva(M: "np.ndarray", x1: int, t1: int, x2: int, t2: int, nf: float, beta: float, te: int, tl: int, ta: int, mu_a: float, mu_e: float, mu_l: float) -> float:
    total = 0.0
    for y in range(t2-te-(tl-1), t2-te+1):
        if t1-ta < y <= t1:
            q = reverse_origin(M, x1, y, t1)[x2]
            total += (1-mu_a)**(t1-y) * q * beta * (1-mu_e)**te * (1-mu_l)**(t2-y-te)
    return total / larval_denominator(nf,beta,te,tl,mu_e,mu_l)

import numpy as np

def adult_denominator(nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float) -> float:
    return float(nf*beta*(1-mu_e)**te*(1-mu_l)**tl*(1-mu_p)**tp*sum((1-mu_a)**a for a in range(ta)))

import numpy as np

def larva_adult_movement(M: "np.ndarray", x1: int, x2: int, y1: int, y2: int, t2: int, te: int, tl: int, tp: int) -> float:
    dev=te+tl+tp
    if y2>=y1:
        return float(adult_transition(M,y1+dev,t2)[x1,x2])
    earlier=reverse_origin(M,x1,y2,y1)
    offspring=adult_transition(M,y2+dev,t2)
    return float(earlier @ offspring[:,x2])

import numpy as np

def sibling_larva_adult(M: "np.ndarray", x1: int, t1: int, x2: int, t2: int, nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float) -> float:
    dev=te+tl+tp;sl=1-mu_l;sa=1-mu_a
    age_norm=sum(sl**a for a in range(tl))
    total=0.0
    for y1 in range(t1-te-(tl-1),t1-te+1):
        age1=t1-y1-te
        p_l=sl**age1/age_norm
        low=max(y1-(ta-1),t2-dev-(ta-1))
        high=min(y1+ta-1,t2-dev)
        for y2 in range(low,high+1):
            adult_age=t2-y2-dev
            offspring=beta*(1-mu_e)**te*sl**tl*(1-mu_p)**tp*sa**adult_age
            move=larva_adult_movement(M,x1,x2,y1,y2,t2,te,tl,tp)
            total+=0.5*p_l*sa**abs(y2-y1)*move*offspring
    return float(total/adult_denominator(nf,beta,te,tl,tp,ta,mu_a,mu_e,mu_l,mu_p))

import numpy as np
import math

def _mixed_binomial_log(p,n,k):
    if not (0<=k<=n and 0<=p<=1):
        raise ValueError("Invalid count or probability")
    if p==0:return 0.0 if k==0 else -math.inf
    if p==1:return 0.0 if k==n else -math.inf
    return k*math.log(p)+(n-k)*math.log1p(-p)

def joint_kin_score(M: "np.ndarray", mo_rows: list, la_rows: list, nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float) -> float:
    total=0.0
    for x1,t1,x2,t2,nfs,nls,k in mo_rows:
        individual=mother_larva(M,x1,t1,x2,t2,nf,beta,te,tl,ta,mu_a,mu_e,mu_l)
        group=-math.expm1(nfs*math.log1p(-individual))
        total+=_mixed_binomial_log(group,nls,k)
    for x1,t1,x2,t2,nas,k in la_rows:
        prob=sibling_larva_adult(M,x1,t1,x2,t2,nf,beta,te,tl,tp,ta,mu_a,mu_e,mu_l,mu_p)
        total+=_mixed_binomial_log(prob,nas,k)
    return float(total)

import numpy as np
from scipy.optimize import differential_evolution

def fit_barrier(coords: "np.ndarray", lam: float, barrier_x: float, mo_rows: list, la_rows: list, nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float, bounds: tuple, p0_bounds: tuple) -> float:
    def _loss(pair):
        M=movement_kernel(coords,lam,barrier_x,pair[0],pair[1])
        return -joint_kin_score(M,mo_rows,la_rows,nf,beta,te,tl,tp,ta,mu_a,mu_e,mu_l,mu_p)
    result=differential_evolution(_loss,[bounds,p0_bounds],seed=7,popsize=12,tol=1e-10,maxiter=160,polish=True)
    return float(result.x[0])
SCICODE_GOLD_EOF
