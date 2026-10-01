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

def compute_scenario_gap(probabilities: np.ndarray, confidence: float) -> float:
    p = np.asarray(probabilities, dtype=float)
    a = float(confidence)
    if p.ndim != 1 or p.size == 0 or np.any(~np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("probabilities must be a nonempty positive finite vector")
    if not np.isclose(p.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    if not (0.0 < a < 1.0):
        raise ValueError("confidence must lie in (0,1)")
    best = 0.0
    for mask in range(1 << p.size):
        mass = float(sum(p[j] for j in range(p.size) if (mask >> j) & 1))
        if mass < a and mass > best:
            best = mass
    return float(a - best)

import numpy as np

def compute_loss_vector(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    R = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if R.ndim != 2 or w.ndim != 1 or R.shape[1] != w.size or R.shape[0] == 0:
        raise ValueError("incompatible return and weight dimensions")
    if np.any(~np.isfinite(R)) or np.any(~np.isfinite(w)):
        raise ValueError("inputs must be finite")
    return -R @ w

import numpy as np

def compute_cvar(losses: np.ndarray, probabilities: np.ndarray, beta: float) -> float:
    L = np.asarray(losses, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    b = float(beta)
    if L.ndim != 1 or p.ndim != 1 or L.size == 0 or L.size != p.size:
        raise ValueError("losses and probabilities must be compatible vectors")
    if np.any(~np.isfinite(L)) or np.any(~np.isfinite(p)) or np.any(p <= 0.0) or not np.isclose(p.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("invalid scenario probabilities or losses")
    if not (0.0 < b < 1.0):
        raise ValueError("beta must lie in (0,1)")
    vals = np.unique(np.sort(L))
    return float(min(float(u + p @ np.maximum(L-u, 0.0)/(1.0-b)) for u in vals))

import numpy as np

def build_dc_risk_terms(cvar_lower: float, cvar_alpha: float, alpha: float, gamma: float, tau: float) -> np.ndarray:
    a=float(alpha); g=float(gamma); t=float(tau)
    lo=float(cvar_lower); hi=float(cvar_alpha)
    if not (0.0 < a < 1.0 and 0.0 < g < a):
        raise ValueError("invalid confidence or gamma")
    A=(1.0-a+g)/g
    B=(1.0-a)/g
    left=A*lo-t
    right=B*hi
    return np.array([A,B,left,right,max(left,right)], dtype=float)

import numpy as np

def compute_cvar_subgradient(returns: np.ndarray, probabilities: np.ndarray, weights: np.ndarray, beta: float) -> np.ndarray:
    R=np.asarray(returns,dtype=float); p=np.asarray(probabilities,dtype=float); w=np.asarray(weights,dtype=float); b=float(beta)
    if R.ndim!=2 or w.ndim!=1 or p.ndim!=1 or R.shape[0]!=p.size or R.shape[1]!=w.size or p.size==0:
        raise ValueError("incompatible dimensions")
    if np.any(~np.isfinite(R)) or np.any(~np.isfinite(p)) or np.any(~np.isfinite(w)) or np.any(p<=0) or not np.isclose(p.sum(),1.0,atol=1e-12,rtol=0.0):
        raise ValueError("invalid finite scenario data")
    if not (0.0<b<1.0):
        raise ValueError("beta must lie in (0,1)")
    L=-R@w
    order=np.argsort(L,kind='stable')
    cum=0.0; u=None
    for j in order:
        cum += p[j]
        if cum >= b-1e-13:
            u=float(L[j]); break
    theta=np.zeros(p.size,dtype=float)
    equal=np.isclose(L,u,atol=1e-10,rtol=0.0)
    tail=L>u+1e-10
    theta[tail]=1.0
    rem=(1.0-b)-float(p[tail].sum())
    for j in np.where(equal)[0]:
        if rem <= 1e-12: break
        take=min(float(p[j]),rem)
        theta[j]=take/float(p[j])
        rem-=take
    if rem>1e-8: raise ValueError("tail mass could not be allocated")
    return -(p*theta)@R/(1.0-b)

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

def _qp_mu(probabilities, returns):
    return np.einsum("ts,tsn->tn", probabilities, returns)

def _qp_dual_extremes(probabilities, beta):
    cap=np.asarray(probabilities,float)/(1.0-beta)
    n=cap.size; out=[]
    for mask in range(1<<n):
        q=np.zeros(n); ids=[i for i in range(n) if (mask>>i)&1]; total=float(cap[ids].sum())
        if np.isclose(total,1.0,atol=1e-12,rtol=0.0) and np.all(cap[ids] <= 1.0+1e-12):
            q[ids]=cap[ids]; out.append(q)
    for j in range(n):
        others=[i for i in range(n) if i!=j]
        for mask in range(1<<len(others)):
            q=np.zeros(n); ids=[others[k] for k in range(len(others)) if (mask>>k)&1]
            total=float(cap[ids].sum()); rem=1.0-total
            if rem>1e-12 and rem<=cap[j]+1e-12 and total<1.0-1e-12:
                q[ids]=cap[ids]; q[j]=rem; out.append(q)
    unique=[]
    for q in out:
        if not any(np.allclose(q,r,atol=1e-12,rtol=0.0) for r in unique): unique.append(q)
    return np.asarray(unique,float)


def _qp_subproblem(probabilities, returns, w_prev, c, lambda1, lambda2, rho, alpha, gamma, tau, nu, w, linearized_h_gradient):
    p=np.asarray(probabilities,float); R=np.asarray(returns,float); wp=np.asarray(w_prev,float); cc=np.asarray(c,float)
    mu=_qp_mu(p,R); A=(1-alpha+gamma)/gamma; B=(1-alpha)/gamma
    pieces=[[ _qp_dual_extremes(p[t],alpha-gamma), _qp_dual_extremes(p[t],alpha) ] for t in range(2)]
    # x(6), transaction auxiliaries d(6), one epigraph variable q per period.
    N=14; qidx=[12,13]
    def obj(z):
        x=z[:6]; d=z[6:12]
        return float(-sum(mu[t]@x[3*t:3*t+3] for t in range(2)) + (lambda2+nu/2.0)*np.dot(x,x)
                     + lambda1*(cc@d[:3]+cc@d[3:]) + rho*(z[12]+z[13]) - linearized_h_gradient@x)
    def jac(z):
        g=np.zeros(N); x=z[:6]
        g[:6]= -np.concatenate(mu)+(2*lambda2+nu)*x-linearized_h_gradient
        g[6:12]=lambda1*np.r_[cc,cc]; g[12:]=rho
        return g
    eq_rows=[]; eq_lb=[]; eq_ub=[]; in_rows=[]; in_lb=[]; in_ub=[]
    def add_eq(row,val=1.0): eq_rows.append(row); eq_lb.append(val); eq_ub.append(val)
    def add_in(row,lo=0.0,hi=np.inf): in_rows.append(row); in_lb.append(lo); in_ub.append(hi)
    for t in range(2):
        row=np.zeros(N); row[3*t:3*t+3]=1.0; add_eq(row,1.0)
    for i in range(3):
        row=np.zeros(N); row[6+i]=1.0; row[i]=-1.0; add_in(row,-wp[i])
        row=np.zeros(N); row[6+i]=1.0; row[i]=1.0; add_in(row,wp[i])
        row=np.zeros(N); row[9+i]=1.0; row[3+i]=-1.0; row[i]=1.0; add_in(row)
        row=np.zeros(N); row[9+i]=1.0; row[3+i]=1.0; row[i]=-1.0; add_in(row)
    for t in range(2):
        xs=slice(3*t,3*t+3); qi=qidx[t]
        for qv in pieces[t][0]:
            a=-qv@R[t]
            row=np.zeros(N); row[qi]=-1.0; row[xs]=A*a; add_in(row,-np.inf,tau)
        for qv in pieces[t][1]:
            a=-qv@R[t]
            row=np.zeros(N); row[qi]=-1.0; row[xs]=B*a; add_in(row,-np.inf,0.0)
    lo=np.full(N,-np.inf); hi=np.full(N,np.inf); lo[:6]=0.0; hi[:6]=1.0; lo[6:12]=0.0
    bounds=Bounds(lo,hi)
    eq_con=LinearConstraint(np.vstack(eq_rows),np.asarray(eq_lb),np.asarray(eq_ub))
    in_con=LinearConstraint(np.vstack(in_rows),np.asarray(in_lb),np.asarray(in_ub))
    constraints=[eq_con,in_con]
    starts=[np.asarray(w,float).copy()]
    eps=1e-3
    for sign in (1.0,-1.0):
        st=np.asarray(w,float).copy(); st[:,0]+=sign*eps; st[:,1]-=sign*eps; starts.append(st)
    best=None
    for start in starts:
        z=np.zeros(N); z[:6]=start.ravel(); z[6:9]=np.abs(start[0]-wp); z[9:12]=np.abs(start[1]-start[0])
        for t in range(2):
            vals1=[A*(-qv@R[t]@start[t])-tau for qv in pieces[t][0]]
            vals2=[B*(-qv@R[t]@start[t]) for qv in pieces[t][1]]
            z[12+t]=max(vals1+vals2)
        res=minimize(obj,z,jac=jac,method='SLSQP',bounds=bounds,constraints=constraints,
                     options={'ftol':1e-13,'maxiter':20000,'disp':False})
        if not np.all(np.isfinite(res.x)): continue
        primal_lo=max(float(np.max(np.maximum(eq_con.lb-eq_con.A@res.x,0.0))),float(np.max(np.maximum(in_con.lb-in_con.A@res.x,0.0))))
        primal_hi=max(float(np.max(np.maximum(eq_con.A@res.x-eq_con.ub,0.0))),float(np.max(np.maximum(in_con.A@res.x-in_con.ub,0.0))))
        if primal_lo>5e-8 or primal_hi>5e-8: continue
        if best is None or res.fun < best.fun-1e-12: best=res
    if best is None: raise ValueError("convex subproblem failed to produce a feasible solution")
    return best.x[:6].reshape(2,3)


def solve_regularized_subproblem(probabilities: np.ndarray, returns: np.ndarray, w_prev: np.ndarray, c: np.ndarray, lambda1: float, lambda2: float, rho: float, alpha: float, gamma: float, tau: float, nu: float, w: np.ndarray, linearized_h_gradient: np.ndarray) -> np.ndarray:
    p=np.asarray(probabilities,float); R=np.asarray(returns,float); wp=np.asarray(w_prev,float); cc=np.asarray(c,float); ww=np.asarray(w,float); h=np.asarray(linearized_h_gradient,float)
    if p.shape!=(2,6) or R.shape!=(2,6,3) or wp.shape!=(3,) or cc.shape!=(3,) or ww.shape!=(2,3) or h.shape!=(6,):
        raise ValueError("invalid shapes")
    if np.any(p<=0) or not np.allclose(p.sum(axis=1),1.0,atol=1e-12,rtol=0.0): raise ValueError("invalid probabilities")
    return _qp_subproblem(p,R,wp,cc,float(lambda1),float(lambda2),float(rho),float(alpha),float(gamma),float(tau),float(nu),ww,h)

import numpy as np


def _project_simplex(v):
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or v.size == 0 or np.any(~np.isfinite(v)):
        raise ValueError("projection input must be a finite nonempty vector")

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    idx = np.nonzero(
        u - cssv / (np.arange(v.size) + 1.0) > 0.0
    )[0][-1]

    return np.maximum(
        v - cssv[idx] / (idx + 1.0),
        0.0,
    )


def _phi(
    probabilities,
    returns,
    w_prev,
    c,
    lambda1,
    lambda2,
    rho,
    alpha,
    gamma,
    tau,
    x,
):
    p = np.asarray(probabilities, dtype=float)
    R = np.asarray(returns, dtype=float)
    wp = np.asarray(w_prev, dtype=float)
    cc = np.asarray(c, dtype=float)
    xx = np.asarray(x, dtype=float)

    mu = np.einsum("ts,tsn->tn", p, R)

    value = -sum(mu[t] @ xx[t] for t in range(2))
    value += float(lambda2) * float(np.dot(xx.ravel(), xx.ravel()))

    value += float(lambda1) * (
        cc @ np.abs(xx[0] - wp)
        + cc @ np.abs(xx[1] - xx[0])
    )

    for t in range(2):
        gap = compute_scenario_gap(p[t], alpha)
        if not (0.0 < gamma < gap):
            raise ValueError(
                "gamma must satisfy the finite-scenario gap condition"
            )

        losses = compute_loss_vector(R[t], xx[t])

        cvar_lower = compute_cvar(
            losses, p[t], alpha - gamma
        )
        cvar_alpha = compute_cvar(
            losses, p[t], alpha
        )

        terms = build_dc_risk_terms(
            cvar_lower,
            cvar_alpha,
            alpha,
            gamma,
            tau,
        )

        value += float(rho) * (
            terms[4] - terms[1] * cvar_alpha
        )

    return float(value)


def take_ibdca_iteration(
    probabilities: np.ndarray,
    returns: np.ndarray,
    w_prev: np.ndarray,
    c: np.ndarray,
    lambda1: float,
    lambda2: float,
    rho: float,
    alpha: float,
    gamma: float,
    tau: float,
    theta: float,
    nu: float,
    sigma: float,
    eta: float,
    bar_lambda: float,
    x_k: np.ndarray,
    x_km1: np.ndarray,
) -> np.ndarray:

    p = np.asarray(probabilities, dtype=float)
    R = np.asarray(returns, dtype=float)
    wp = np.asarray(w_prev, dtype=float)
    cc = np.asarray(c, dtype=float)
    x = np.asarray(x_k, dtype=float)
    xm1 = np.asarray(x_km1, dtype=float)

    if (
        p.shape != (2, 6)
        or R.shape != (2, 6, 3)
        or wp.shape != (3,)
        or cc.shape != (3,)
        or x.shape != (2, 3)
        or xm1.shape != (2, 3)
    ):
        raise ValueError("invalid shapes")

    # Inertial extrapolation followed by blockwise simplex projection.
    extrapolated = x + float(theta) * (x - xm1)

    w = np.vstack(
        [_project_simplex(extrapolated[t]) for t in range(2)]
    )

    # Objective safeguard.
    if (
        _phi(
            p, R, wp, cc,
            lambda1, lambda2, rho,
            alpha, gamma, tau, w
        )
        >
        _phi(
            p, R, wp, cc,
            lambda1, lambda2, rho,
            alpha, gamma, tau, x
        ) + 1e-14
    ):
        w = x.copy()

    # CVaR subgradients from the earlier sub-problem.
    B = (1.0 - float(alpha)) / float(gamma)

    subgradients = np.vstack([
        float(rho)
        * B
        * compute_cvar_subgradient(
            R[t], p[t], w[t], alpha
        )
        for t in range(2)
    ])

    u = subgradients + float(nu) * w

    # Strongly-convex DCA subproblem from step 6.
    y = solve_regularized_subproblem(
        p,
        R,
        wp,
        cc,
        lambda1,
        lambda2,
        rho,
        alpha,
        gamma,
        tau,
        nu,
        w,
        u.ravel(),
    )

    d = y - w

    # Maximum feasible extrapolation along d.
    lambda_max = np.inf

    for t in range(2):
        for i in range(3):
            if d[t, i] < 0.0:
                lambda_max = min(
                    lambda_max,
                    -w[t, i] / d[t, i],
                )

    lam = min(
        float(bar_lambda),
        float(lambda_max),
    )

    # Boosted Armijo line search.
    while lam > 1.0:
        trial = w + lam * d

        dnorm2 = float(np.dot(d.ravel(), d.ravel()))

        rhs = min(
            _phi(
                p, R, wp, cc,
                lambda1, lambda2, rho,
                alpha, gamma, tau, y
            ),
            _phi(
                p, R, wp, cc,
                lambda1, lambda2, rho,
                alpha, gamma, tau, w
            )
            - float(sigma) * lam * lam * dnorm2,
        )

        trial_value = _phi(
            p, R, wp, cc,
            lambda1, lambda2, rho,
            alpha, gamma, tau, trial
        )

        if trial_value <= rhs + 1e-12:
            break

        lam = max(
            1.0,
            float(eta) * lam,
        )

    return w + lam * d

import numpy as np


def run_ibdca_target(
    probabilities: np.ndarray,
    returns: np.ndarray,
    w_prev: np.ndarray,
    c: np.ndarray,
    lambda1: float,
    lambda2: float,
    rho: float,
    alpha: float,
    gamma: float,
    tau: float,
    theta: float,
    nu: float,
    sigma: float,
    eta: float,
    bar_lambda: float,
    x0: np.ndarray,
    K: int,
) -> float:

    p = np.asarray(probabilities, dtype=float)
    R = np.asarray(returns, dtype=float)
    wp = np.asarray(w_prev, dtype=float)
    cc = np.asarray(c, dtype=float)

    x = np.asarray(x0, dtype=float).copy()
    xm1 = np.asarray(x0, dtype=float).copy()

    k = int(K)

    if (
        p.shape != (2, 6)
        or R.shape != (2, 6, 3)
        or wp.shape != (3,)
        or cc.shape != (3,)
        or x.shape != (2, 3)
        or k < 0
    ):
        raise ValueError("invalid inputs")

    for _ in range(k):
        xn = take_ibdca_iteration(
            p,
            R,
            wp,
            cc,
            lambda1,
            lambda2,
            rho,
            alpha,
            gamma,
            tau,
            theta,
            nu,
            sigma,
            eta,
            bar_lambda,
            x,
            xm1,
        )

        xm1, x = x, xn

    return float(x[1, 0])
SCICODE_GOLD_EOF
