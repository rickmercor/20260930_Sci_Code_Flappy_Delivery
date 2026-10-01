"""
Solve the strongly convex allocation subproblem associated with one expansion point of the penalized portfolio objective.

The allocation domain is the product of two three-asset simplexes, with transaction costs linking the two period blocks and a quadratic term contributing to strong convexity.

Returns
-------
The returned value is the unique two-period allocation selected by the convex subproblem.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_regularized_subproblem(probabilities: np.ndarray, returns: np.ndarray, w_prev: np.ndarray, c: np.ndarray, lambda1: float, lambda2: float, rho: float, alpha: float, gamma: float, tau: float, nu: float, w: np.ndarray, linearized_h_gradient: np.ndarray) -> np.ndarray:
    """Return the allocation solving the regularized convex subproblem.

    Parameters
    ----------
    probabilities : np.ndarray
        Scenario probabilities, shape (2, 6).
    returns : np.ndarray
        Scenario returns, shape (2, 6, 3).
    w_prev : np.ndarray
        Previous holding, shape (3,).
    c : np.ndarray
        Asset transaction-cost coefficients, shape (3,).
    lambda1, lambda2, rho, alpha, gamma, tau, nu : float
        Model and decomposition controls.
    w : np.ndarray
        Current expansion allocation, shape (2, 3).
    linearized_h_gradient : np.ndarray
        Asset-space linearization vector for the concave DC component, length 6.

    Returns
    -------
    y : np.ndarray
        Unique two-period allocation returned by the convex subproblem. The
        subproblem must be solved to high accuracy (objective tolerance 1e-9 or
        tighter); the returned allocation is compared at 1e-6.

    Raises
    ------
    ValueError
        If dimensions are incompatible or the supplied data violate the tested input contract.
    """
    return y

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_solve_regularized_subproblem(probabilities: np.ndarray, returns: np.ndarray, w_prev: np.ndarray, c: np.ndarray, lambda1: float, lambda2: float, rho: float, alpha: float, gamma: float, tau: float, nu: float, w: np.ndarray, linearized_h_gradient: np.ndarray) -> np.ndarray:
    p=np.asarray(probabilities,float); R=np.asarray(returns,float); wp=np.asarray(w_prev,float); cc=np.asarray(c,float); ww=np.asarray(w,float); h=np.asarray(linearized_h_gradient,float)
    if p.shape!=(2,6) or R.shape!=(2,6,3) or wp.shape!=(3,) or cc.shape!=(3,) or ww.shape!=(2,3) or h.shape!=(6,):
        raise ValueError("invalid shapes")
    if np.any(p<=0) or not np.allclose(p.sum(axis=1),1.0,atol=1e-12,rtol=0.0): raise ValueError("invalid probabilities")
    return _qp_subproblem(p,R,wp,cc,float(lambda1),float(lambda2),float(rho),float(alpha),float(gamma),float(tau),float(nu),ww,h)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three deterministic subproblem cases."""
    base = """import numpy as np
p=np.array([[0.05,0.10,0.15,0.20,0.22,0.28],[0.08,0.09,0.12,0.16,0.24,0.31]],float)
R=np.array([[[-0.15,0.05,0.03],[0.04,-0.13,0.02],[0.03,0.02,-0.12],[-0.05,-0.04,0.06],[0.08,0.07,0.09],[-0.02,0.03,-0.06]],[[ -0.12,0.04,0.02],[0.05,-0.11,0.03],[0.02,0.03,-0.10],[-0.04,-0.03,0.05],[0.09,0.08,0.10],[-0.03,0.02,-0.05]]],float)
w_prev=np.array([0.40,0.35,0.25],float)
x0=np.array([[0.45,0.30,0.25],[0.30,0.45,0.25]],float)
c=np.array([0.06,0.10,0.14],float)
alpha=0.65; tau=0.02; lambda1=0.07; lambda2=0.02; rho=6.0; gamma=0.005; theta=0.05; nu=0.04; sigma=0.004; eta=0.5; bar_lambda=1.2"""
    return [
        {"setup": base, "call": "tuple(solve_regularized_subproblem(p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,nu,x0,np.zeros(6)))", "gold_call": "tuple(_oracle_solve_regularized_subproblem(p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,nu,x0,np.zeros(6)))", "tol": 1e-6},
        {"setup": base+"\nx_alt=np.array([[0.45,0.30,0.25],[0.30,0.45,0.25]],float); h=np.array([5.,0.,0.,5.,0.,0.])", "call": "tuple(solve_regularized_subproblem(p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,nu,x_alt,h))", "gold_call": "tuple(_oracle_solve_regularized_subproblem(p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,nu,x_alt,h))", "tol": 1e-6},
        {"setup": base+"\nx_edge=np.array([[0.45,0.30,0.25],[0.30,0.45,0.25]],float); h=np.array([0.,5.,0.,0.,5.,0.])", "call": "tuple(solve_regularized_subproblem(p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,nu,x_edge,h))", "gold_call": "tuple(_oracle_solve_regularized_subproblem(p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,nu,x_edge,h))", "tol": 1e-6},
    ]
