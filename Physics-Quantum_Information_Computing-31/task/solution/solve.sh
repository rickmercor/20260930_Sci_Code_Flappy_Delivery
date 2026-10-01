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


def pi_perp_error_block(Z: "np.ndarray", K: int, N: int, V_perp: "np.ndarray") -> "np.ndarray":
    m = V_perp.shape[1]
    if Z.shape[0] != m:
        raise ValueError("Z's row count must match V_perp's column count")
    if N <= 0:
        raise ValueError("N must be positive")
    E_coords = (Z @ Z.conj().T - K * np.eye(m)) / N
    return V_perp @ E_coords @ V_perp.conj().T

import numpy as np


def project_rank_r_density_matrix(Ybar: "np.ndarray", r: int) -> "np.ndarray":
    d = Ybar.shape[0]
    if Ybar.shape[0] != Ybar.shape[1]:
        raise ValueError("Ybar must be square")
    if not np.allclose(Ybar, Ybar.conj().T, atol=1e-8):
        raise ValueError("Ybar must be Hermitian")
    if not (1 <= r <= d):
        raise ValueError("r must satisfy 1 <= r <= d")

    w, V = np.linalg.eigh(Ybar)
    idx = np.argsort(w)[::-1]
    w_sorted = w[idx]
    V_sorted = V[:, idx]
    top_r = w_sorted[:r]

    u = np.sort(top_r)[::-1]
    css = np.cumsum(u)
    candidates = np.nonzero(u * np.arange(1, r + 1) > (css - 1))[0]
    rho_idx = candidates[-1]
    theta = (css[rho_idx] - 1) / (rho_idx + 1)
    proj = np.maximum(top_r - theta, 0)

    sigma = (V_sorted[:, :r] * proj) @ V_sorted[:, :r].conj().T
    return sigma

import numpy as np

def _regularized_loss_gradient(F, A_ops, y, lam):
    rho=F@F.conj().T
    residual=np.einsum('kij,ji->k',A_ops,rho).real-y
    loss=0.5*np.dot(residual,residual)+lam*np.vdot(F,F).real
    adjoint=np.einsum('k,kij->ij',residual,A_ops)
    return float(loss),2.0*(adjoint@F+lam*F)

def regularized_backtrack_step(F: "np.ndarray", A_ops: "np.ndarray", y: "np.ndarray", lam: float) -> "np.ndarray":
    if lam<0 or len(A_ops)!=len(y):
        raise ValueError('nonnegative lam and matching measurement lengths required')
    value,grad=_regularized_loss_gradient(F,A_ops,y,lam)
    norm2=np.vdot(grad,grad).real
    step=1.0
    while True:
        trial=F-step*grad
        trial_value,_=_regularized_loss_gradient(trial,A_ops,y,lam)
        if trial_value<=value-1e-4*step*norm2:
            return trial
        step*=0.5
        if step==0.0:
            raise FloatingPointError('Armijo trial step underflowed')

import numpy as np

def regularized_recovery(rho_init: "np.ndarray", r: int, A_ops: "np.ndarray", y: "np.ndarray", n_iters: int, lam: float) -> "np.ndarray":
    if n_iters<0 or not 1<=r<=len(rho_init) or lam<0 or len(A_ops)!=len(y):
        raise ValueError('invalid rank, iterations, regularization, or measurement lengths')
    values,vectors=np.linalg.eigh(rho_init)
    indices=np.argsort(values)[::-1][:r]
    F=vectors[:,indices]*np.sqrt(np.maximum(values[indices],0))
    for _ in range(n_iters):
        F=regularized_backtrack_step(F,A_ops,y,lam)
    return F@F.conj().T

import numpy as np


def adjoint_operator_norm(xi: "np.ndarray", A_ops: "np.ndarray") -> float:
    if xi.shape[0] != A_ops.shape[0]:
        raise ValueError("xi and A_ops must have the same length")
    d = A_ops.shape[1]
    A_star_xi = np.zeros((d, d), dtype=complex)
    for i in range(xi.shape[0]):
        A_star_xi = A_star_xi + xi[i] * A_ops[i]
    eigs = np.linalg.eigvalsh(A_star_xi)
    return float(np.max(np.abs(eigs)))

import numpy as np

def _canonical_hermitian_basis(d):
    basis=[]
    for i in range(d):
        B=np.zeros((d,d),complex); B[i,i]=1
        basis.append(B)
    for i in range(d):
        for j in range(i+1,d):
            B=np.zeros((d,d),complex); B[i,j]=B[j,i]=1/np.sqrt(2)
            basis.append(B)
            B=np.zeros((d,d),complex); B[i,j]=1j/np.sqrt(2); B[j,i]=-1j/np.sqrt(2)
            basis.append(B)
    return np.array(basis)

def sensing_isometry_constants(A_ops: "np.ndarray", V_support: "np.ndarray") -> "np.ndarray":
    d,r_star=V_support.shape
    basis=_canonical_hermitian_basis(d)
    coefficients=np.einsum('aij,bji->ab',A_ops,basis).real
    eigenvalues=np.linalg.eigvalsh(coefficients.T@coefficients)
    Q,_=np.linalg.qr(V_support,mode='complete')
    tangent=[]
    for B in basis:
        if np.linalg.norm(B[r_star:,r_star:])<1e-14:
            tangent.append(Q@B@Q.conj().T)
    C=np.einsum('aij,bji->ab',A_ops,np.array(tangent)).real
    beta=np.linalg.eigvalsh(C.T@C)[-1]
    return np.array([eigenvalues[0],beta,eigenvalues[-1]],float)

import numpy as np


def certification_bound(op_norm: float, r_star: int, r: int, alpha: float, beta: float, lam: float) -> float:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if beta < alpha:
        raise ValueError("beta must be at least alpha")
    if beta / alpha >= 6.0 / (np.sqrt(5.0) + 2.0):
        raise ValueError("beta/alpha exceeds the bound's applicability threshold")

    numerator = (36.0 * lam + 12.0 * op_norm) * np.sqrt(r_star) + 10.0 * np.sqrt(r + r_star) * max(op_norm - lam, 0.0)
    denominator = 6.0 * alpha - (np.sqrt(5.0) + 2.0) * beta
    return float(numerator / denominator)

import numpy as np

def regularization_from_budget(op_norm: float, r_star: int, r: int, alpha: float, beta: float, budget: float) -> float:
    base=certification_bound(op_norm,r_star,r,alpha,beta,0.0)
    if budget<base:
        raise ValueError('conditional error budget is infeasible')
    knot=certification_bound(op_norm,r_star,r,alpha,beta,op_norm)
    denominator=6*alpha-(np.sqrt(5)+2)*beta
    if budget<=knot:
        slope=(36*np.sqrt(r_star)-10*np.sqrt(r+r_star))/denominator
        return float((budget-base)/slope)
    slope=36*np.sqrt(r_star)/denominator
    return float(op_norm+(budget-knot)/slope)

import numpy as np

def _recovery_bound_components(seed_rho,eigvals,seed_Z,J,N,weights,seed_noise,noise_scale,n_iters,budget_fraction):
    rng=np.random.default_rng(seed_rho)
    raw=rng.standard_normal((4,4))+1j*rng.standard_normal((4,4))
    Q,_=np.linalg.qr(raw)
    rho=(Q*np.r_[eigvals,0.0,0.0])@Q.conj().T
    rho=(rho+rho.conj().T)/2
    K=int(np.sum(J)); rng=np.random.default_rng(seed_Z)
    Z=(rng.standard_normal((2,K))+1j*rng.standard_normal((2,K)))/np.sqrt(2)
    Y=rho+pi_perp_error_block(Z,K,N,Q[:,2:])
    initial=project_rank_r_density_matrix(Y,2)
    canonical=_canonical_hermitian_basis(4)
    A_ops=np.array([Q@B@Q.conj().T for B in canonical])*np.sqrt(weights)[:,None,None]
    xi=np.random.default_rng(seed_noise).normal(scale=noise_scale,size=16)
    y=np.einsum('kij,ji->k',A_ops,rho).real+xi
    eta=adjoint_operator_norm(xi,A_ops)
    alpha,beta,beta_global=sensing_isometry_constants(A_ops,Q[:,:2])
    budget=budget_fraction*float(np.linalg.norm(initial-rho,'fro'))
    lam=regularization_from_budget(eta,2,2,alpha,beta,budget)
    recovered=regularized_recovery(initial,2,A_ops,y,n_iters,lam)
    bound=certification_bound(eta,2,2,alpha,beta,lam)
    if bound==0:
        raise ValueError('the selected bound must be nonzero')
    error=float(np.linalg.norm(recovered-rho,'fro'))
    return {'initial_error':float(np.linalg.norm(initial-rho,'fro')),'error':error,
            'eta':eta,'alpha':float(alpha),'beta':float(beta),'beta_global':float(beta_global),
            'budget':budget,'lambda':lam,'bound_zero':certification_bound(eta,2,2,alpha,beta,0.0),'bound':bound,'ratio':error/bound,'trace':float(np.trace(recovered).real)}

def recovery_bound_utilization(seed_rho: int, eigvals: "np.ndarray", seed_Z: int, J: "np.ndarray", N: int, weights: "np.ndarray", seed_noise: int, noise_scale: float, n_iters: int, budget_fraction: float) -> float:
    parts=_recovery_bound_components(seed_rho,eigvals,seed_Z,J,N,weights,seed_noise,noise_scale,n_iters,budget_fraction)
    return float(parts['ratio'])
SCICODE_GOLD_EOF
