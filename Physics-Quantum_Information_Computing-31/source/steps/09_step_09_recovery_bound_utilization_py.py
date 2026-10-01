"""
Run the complete acquisition-to-recovery bound-utilization pipeline.

The final orchestrator measures finite reconstruction error against the conditional landscape envelope. The acquisition estimate supplies the warm start, and the realized weighted adjoint noise enters the theorem.

Returns
-------
float, the empirical recovery error divided by its conditional landscape bound.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def recovery_bound_utilization(seed_rho: int, eigvals: "np.ndarray", seed_Z: int, J: "np.ndarray", N: int, weights: "np.ndarray", seed_noise: int, noise_scale: float, n_iters: int, budget_fraction: float) -> float:
    """Combine acquisition, regularized sensing, and conditional landscape analysis.

    Parameters
    ----------
    seed_rho : int
        Seed for a 4-by-4 complex standard-normal matrix (real array first,
        imaginary array second); use its unadjusted NumPy QR factor Q.
    eigvals : np.ndarray
        Two positive target eigenvalues summing to one; remaining eigenvalues zero.
    seed_Z : int
        Seed for a (2,sum(J)) standard complex Gaussian draw, real array first.
    J : np.ndarray
        Positive integer round outcomes. Acquisition uses their sum K.
    N : int
        Positive total acquisition sample count. Use the supplied out-of-support
        error (Z Z-dagger minus K times identity) divided by N, embedded with
        Q's last two columns; other error blocks vanish by task convention.
    weights : np.ndarray
        Sixteen positive squared measurement weights: four diagonals first,
        then real and positive-upper-imaginary off-diagonal Hermitian elements
        for lexicographic index pairs. Rotate each canonical element by Q and
        scale by the square root of its weight. The first weight must be minimal,
        making the full-space lower constant sharp on rank-2 PSD errors.
        The sharp tangent-to-lower ratio must satisfy the source threshold.
    seed_noise : int
        Seed for independent real normal noise, in measurement-operator order.
    noise_scale : float
        Nonnegative standard deviation of the sixteen noise entries.
    n_iters : int
        Nonnegative count of unconstrained regularized Armijo updates. Start
        with the top-two-eigenpair factor of the nearest rank-2 density matrix
        to the acquisition estimate.
    budget_fraction : float
        Positive fraction of the acquisition Frobenius error defining the
        conditional recovery-error budget. Choose the largest nonnegative
        regularization satisfying that budget. The budget must be feasible.
        At least one of noise_scale and the resulting regularization is positive.

    Returns
    -------
    result : float
        Actual recovered-matrix Frobenius error divided by the landscape theorem's
        conditional bound, as a finite native float. Values above one are allowed:
        a finite iterate need not be an exact second-order critical point.
        All randomness uses separate generators.

    Raises
    ------
    ValueError
        If the error budget is infeasible, the selected bound is zero, or
        the isometry constants violate the bound applicability condition.
    FloatingPointError
        If a recovery line search underflows.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _recovery_bound_components(seed_rho,eigvals,seed_Z,J,N,weights,seed_noise,noise_scale,n_iters,budget_fraction):
    rng=np.random.default_rng(seed_rho)
    raw=rng.standard_normal((4,4))+1j*rng.standard_normal((4,4))
    Q,_=np.linalg.qr(raw)
    rho=(Q*np.r_[eigvals,0.0,0.0])@Q.conj().T
    rho=(rho+rho.conj().T)/2
    K=int(np.sum(J)); rng=np.random.default_rng(seed_Z)
    Z=(rng.standard_normal((2,K))+1j*rng.standard_normal((2,K)))/np.sqrt(2)
    Y=rho+_oracle_pi_perp_error_block(Z,K,N,Q[:,2:])
    initial=_oracle_project_rank_r_density_matrix(Y,2)
    canonical=_canonical_hermitian_basis(4)
    A_ops=np.array([Q@B@Q.conj().T for B in canonical])*np.sqrt(weights)[:,None,None]
    xi=np.random.default_rng(seed_noise).normal(scale=noise_scale,size=16)
    y=np.einsum('kij,ji->k',A_ops,rho).real+xi
    eta=_oracle_adjoint_operator_norm(xi,A_ops)
    alpha,beta,beta_global=_oracle_sensing_isometry_constants(A_ops,Q[:,:2])
    budget=budget_fraction*float(np.linalg.norm(initial-rho,'fro'))
    lam=_oracle_regularization_from_budget(eta,2,2,alpha,beta,budget)
    recovered=_oracle_regularized_recovery(initial,2,A_ops,y,n_iters,lam)
    bound=_oracle_certification_bound(eta,2,2,alpha,beta,lam)
    if bound==0:
        raise ValueError('the selected bound must be nonzero')
    error=float(np.linalg.norm(recovered-rho,'fro'))
    return {'initial_error':float(np.linalg.norm(initial-rho,'fro')),'error':error,
            'eta':eta,'alpha':float(alpha),'beta':float(beta),'beta_global':float(beta_global),
            'budget':budget,'lambda':lam,'bound_zero':_oracle_certification_bound(eta,2,2,alpha,beta,0.0),'bound':bound,'ratio':error/bound,'trace':float(np.trace(recovered).real)}

def _oracle_recovery_bound_utilization(seed_rho: int, eigvals: "np.ndarray", seed_Z: int, J: "np.ndarray", N: int, weights: "np.ndarray", seed_noise: int, noise_scale: float, n_iters: int, budget_fraction: float) -> float:
    parts=_recovery_bound_components(seed_rho,eigvals,seed_Z,J,N,weights,seed_noise,noise_scale,n_iters,budget_fraction)
    return float(parts['ratio'])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return separate normal, boundary, and edge comparison cases."""
    return [{'setup': 'import numpy as np\n'
               'eigs=np.array([0.6,0.4]); J=np.array([2,2,2]); N=12\n'
               'w=np.array([1.00,1.08,9.0,6.0,1.12,1.20,1.04,1.10,1.16,1.06,1.14,1.18,1.02,1.09,7.0,8.0])\n'
               'seed=31; noise=0.001; n=50; budget_fraction=0.5',
      'call': 'recovery_bound_utilization(seed,eigs.copy(),4100,J.copy(),N,w.copy(),2026,noise,n,budget_fraction)',
      'gold_call': '_oracle_recovery_bound_utilization(seed,eigs.copy(),4100,J.copy(),N,w.copy(),2026,noise,n,budget_fraction)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'eigs=np.array([0.6,0.4]); J=np.array([2,2,2]); N=12\n'
               'w=np.array([1.00,1.08,9.0,6.0,1.12,1.20,1.04,1.10,1.16,1.06,1.14,1.18,1.02,1.09,7.0,8.0])\n'
               'seed=32; noise=0.0; n=0; budget_fraction=0.5',
      'call': 'recovery_bound_utilization(seed,eigs.copy(),4100,J.copy(),N,w.copy(),2026,noise,n,budget_fraction)',
      'gold_call': '_oracle_recovery_bound_utilization(seed,eigs.copy(),4100,J.copy(),N,w.copy(),2026,noise,n,budget_fraction)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'eigs=np.array([0.6,0.4]); J=np.array([2,2,2]); N=12\n'
               'w=np.array([1.00,1.08,9.0,6.0,1.12,1.20,1.04,1.10,1.16,1.06,1.14,1.18,1.02,1.09,7.0,8.0])\n'
               'seed=33; noise=0.0002; n=100; budget_fraction=0.8',
      'call': 'recovery_bound_utilization(seed,eigs.copy(),4100,J.copy(),N,w.copy(),2026,noise,n,budget_fraction)',
      'gold_call': '_oracle_recovery_bound_utilization(seed,eigs.copy(),4100,J.copy(),N,w.copy(),2026,noise,n,budget_fraction)',
      'tol': 1e-08}]
