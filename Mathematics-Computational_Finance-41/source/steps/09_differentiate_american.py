"""
Derive and propagate first and second jump-mean derivatives through the complete discrete American IMEX-BDF1/BDF2 and IT evolution. Account for the derivative of the implicit matrix, jump extrapolation and carried constraint multiplier.

The source Section 5 supplies the coupled intermediate solve and IT complementarity correction. This task differentiates that exact finite algorithm. On a stable branch pattern, implicit differentiation and differentiation of the complementarity branches give its ordinary parameter derivatives. The base trajectory must remain coupled to the multiplier.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def differentiate_american(nodes: 'np.ndarray', D_jets: 'np.ndarray', J_jets: 'np.ndarray', R_jets: 'np.ndarray', prices: 'np.ndarray', multipliers: 'np.ndarray', intensity: 'float', maturity: 'float') -> 'tuple':
    r"""Differentiate the complete IMEX/IT trajectory twice in jump mean.

    The supplied prices and multipliers are the base histories from
    imex_american, at the same nodes and model parameters used to form
    merton_mean_jets. Differentiate that finite algorithm, including its
    implicit systems, explicit jump extrapolation, carried multiplier,
    and the IT complementarity correction. The payoff, finite endpoint
    values, grid, quadrature and time intervals are independent of jump mean.
    Use analytic differentiation; do not estimate these derivatives from
    perturbed base solves, automatic differentiation packages, or complex steps.
    Ordinary derivatives and their product rules apply, not factorial-scaled
    Taylor coefficients. The base histories and operator jets are assumed
    mutually consistent; no reconstruction or reoptimization of them is needed.

    At each interior correction, a price strictly above its initial payoff
    identifies continuation. A price equal to payoff identifies exercise;
    for a degenerate exact tie with zero multiplier, use the exercise branch
    for derivative propagation. This makes the interface deterministic at
    ties. On a locally stable base branch pattern it equals differentiation
    of the full finite algorithm. The default task must be checked for this
    stability before interpreting the result as an ordinary derivative.
    Initial derivatives and all endpoint derivatives are exactly zero.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite increasing log-price nodes, $M\geq2$, straddling zero.
    D_jets, J_jets : array_like, shape $(3,M,M)$
        Finite matrices in derivative-order, evaluation-node, value-node order.
    R_jets : array_like, shape $(3,M)$
        Finite intensity-free exterior integral derivatives.
    prices, multipliers : array_like, shape $(N+1,M)$
        Consistent base histories, $N\geq1$. Row zero gives payoff and zero.
    intensity : float
        Finite nonnegative jump intensity, held fixed.
    maturity : float
        Finite positive horizon; $\Delta t=T/N$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Price jets and multiplier jets, each shape $(3,N+1,M)$.
        Slot zero is a copy of the supplied history. Slots one and two
        are its first and second jump-mean derivatives, without rounding.

    Raises
    ------
    ValueError
        For invalid shape, ordering, finiteness or stated sign conditions.

    The boundary-modified linear systems must be nonsingular and all
    analytic derivatives must remain finite in binary64 arithmetic.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import ndtr, roots_legendre
from scipy.linalg import lu_factor, lu_solve

def _oracle_differentiate_american(nodes: "np.ndarray", D_jets: "np.ndarray",
                           J_jets: "np.ndarray", R_jets: "np.ndarray",
                           prices: "np.ndarray", multipliers: "np.ndarray",
                           intensity: float, maturity: float) -> tuple:
    r"""Differentiate the complete IMEX/IT trajectory twice in jump mean.

    The supplied prices and multipliers are the base histories from
    imex_american, at the same nodes and model parameters used to form
    merton_mean_jets. Differentiate that finite algorithm, including its
    implicit systems, explicit jump extrapolation, carried multiplier,
    and the IT complementarity correction. The payoff, finite endpoint
    values, grid, quadrature and time intervals are independent of jump mean.
    Use analytic differentiation; do not estimate these derivatives from
    perturbed base solves, automatic differentiation packages, or complex steps.
    Ordinary derivatives and their product rules apply, not factorial-scaled
    Taylor coefficients. The base histories and operator jets are assumed
    mutually consistent; no reconstruction or reoptimization of them is needed.

    At each interior correction, a price strictly above its initial payoff
    identifies continuation. A price equal to payoff identifies exercise;
    for a degenerate exact tie with zero multiplier, use the exercise branch
    for derivative propagation. This makes the interface deterministic at
    ties. On a locally stable base branch pattern it equals differentiation
    of the full finite algorithm. The default task must be checked for this
    stability before interpreting the result as an ordinary derivative.
    Initial derivatives and all endpoint derivatives are exactly zero.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite increasing log-price nodes, $M\geq2$, straddling zero.
    D_jets, J_jets : array_like, shape $(3,M,M)$
        Finite matrices in derivative-order, evaluation-node, value-node order.
    R_jets : array_like, shape $(3,M)$
        Finite intensity-free exterior integral derivatives.
    prices, multipliers : array_like, shape $(N+1,M)$
        Consistent base histories, $N\geq1$. Row zero gives payoff and zero.
    intensity : float
        Finite nonnegative jump intensity, held fixed.
    maturity : float
        Finite positive horizon; $\Delta t=T/N$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Price jets and multiplier jets, each shape $(3,N+1,M)$.
        Slot zero is a copy of the supplied history. Slots one and two
        are its first and second jump-mean derivatives, without rounding.

    Raises
    ------
    ValueError
        For invalid shape, ordering, finiteness or stated sign conditions.

    The boundary-modified linear systems must be nonsingular and all
    analytic derivatives must remain finite in binary64 arithmetic.
    """
    x = np.asarray(nodes, dtype=float)
    D, J, R, U, phi = map(lambda a: np.asarray(a, dtype=float),
                          (D_jets, J_jets, R_jets, prices, multipliers))
    M = x.size
    if (x.ndim != 1 or M < 2 or not np.isfinite(x).all()
            or np.any(np.diff(x) <= 0) or not x[0] < 0 < x[-1]):
        raise ValueError('increasing finite nodes straddling zero required')
    if (D.shape != (3,M,M) or J.shape != D.shape or R.shape != (3,M)
            or U.ndim != 2 or U.shape[1] != M or U.shape[0] < 2
            or phi.shape != U.shape or not all(np.isfinite(a).all() for a in (D,J,R,U,phi))):
        raise ValueError('finite jet tensors and matching histories required')
    lam, T = float(intensity), float(maturity)
    if not np.isfinite([lam,T]).all() or lam < 0 or T <= 0:
        raise ValueError('nonnegative intensity and positive maturity required')
    N = U.shape[0]-1
    dt = T/N
    V = np.zeros((3,N+1,M)); psi = np.zeros_like(V)
    V[0], psi[0] = U, phi
    I = np.eye(M)
    A1, A2 = I-dt*D[0], 3*I-2*dt*D[0]
    for A in (A1,A2):
        A[[0,-1]] = I[[0,-1]]
    factor1 = lu_factor(A1)
    factor2 = lu_factor(A2) if N > 1 else None
    for n in range(N):
        first = n == 0
        c, h = (dt,dt) if first else (2*dt,2*dt/3)
        ex = V[:,n] if first else 2*V[:,n]-V[:,n-1]
        pred = V[:,n] if first else 4*V[:,n]-V[:,n-1]
        base_tilde = U[n+1]-h*(phi[n+1]-phi[n])
        rhs1 = pred[1]+c*(lam*(J[1]@ex[0]+J[0]@ex[1]+R[1])+psi[1,n]+D[1]@base_tilde)
        rhs1[[0,-1]] = 0.0
        fac = factor1 if first else factor2
        tilde1 = lu_solve(fac,rhs1)
        rhs2 = pred[2]+c*(lam*(J[2]@ex[0]+2*J[1]@ex[1]+J[0]@ex[2]+R[2])+psi[2,n]
                           +D[2]@base_tilde+2*D[1]@tilde1)
        rhs2[[0,-1]] = 0.0
        tilde2 = lu_solve(fac,rhs2)
        free = U[n+1] > U[0]
        free[[0,-1]] = False
        for q, tilde in ((1,tilde1),(2,tilde2)):
            V[q,n+1] = np.where(free,tilde-h*psi[q,n],0.0)
            psi[q,n+1] = np.where(free,0.0,psi[q,n]-tilde/h)
            V[q,n+1,[0,-1]] = 0.0
            psi[q,n+1,[0,-1]] = 0.0
    return V, psi

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'nodes=[-1.5,-.7,-.1,.4,1.2]\nD,J,R=_oracle_merton_mean_jets(nodes,3,23,.04,.21,.35,-.4,.32,100.)\nU,phi=_oracle_imex_american(nodes,D[0],J[0],100.,.35,-.4,.32,.18,1)', 'call': 'differentiate_american(nodes,D,J,R,U,phi,.35,.18)', 'gold_call': '_oracle_differentiate_american(nodes,D,J,R,U,phi,.35,.18)', 'tol': 2e-07}, {'setup': 'nodes=[-1.5,-.7,-.1,.4,1.2]\nD,J,R=_oracle_merton_mean_jets(nodes,3,23,.04,.21,.35,-.4,.32,100.)\nU,phi=_oracle_imex_american(nodes,D[0],J[0],100.,.35,-.4,.32,.18,2)', 'call': 'differentiate_american(nodes,D,J,R,U,phi,.35,.18)', 'gold_call': '_oracle_differentiate_american(nodes,D,J,R,U,phi,.35,.18)', 'tol': 2e-07}, {'setup': 'nodes=[-1.5,-.7,-.1,.4,1.2]\nD,J,R=_oracle_merton_mean_jets(nodes,3,23,.04,.21,.35,-.4,.32,100.)\nU,phi=_oracle_imex_american(nodes,D[0],J[0],100.,.35,-.4,.32,.18,7)', 'call': 'differentiate_american(nodes,D,J,R,U,phi,.35,.18)', 'gold_call': '_oracle_differentiate_american(nodes,D,J,R,U,phi,.35,.18)', 'tol': 2e-07}, {'setup': 'nodes=_oracle_clustered_grid(-1.5,1.5,49,.4)\nD,J,R=_oracle_merton_mean_jets(nodes,3,96,.05,.15,.1,-.9,.45,100.)\nU,phi=_oracle_imex_american(nodes,D[0],J[0],100.,.1,-.9,.45,.25,31)', 'call': 'differentiate_american(nodes,D,J,R,U,phi,.1,.25)', 'gold_call': '_oracle_differentiate_american(nodes,D,J,R,U,phi,.1,.25)', 'tol': 2e-07}, {'setup': 'nodes=_oracle_clustered_grid(-1.7,1.7,25,0.)\nD,J,R=_oracle_merton_mean_jets(nodes,1,40,.06,.19,0.,.12,.5,80.)\nU,phi=_oracle_imex_american(nodes,D[0],J[0],80.,0.,.12,.5,.18,9)', 'call': 'differentiate_american(nodes,D,J,R,U,phi,0.,.18)', 'gold_call': '_oracle_differentiate_american(nodes,D,J,R,U,phi,0.,.18)', 'tol': 2e-07}]
