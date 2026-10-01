"""
Advance both corrected prices and constraint multipliers with one IMEX-BDF1 startup and one IT correction per layer, followed by IMEX-BDF2. Return both complete histories and respect the explicitly specified boundary convention.

Section 5 introduces an auxiliary nonnegative multiplier for early exercise. The intermediate linear solve uses its preceding value, then the local IT correction enforces obstacle complementarity. Carrying the multiplier and corrected price history is essential; clipping a European price at the payoff is a different scheme. The raw docstring specifies both startup and multistep formulas completely.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def imex_american(nodes: 'np.ndarray', D: 'np.ndarray', J: 'np.ndarray', strike: 'float', intensity: 'float', jump_mean: 'float', jump_std: 'float', maturity: 'float', steps: 'int') -> 'tuple':
    r"""Advance price and multiplier with exactly one IT correction per layer.

    Set $P_i=\max(K-Ke^{x_i},0)$, $U^0=P$, $\phi^0=0$ and
    $\Delta t=T/N$. Obtain $b,R$ from american_boundary_tail.
    The startup intermediate interior equations are
    $$(I-\Delta t D)\widetilde U^1=U^0+
      \Delta t\{\lambda(JU^0+R)+\phi^0\}.$$
    At later layers, for $n=1,\ldots,N-1$, use
    $$(3I-2\Delta t D)\widetilde U^{n+1}=4U^n-U^{n-1}
      +2\Delta t\{\lambda[J(2U^n-U^{n-1})+R]+\phi^n\}.$$
    In each linear system, impose the two endpoint values $b$ and retain
    their column contributions in the interior equations. Use the corrected
    histories in the extrapolation, not intermediate histories.
    For each interior component perform the Section 5 IT correction
    $$U^{n+1}=\max(P,\widetilde U^{n+1}-h\phi^n),\qquad
      \phi^{n+1}=\max(0,\phi^n+(P-\widetilde U^{n+1})/h),$$
    where $h=\Delta t$ at startup and $h=2\Delta t/3$ thereafter.
    Maxima are componentwise. Keep corrected endpoint prices equal to $b$
    and endpoint multipliers exactly zero. There are no inner iterations,
    multiplier extrapolation, damping steps or post-hoc clipping elsewhere.
    The multiplier formula is algebraically equivalent to Equation (5.4)
    and avoids cancellation on continuation nodes.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite increasing log-price nodes, $M\geq2$, $x_0<0<x_{M-1}$.
    D : array_like, shape $(M,M)$
        Finite local differential matrix, including compensated drift and loss.
    J : array_like, shape $(M,M)$
        Finite interior jump matrix without intensity or exterior tail.
    strike : float
        Finite positive strike $K$.
    intensity : float
        Finite jump intensity $\lambda\geq0$.
    jump_mean : float
        Finite normal log-jump mean $\mu_J$.
    jump_std : float
        Finite positive normal log-jump standard deviation $s_J$.
    maturity : float
        Finite positive final elapsed time $T$.
    steps : int
        Number of time intervals $N\geq1$. A single interval uses only startup.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Price history then multiplier history, each of shape $(N+1,M)$,
        with time rows and node columns. Row zero is exactly payoff and zero,
        respectively. Outputs are unrounded binary64 arrays.

    Raises
    ------
    ValueError
        If shapes, finiteness, ordering, signs or integer count are invalid.

    The boundary-modified systems must be nonsingular and all intermediate
    expressions finite in binary64. Behavior outside that domain is unspecified.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import lu_factor, lu_solve

def _oracle_imex_american(nodes: "np.ndarray", D: "np.ndarray", J: "np.ndarray",
                  strike: float, intensity: float, jump_mean: float,
                  jump_std: float, maturity: float, steps: int) -> tuple:
    r"""Advance price and multiplier with exactly one IT correction per layer.

    Set $P_i=\max(K-Ke^{x_i},0)$, $U^0=P$, $\phi^0=0$ and
    $\Delta t=T/N$. Obtain $b,R$ from american_boundary_tail.
    The startup intermediate interior equations are
    $$(I-\Delta t D)\widetilde U^1=U^0+
      \Delta t\{\lambda(JU^0+R)+\phi^0\}.$$
    At later layers, for $n=1,\ldots,N-1$, use
    $$(3I-2\Delta t D)\widetilde U^{n+1}=4U^n-U^{n-1}
      +2\Delta t\{\lambda[J(2U^n-U^{n-1})+R]+\phi^n\}.$$
    In each linear system, impose the two endpoint values $b$ and retain
    their column contributions in the interior equations. Use the corrected
    histories in the extrapolation, not intermediate histories.
    For each interior component perform the Section 5 IT correction
    $$U^{n+1}=\max(P,\widetilde U^{n+1}-h\phi^n),\qquad
      \phi^{n+1}=\max(0,\phi^n+(P-\widetilde U^{n+1})/h),$$
    where $h=\Delta t$ at startup and $h=2\Delta t/3$ thereafter.
    Maxima are componentwise. Keep corrected endpoint prices equal to $b$
    and endpoint multipliers exactly zero. There are no inner iterations,
    multiplier extrapolation, damping steps or post-hoc clipping elsewhere.
    The multiplier formula is algebraically equivalent to Equation (5.4)
    and avoids cancellation on continuation nodes.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite increasing log-price nodes, $M\geq2$, $x_0<0<x_{M-1}$.
    D : array_like, shape $(M,M)$
        Finite local differential matrix, including compensated drift and loss.
    J : array_like, shape $(M,M)$
        Finite interior jump matrix without intensity or exterior tail.
    strike : float
        Finite positive strike $K$.
    intensity : float
        Finite jump intensity $\lambda\geq0$.
    jump_mean : float
        Finite normal log-jump mean $\mu_J$.
    jump_std : float
        Finite positive normal log-jump standard deviation $s_J$.
    maturity : float
        Finite positive final elapsed time $T$.
    steps : int
        Number of time intervals $N\geq1$. A single interval uses only startup.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Price history then multiplier history, each of shape $(N+1,M)$,
        with time rows and node columns. Row zero is exactly payoff and zero,
        respectively. Outputs are unrounded binary64 arrays.

    Raises
    ------
    ValueError
        If shapes, finiteness, ordering, signs or integer count are invalid.

    The boundary-modified systems must be nonsingular and all intermediate
    expressions finite in binary64. Behavior outside that domain is unspecified.
    """
    x = np.asarray(nodes, dtype=float)
    boundary, tail = _oracle_american_boundary_tail(x, strike, jump_mean, jump_std)
    D, J = np.asarray(D, dtype=float), np.asarray(J, dtype=float)
    if D.shape != (x.size, x.size) or J.shape != D.shape or not np.isfinite(D).all() or not np.isfinite(J).all():
        raise ValueError("finite matching square matrices required")
    maturity, intensity = float(maturity), float(intensity)
    if (not np.isfinite([maturity, intensity]).all() or maturity <= 0 or intensity < 0
            or isinstance(steps, (bool, np.bool_)) or not np.isscalar(steps)
            or not np.isfinite(steps) or int(steps) != steps or steps < 1):
        raise ValueError("positive maturity, nonnegative intensity and positive integer steps required")
    N = int(steps)
    dt = maturity/N
    payoff = np.maximum(float(strike)-float(strike)*np.exp(x), 0.0)
    U, phi = np.empty((N+1, x.size)), np.zeros((N+1, x.size))
    U[0] = payoff
    identity = np.eye(x.size)
    A1, A2 = identity-dt*D, 3*identity-2*dt*D
    for A in (A1, A2):
        A[0], A[-1] = identity[0], identity[-1]
    factors1 = lu_factor(A1)
    factors2 = lu_factor(A2) if N > 1 else None
    for n in range(N):
        if n == 0:
            h = dt
            rhs = U[0]+dt*(intensity*(J@U[0]+tail)+phi[0])
            factors = factors1
        else:
            h = 2*dt/3
            rhs = 4*U[n]-U[n-1]+2*dt*(intensity*(J@(2*U[n]-U[n-1])+tail)+phi[n])
            factors = factors2
        rhs[0], rhs[-1] = boundary
        intermediate = lu_solve(factors, rhs)
        U[n+1] = np.maximum(payoff, intermediate-h*phi[n])
        phi[n+1] = np.maximum(0.0, phi[n]+(payoff-intermediate)/h)
        U[n+1, 0], U[n+1, -1] = boundary
        phi[n+1, 0] = phi[n+1, -1] = 0.0
    return U, phi

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'nodes=[-1.5,-.7,-.1,.4,1.2]\nD,J=_oracle_merton_operators(nodes,3,23,.04,.21,.35,-.4,.32)', 'call': '[p.tolist() for p in imex_american(nodes,D,J,100.,.35,-.4,.32,.18,1)]', 'gold_call': '[p.tolist() for p in _oracle_imex_american(nodes,D,J,100.,.35,-.4,.32,.18,1)]', 'tol': 2e-09}, {'setup': 'nodes=[-1.5,-.7,-.1,.4,1.2]\nD,J=_oracle_merton_operators(nodes,3,23,.04,.21,.35,-.4,.32)', 'call': '[p.tolist() for p in imex_american(nodes,D,J,100.,.35,-.4,.32,.18,2)]', 'gold_call': '[p.tolist() for p in _oracle_imex_american(nodes,D,J,100.,.35,-.4,.32,.18,2)]', 'tol': 2e-09}, {'setup': 'nodes=[-1.5,-.7,-.1,.4,1.2]\nD,J=_oracle_merton_operators(nodes,3,23,.04,.21,.35,-.4,.32)', 'call': '[p.tolist() for p in imex_american(nodes,D,J,100.,.35,-.4,.32,.18,7)]', 'gold_call': '[p.tolist() for p in _oracle_imex_american(nodes,D,J,100.,.35,-.4,.32,.18,7)]', 'tol': 2e-09}, {'setup': 'nodes=_oracle_clustered_grid(-1.5,1.5,49,.4)\nD,J=_oracle_merton_operators(nodes,3,96,.05,.15,.1,-.9,.45)', 'call': '[p.tolist() for p in imex_american(nodes,D,J,100.,.1,-.9,.45,.25,31)]', 'gold_call': '[p.tolist() for p in _oracle_imex_american(nodes,D,J,100.,.1,-.9,.45,.25,31)]', 'tol': 2e-07}, {'setup': 'import numpy as np\nnodes=np.array([-1.2,-.4,.2,1.1])\nD=np.array([[-.4,.1,0.,0.],[.3,-.7,.2,.1],[.05,.15,-.6,.25],[0.,0.,.1,-.2]])\nJ=np.zeros((4,4))', 'call': '[p.tolist() for p in imex_american(nodes,D,J,120.,0.,-.3,.4,.25,4)]', 'gold_call': '[p.tolist() for p in _oracle_imex_american(nodes,D,J,120.,0.,-.3,.4,.25,4)]', 'tol': 2e-09}]
