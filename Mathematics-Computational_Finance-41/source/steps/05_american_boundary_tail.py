"""
Compute the undiscounted American put boundary and analytic Gaussian exterior tail, using the complete domain, formula and return-shape contract in the raw docstring.

Section 4 integrates the American far-left payoff against the Gaussian log-jump density. Unlike the European tail it is time-independent and undiscounted. The task selects the finite-left-boundary convention in Section 2, resolving the different limiting value printed in Equation (5.2).

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def american_boundary_tail(nodes: 'np.ndarray', strike: 'float', jump_mean: 'float', jump_std: 'float') -> 'tuple':
    r"""Return the time-independent American Merton boundary and exterior tail.

    Write $a=x_0$ and $z_i=(a-x_i-\mu_J)/s_J$. The task adopts the
    finite-boundary convention in Section 2, not the limiting strike value
    printed in Equation (5.2): $b=(K-Ke^a,0)$. Section 4 gives
    $$R_i=K\Phi(z_i)-K\exp(x_i+\mu_J+s_J^2/2)\Phi(z_i-s_J).$$
    Here $\Phi$ is the standard normal CDF. Neither discounting nor jump
    intensity is included. The exterior payoff is used for all elapsed times.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing log-price nodes, $M\geq2$, with
        $x_0<0<x_{M-1}$.
    strike : float
        Finite strike $K>0$.
    jump_mean : float
        Finite normal log-jump mean $\mu_J$.
    jump_std : float
        Finite normal log-jump standard deviation $s_J>0$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Boundary values of shape $(2,)$, left then right, and exterior
        integral of shape $(M,)$, in node order, without rounding.

    Raises
    ------
    ValueError
        For invalid shape, ordering, finiteness or stated sign conditions.

    All displayed expressions must remain finite in binary64; behavior
    outside that representable domain is unspecified.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import ndtr

def _oracle_american_boundary_tail(nodes: "np.ndarray", strike: float,
                           jump_mean: float, jump_std: float) -> tuple:
    r"""Return the time-independent American Merton boundary and exterior tail.

    Write $a=x_0$ and $z_i=(a-x_i-\mu_J)/s_J$. The task adopts the
    finite-boundary convention in Section 2, not the limiting strike value
    printed in Equation (5.2): $b=(K-Ke^a,0)$. Section 4 gives
    $$R_i=K\Phi(z_i)-K\exp(x_i+\mu_J+s_J^2/2)\Phi(z_i-s_J).$$
    Here $\Phi$ is the standard normal CDF. Neither discounting nor jump
    intensity is included. The exterior payoff is used for all elapsed times.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing log-price nodes, $M\geq2$, with
        $x_0<0<x_{M-1}$.
    strike : float
        Finite strike $K>0$.
    jump_mean : float
        Finite normal log-jump mean $\mu_J$.
    jump_std : float
        Finite normal log-jump standard deviation $s_J>0$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Boundary values of shape $(2,)$, left then right, and exterior
        integral of shape $(M,)$, in node order, without rounding.

    Raises
    ------
    ValueError
        For invalid shape, ordering, finiteness or stated sign conditions.

    All displayed expressions must remain finite in binary64; behavior
    outside that representable domain is unspecified.
    """
    x = np.asarray(nodes, dtype=float)
    if (x.ndim != 1 or x.size < 2 or not np.isfinite(x).all()
            or np.any(np.diff(x) <= 0) or not x[0] < 0 < x[-1]):
        raise ValueError("ordered finite nodes straddling zero required")
    strike, jump_mean, jump_std = map(float, (strike, jump_mean, jump_std))
    if not np.isfinite([strike, jump_mean, jump_std]).all() or strike <= 0 or jump_std <= 0:
        raise ValueError("finite positive strike and jump_std required")
    z = (x[0]-x-jump_mean)/jump_std
    boundary = np.array([strike*(1-np.exp(x[0])), 0.0])
    tail = strike*ndtr(z)-strike*np.exp(x+jump_mean+0.5*jump_std**2)*ndtr(z-jump_std)
    return boundary, tail

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'nodes=[-1.5,-.7,0.,.4,1.2]', 'call': '[p.tolist() for p in american_boundary_tail(nodes,100.,-.9,.45)]', 'gold_call': '[p.tolist() for p in _oracle_american_boundary_tail(nodes,100.,-.9,.45)]', 'tol': 2e-09}, {'setup': 'nodes=[-1.2,-.3,.2,1.4]', 'call': '[p.tolist() for p in american_boundary_tail(nodes,110.,.12,.32)]', 'gold_call': '[p.tolist() for p in _oracle_american_boundary_tail(nodes,110.,.12,.32)]', 'tol': 2e-09}, {'setup': 'nodes=[-2.,-1.,-.2,.3,.9,1.6]', 'call': '[p.tolist() for p in american_boundary_tail(nodes,75.,-.45,.65)]', 'gold_call': '[p.tolist() for p in _oracle_american_boundary_tail(nodes,75.,-.45,.65)]', 'tol': 2e-09}]
