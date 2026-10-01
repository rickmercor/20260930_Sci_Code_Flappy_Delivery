"""
Continuous delay-density normalization for population mixtures

The formation-independent delay density is proportional to $$\tau^\alpha$$ on the inclusive support $$[\tau_{\min},\tau_{\max}]$$ and zero elsewhere. Normalize it by its continuous integral over that support, including the admissible exponent $$\alpha=-1$$ and its neighborhood. Later steps mix the resulting component densities as population fractions, so their normalization must have a meaning independent of the grid on which they are sampled. Time-grid quadrature weights are introduced by the inverse operator.

Returns
-------
np.ndarray of shape (N,), the sampled continuous-normalized delay density.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def delay_time_distribution(tau_grid: np.ndarray, alpha: float,
                            tau_min: float, tau_max: float) -> np.ndarray:
    r'''Sample a continuously normalized power-law delay density.

    Parameters
    ----------
    tau_grid : np.ndarray
        Nonempty one-dimensional finite real delay times >= 0, in Gyr.
    alpha : float
        Finite real exponent, including -1 and its neighborhood.
    tau_min, tau_max : float
        Finite real support limits in Gyr satisfying 0 < tau_min < tau_max.

    Returns
    -------
    density : np.ndarray
        Finite shape (N,), in Gyr^-1. Inclusive support, zero outside. Continuous normalization is preserved without rescaling the sampled sum.

    Raises
    ------
    ValueError
        If the input domains fail or the density cannot be evaluated finitely.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_delay_time_distribution(tau_grid: np.ndarray, alpha: float,
                                    tau_min: float, tau_max: float) -> np.ndarray:
    if not np.isrealobj(tau_grid):
        raise ValueError("delay samples must be real")
    try:
        tau = np.asarray(tau_grid,dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("delay samples must be finite real values") from exc
    if tau.ndim != 1 or tau.size == 0 or not np.all(np.isfinite(tau)) or np.any(tau<0):
        raise ValueError("nonempty finite nonnegative delay grid required")
    if any(not np.isscalar(v) or not np.isrealobj(v) for v in (alpha,tau_min,tau_max)):
        raise ValueError("exponent and limits must be real scalars")
    try:
        exponent, lo, hi = map(float,(alpha,tau_min,tau_max))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("exponent and limits must be finite real scalars") from exc
    if not np.all(np.isfinite([exponent,lo,hi])) or not 0<lo<hi:
        raise ValueError("finite exponent and 0 < lower < upper support required")
    width = np.log(hi)-np.log(lo)
    beta = exponent+1
    x = beta*width
    if not np.isfinite(x):
        raise ValueError("exponent-support product must be finite")
    if x == 0:
        relative = 0.0
    elif x > 50:
        relative = x+np.log1p(-np.exp(-x))-np.log(x)
    elif x < -50:
        relative = np.log1p(-np.exp(x))-np.log(-x)
    else:
        relative = np.log(np.expm1(x)/x)
    log_z = beta*np.log(lo)+np.log(width)+relative
    inside = (tau>=lo)&(tau<=hi)
    result = np.zeros_like(tau)
    with np.errstate(over='ignore',invalid='ignore'):
        result[inside] = np.exp(exponent*np.log(tau[inside])-log_z)
    if not np.all(np.isfinite(result)):
        raise ValueError("density must evaluate finitely")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent input copies in explicit scientific case specifications."""
    return [{'setup': 'import numpy as np\n'
               't=np.array([0, 0.005, 0.01, 0.02, 0.1, 1, 5, 13.8, 14],dtype=float)\n'
               'a,lo,hi=-1.73,0.01,13.8\n',
      'call': 'delay_time_distribution(t.copy(),a,lo,hi)',
      'gold_call': '_oracle_delay_time_distribution(t.copy(),a,lo,hi)',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               't=np.array([0.01, 0.03, 0.1, 1, 4, 13.8],dtype=float)\n'
               'a,lo,hi=-1,0.01,13.8\n',
      'call': 'delay_time_distribution(t.copy(),a,lo,hi)',
      'gold_call': '_oracle_delay_time_distribution(t.copy(),a,lo,hi)',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               't=np.array([0.01, 0.1, 1, 13.8],dtype=float)\n'
               'a,lo,hi=-0.99999999,0.01,13.8\n',
      'call': 'delay_time_distribution(t.copy(),a,lo,hi)',
      'gold_call': '_oracle_delay_time_distribution(t.copy(),a,lo,hi)',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               't=np.array([0.01, 0.1, 1, 13.8],dtype=float)\n'
               'a,lo,hi=-1.00000001,0.01,13.8\n',
      'call': 'delay_time_distribution(t.copy(),a,lo,hi)',
      'gold_call': '_oracle_delay_time_distribution(t.copy(),a,lo,hi)',
      'tol': 1e-06},
     {'setup': 'import numpy as np\nt=np.array([0, 0.2, 0.5, 1, 2, 5, 6],dtype=float)\na,lo,hi=-0.7,0.2,5\n',
      'call': 'delay_time_distribution(t.copy(),a,lo,hi)',
      'gold_call': '_oracle_delay_time_distribution(t.copy(),a,lo,hi)',
      'tol': 1e-06},
     {'setup': 'import numpy as np\nt=np.array([0.01, 0.02, 0.1, 0.4],dtype=float)\na,lo,hi=1.3,0.01,0.4\n',
      'call': 'delay_time_distribution(t.copy(),a,lo,hi)',
      'gold_call': '_oracle_delay_time_distribution(t.copy(),a,lo,hi)',
      'tol': 1e-06}]
