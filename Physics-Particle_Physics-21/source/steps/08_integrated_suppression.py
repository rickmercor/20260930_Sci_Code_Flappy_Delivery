"""
Aggregate the two constrained Gaussian measures and a branch-one mean.

The finite branch integrals are constructed from the constraint measure motivated by Eq. 10. Quadrature is in xi, and the logarithmic representation keeps the Gaussian ratios evaluable.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def integrated_suppression(fractions: "np.ndarray", weights: "np.ndarray", logweights: "np.ndarray", xi_critical: float) -> "np.ndarray":
    """Aggregate the two constrained Gaussian measures and a branch-one mean.

    fractions : ndarray, shape (M,)
        Dimensionless quadrature coordinates q=xi/xi_c.
    weights : ndarray, shape (M,)
        Positive quadrature weights for dq, in the same order.
    logweights : ndarray, shape (M,2)
        Logarithms of G, with columns constrained branches b=0 and b=1.
    xi_critical : float > 0
        Dimensionless xi_c multiplying dq to give dxi.
    Returns
    -------
    ndarray, length 4
        [B,log(Z_0),log(Z_1),mean_q_1], with B=log(Z_0)-log(Z_1). The mean uses the normalized branch-1 Gaussian measure. Inputs are finite; logweights may be too negative for direct exponentiation.

    Raises
    ------
    ValueError
        If numerical input domains or array shapes are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_banded, eigvalsh_tridiagonal
from scipy.optimize import root, brentq
from scipy.special import logsumexp

def _oracle_integrated_suppression(fractions: "np.ndarray", weights: "np.ndarray", logweights: "np.ndarray", xi_critical: float) -> "np.ndarray":
    fractions=np.asarray(fractions,dtype=float);weights=np.asarray(weights,dtype=float);logweights=np.asarray(logweights,dtype=float)
    if fractions.ndim!=1 or not len(fractions) or weights.shape!=fractions.shape or logweights.shape!=(len(fractions),2) or not np.all(np.isfinite(fractions)) or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(logweights)) or np.any(weights<=0) or not np.isfinite(xi_critical) or xi_critical<=0:
        raise ValueError("Invalid quadrature inputs")
    logs=logsumexp(np.log(np.asarray(weights))[:,None]+logweights,axis=0)+np.log(xi_critical)
    posterior=np.exp(np.log(weights)+logweights[:,1]-logs[1]+np.log(xi_critical))
    mean_fraction=posterior@np.asarray(fractions)
    return np.array([logs[0]-logs[1],logs[0],logs[1],mean_fraction])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Constructed scientific cases, not published numerical examples."""
    return [{'setup': 'import numpy as np\n'
               'fractions = np.array([0.2, 0.8], dtype=float)\n'
               'weights = np.array([0.5, 0.5], dtype=float)\n'
               'logweights = np.array([[0, 0], [0, 0]], dtype=float)\n'
               'xi_critical = 2.0\n',
      'call': 'integrated_suppression(fractions.copy(), weights.copy(), logweights.copy(), xi_critical)',
      'gold_call': '_oracle_integrated_suppression(fractions.copy(), weights.copy(), logweights.copy(), xi_critical)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'fractions = np.array([0.2, 0.5, 0.8], dtype=float)\n'
               'weights = np.array([0.2, 0.5, 0.3], dtype=float)\n'
               'logweights = np.array([[-2, -5], [-1, -4], [-3, -6]], dtype=float)\n'
               'xi_critical = 4.0\n',
      'call': 'integrated_suppression(fractions.copy(), weights.copy(), logweights.copy(), xi_critical)',
      'gold_call': '_oracle_integrated_suppression(fractions.copy(), weights.copy(), logweights.copy(), xi_critical)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'fractions = np.array([0.2, 0.5, 0.8], dtype=float)\n'
               'weights = np.array([0.2, 0.5, 0.3], dtype=float)\n'
               'logweights = np.array([[-1002, -1005], [-1001, -1004], [-1003, -1006]], dtype=float)\n'
               'xi_critical = 4.0\n',
      'call': 'integrated_suppression(fractions.copy(), weights.copy(), logweights.copy(), xi_critical)',
      'gold_call': '_oracle_integrated_suppression(fractions.copy(), weights.copy(), logweights.copy(), xi_critical)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'fractions = np.array([0.1, 0.4, 0.9], dtype=float)\n'
               'weights = np.array([0.1, 0.7, 0.2], dtype=float)\n'
               'logweights = np.array([[1, -2], [-3, 2], [2, -1]], dtype=float)\n'
               'xi_critical = 1.0\n',
      'call': 'integrated_suppression(fractions.copy(), weights.copy(), logweights.copy(), xi_critical)',
      'gold_call': '_oracle_integrated_suppression(fractions.copy(), weights.copy(), logweights.copy(), xi_critical)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'fractions = np.array([0.2, 0.7], dtype=float)\n'
               'weights = np.array([0.3, 0.7], dtype=float)\n'
               'logweights = np.array([[0, -1], [1, 2]], dtype=float)\n'
               'xi_critical = 13.0\n',
      'call': 'integrated_suppression(fractions.copy(), weights.copy(), logweights.copy(), xi_critical)',
      'gold_call': '_oracle_integrated_suppression(fractions.copy(), weights.copy(), logweights.copy(), xi_critical)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'fractions = np.array([0.9, 0.1, 0.4], dtype=float)\n'
               'weights = np.array([0.2, 0.1, 0.7], dtype=float)\n'
               'logweights = np.array([[2, -1], [1, -2], [-3, 2]], dtype=float)\n'
               'xi_critical = 1.0\n',
      'call': 'integrated_suppression(fractions.copy(), weights.copy(), logweights.copy(), xi_critical)',
      'gold_call': '_oracle_integrated_suppression(fractions.copy(), weights.copy(), logweights.copy(), xi_critical)',
      'tol': 2e-06}]
