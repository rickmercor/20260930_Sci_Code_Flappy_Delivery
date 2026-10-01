"""
Determine the exact coherent-state mixture spectrum in cyclic order.

Sections IV.1–IV.2 orthogonalize the lost coherent states. The basis convention is <psi_s|gamma_k>=sqrt(w_s)*exp(2*pi*i*s*k/N); gamma_0 has positive coefficients in that basis.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cyclic_weights(N: int, mean_photons: float) -> np.ndarray:
    """Determine the exact coherent-state mixture spectrum in cyclic order.

    N : int
        Alphabet size, 2 or 4.
    mean_photons : float
        Mean photon number of Eve's coherent state, in [0,16].
    Returns
    -------
    ndarray, shape (N,), float
        The normalized nonnegative spectrum w_s of the uniform lost-state mixture,
        ordered by the cyclic index s=0,...,N-1 in the specified basis.
        At zero photons, w_0=1 and all remaining weights vanish.
    Raises
    ------
    ValueError
        If N or mean_photons is outside its domain.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def _oracle_cyclic_weights(N: int, mean_photons: float) -> np.ndarray:
    if N not in (2,4) or not np.isfinite(mean_photons) or not 0<=mean_photons<=16:
        raise ValueError('N in {2,4}, mean_photons in [0,16]')
    w=np.zeros(N);term=np.exp(-mean_photons);w[0]=term
    for k in range(1,192):
        term*=mean_photons/k;w[k%N]+=term
    return w/w.sum()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight distinct deterministic scientific regimes."""
    return [{'setup': 'import numpy as np', 'call': 'cyclic_weights(2,0.0)', 'gold_call': '_oracle_cyclic_weights(2,0.0)', 'tol': 1e-08},
     {'setup': 'import numpy as np',
      'call': 'cyclic_weights(2,1e-08)',
      'gold_call': '_oracle_cyclic_weights(2,1e-08)',
      'tol': 1e-08},
     {'setup': 'import numpy as np', 'call': 'cyclic_weights(2,0.37)', 'gold_call': '_oracle_cyclic_weights(2,0.37)', 'tol': 1e-08},
     {'setup': 'import numpy as np', 'call': 'cyclic_weights(2,16.0)', 'gold_call': '_oracle_cyclic_weights(2,16.0)', 'tol': 1e-08},
     {'setup': 'import numpy as np',
      'call': 'cyclic_weights(4,1e-05)',
      'gold_call': '_oracle_cyclic_weights(4,1e-05)',
      'tol': 1e-08},
     {'setup': 'import numpy as np',
      'call': 'cyclic_weights(4,1.5707963267948966)',
      'gold_call': '_oracle_cyclic_weights(4,1.5707963267948966)',
      'tol': 1e-08},
     {'setup': 'import numpy as np',
      'call': 'cyclic_weights(4,3.141592653589793)',
      'gold_call': '_oracle_cyclic_weights(4,3.141592653589793)',
      'tol': 1e-08},
     {'setup': 'import numpy as np', 'call': 'cyclic_weights(4,16.0)', 'gold_call': '_oracle_cyclic_weights(4,16.0)', 'tol': 1e-08}]
