"""
Compute the three optical population-decay rates.

The four-level basis is [0,0′,X,X′]. gamma0 is the actual zero-phonon population-decay rate. Use the relative displaced-oscillator transition probabilities, retaining the full (1-S)^2 factor. The common overlap normalization is already in gamma0 is the actual zero-phonon population-decay rate. gamma_prime is the rate of each phonon-assisted optical transition, X→0′ and X′→0, and gamma1 is the rate of X′→0′.

Returns
-------
return result  # real ndarray, shape (3,), ordered [gamma0 (X→0), gamma_prime (each of X→0′ and X′→0), gamma1 (X′→0′)], in ps^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def polaron_optical_rates(huang_rhys: float, gamma0: float) -> np.ndarray:
    """Compute the three optical population-decay rates.

    Parameters
    ----------
    huang_rhys : float in [0,1]
        Dimensionless S for the finite four-level model.
    gamma0 : nonnegative float
        Zero-phonon population-decay rate in ps^-1.

    Returns
    -------
    real ndarray, shape (3,), ordered [gamma0 (X→0), gamma_prime (each of X→0′ and X′→0), gamma1 (X′→0′)], in ps^-1.

    Raises
    ------
    ValueError
        If inputs fall outside the documented numerical domain.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def _oracle_polaron_optical_rates(huang_rhys: float, gamma0: float) -> np.ndarray:
    if not (np.isfinite(huang_rhys) and 0<=huang_rhys<=1 and np.isfinite(gamma0) and gamma0>=0):raise ValueError("invalid coupling or population rate")
    return gamma0*np.array([1.,huang_rhys,(1-huang_rhys)**2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nhuang_rhys=0.0\ngamma0=0.003\n', 'call': 'polaron_optical_rates(huang_rhys,gamma0)', 'gold_call': '_oracle_polaron_optical_rates(huang_rhys,gamma0)'}, {'setup': 'import numpy as np\nhuang_rhys=0.08\ngamma0=0.004\n', 'call': 'polaron_optical_rates(huang_rhys,gamma0)', 'gold_call': '_oracle_polaron_optical_rates(huang_rhys,gamma0)'}, {'setup': 'import numpy as np\nhuang_rhys=1.0\ngamma0=0.007\n', 'call': 'polaron_optical_rates(huang_rhys,gamma0)', 'gold_call': '_oracle_polaron_optical_rates(huang_rhys,gamma0)'}, {'setup': 'import numpy as np\nhuang_rhys=0.12\ngamma0=0.0\n', 'call': 'polaron_optical_rates(huang_rhys,gamma0)', 'gold_call': '_oracle_polaron_optical_rates(huang_rhys,gamma0)'}, {'setup': 'import numpy as np\nhuang_rhys=0.3819660112501051\ngamma0=0.005\n', 'call': 'polaron_optical_rates(huang_rhys,gamma0)', 'gold_call': '_oracle_polaron_optical_rates(huang_rhys,gamma0)'}]
