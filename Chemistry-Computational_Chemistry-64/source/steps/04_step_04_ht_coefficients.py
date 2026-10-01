"""
Resolve a Gaussian dipole moment into powers of the common HT scale lambda.

The power of lambda counts HT vertices, not vibrational quanta. Six linear dipoles produce terms through sixth order in lambda, even when each dipole is only linear in coordinates.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ht_coefficients(mu: np.ndarray, L: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Resolve a Gaussian dipole moment into powers of the common HT scale lambda.
    
    Parameters
    ----------
    mu : complex ndarray, shape (J,)
        Condon coefficient at each insertion, 1 <= J <= 8; zeros are allowed.
    L : complex ndarray, shape (J,)
        Linear coefficients of the normalized Gaussian source generating function.
    K : complex ndarray, shape (J,J)
        Complex symmetric Hessian, not a Hermitian matrix.
    
    Returns
    -------
    result : complex ndarray, shape (J+1,)
        Coefficients c_r, r=0,...,J, in
        [prod_i (mu_i+lambda*d/ds_i) exp(L.s+0.5*s.T*K*s)]_(s=0).
        Each source can be differentiated at most once. The full polynomial,
        including mixed-mode terms, is required. No coefficient is conjugated.
        Inputs satisfy the stated shapes.
        Numeric outputs are tested with absolute and relative tolerances
        of 1e-9; integer outputs must be exact.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _oracle_ht_coefficients(mu: np.ndarray, L: np.ndarray, K: np.ndarray) -> np.ndarray:
    mu, L, K = map(np.asarray, (mu, L, K))
    n = len(mu)
    moments = np.zeros(1 << n, dtype=complex)
    moments[0] = 1
    for mask in range(1, 1 << n):
        low = mask & -mask
        i = low.bit_length()-1
        rest = mask ^ low
        value = L[i]*moments[rest]
        bits = rest
        while bits:
            bit = bits & -bits
            j = bit.bit_length()-1
            value += K[i, j]*moments[rest ^ bit]
            bits ^= bit
        moments[mask] = value
    result = np.zeros(n+1, dtype=complex)
    for mask in range(1 << n):
        weight = 1+0j
        for i in range(n):
            if not (mask >> i) & 1:
                weight *= mu[i]
        result[mask.bit_count()] += weight*moments[mask]
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'mu=np.array([0.7,0.8])\n'
               'L=np.array([0.2j,0.3])\n'
               'K=np.array([[3,0.4-0.1j],[0.4-0.1j,7]])',
      'call': 'ht_coefficients(*deepcopy((mu, L, K)))',
      'gold_call': '_oracle_ht_coefficients(*deepcopy((mu, L, K)))',
      'name': 'legacy_1',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'mu=np.zeros(4)\n'
               'L=np.zeros(4)\n'
               'K=np.ones((4,4))',
      'call': 'ht_coefficients(*deepcopy((mu, L, K)))',
      'gold_call': '_oracle_ht_coefficients(*deepcopy((mu, L, K)))',
      'name': 'legacy_2',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'mu=np.array([0.,0.8,-0.2])\n'
               'L=np.array([0.3,-0.1j,0.2])\n'
               'K=np.array([[0,0.5j,0.1],[0.5j,0,-0.4],[0.1,-0.4,0]])',
      'call': 'ht_coefficients(*deepcopy((mu, L, K)))',
      'gold_call': '_oracle_ht_coefficients(*deepcopy((mu, L, K)))',
      'name': 'legacy_3',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'mu=np.array([.12,.83,-.17,.64,.09,-.21])\n'
               'L=np.array([.1+.2j,.2-.3j,-.3,.4j,.05,.1j])\n'
               'a=np.arange(36).reshape(6,6)/200\n'
               'K=(a+a.T)*(1+0.3j)',
      'call': 'ht_coefficients(*deepcopy((mu, L, K)))',
      'gold_call': '_oracle_ht_coefficients(*deepcopy((mu, L, K)))',
      'name': 'legacy_4',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'mu=np.array([0.3+0.2j])\n'
               'L=np.array([-0.4j])\n'
               'K=np.array([[9.]])',
      'call': 'ht_coefficients(*deepcopy((mu, L, K)))',
      'gold_call': '_oracle_ht_coefficients(*deepcopy((mu, L, K)))',
      'name': 'legacy_5',
      'tol': 1e-09}]
