"""
Expand Tr[mu_order [mu_(order-1), [... [mu_0, rho] ...]]].

Each commutator contributes a left and a right action on the density matrix. Trace cyclicity closes each branch into a dipole correlation function. The branch sign is $$(-1)^{N_R}.$$

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def commutator_words(order: int) -> np.ndarray:
    """Expand Tr[mu_order [mu_(order-1), [... [mu_0, rho] ...]]].
    
    Parameters
    ----------
    order : int
        Number of commutators, 1 <= order <= 7. Labels 0,...,order identify
        chronological dipole times; rho is at the far right after cyclic rotation.
    
    Returns
    -------
    result : integer ndarray, shape (2**order, order+2)
        Each row contains sign followed by the left-to-right dipole word.
        Enumerate branch masks b=0,...,2**order-1. Bit k=0 chooses left
        multiplication by mu_k; bit k=1 chooses right multiplication and a minus
        sign. After all order interactions, put mu_order on the left, then move
        the factors to the right of rho to the front by trace cyclicity.
        Do not sort, reverse, merge or discard words. Inputs satisfy this domain.
        Numeric outputs are tested with absolute and relative tolerances
        of 1e-9; integer outputs must be exact.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _oracle_commutator_words(order: int) -> np.ndarray:
    rows = []
    for mask in range(1 << order):
        left, right, sign = [], [], 1
        for k in range(order):
            if (mask >> k) & 1:
                right.append(k)
                sign = -sign
            else:
                left.insert(0, k)
        rows.append([sign] + right + [order] + left)
    return np.asarray(rows, dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n',
      'call': 'commutator_words(1)',
      'gold_call': '_oracle_commutator_words(1)',
      'name': 'legacy_1',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n',
      'call': 'commutator_words(2)',
      'gold_call': '_oracle_commutator_words(2)',
      'name': 'legacy_2',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n',
      'call': 'commutator_words(3)',
      'gold_call': '_oracle_commutator_words(3)',
      'name': 'legacy_3',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n',
      'call': 'commutator_words(5)',
      'gold_call': '_oracle_commutator_words(5)',
      'name': 'legacy_4',
      'tol': 1e-09}]
