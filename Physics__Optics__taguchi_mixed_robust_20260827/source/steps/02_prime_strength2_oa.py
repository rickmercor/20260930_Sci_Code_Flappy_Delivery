"""
Construct the canonical prime-level strength-two orthogonal array.

The finite-field construction fixes the experiment row and column order.

Returns
-------
np.ndarray of shape (q*q,k), int: canonical zero-based OA.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def prime_strength2_oa(q: int, k: int) -> np.ndarray:
    """Return the canonical `OA(q^2,k,q,2)` for prime `q`.

Parameters
----------
q : int
    Prime number of levels.
k : int
    Number of factors, satisfying `2 <= k <= q+1`.

Returns
-------
oa : numpy.ndarray
    Integer array of shape `(q*q, k)`. Rows are lexicographic in
    `(a,b)`; columns are `[a,b,a+b,a+2b,...]` modulo `q`.

Conventions
-----------
The signature returns an integer placeholder with shape (q*q,k); it does not implement the OA construction."""
    return np.zeros((q * q, k), dtype=int)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _is_prime_integer(n: int) -> bool:
    if n < 2:
        return False
    return all(n % d for d in range(2, int(np.sqrt(n)) + 1))

def _oracle_prime_strength2_oa(q: int, k: int) -> np.ndarray:
    q = int(q)
    k = int(k)
    if not _is_prime_integer(q) or not 2 <= k <= q + 1:
        raise ValueError("Require prime q and 2 <= k <= q+1")
    rows = []
    for a in range(q):
        for b in range(q):
            row = [a, b]
            row.extend((a + m * b) % q for m in range(1, k - 1))
            rows.append(row)
    return np.asarray(rows, dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Eight regimes vary prime field order, row count, and the number of OA columns."""
    return [{'setup': '', 'call': 'prime_strength2_oa(3, 2)', 'gold_call': '_oracle_prime_strength2_oa(3, 2)'},
        {'setup': '', 'call': 'prime_strength2_oa(3, 4)', 'gold_call': '_oracle_prime_strength2_oa(3, 4)'},
        {'setup': '', 'call': 'prime_strength2_oa(2, 3)', 'gold_call': '_oracle_prime_strength2_oa(2, 3)'},
        {'setup': '', 'call': 'prime_strength2_oa(5, 6)', 'gold_call': '_oracle_prime_strength2_oa(5, 6)'},
        {'setup': '', 'call': 'prime_strength2_oa(5, 2)', 'gold_call': '_oracle_prime_strength2_oa(5, 2)'},
        {'setup': '', 'call': 'prime_strength2_oa(7, 5)', 'gold_call': '_oracle_prime_strength2_oa(7, 5)'},
        {'setup': '', 'call': 'prime_strength2_oa(11, 12)', 'gold_call': '_oracle_prime_strength2_oa(11, 12)'},
        {'setup': '', 'call': 'prime_strength2_oa(13, 7)', 'gold_call': '_oracle_prime_strength2_oa(13, 7)'}]
