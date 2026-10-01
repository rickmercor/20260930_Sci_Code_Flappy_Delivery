"""
Enumerate the paired determinant space.

A seniority-zero determinant has the same occupied spatial orbitals in both spin sectors. It takes one sorted occupation tuple to identify it.

Returns
-------
return occupations
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_occupations(norb: int, npair: int) -> "np.ndarray":
    """Enumerate the paired determinant space.
    
    Parameters
    ----------
    norb : int
        Number of spatial orbitals, 1 through 6.
    npair : int
        Number of doubly occupied orbitals, 1 through norb.
    Returns
    -------
    occupations : integer ndarray, shape (binom(norb,npair),npair)
        Strictly increasing zero-based tuples, in lexicographic order.
    Raises
    ------
    ValueError
        If either argument is not an integer or 1 <= npair <= norb <= 6 fails.
    """
    return occupations

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from itertools import combinations
import numpy as np


def _oracle_pair_occupations(norb: int, npair: int) -> "np.ndarray":
    if not isinstance(norb, (int, np.integer)) or not isinstance(npair, (int, np.integer)) or not 1 <= npair <= norb <= 6:
        raise ValueError('Require 1 <= npair <= norb <= 6.')
    return np.array(list(combinations(range(norb), npair)), dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge comparisons against the oracle."""
    checks = "\ndef _checked(fn, *args):\n    copied = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args]\n    before = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in copied]\n    try:\n        result = fn(*copied)\n    finally:\n        for original, current in zip(before, copied):\n            if isinstance(original, np.ndarray) and not np.array_equal(original, current):\n                raise AssertionError('Input arrays must not be modified')\n    return result\n"
    setup_0 = 'import numpy as np\ndef pack(z):\n    z=np.asarray(z)\n    return np.stack((z.real,z.imag),axis=-1)\n\n'
    setup_1 = "import numpy as np\ndef pack(z):\n    z=np.asarray(z)\n    return np.stack((z.real,z.imag),axis=-1)\n\n\ndef _raises(fn,*args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    raise AssertionError('Expected ValueError')\n"
    return [
        {
            "setup": setup_0 + checks,
            "call": '_checked(pair_occupations, 4, 2)',
            "gold_call": '_checked(_oracle_pair_occupations, 4, 2)',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(pair_occupations, 5, 3)',
            "gold_call": '_checked(_oracle_pair_occupations, 5, 3)',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(pair_occupations, 1, 1)',
            "gold_call": '_checked(_oracle_pair_occupations, 1, 1)',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(pair_occupations, 6, 6)',
            "gold_call": '_checked(_oracle_pair_occupations, 6, 6)',
            "tol": 2e-08,
        },
        {
            "setup": setup_1 + checks,
            "call": '_raises(_checked, pair_occupations, 3, 4)',
            "gold_call": '_raises(_checked, _oracle_pair_occupations, 3, 4)',
            "tol": 2e-08,
        },
    ]
