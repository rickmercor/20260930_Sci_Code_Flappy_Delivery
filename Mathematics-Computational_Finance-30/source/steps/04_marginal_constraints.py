"""
Return the equality constraint block, as coefficient matrix and right-hand side, that the source requires of every per-expiry marginal in its program (SMP) in Section 4.1. Rows in the order the source lists them. This block is the same for every expiry, so build it once for the ladder.

The source works with marginals of a normalised driver, and Section 4.1 lists under 'Constraints' what that requires of each marginal beyond non-negativity. More than one equality is required.

Returns
-------
A tuple (A, b): float64 coefficient matrix with one column per ladder point, and the matching float64 right-hand-side vector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def marginal_constraints(strikes: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Return the equality block every per-expiry marginal must satisfy in the source's program.

    Parameters
    ----------
    strikes : numpy.ndarray
        One-dimensional strike ladder, strictly increasing and strictly positive.

    Returns
    -------
    A, b : tuple of numpy.ndarray
        A is the float64 coefficient matrix with one column per ladder point and one row per equality, in the source's order; b is the matching float64 right-hand-side vector.

    Raises
    ------
    ValueError
        If the ladder is not one-dimensional, not strictly increasing, not strictly positive, or contains a non-finite value.
    """
    return A, b

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _ladder(strikes):
    """Validate and return a strike ladder. Shared by several steps."""
    K = np.asarray(strikes, dtype=np.float64)
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(K)):
        raise ValueError("strikes must all be finite")
    if np.any(K <= 0.0):
        raise ValueError("strikes must be strictly positive")
    if K.size > 1 and not np.all(np.diff(K) > 0.0):
        raise ValueError("strikes must be strictly increasing")
    return K


def _oracle_marginal_constraints(strikes: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Two equality rows per expiry: 1'q = 1 (mass) and K'q = 1 (unit mean)."""
    K = _ladder(strikes)
    A = np.vstack([np.ones_like(K), K]).astype(np.float64)
    b = np.array([1.0, 1.0], dtype=np.float64)
    if A.shape[0] != b.size or A.shape[1] != K.size:
        raise ValueError("the equality block and its right-hand side disagree in shape")
    return A, b

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\n',
            'call': 'marginal_constraints(K)[0]',
            'gold_call': '_oracle_marginal_constraints(K)[0]',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\n',
            'call': 'marginal_constraints(K)[1]',
            'gold_call': '_oracle_marginal_constraints(K)[1]',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.5, 1.0, 2.0])\n',
            'call': 'marginal_constraints(K)[0]',
            'gold_call': '_oracle_marginal_constraints(K)[0]',
        },
        {
            'setup': 'import numpy as np\nK = np.array([1.0])\n',
            'call': 'marginal_constraints(K)[0]',
            'gold_call': '_oracle_marginal_constraints(K)[0]',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try:\n        marginal_constraints(np.array([-1.0, 0.5]))\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_marginal_constraints(np.array([-1.0, 0.5]))\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': 'run_oracle()',
        },
    ]
