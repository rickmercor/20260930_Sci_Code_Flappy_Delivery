"""
Given the fertilities $F_1$ to $F_n$ and the survival probabilities $P_1$ to $P_{n-1}$ of an irreducible Leslie model and the number of sub-classes $m$ per original class, return the Leslie projection matrix for the demographic timing specified above on $n m$ equal-width sub-classes ordered from youngest to oldest. Each original age class contains $m$ consecutive sub-classes, and one projection interval is the time required to traverse one sub-class. With $m$ equal to one the result is the original Leslie matrix.

Merging adjacent age classes of a Leslie model is straightforward only when the number of original classes is a whole multiple of the number of reduced classes, because each reduced class is then a block of whole original classes and the reduced model projects over a whole number of original intervals. When the ratio is not an integer, a reduced class boundary falls inside an original age class.

The finer age classification must preserve the timing of the original demographic events. Between consecutive original age boundaries, an individual only grows older, with no mortality or reproduction. On completing an original age class, its fertility is the expected number of newborns per individual present immediately before that boundary, and its survival probability is the probability of entering the next original age class. Newborns enter the youngest resolved class, and no individual survives beyond the terminal age of the original model.

Returns
-------
np.ndarray of shape (n m, n m), the resolved Leslie matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def disaggregate_leslie(
    fertility: np.ndarray,
    survival: np.ndarray,
    subdivisions: int,
) -> np.ndarray:
    """Resolve each age class of a Leslie model into equal sub-classes.

    Parameters
    ----------
    fertility : np.ndarray
        Fertilities F_1 to F_n, shape (n,), nonnegative with F_n positive.
    survival : np.ndarray
        Survival probabilities P_1 to P_(n-1), shape (n - 1,), each in (0, 1].
    subdivisions : int
        Number m of sub-classes per original class, at least one.

    Returns
    -------
    np.ndarray
        The resolved Leslie matrix, shape (n m, n m).

    Raises
    ------
    ValueError
        When the vital rates have the wrong shapes, are not finite, violate their ranges, or
        when subdivisions is not an integer of at least one.
        A count given as a bool, or as a float even when its value is integral, is
        not accepted as an integer.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numbers

import numpy as np


def _check_vital_rates(fertility, survival):
    """Validate the vital rates of an irreducible Leslie model and return them as float arrays."""
    f = np.asarray(fertility)
    p = np.asarray(survival)
    if np.iscomplexobj(f) or np.iscomplexobj(p):
        raise ValueError("vital rates must be real")
    f = f.astype(float)
    p = p.astype(float)
    if f.ndim != 1 or f.size < 1:
        raise ValueError("fertility must be a one-dimensional array of length at least one")
    if p.ndim != 1 or p.size != f.size - 1:
        raise ValueError("survival must be a one-dimensional array one shorter than fertility")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(p))):
        raise ValueError("vital rates must be finite")
    if np.any(f < 0.0) or f[-1] <= 0.0:
        raise ValueError("fertilities must be nonnegative with the oldest class fertile")
    if np.any(p <= 0.0) or np.any(p > 1.0):
        raise ValueError("survival probabilities must lie in (0, 1]")
    return f, p


def _oracle_disaggregate_leslie(
    fertility: np.ndarray,
    survival: np.ndarray,
    subdivisions: int,
) -> np.ndarray:
    """Reference implementation."""
    f, p = _check_vital_rates(fertility, survival)
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, numbers.Integral):
        raise ValueError("subdivisions must be an integer")
    m = int(subdivisions)
    if m < 1:
        raise ValueError("subdivisions must be at least one")

    n = f.size
    size = n * m
    resolved = np.zeros((size, size))
    for l in range(1, n + 1):
        resolved[0, l * m - 1] = f[l - 1]
    for i in range(1, size):
        resolved[i, i - 1] = p[i // m - 1] if i % m == 0 else 1.0
    return resolved

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return self-contained, single-invocation numeric test cases."""
    setup = "import numpy as np\n\ndef _flatten(value):\n    if isinstance(value, dict):\n        parts = [_flatten(value[key]) for key in sorted(value)]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    if isinstance(value, (list, tuple)):\n        parts = [_flatten(item) for item in value]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    return np.asarray(value, dtype=float).ravel()\n\ndef project(value):\n    raw = _flatten(value)\n    missing = np.isnan(raw)\n    return np.concatenate((np.where(missing, 0.0, raw), missing.astype(float)))\n\ndef expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef expect_runtime_error(fn, *args):\n    try:\n        fn(*args)\n    except RuntimeError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef leslie(fertility, survival):\n    matrix = np.zeros((len(fertility), len(fertility)))\n    matrix[0, :] = fertility\n    for index, probability in enumerate(survival):\n        matrix[index + 1, index] = probability\n    return matrix\n\nF = np.array([0.0, 0.4, 1.0, 1.4, 1.6, 1.8, 1.6])\nP = np.array([0.60, 0.72, 0.78, 0.76, 0.70, 0.50])\n"
    return [
        {
            "setup": setup + "",
            "call": "project(disaggregate_leslie(np.array([0.0, 1.0, 5.0]), np.array([0.3, 0.5]), 2))",
            "gold_call": "project(_oracle_disaggregate_leslie(np.array([0.0, 1.0, 5.0]), np.array([0.3, 0.5]), 2))",
        },
        {
            "setup": setup + "",
            "call": "project(disaggregate_leslie(F, P, 3))",
            "gold_call": "project(_oracle_disaggregate_leslie(F, P, 3))",
        },
        {
            "setup": setup + "",
            "call": "project(disaggregate_leslie(np.array([1.3]), np.array([]), 4))",
            "gold_call": "project(_oracle_disaggregate_leslie(np.array([1.3]), np.array([]), 4))",
        },
        {
            "setup": setup + "",
            "call": "expect_value_error(disaggregate_leslie, F, np.append(P[:-1], 0.0), 3)",
            "gold_call": "expect_value_error(_oracle_disaggregate_leslie, F, np.append(P[:-1], 0.0), 3)",
        },
    ]
