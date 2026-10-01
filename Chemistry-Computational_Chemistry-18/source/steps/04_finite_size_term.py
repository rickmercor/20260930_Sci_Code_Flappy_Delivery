"""
Evaluates the leading correction that separates a finite cubic crystal from the macroscopic limit of the same shape.

A crystal of finite size differs from the macroscopic crystal of identical proportions by the cells that have not yet been included. Expanding their contribution in multipoles about the origin and replacing their sum by an integral over the region outside the sample leaves, for a cubic sample of a cubic lattice, the closed-form leading term nu_corr = [24 (r . r)^2 - 40 (x^4 + y^4 + z^4)] / [9 sqrt(3) (2p+1)^2 l^5], where x, y and z are the components of r in the same length units as l; it shrinks as the inverse square of the number 2p+1 of cells along an edge and depends on the displacement through its fourth powers.

Returns
-------
A numpy float64 array of shape (4,) holding the cube's leading finite-size correction, its scaled value, and the two displacement invariants.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def finite_size_term(r: "np.ndarray", p: int, l: float) -> "np.ndarray":
    """Evaluates the leading correction that separates a finite cubic crystal from the macroscopic limit of the same shape.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as l.
        p: non-negative integer, the size index of the cubic sample.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (4,): the correction in inverse length units, the reciprocal of the unit in which r and l are given (the absolute potential, which for l = 1 coincides with units of 1/l), the correction multiplied by
        (2p+1)^2, the squared length of the displacement, and the sum of the fourth powers of its components.

    Raises:
        ValueError: if r is not a finite real vector of length three, if p is not a non-negative integer,
            or if l is not a positive finite real number.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _real(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError("invalid " + name)
    v = float(v)
    if not np.isfinite(v):
        raise ValueError("invalid " + name)
    return v


def _pos_float(v, name):
    v = _real(v, name)
    if v <= 0.0:
        raise ValueError("invalid " + name)
    return v


def _nonneg_int(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0:
        raise ValueError("invalid " + name)
    return int(v)


def _vec3(r, name):
    a = np.asarray(r, dtype=np.float64)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError("invalid " + name)
    return a


def _oracle_finite_size_term(r: "np.ndarray", p: int, l: float) -> "np.ndarray":
    r = _vec3(r, "r")
    p = _nonneg_int(p, "p")
    l = _pos_float(l, "l")
    rr = float(r @ r)
    q4 = float(np.sum(r ** 4))
    K = 2 * p + 1
    corr = (24.0 * rr ** 2 - 40.0 * q4) / (9.0 * np.sqrt(3.0) * K ** 2 * l ** 5)
    return np.array([corr, corr * K ** 2, rr, q4], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\n',
         'call': 'finite_size_term(np.array([0.5, 0.5, 0.5]), 1, 1.0)',
         'gold_call': '_oracle_finite_size_term(np.array([0.5, 0.5, 0.5]), 1, 1.0)',
         'tol': 1e-12},
        {'setup': 'import numpy as np\n',
         'call': 'finite_size_term(np.array([0.37, 0.21, 0.13]), 8, 1.0)',
         'gold_call': '_oracle_finite_size_term(np.array([0.37, 0.21, 0.13]), 8, 1.0)',
         'tol': 1e-12},
        {'setup': 'import numpy as np\n# an axial displacement, where the quartic term changes sign, with a lattice constant of 2\n',
         'call': 'finite_size_term(np.array([0.5, 0.0, 0.0]), 4, 2.0)',
         'gold_call': '_oracle_finite_size_term(np.array([0.5, 0.0, 0.0]), 4, 2.0)',
         'tol': 1e-12},
        {'setup': 'import numpy as np\n# boundary: the central cell alone, p = 0\n',
         'call': 'finite_size_term(np.array([0.37, 0.21, 0.13]), 0, 1.0)',
         'gold_call': '_oracle_finite_size_term(np.array([0.37, 0.21, 0.13]), 0, 1.0)',
         'tol': 1e-12},
        {'setup': 'import numpy as np\n# invalid input: a negative size index must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: finite_size_term(np.array([0.5, 0.5, 0.5]), -1, 1.0))',
         'gold_call': '_catches_value_error(lambda: _oracle_finite_size_term(np.array([0.5, 0.5, 0.5]), -1, 1.0))'},
    ]
