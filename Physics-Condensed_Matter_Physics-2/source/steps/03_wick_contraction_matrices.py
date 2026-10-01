"""
Assemble the ordered contractions of Majorana strings.

Wick contractions carry the ordering of the original operator string.

Repeated fields at equal time still obey the Clifford algebra.

Returns
-------
complex ndarray (B,2m,2m): skew contraction matrices
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def wick_contraction_matrices(
    kernel: np.ndarray, indices: np.ndarray, slots: np.ndarray
) -> np.ndarray:
    """Build the skew matrices whose Pfaffians are Gaussian string averages.

    Parameters
    ----------
    kernel : ndarray, shape (T,T,D,D)
        Finite complex time-ordered thermal contraction tensor, T,D>0.
        Equal-time entries describe the ordered product and include the
        diagonal contraction one, as in thermal_majorana_kernel.
    indices : ndarray, shape (B,2m)
        Finite integer-valued Majorana indices in [0,D), one operator
        string per row. B or m may be zero. Repeated fields are allowed.
    slots : ndarray, shape (2m,)
        Finite integer-valued indices in [0,T), specifying the kernel
        time slot for each string position; shared across batch rows.

    Returns
    -------
    matrices : complex ndarray, shape (B,2m,2m)
        Zero diagonal. For u<v, entry [r,u,v] equals
        kernel[slots[u],slots[v],indices[r,u],indices[r,v]]. The lower
        triangle is its negative transpose. Preserve the input string
        order; it determines the signs even when times coincide.

    Raises
    ------
    ValueError
        For nonfinite data, incompatible shapes, odd string length,
        noninteger indices/slots, or indices/slots outside their ranges.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def _oracle_wick_contraction_matrices(
    kernel: np.ndarray, indices: np.ndarray, slots: np.ndarray
) -> np.ndarray:
    k = _numeric(kernel, "kernel")
    x = _numeric(indices, "indices", True)
    s = _numeric(slots, "slots", True)
    if (
        k.ndim != 4
        or k.shape[0] != k.shape[1]
        or k.shape[2] != k.shape[3]
        or min(k.shape) == 0
    ):
        raise ValueError(
            ("kernel must have shape (T,T,D,D), with " "T,D positive")
        )
    if x.ndim != 2 or x.shape[1] % 2 or s.shape != (x.shape[1],):
        raise ValueError("indices must have shape (B,2m), slots shape (2m,)")
    if np.any(x != np.floor(x)) or np.any(s != np.floor(s)):
        raise ValueError("indices and slots must be integer-valued")
    if np.any((x < 0) | (x >= k.shape[2])) or np.any(
        (s < 0) | (s >= k.shape[0])
    ):
        raise ValueError("index or slot out of range")
    x, s = x.astype(int), s.astype(int)
    batch, size = x.shape
    result = np.zeros((batch, size, size), complex)
    upper = np.triu_indices(size, 1)
    values = k[s[upper[0]], s[upper[1]], x[:, upper[0]], x[:, upper[1]]]
    result[:, upper[0], upper[1]] = values
    result[:, upper[1], upper[0]] = -values
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    ("Return normal, boundary, edge and " "invalid controls.")
    return [
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

beta = 1.7
times = np.array([0.93, 0.2, 0.0, 0.93])
kernel = np.zeros((4, 4, 6, 6), complex)
for u in range(4):
    for v in range(4):
        delta = times[u] - times[v]
        sign = -1 if delta < 0 else 1
        if delta < 0:
            delta += beta
        for c, a in enumerate([0.23, -0.4, 0.17]):
            z = a * (beta - 2 * delta)
            block = np.array(
                [[np.cosh(z), 1j * np.sinh(z)], [-1j * np.sinh(z), np.cosh(z)]]
            )
            kernel[u, v, 2 * c : 2 * c + 2, 2 * c : 2 * c + 2] = (
                sign * block / np.cosh(beta * a)
            )
""",
            "call": (
                "wick_contraction_matrices(kernel.copy(),"
                " np.array([[0,2,0,2],[1,3,1,3]]), "
                "np.array([0,0,2,2]))"
            ),
            "gold_call": (
                "_oracle_wick_contraction_matrices(kernel.copy(),"
                " np.array([[0,2,0,2],[1,3,1,3]]), "
                "np.array([0,0,2,2]))"
            ),
            "tol": 1e-09,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

beta = 1.7
times = np.array([0.93, 0.2, 0.0, 0.93])
kernel = np.zeros((4, 4, 6, 6), complex)
for u in range(4):
    for v in range(4):
        delta = times[u] - times[v]
        sign = -1 if delta < 0 else 1
        if delta < 0:
            delta += beta
        for c, a in enumerate([0.23, -0.4, 0.17]):
            z = a * (beta - 2 * delta)
            block = np.array(
                [[np.cosh(z), 1j * np.sinh(z)], [-1j * np.sinh(z), np.cosh(z)]]
            )
            kernel[u, v, 2 * c : 2 * c + 2, 2 * c : 2 * c + 2] = (
                sign * block / np.cosh(beta * a)
            )
""",
            "call": (
                "wick_contraction_matrices(kernel.copy(),"
                " np.array([[0,0,2,2],[0,1,1,0]]), "
                "np.array([0,3,0,3]))"
            ),
            "gold_call": (
                "_oracle_wick_contraction_matrices(kernel.copy(),"
                " np.array([[0,0,2,2],[0,1,1,0]]), "
                "np.array([0,3,0,3]))"
            ),
            "tol": 1e-09,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

beta = 1.7
times = np.array([0.93, 0.2, 0.0, 0.93])
kernel = np.zeros((4, 4, 6, 6), complex)
for u in range(4):
    for v in range(4):
        delta = times[u] - times[v]
        sign = -1 if delta < 0 else 1
        if delta < 0:
            delta += beta
        for c, a in enumerate([0.23, -0.4, 0.17]):
            z = a * (beta - 2 * delta)
            block = np.array(
                [[np.cosh(z), 1j * np.sinh(z)], [-1j * np.sinh(z), np.cosh(z)]]
            )
            kernel[u, v, 2 * c : 2 * c + 2, 2 * c : 2 * c + 2] = (
                sign * block / np.cosh(beta * a)
            )
""",
            "call": (
                "wick_contraction_matrices(kernel.copy(),"
                " np.array([[1,0,3,2,5,4,0,1]]), "
                "np.array([2,0,1,0,3,2,1,3]))"
            ),
            "gold_call": (
                "_oracle_wick_contraction_matrices(kernel.copy(),"
                " np.array([[1,0,3,2,5,4,0,1]]), "
                "np.array([2,0,1,0,3,2,1,3]))"
            ),
            "tol": 1e-09,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

beta = 1.7
times = np.array([0.93, 0.2, 0.0, 0.93])
kernel = np.zeros((4, 4, 6, 6), complex)
for u in range(4):
    for v in range(4):
        delta = times[u] - times[v]
        sign = -1 if delta < 0 else 1
        if delta < 0:
            delta += beta
        for c, a in enumerate([0.23, -0.4, 0.17]):
            z = a * (beta - 2 * delta)
            block = np.array(
                [[np.cosh(z), 1j * np.sinh(z)], [-1j * np.sinh(z), np.cosh(z)]]
            )
            kernel[u, v, 2 * c : 2 * c + 2, 2 * c : 2 * c + 2] = (
                sign * block / np.cosh(beta * a)
            )
""",
            "call": (
                "wick_contraction_matrices(kernel.copy(),"
                " np.empty((2,0)), np.array([]))"
            ),
            "gold_call": (
                "_oracle_wick_contraction_matrices(kernel.copy(),"
                " np.empty((2,0)), np.array([]))"
            ),
            "tol": 1e-09,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

beta = 1.7
times = np.array([0.93, 0.2, 0.0, 0.93])
kernel = np.zeros((4, 4, 6, 6), complex)
for u in range(4):
    for v in range(4):
        delta = times[u] - times[v]
        sign = -1 if delta < 0 else 1
        if delta < 0:
            delta += beta
        for c, a in enumerate([0.23, -0.4, 0.17]):
            z = a * (beta - 2 * delta)
            block = np.array(
                [[np.cosh(z), 1j * np.sinh(z)], [-1j * np.sinh(z), np.cosh(z)]]
            )
            kernel[u, v, 2 * c : 2 * c + 2, 2 * c : 2 * c + 2] = (
                sign * block / np.cosh(beta * a)
            )


def _expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return 1
    return 0
""",
            "call": (
                "_expect_value_error(lambda: "
                "wick_contraction_matrices(kernel, "
                "np.array([[0,6]]), np.array([0,1])))"
            ),
            "gold_call": (
                "_expect_value_error(lambda: "
                "_oracle_wick_contraction_matrices(kernel,"
                " np.array([[0,6]]), np.array([0,1])))"
            ),
            "tol": 1e-09,
        },
    ]
