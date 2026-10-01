"""
Evaluate signed complex Gaussian contraction weights.

The sign and complex phase of a fermionic contraction survive in a

Pfaffian. Singular contraction matrices occur in physical limits.

Returns
-------
complex ndarray (B,): signed Pfaffian weights
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def pfaffian_batch(matrices: np.ndarray) -> np.ndarray:
    """Return Pfaffians with their signs and complex phases.

    Parameters
    ----------
    matrices : ndarray, shape (B,2m,2m)
        Finite complex skew-symmetric matrices. The batch may be empty,
        m may be zero, and individual matrices may be singular or have
        a zero leading pivot. Inputs are not mutated.

    Returns
    -------
    weights : complex ndarray, shape (B,)
        Pfaffians in input order, with Pf([[0,a],[-a,0]])=a and
        Pf of the 0-by-0 matrix equal to one. A singular matrix has
        Pfaffian zero. The transpose in skew symmetry has no conjugation.

    Raises
    ------
    ValueError
        For nonfinite data, incorrect shape, odd matrix dimension,
        or matrices that are not skew-symmetric.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from itertools import product
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss


def _oracle_pfaffian_batch(matrices: np.ndarray) -> np.ndarray:
    a = _numeric(matrices, "matrices").copy()
    if a.ndim != 3 or a.shape[1] != a.shape[2] or a.shape[1] % 2:
        raise ValueError("matrices must have shape (B,2m,2m)")
    if not np.allclose(a, -a.swapaxes(1, 2), atol=1e-12, rtol=1e-12):
        raise ValueError("matrices must be skew-symmetric")
    batch, size, unused = a.shape
    result = np.zeros(batch, complex)
    active = np.arange(batch)
    weight = np.ones(batch, complex)
    for k in range(0, size, 2):
        rows = np.arange(len(active))
        if not len(active):
            break
        pivot = k + 1 + np.argmax(np.abs(a[:, k, k + 1 :]), axis=1)
        saved = a[:, k + 1, :].copy()
        a[:, k + 1, :] = a[rows, pivot, :]
        a[rows, pivot, :] = saved
        saved = a[:, :, k + 1].copy()
        a[:, :, k + 1] = a[rows, :, pivot]
        a[rows, :, pivot] = saved
        weight *= np.where(pivot == k + 1, 1, -1)
        value = a[:, k, k + 1]
        weight *= value
        keep = value != 0
        a, active, weight, value = (
            a[keep],
            active[keep],
            weight[keep],
            value[keep],
        )
        u = a[:, k, k + 2 :].copy()
        v = a[:, k + 1, k + 2 :].copy()
        a[:, k + 2 :, k + 2 :] -= (
            u[:, :, None] * v[:, None, :] - v[:, :, None] * u[:, None, :]
        ) / value[:, None, None]
    result[active] = weight
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

rng = np.random.default_rng(871)
x = rng.normal(size=(4, 12, 12)) + 1j * rng.normal(size=(4, 12, 12))
a = x - x.swapaxes(1, 2)

def _with_preservation(fn, matrices):
    before = matrices.copy()
    values = fn(matrices)
    return np.r_[values, float(np.array_equal(matrices, before))]
""",
            "call": "_with_preservation(pfaffian_batch, a.copy())",
            "gold_call": "_with_preservation(_oracle_pfaffian_batch, a.copy())",
            "tol": 1e-08,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

a = np.array(
    [[[0, 0, 3, 1], [0, 0, 2, -4], [-3, -2, 0, 5], [-1, 4, -5, 0]]], complex
)
""",
            "call": "pfaffian_batch(a.copy())",
            "gold_call": "_oracle_pfaffian_batch(a.copy())",
            "tol": 1e-09,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

x = np.arange(36).reshape(1, 6, 6).astype(complex)
a = x - x.swapaxes(1, 2)
""",
            "call": "pfaffian_batch(a.copy())",
            "gold_call": "_oracle_pfaffian_batch(a.copy())",
            "tol": 1e-09,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss
""",
            "call": "pfaffian_batch(np.empty((3,0,0)))",
            "gold_call": "_oracle_pfaffian_batch(np.empty((3,0,0)))",
            "tol": 1e-09,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss
""",
            "call": "pfaffian_batch(np.empty((0,6,6)))",
            "gold_call": "_oracle_pfaffian_batch(np.empty((0,6,6)))",
            "tol": 1e-09,
        },
        {
            "setup": """from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss


def _expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return 1
    return 0
""",
            "call": (
                "_expect_value_error(lambda: "
                "pfaffian_batch(np.eye(4)[None]))"
            ),
            "gold_call": (
                "_expect_value_error(lambda: "
                "_oracle_pfaffian_batch(np.eye(4)[None]))"
            ),
            "tol": 1e-09,
        },
    ]
