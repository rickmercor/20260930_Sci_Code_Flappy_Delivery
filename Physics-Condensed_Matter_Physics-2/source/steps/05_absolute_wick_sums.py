"""
Compute the total magnitude of individual perfect-pairing products for a batch of complex skew matrices, including empty and singular cases.

A phase-cancelled Gaussian moment does not determine its total individual-pairing magnitude.

$A(K)=\sum_{P\text{ perfect matching}}\prod_{(u,v)\in P}|K_{uv}|=\operatorname{haf}(|K|)$; the empty matching contributes $1$.

Returns
-------
real ndarray (B,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def absolute_wick_sums(matrices: np.ndarray) -> np.ndarray:
    """Sum magnitudes of individual perfect-pairing products.

    Parameters
    ----------
    matrices : complex ndarray, shape (B, 2m, 2m)
        Finite skew-symmetric matrices (transpose, not adjoint), with
        zero diagonal, within absolute tolerance 1e-12. B or m may be
        zero. Repeated fields and singular matrices are permitted.

    Returns
    -------
    sums : real ndarray, shape (B,)
        For each matrix K, sum over all perfect matchings P of
        product(abs(K[u, v]) for (u, v) in P), with u < v.
        The empty matching has weight one. Take absolute values before
        summing matchings. Inputs are unchanged.

    Raises
    ------
    ValueError
        For nonfinite data, incorrect shape, odd dimension, nonzero
        diagonal or failure of skew symmetry at the stated tolerance.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss


def _oracle_absolute_wick_sums(matrices: np.ndarray) -> np.ndarray:
    """Sum magnitudes of individual perfect-pairing products.

    Parameters
    ----------
    matrices : complex ndarray, shape (B, 2m, 2m)
        Finite skew-symmetric matrices (transpose, not adjoint), with
        zero diagonal, within absolute tolerance 1e-12. B or m may be
        zero. Repeated fields and singular matrices are permitted.

    Returns
    -------
    sums : real ndarray, shape (B,)
        For each matrix K, sum over all perfect matchings P of
        product(abs(K[u, v]) for (u, v) in P), with u < v.
        The empty matching has weight one. Take absolute values before
        summing matchings. Inputs are unchanged.

    Raises
    ------
    ValueError
        For nonfinite data, incorrect shape, odd dimension, nonzero
        diagonal or failure of skew symmetry at the stated tolerance.
    """
    matrix = _numeric(matrices, "matrices")
    if (
        matrix.ndim != 3
        or matrix.shape[1] != matrix.shape[2]
        or matrix.shape[1] % 2
    ):
        raise ValueError("matrices must have shape (B, 2m, 2m)")
    if not np.allclose(
        matrix + matrix.swapaxes(1, 2), 0, atol=1e-12, rtol=0
    ) or not np.allclose(
        np.diagonal(matrix, axis1=1, axis2=2), 0, atol=1e-12, rtol=0
    ):
        raise ValueError("matrices must be skew with zero diagonal")
    weights = np.abs(matrix)
    batch, size, _ = weights.shape

    @lru_cache(None)
    def _match(mask):
        if mask == 0:
            return np.ones(batch)
        first = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << first)
        remaining = rest
        total = np.zeros(batch)
        while remaining:
            bit = remaining & -remaining
            second = bit.bit_length() - 1
            edge = weights[:, first, second]
            if np.any(edge):
                total += edge * _match(rest ^ bit)
            remaining ^= bit
        return total

    return _match((1 << size) - 1).copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

rng = np.random.default_rng(260914)
x = rng.normal(size=(3, 8, 8)) + 1j * rng.normal(size=(3, 8, 8))
a = x - x.swapaxes(1, 2)


def with_preservation(fn, *args):
    before = [
        arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args
    ]
    result = fn(*args)
    preserved = all(np.array_equal(arg, old) for arg, old in zip(args, before))
    return np.r_[np.asarray(result).ravel(), float(preserved)]
""",
            "call": ("with_preservation(absolute_wick_sums, a)"),
            "gold_call": ("with_preservation(_oracle_absolute_wick_sums, a)"),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

a = np.array(
    [[[0, 1, 1, 0], [-1, 0, 0, 1], [-1, 0, 0, 1], [0, -1, -1, 0]]], complex
)
""",
            "call": ("absolute_wick_sums(a)"),
            "gold_call": ("_oracle_absolute_wick_sums(a)"),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

import numpy as np
""",
            "call": ("absolute_wick_sums(np.empty((2, 0, 0)))"),
            "gold_call": ("_oracle_absolute_wick_sums(np.empty((2, 0, 0)))"),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

import numpy as np
""",
            "call": ("absolute_wick_sums(np.empty((0, 6, 6)))"),
            "gold_call": ("_oracle_absolute_wick_sums(np.empty((0, 6, 6)))"),
            "tol": 1e-08,
        },
        {
            "setup": """from functools import lru_cache
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss

import numpy as np


def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
""",
            "call": (
                "expect_value_error(absolute_wick_sums, np.ones((1, " "2, 2)))"
            ),
            "gold_call": (
                "expect_value_error(_oracle_absolute_wick_sums, "
                "np.ones((1, 2, 2)))"
            ),
            "tol": 1e-08,
        },
    ]
