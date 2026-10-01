"""
Determine one NRSAI column's initial active rows and coefficient positions from the exact numerical patterns of the identity, C, and C squared.

Algorithm 1 initializes each inverse column from a union of matrix-power patterns. Products contributing to a matrix-power entry can cancel; Boolean reachability is not a substitute for the numerical support.




$$

J={k}\cup\operatorname{supp}(C_{:,k})\cup\operatorname{supp}((C^2)*{:,k}),\quad I={i:C*{i,j}\ne0\text{ for some }j\in J}.

$$

Returns
-------
tuple: (I, J), two increasing tuples of native integer indices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def initial_pattern(C: np.ndarray, k: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the initial active row and coefficient indices for column k.

    Parameters
    ----------
    C : np.ndarray
        Nonempty real square matrix with nonzero diagonal, nonsingular when
        its binary64 entries are interpreted as exact rational numbers.
        Conversion to binary64 occurs before all exact support calculations.
    k : int
        Zero-based column index, 0 <= k < C.shape[0].

    Returns
    -------
    I, J : tuple[tuple[int, ...], tuple[int, ...]]
        Increasing, duplicate-free global row and coefficient indices.
        J is the union of the separate numerical supports of I_n, C, C^2.
        I contains exactly the nonzero rows of C[:, J].

    Raises
    ------
    ValueError
        If C is not a nonempty finite real square matrix after binary64
        conversion, has a zero diagonal entry, or is exactly singular;
        or if k is not a non-Boolean integer in the stated range.
    """
    return (), ()

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
from fractions import Fraction

import numpy as np


# Private exact-arithmetic utilities. The cache contains oracle-side data only.
def _nr_integer(value, name, low=0, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _nr_indices(values, n, name, sorted_required=True, nonempty=False):
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a tuple or list of indices")
    result = tuple(_nr_integer(x, name, 0, n - 1) for x in values)
    if len(set(result)) != len(result) or (nonempty and not result):
        raise ValueError(f"{name} must contain distinct indices")
    if sorted_required and result != tuple(sorted(result)):
        raise ValueError(f"{name} must be increasing")
    return result


def _nr_real_array(value, name, ndim):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'iuf' or raw.ndim != ndim or raw.size == 0:
            raise ValueError(f"{name} must be a nonempty real array of dimension {ndim}")
        with np.errstate(over='raise', invalid='raise'):
            array = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
    except (TypeError, OverflowError, FloatingPointError) as exc:
        raise ValueError(f"{name} cannot be represented as finite binary64") from exc
    return array


def _nr_bareiss_solve(gram, rhs):
    # Fraction-free elimination for an integer positive-definite Gram matrix.
    n = len(rhs)
    a = [[int(x) for x in row] + [int(rhs[i])] for i, row in enumerate(gram)]
    previous = 1
    for j in range(n - 1):
        pivot = a[j][j]
        if pivot == 0:
            raise ValueError("the matrix or selected column system is singular")
        for i in range(j + 1, n):
            entry = a[i][j]
            for k in range(j + 1, n + 1):
                a[i][k] = (pivot * a[i][k] - entry * a[j][k]) // previous
            a[i][j] = 0
        previous = pivot
    if a[-1][-2] == 0:
        raise ValueError("the matrix or selected column system is singular")
    result = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        result[i] = (Fraction(a[i][-1]) - sum(
            (a[i][j] * result[j] for j in range(i + 1, n)), Fraction(0)
        )) / a[i][i]
    return tuple(result)


def _nr_matrix(C):
    array = _nr_real_array(C, 'C', 2)
    if array.shape[0] != array.shape[1] or np.any(np.diag(array) == 0):
        raise ValueError("C must be square with a nonzero diagonal")
    key = (array.shape, array.tobytes())
    cache = getattr(_nr_matrix, '_cache', None)
    if cache is None:
        cache = {}
        _nr_matrix._cache = cache
    if key in cache:
        return cache[key]
    n = len(array)
    values = [Fraction(float(x)) for x in array.flat]
    den = math.lcm(*(v.denominator for v in values))
    X = np.array([int(v * den) for v in values], dtype=object).reshape(n, n)
    gram = X.T @ X
    _nr_bareiss_solve(gram.tolist(), [0] * n)  # Exact nonsingularity check.
    data = {
        'n': n, 'X': X, 'den': den, 'gram': gram,
        'rows': tuple(tuple(int(j) for j in np.flatnonzero(X[i])) for i in range(n)),
        'cols': tuple(tuple(int(i) for i in np.flatnonzero(X[:, j])) for j in range(n)),
        'fits': {},
    }
    cache[key] = data
    return data


def _nr_pair(value):
    value = Fraction(value)
    return int(value.numerator), int(value.denominator)


def _nr_fraction(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("an exact rational must be a (numerator, denominator) pair")
    p, q = value
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
           for x in (p, q)):
        raise ValueError("rational components must be integers")
    p, q = int(p), int(q)
    if q <= 0 or math.gcd(abs(p), q) != 1:
        raise ValueError("rational pairs must be reduced with a positive denominator")
    return Fraction(p, q)


def _oracle_initial_pattern(C: np.ndarray, k: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    data = _nr_matrix(C)
    n, X = data['n'], data['X']
    k = _nr_integer(k, 'k', 0, n - 1)
    # The common positive denominator cannot change numerical nonzero tests.
    power_column = X @ X[:, k]
    J = tuple(j for j in range(n) if j == k or X[j, k] != 0 or power_column[j] != 0)
    I = tuple(sorted(set().union(*(data['cols'][j] for j in J))))
    return I, J

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic setup/call/gold_call test specifications."""
    return [
        # normal: nonsymmetric banded pattern
        {
            "setup": """import numpy as np
C=np.array([[4,1,0,0],[2,4,1,0],[0,2,4,1],[0,0,2,4]],float)
k=0

expected = ((0, 1, 2, 3), (0, 1, 2))
""",
            "call": 'int(initial_pattern(C, k) == expected)',
            "gold_call": 'int(_oracle_initial_pattern(C, k) == expected)',
        },
        # boundary: one-dimensional matrix
        {
            "setup": """import numpy as np
C=np.array([[2.0]])
k=0

expected = ((0,), (0,))
""",
            "call": 'int(initial_pattern(C, k) == expected)',
            "gold_call": 'int(_oracle_initial_pattern(C, k) == expected)',
        },
        # edge: two nonzero paths cancel exactly
        {
            "setup": """import numpy as np
n=8
i=np.arange(n)
C=3.0*np.eye(n)
C[i,(i+1)%n]=-1.0
C[i,(i+3)%n]=(-1.0)**i
k=0

expected = ((0, 1, 2, 3, 4, 5, 6, 7), (0, 2, 5, 6, 7))
""",
            "call": 'int(initial_pattern(C, k) == expected)',
            "gold_call": 'int(_oracle_initial_pattern(C, k) == expected)',
        },
        # invalid: singular matrix
        {
            "setup": """import numpy as np
C=np.ones((2,2))
k=0

def _capture_value_error(fn):
    try:
        fn()
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_capture_value_error(lambda: initial_pattern(C, k))',
            "gold_call": '_capture_value_error(lambda: _oracle_initial_pattern(C, k))',
        },
        # invalid: nonsquare matrix
        {
            "setup": """import numpy as np
C=np.ones((2,3))
k=0

def _capture_value_error(fn):
    try:
        fn()
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_capture_value_error(lambda: initial_pattern(C, k))',
            "gold_call": '_capture_value_error(lambda: _oracle_initial_pattern(C, k))',
        },
    ]
