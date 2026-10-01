"""
Fit one inverse column on a prescribed coefficient support and return its coefficients, full residual, and squared residual norm exactly.

The Frobenius-norm objective separates into column least-squares problems. Retaining all rows touched by the active columns preserves the full residual norm; existing coefficients remain free whenever the pattern grows.




$$

v=\arg\min_u|C(I,J)u-e_k(I)|_2,\quad r=C(:,J)v-e_k,\quad\beta=r^{\mathsf T}r.

$$

Returns
-------
tuple: (coefficients, full_residual, squared_norm); all scalars use reduced integer rational pairs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fit_column(C: np.ndarray, k: int, J: tuple[int, ...]) -> tuple[tuple, tuple, tuple[int, int]]:
    """Return exact coefficients, the full residual, and its squared norm.

    Parameters
    ----------
    C : np.ndarray
        Finite real nonempty square matrix with nonzero diagonal, nonsingular
        in exact arithmetic after conversion to binary64.
    k : int
        Non-Boolean integer column index in [0, n).
    J : tuple[int, ...]
        Nonempty increasing distinct coefficient indices in [0, n), containing k.

    Returns
    -------
    coefficients, residual, beta : tuple
        coefficients has len(J) entries in J order; residual has n entries.
        Each entry, and beta, is a reduced (numerator, denominator) tuple of
        native integers with positive denominator. The residual is C*m-e_k.
        beta is the squared Euclidean norm, not the norm or a rounded value.
        All coefficients in J are refitted without regularization.

    Raises
    ------
    ValueError
        If C fails the matrix conditions, k is invalid, or J is empty,
        unsorted, duplicated, out of range, or does not contain k.
    """
    return (), (), (0, 1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from fractions import Fraction

import numpy as np


def _oracle_fit_column(
    C: np.ndarray,
    k: int,
    J: tuple[int, ...],
) -> tuple[tuple, tuple, tuple[int, int]]:
    data = _nr_matrix(C)
    n, X, den, gram = data['n'], data['X'], data['den'], data['gram']
    k = _nr_integer(k, 'k', 0, n - 1)
    J = _nr_indices(J, n, 'J', nonempty=True)
    if k not in J:
        raise ValueError("J must contain k")
    key = (k, J)
    if key in data['fits']:
        return data['fits'][key]
    I = tuple(sorted(set().union(*(data['cols'][j] for j in J))))
    local_gram = [[gram[a, b] for b in J] for a in J]
    local_rhs = [den * X[k, j] for j in J]
    v = _nr_bareiss_solve(local_gram, local_rhs)
    r = [Fraction(0)] * n
    for i in I:
        r[i] = sum((Fraction(int(X[i, j]), den) * z
                    for j, z in zip(J, v) if X[i, j] != 0), Fraction(0)) - int(i == k)
    beta = sum((z * z for z in r), Fraction(0))
    result = (tuple(_nr_pair(z) for z in v), tuple(_nr_pair(z) for z in r), _nr_pair(beta))
    data['fits'][key] = result
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic setup/call/gold_call test specifications."""
    return [
        # normal: exact nonzero residual
        {
            "setup": """import numpy as np
n = 4
C = 2.0 * np.eye(n)
C[np.arange(n), (np.arange(n)+1) % n] = -1.0
k=0
J=(0,3)

expected = (((10, 21), (4, 21)), ((-1, 21), (0, 1), (-4, 21), (-2, 21)), (1, 21))
""",
            "call": 'int(fit_column(C, k, J) == expected)',
            "gold_call": 'int(_oracle_fit_column(C, k, J) == expected)',
        },
        # boundary: exact one-dimensional inverse
        {
            "setup": """import numpy as np
C=np.array([[2.]])
k=0
J=(0,)

expected = (((1, 2),), ((0, 1),), (0, 1))
""",
            "call": 'int(fit_column(C, k, J) == expected)',
            "gold_call": 'int(_oracle_fit_column(C, k, J) == expected)',
        },
        # edge: retained active coefficient is zero
        {
            "setup": """import numpy as np
C=np.array([[1.,1.],[0.,1.]])
k=0
J=(0,1)

expected = (((1, 1), (0, 1)), ((0, 1), (0, 1)), (0, 1))
""",
            "call": 'int(fit_column(C, k, J) == expected)',
            "gold_call": 'int(_oracle_fit_column(C, k, J) == expected)',
        },
        # invalid: required diagonal position is missing
        {
            "setup": """import numpy as np
n = 4
C = 2.0 * np.eye(n)
C[np.arange(n), (np.arange(n)+1) % n] = -1.0
k=0
J=(1,2)

def _capture_value_error(fn):
    try:
        fn()
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_capture_value_error(lambda: fit_column(C, k, J))',
            "gold_call": '_capture_value_error(lambda: _oracle_fit_column(C, k, J))',
        },
    ]
