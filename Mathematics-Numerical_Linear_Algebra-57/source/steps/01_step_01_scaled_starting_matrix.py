"""
The coupled inverse Newton iteration for the inverse `$p$`-th root carries a pair of matrices rather than a single iterate. The pair stays tied by `$M = X ** p A$` at every iteration, so the residual `$R = I - M$` measures how far `$X$` is from the inverse `$p$$-th root of$$A$` without `$X$` ever being formed, and the whole run can be reported from `$M$` alone.



Both members of the pair start from one positive scaling constant: `$X$` starts from the identity divided by that constant, and `$M$` from `$A$` divided by the `$p$$-th power of the same constant, which is what makes the tie hold at$$k = 0$`. The published constant is the one that normalises the starting `$M$` exactly, that is, the unique positive scalar for which the Frobenius norm of the starting `$M$` equals `$(p + 1) / 2$`. Because the Frobenius norm dominates every eigenvalue of a symmetric matrix, that choice places the whole spectrum of the starting `$M$` inside ``(0, (p + 1) / 2]``, which is the region in which the iteration contracts. That bound is one-sided: it holds the starting residual spectrum below one but not above zero, so a negative eigenvalue in the starting residual is expected rather than a symptom of a wrong constant.



This step returns the starting `$M$` of that scaled pair. Its residual is read off it by the defining relation and is not returned here.



The normalization is homogeneous, so multiplying `$A$` by any positive scalar must leave the returned `$M$` unchanged. That invariant is required over the full finite float64 range: the implementation must not first square entries in a way that makes a representable nonzero matrix look like zero, or makes its Frobenius norm overflow. If an input is accepted by the `$1e-12$` symmetry tolerance, it represents the symmetric matrix ``(A + A.T) / 2``; that projection is made before the norm and the scaling are formed so the returned member still has the symmetry on which the coupled construction relies.

Returns
-------
np.ndarray of shape (n, n), the residual-bearing member of the scaled starting pair
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scaled_starting_matrix(A: np.ndarray, p: int) -> np.ndarray:
    """Return the residual-bearing member of the scaled starting pair.

    Raises ``ValueError`` unless every one of the following holds: ``A`` is real
    rather than complex; ``A`` is a nonempty square two-dimensional array; every
    entry of ``A`` is finite; ``A`` is symmetric to an absolute tolerance of
    ``1e-12``, so a matrix that is asymmetric only at the ``1e-13`` level is
    accepted and projected to its symmetric part rather than rejected; that
    symmetric part is nonzero; and ``p`` is an integer, not a bool, with ``p >=
    1``. The finite-input contract includes magnitudes for which a naive
    sum-of-squares Frobenius norm overflows or underflows.

    Parameters
    ----------
    A : np.ndarray
        Nonzero finite real symmetric matrix of shape ``(n, n)``.
    p : int
        Root order. The iteration targets the inverse ``p``-th root of ``A``.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n, n)`` holding the starting ``M``.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_scaled_starting_matrix(A: np.ndarray, p: int) -> np.ndarray:
    """Reference starting matrix of the scaled coupled pair."""
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
        raise ValueError("p must be an integer")
    if int(p) < 1:
        raise ValueError("p must be at least one")
    order = int(p)
    matrix = np.asarray(A)
    if np.iscomplexobj(matrix):
        raise ValueError("A must be real")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("A must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("A must be symmetric")
    projected = matrix.copy()
    for row in range(matrix.shape[0]):
        for column in range(row + 1, matrix.shape[1]):
            left = float(matrix[row, column])
            right = float(matrix[column, row])
            if np.signbit(left) == np.signbit(right):
                midpoint = left + 0.5 * (right - left)
            else:
                midpoint = 0.5 * (left + right)
            projected[row, column] = midpoint
            projected[column, row] = midpoint
    matrix = projected
    magnitude = float(np.max(np.abs(matrix)))
    if magnitude == 0.0:
        raise ValueError("the symmetric part of A must be nonzero")
    scaled = matrix / magnitude
    scaled_norm = float(np.linalg.norm(scaled, ord="fro"))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        frobenius = float(np.linalg.norm(matrix, ord="fro"))
    if (
        np.isfinite(frobenius)
        and frobenius > 0.0
        and frobenius <= np.finfo(float).max / 2.0
    ):
        scale = float((2.0 * frobenius / (order + 1.0)) ** (1.0 / order))
        return matrix / scale**order
    return (scaled / scaled_norm) * ((order + 1.0) / 2.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return ordinary, homogeneous extreme-scale, projected-symmetry and invalid cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5, -0.25], [0.5, 1.5, 0.125], [-0.25, 0.125, 3.0]])
p = 3
""",
            "call": "scaled_starting_matrix(A, p)",
            "gold_call": "_oracle_scaled_starting_matrix(A, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[4.0]])
p = 2
""",
            "call": "scaled_starting_matrix(A, p)",
            "gold_call": "_oracle_scaled_starting_matrix(A, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5, -0.25], [0.5, 1.5, 0.125], [-0.25, 0.125, 3.0]])
p = 5
""",
            "call": "scaled_starting_matrix(A, p)",
            "gold_call": "_oracle_scaled_starting_matrix(A, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5, -0.25], [0.5, 1.5, 0.125], [-0.25, 0.125, 3.0]])
A[0, 1] += 5e-13
p = 3
""",
            "call": "scaled_starting_matrix(A, p)",
            "gold_call": "_oracle_scaled_starting_matrix(A, p)",
        },
        {
            "setup": """import numpy as np
A = 1e300 * np.array([[2.0, 0.5, -0.25], [0.5, 1.5, 0.125], [-0.25, 0.125, 3.0]])
p = 7
""",
            "call": "scaled_starting_matrix(A, p)",
            "gold_call": "_oracle_scaled_starting_matrix(A, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.6e308, 3e307], [3e307, 1.2e308]])
p = 29
""",
            "call": "scaled_starting_matrix(A, p)",
            "gold_call": "_oracle_scaled_starting_matrix(A, p)",
        },
        {
            "setup": """import numpy as np
A = 1e-300 * np.array([[2.0, -0.25], [-0.25, 1.0]])
p = 4
""",
            "call": "scaled_starting_matrix(A, p)",
            "gold_call": "_oracle_scaled_starting_matrix(A, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[np.nextafter(0.0, 1.0)]])
p = 401
""",
            "call": "scaled_starting_matrix(A, p)",
            "gold_call": "_oracle_scaled_starting_matrix(A, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2e-12, 9e-13], [1e-13, 1e-12]])
p = 3
""",
            "call": "scaled_starting_matrix(A, p)",
            "gold_call": "_oracle_scaled_starting_matrix(A, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0], [0.0, 1.0]])
p = 3
def run_model_asym():
    try:
        scaled_starting_matrix(A, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_asym():
    try:
        _oracle_scaled_starting_matrix(A, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_asym()",
            "gold_call": "run_oracle_asym()",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5], [0.5, 1.5]])
p = True
def run_model_bool():
    try:
        scaled_starting_matrix(A, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_bool():
    try:
        _oracle_scaled_starting_matrix(A, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_bool()",
            "gold_call": "run_oracle_bool()",
        },
        {
            "setup": """import numpy as np
A = np.zeros((2, 2))
p = 3
def run_model_zero():
    try:
        scaled_starting_matrix(A, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_zero():
    try:
        _oracle_scaled_starting_matrix(A, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_zero()",
            "gold_call": "run_oracle_zero()",
        },
        {
            "setup": """import numpy as np
v = np.array([[1.0], [-2.0], [0.5]])
A = v @ v.T
p = 4
""",
            "call": "scaled_starting_matrix(A, p)",
            "gold_call": "_oracle_scaled_starting_matrix(A, p)",
        },
    ]
