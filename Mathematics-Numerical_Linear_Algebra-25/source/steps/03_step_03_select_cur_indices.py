"""
Select rank-ell CUR row and column indices from a reused sketch. With empty index sets the sketched residual is E_row = Y. Columns J are the first ell LUPP row pivots of Y^T; rows I are the first ell LUPP row pivots of A(:, J). Return the concatenated vector [I, J] as floats.

Residual-driven pivoting first identifies informative columns from the transposed sketched residual and then rows from the matching column residual. The packed vector stores zero-based row indices followed by zero-based column indices.

Returns
-------
ndarray of shape (2*ell,): zero-based row indices followed by column indices
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def select_cur_indices(A: np.ndarray, Y: np.ndarray, ell: int) -> np.ndarray:
    """Residual-update CUR indices from one sketch.

    Parameters
    ----------
    A : np.ndarray
        Data matrix, shape (m, n).
    Y : np.ndarray
        Sketch Y = S A, shape (n_sketch, n) with n_sketch >= ell.
    ell : int
        Block size / target rank, 1 <= ell <= min(m, n, n_sketch).

    Returns
    -------
    index_vector : np.ndarray
        Length-2*ell vector [I, J] of 0-based indices stored as floats.

    Raises
    ------
    ValueError
        If A or Y is not 2D, if Y does not have n columns matching A, if ell is
        not an integer >= 1, or if ell exceeds min(m, n, n_sketch).
    """
    return np.zeros(2 * ell)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _lu_row_pivots(M: np.ndarray, k: int) -> np.ndarray:
    np = __import__("numpy")
    A = np.array(M, dtype=float, copy=True)
    m, n = A.shape
    piv = np.arange(m)
    for i in range(min(m, n)):
        j = i + int(np.argmax(np.abs(A[i:, i])))
        if j != i:
            A[[i, j]] = A[[j, i]]
            piv[[i, j]] = piv[[j, i]]
        pivot = A[i, i]
        if abs(pivot) > 0.0:
            A[i + 1 :, i] /= pivot
            A[i + 1 :, i + 1 :] -= np.outer(A[i + 1 :, i], A[i, i + 1 :])
    return piv[:k]


def _oracle_select_cur_indices(A: np.ndarray, Y: np.ndarray, ell: int) -> np.ndarray:
    np = __import__("numpy")
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if A.ndim != 2 or Y.ndim != 2:
        raise ValueError("A and Y must be 2D")
    m, n = A.shape
    if Y.shape[1] != n:
        raise ValueError("Y must have n columns matching A")
    if not isinstance(ell, (int, np.integer)) or int(ell) < 1:
        raise ValueError("ell must be an integer >= 1")
    ell = int(ell)
    if ell > min(m, n, Y.shape[0]):
        raise ValueError("ell cannot exceed min(m, n, n_sketch)")
    J = _lu_row_pivots(Y.T, ell)
    I = _lu_row_pivots(A[:, J], ell)
    return np.concatenate([I, J]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
rng = np.random.default_rng(7)
U, _ = np.linalg.qr(rng.standard_normal((12, 8)), mode='reduced')
V, _ = np.linalg.qr(rng.standard_normal((8, 8)), mode='reduced')
s = np.array([10.0, 9.0, 8.0, 0.45, 0.3, 0.22, 0.15, 0.1])
A = U @ np.diag(s) @ V.T
Y = np.random.default_rng(11).standard_normal((3, 12)) @ A
ell = 2
""",
            "call": "select_cur_indices(A, Y, ell)",
            "gold_call": "_oracle_select_cur_indices(A, Y, ell)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.1], [0.2, 3.0], [1.0, 0.0], [0.0, 1.5]], dtype=float)
Y = np.array([[4.0, 1.0], [0.5, 5.0]], dtype=float)
ell = 1
""",
            "call": "select_cur_indices(A, Y, ell)",
            "gold_call": "_oracle_select_cur_indices(A, Y, ell)",
        },
        {
            "setup": """import numpy as np
A = np.eye(3)
Y = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 0.1]], dtype=float)
ell = 2
""",
            "call": "select_cur_indices(A, Y, ell)",
            "gold_call": "_oracle_select_cur_indices(A, Y, ell)",
        },
        {
            "setup": """import numpy as np
A = np.ones((4, 3))
Y = np.ones((2, 3))
def run_model():
    try:
        select_cur_indices(A, Y, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_cur_indices(A, Y, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.ones((4, 3))
Y = np.ones((2, 2))
def run_model():
    try:
        select_cur_indices(A, Y, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_cur_indices(A, Y, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
