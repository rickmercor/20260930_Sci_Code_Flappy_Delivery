"""
Form the CUR-CA core from packed indices: unpack [I, J] and return U = A(I, J)^+, the Moore-Penrose pseudoinverse of the intersection submatrix.

The CUR cross-approximation core is the Moore-Penrose pseudoinverse of the selected intersection. Supplying this core to later steps keeps the factorization dataflow explicit.

Returns
-------
ndarray of shape (ell, ell): the CUR core U
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def cur_core_matrix(A: np.ndarray, index_vector: np.ndarray) -> np.ndarray:
    """CUR-CA core U = A(I, J)^+.

    Parameters
    ----------
    A : np.ndarray
        Data matrix, shape (m, n).
    index_vector : np.ndarray
        Length-2*ell vector [I, J] of 0-based indices.

    Returns
    -------
    U : np.ndarray
        Core matrix of shape (ell, ell).

    Raises
    ------
    ValueError
        If A is not a 2D array, if index_vector does not have even positive
        length, if any index is out of range, or if I and J do not each
        contain ell distinct indices.
    """
    return np.zeros((len(index_vector) // 2, len(index_vector) // 2))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _unpack_indices(index_vector: np.ndarray, m: int, n: int):
    np = __import__("numpy")
    idx = np.asarray(index_vector, dtype=float).reshape(-1)
    if idx.size < 2 or idx.size % 2 != 0:
        raise ValueError("index_vector must have even positive length")
    ell = idx.size // 2
    I = np.rint(idx[:ell]).astype(int)
    J = np.rint(idx[ell:]).astype(int)
    if np.any(I < 0) or np.any(I >= m) or np.any(J < 0) or np.any(J >= n):
        raise ValueError("indices out of range")
    if len(np.unique(I)) != ell or len(np.unique(J)) != ell:
        raise ValueError("I and J must each contain ell distinct indices")
    return I, J, ell


def _oracle_cur_core_matrix(A: np.ndarray, index_vector: np.ndarray) -> np.ndarray:
    np = __import__("numpy")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) < 1:
        raise ValueError("A must be a 2D array")
    I, J, _ = _unpack_indices(index_vector, A.shape[0], A.shape[1])
    Aij = A[np.ix_(I, J)]
    return np.linalg.pinv(Aij)

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
index_vector = np.array([2.0, 6.0, 0.0, 6.0])
""",
            "call": "cur_core_matrix(A, index_vector)",
            "gold_call": "_oracle_cur_core_matrix(A, index_vector)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 4.0]])
index_vector = np.array([0.0, 0.0])
""",
            "call": "cur_core_matrix(A, index_vector)",
            "gold_call": "_oracle_cur_core_matrix(A, index_vector)",
        },
        {
            "setup": """import numpy as np
A = np.diag([4.0, 2.0, 1.0, 0.5])
index_vector = np.array([0.0, 1.0, 0.0, 1.0])
""",
            "call": "cur_core_matrix(A, index_vector)",
            "gold_call": "_oracle_cur_core_matrix(A, index_vector)",
        },
        {
            "setup": """import numpy as np
A = np.eye(3)
index_vector = np.array([0.0, 5.0])
def run_model():
    try:
        cur_core_matrix(A, index_vector)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_cur_core_matrix(A, index_vector)
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
A = np.eye(3)
index_vector = np.array([0.0, 0.0, 1.0])
def run_model():
    try:
        cur_core_matrix(A, index_vector)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_cur_core_matrix(A, index_vector)
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
