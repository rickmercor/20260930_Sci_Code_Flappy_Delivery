"""
Compute the captured singular values of a CUR surrogate from its supplied core.

The nonzero singular values of a CUR surrogate can be recovered from a small matrix formed with a Cholesky factor of the sampled columns, the supplied CUR core, and a reduced QR factor of the sampled rows.

Returns
-------
ndarray of shape (ell,): captured singular values in descending order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def cur_captured_singular_values(
    A: np.ndarray,
    index_vector: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    """Return the singular values captured by the supplied CUR factors.

    Parameters
    ----------
    A : np.ndarray
        Data matrix, shape (m, n).
    index_vector : np.ndarray
        Length-2*ell vector [I, J] of 0-based indices.
    U : np.ndarray
        CUR core from the preceding step, shape (ell, ell).

    Returns
    -------
    sigma : np.ndarray
        Captured singular values, shape (ell,), in descending order.

    Raises
    ------
    ValueError
        If A is not a 2D array, if index_vector does not have even positive
        length, if any index is out of range, if I and J do not each contain
        ell distinct indices, if U is not a finite array of shape
        (ell, ell), or if C^T C is not symmetric positive definite.
    """
    return np.zeros(len(index_vector) // 2)

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


def _oracle_cur_captured_singular_values(
    A: np.ndarray,
    index_vector: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    np = __import__("numpy")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) < 1:
        raise ValueError("A must be a 2D array")
    I, J, ell = _unpack_indices(index_vector, A.shape[0], A.shape[1])
    U = np.asarray(U, dtype=float)
    if U.shape != (ell, ell) or not np.all(np.isfinite(U)):
        raise ValueError("U must be a finite array of shape (ell, ell)")

    C = A[:, J]
    R = A[I, :]
    gram = C.T @ C
    try:
        T_C = np.linalg.cholesky(gram).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("C^T C must be SPD") from exc
    _, T_R = np.linalg.qr(R.T, mode="reduced")
    M = T_C @ U @ T_R.T
    sigma = np.linalg.svd(M, compute_uv=False, full_matrices=False)
    return np.asarray(sigma, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
rng = np.random.default_rng(7)
left, _ = np.linalg.qr(rng.standard_normal((12, 8)), mode='reduced')
right, _ = np.linalg.qr(rng.standard_normal((8, 8)), mode='reduced')
s = np.array([10.0, 9.0, 8.0, 0.45, 0.3, 0.22, 0.15, 0.1])
A = left @ np.diag(s) @ right.T
index_vector = np.array([2.0, 6.0, 0.0, 6.0])
U = _oracle_cur_core_matrix(A, index_vector)
""",
            "call": "cur_captured_singular_values(A, index_vector, U)",
            "gold_call": "_oracle_cur_captured_singular_values(A, index_vector, U)",
        },
        {
            "setup": """import numpy as np
A = np.diag([5.0, 3.0, 1.0])
index_vector = np.array([0.0, 0.0])
U = _oracle_cur_core_matrix(A, index_vector)
""",
            "call": "cur_captured_singular_values(A, index_vector, U)",
            "gold_call": "_oracle_cur_captured_singular_values(A, index_vector, U)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 4.0], [1.0, 0.0, 0.0]])
index_vector = np.array([0.0, 2.0, 1.0, 2.0])
U = _oracle_cur_core_matrix(A, index_vector)
""",
            "call": "cur_captured_singular_values(A, index_vector, U)",
            "gold_call": "_oracle_cur_captured_singular_values(A, index_vector, U)",
        },
        {
            "setup": """import numpy as np
A = np.eye(3)
index_vector = np.array([0.0, 0.0])
U = np.eye(2)
def run_model():
    try:
        cur_captured_singular_values(A, index_vector, U)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_cur_captured_singular_values(A, index_vector, U)
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
A = np.zeros((4, 3))
index_vector = np.array([0.0, 1.0, 0.0, 1.0])
U = np.zeros((2, 2))
def run_model():
    try:
        cur_captured_singular_values(A, index_vector, U)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_cur_captured_singular_values(A, index_vector, U)
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
