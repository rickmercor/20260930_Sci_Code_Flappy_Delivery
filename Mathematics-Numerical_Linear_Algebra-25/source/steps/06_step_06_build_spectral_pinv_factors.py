"""
Build compact numerical factors for the implicit rank-ell spectral inverse.

The captured regularized singular values are scaled to their smallest captured value. The returned array packs those inverse coefficients above the captured right basis so later matvecs do not rebuild or explicitly invert a dense preconditioner.

Returns
-------
ndarray of shape (n+1, ell): inverse coefficients followed by the captured right basis
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_spectral_pinv_factors(
    A: np.ndarray,
    index_vector: np.ndarray,
    U: np.ndarray,
    sigma: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Pack the inverse scales and captured right basis.

    Parameters
    ----------
    A : np.ndarray
        Data matrix, shape (m, n).
    index_vector : np.ndarray
        Length-2*ell vector [I, J] of 0-based indices.
    U : np.ndarray
        CUR core from the preceding step, shape (ell, ell).
    sigma : np.ndarray
        Captured singular values from the preceding step, shape (ell,).
    mu : float
        Regularization parameter, mu >= 0.

    Returns
    -------
    factors : np.ndarray
        Array of shape (n+1, ell). The first row stores inverse scaling
        coefficients and the remaining rows store the captured right basis.

    Raises
    ------
    ValueError
        If A is not a 2D array, if index_vector does not have even positive
        length, if any index is out of range, if I and J do not each contain
        ell distinct indices, if U is not a finite array of shape
        (ell, ell), if sigma does not contain ell positive finite values, if
        mu is not a finite number >= 0, if C^T C is not symmetric positive
        definite, or if sigma is inconsistent with A, index_vector and U.
    """
    return np.zeros((A.shape[1] + 1, len(index_vector) // 2))

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


def _oracle_build_spectral_pinv_factors(
    A: np.ndarray,
    index_vector: np.ndarray,
    U: np.ndarray,
    sigma: np.ndarray,
    mu: float,
) -> np.ndarray:
    np = __import__("numpy")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) < 1:
        raise ValueError("A must be a 2D array")
    if not np.isfinite(mu) or float(mu) < 0.0:
        raise ValueError("mu must be a finite number >= 0")
    I, J, ell = _unpack_indices(index_vector, A.shape[0], A.shape[1])

    U = np.asarray(U, dtype=float)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    if U.shape != (ell, ell) or not np.all(np.isfinite(U)):
        raise ValueError("U must be a finite array of shape (ell, ell)")
    if sigma.shape != (ell,) or np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must contain ell positive finite values")

    C = A[:, J]
    R = A[I, :]
    gram = C.T @ C
    try:
        T_C = np.linalg.cholesky(gram).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("C^T C must be SPD") from exc
    Q_R, T_R = np.linalg.qr(R.T, mode="reduced")
    M = T_C @ U @ T_R.T
    _, sigma_from_U, Vh = np.linalg.svd(M, full_matrices=False)
    if not np.allclose(sigma, sigma_from_U, rtol=1e-10, atol=1e-12):
        raise ValueError("sigma is inconsistent with A, index_vector, and U")

    Vhat = Q_R @ Vh.T
    gamma = np.sqrt(sigma**2 + float(mu) ** 2)
    inverse_scale = gamma[-1] / gamma - 1.0
    return np.vstack([inverse_scale, Vhat])

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
sigma = _oracle_cur_captured_singular_values(A, index_vector, U)
mu = 0.05
""",
            "call": "build_spectral_pinv_factors(A, index_vector, U, sigma, mu)",
            "gold_call": "_oracle_build_spectral_pinv_factors(A, index_vector, U, sigma, mu)",
        },
        {
            "setup": """import numpy as np
A = np.diag([4.0, 2.0, 1.0])
index_vector = np.array([0.0, 0.0])
U = _oracle_cur_core_matrix(A, index_vector)
sigma = _oracle_cur_captured_singular_values(A, index_vector, U)
mu = 0.0
""",
            "call": "build_spectral_pinv_factors(A, index_vector, U, sigma, mu)",
            "gold_call": "_oracle_build_spectral_pinv_factors(A, index_vector, U, sigma, mu)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 1.0], [0.0, 3.0], [1.0, 0.0]], dtype=float)
index_vector = np.array([0.0, 2.0, 0.0, 1.0])
U = _oracle_cur_core_matrix(A, index_vector)
sigma = _oracle_cur_captured_singular_values(A, index_vector, U)
mu = 1.0
""",
            "call": "build_spectral_pinv_factors(A, index_vector, U, sigma, mu)",
            "gold_call": "_oracle_build_spectral_pinv_factors(A, index_vector, U, sigma, mu)",
        },
        {
            "setup": """import numpy as np
A = np.diag([3.0, 2.0, 1.0])
index_vector = np.array([0.0, 0.0])
U = np.array([[1.0 / 3.0]])
sigma = np.array([99.0])
def run_model():
    try:
        build_spectral_pinv_factors(A, index_vector, U, sigma, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "1",
        },
        {
            "setup": """import numpy as np
A = np.eye(3)
index_vector = np.array([0.0, 0.0])
U = np.ones((2, 2))
sigma = np.ones(1)
def run_model():
    try:
        build_spectral_pinv_factors(A, index_vector, U, sigma, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_spectral_pinv_factors(A, index_vector, U, sigma, 0.1)
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
