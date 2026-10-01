"""
From a completed Krylov pack (all Hessenberg columns written), return the coefficients that make the leftover Euclidean-orthogonal to the accepted basis.

The leftover after sketched Arnoldi is only Omega-orthogonal to range(U). Restoring similarity requires Euclidean orthogonality of that leftover, which is a different inner product than the one used to build the basis. A sketched Galerkin condition on Omega, or a pseudoinverse on a rank-deficient Gram matrix, is a different algorithm.

Returns
-------
ndarray of shape (m,): Euclidean leftover coefficients
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def leftover_euclidean_coeffs(state: np.ndarray) -> np.ndarray:
    """Return coefficients making the leftover Euclidean-orthogonal to range(U).

    Parameters
    ----------
    state : np.ndarray
        Completed Krylov pack (n_hess = m).

    Returns
    -------
    hhat : np.ndarray
        Vector of length m.

    Raises
    ------
    ValueError
        If the Krylov pack is short, the wrong length, or has an invalid
        header, if n_hess is not m, or if U.T @ U is not SPD.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _unpack_leftover_state(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("Krylov pack is too short")
    n, m, n_hess = [int(round(float(v))) for v in p[:3]]
    if n < 2 or m < 1 or n_hess < 0 or n_hess > m:
        raise ValueError("Krylov pack header is invalid")
    need = 3 + n * m + m * m + n + 1
    if p.size != need:
        raise ValueError("Krylov pack length does not match header")
    U = p[3 : 3 + n * m].reshape(n, m)
    leftover = p[3 + n * m + m * m : 3 + n * m + m * m + n]
    return n, m, n_hess, U, leftover

def _oracle_leftover_euclidean_coeffs(state):
    _n, m, n_hess, U, leftover = _unpack_leftover_state(state)
    if n_hess != m:
        raise ValueError("Krylov pack is incomplete")
    G = U.T @ U
    try:
        return np.linalg.solve(G, U.T @ leftover)
    except np.linalg.LinAlgError as exc:
        raise ValueError("U^T U must be SPD") from exc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, m, n_hess = 4, 2, 2
U = np.array([[1.0, 0.1], [0.0, 1.0], [0.2, 0.0], [0.0, 0.3]])
H = np.array([[1.0, 0.5], [0.4, 2.0]])
leftover = np.array([0.3, -0.2, 0.1, 0.4])
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), leftover, [0.8]))
""",
            "call": "leftover_euclidean_coeffs(state)",
            "gold_call": "_oracle_leftover_euclidean_coeffs(state)",
        },
        {
            "setup": """import numpy as np
n, m, n_hess = 3, 2, 2
U = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
H = np.eye(2)
leftover = np.array([0.2, -0.1, 1.0])
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), leftover, [0.5]))
""",
            "call": "leftover_euclidean_coeffs(state)",
            "gold_call": "_oracle_leftover_euclidean_coeffs(state)",
        },
        {
            "setup": """import numpy as np
n, m, n_hess = 3, 2, 1
U = np.eye(3, 2)
H = np.eye(2)
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), np.ones(3), [0.5]))
def run_model():
    try:
        leftover_euclidean_coeffs(state)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_leftover_euclidean_coeffs(state)
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
n, m, n_hess = 3, 2, 2
U = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
H = np.eye(2)
leftover = np.array([1.0, 0.0, 0.0])
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), leftover, [1.0]))
def run_model():
    try:
        leftover_euclidean_coeffs(state)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_leftover_euclidean_coeffs(state)
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
def run_model():
    try:
        leftover_euclidean_coeffs(np.ones(3))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_leftover_euclidean_coeffs(np.ones(3))
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
