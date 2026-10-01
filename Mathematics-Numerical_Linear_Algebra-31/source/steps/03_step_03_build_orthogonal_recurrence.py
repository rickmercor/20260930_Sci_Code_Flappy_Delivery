"""
Build the reorthogonalized symmetric Krylov basis and finite recurrence matrix.

For a symmetric matrix, the Krylov basis admits a short recurrence. The prescribed system is extremely ill-conditioned, so one explicit reorthogonalization against the accumulated basis is used to preserve the orthonormal-basis and recurrence identities in floating-point arithmetic.

Returns
-------
tuple[np.ndarray, np.ndarray], Q with shape (n, k+1) and T with shape (k+1, k)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_orthogonal_recurrence(
    A: np.ndarray,
    b: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable ``Q`` and ``T`` satisfying ``A Q[:, :k] = Q T``.

    Initialize ``q_prev = 0``, ``q = b / ||b||_2``, and ``beta = 0``.
    For columns ``j = 0, ..., k-1``, compute

    ``w = A q - beta q_prev``, ``alpha = q.T w``, and
    ``w = w - alpha q``. Reorthogonalize once against all columns already
    stored in ``Q`` by replacing ``w`` with
    ``w - Q[:, :j+1] @ (Q[:, :j+1].T @ w)``. Then set
    ``beta_next = ||w||_2`` and ``q_next = w / beta_next``.

    Store ``alpha`` on row ``j`` of column ``j`` in ``T``, the previous
    ``beta`` on row ``j-1`` when ``j > 0``, and ``beta_next`` on row
    ``j+1``. The off-diagonal coefficients are nonnegative by construction.

    Parameters
    ----------
    A : np.ndarray
        Finite symmetric matrix of shape ``(n, n)``.
    b : np.ndarray
        Finite nonzero vector of shape ``(n,)``.
    iterations : int
        Number ``k`` of recurrence columns, with ``1 <= k < n``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``Q`` with shape ``(n, k+1)`` and orthonormal columns, and ``T``
        with shape ``(k+1, k)``.

    Raises
    ------
    ValueError
        If ``A`` is not a nonempty square matrix, if ``A`` or ``b`` is
        non-finite or their shapes are incompatible, if ``A`` is not
        symmetric, if ``iterations`` does not satisfy ``1 <= iterations < n``,
        if ``b`` is zero, or if the recurrence breaks down before the
        requested iteration.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_orthogonal_recurrence(
    A: np.ndarray,
    b: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic reorthogonalized recurrence."""
    np = __import__("numpy")

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    n = A.shape[0]
    if b.shape != (n,) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("inputs must be finite with compatible shapes")
    if not np.allclose(A, A.T, rtol=0.0, atol=1e-12):
        raise ValueError("A must be symmetric")
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (int, np.integer))
        or not 1 <= iterations < n
    ):
        raise ValueError("iterations must satisfy 1 <= iterations < n")
    norm_b = float(np.linalg.norm(b))
    if norm_b == 0.0:
        raise ValueError("b must be nonzero")

    Q = np.zeros((n, iterations + 1), dtype=float)
    T = np.zeros((iterations + 1, iterations), dtype=float)
    Q[:, 0] = b / norm_b
    q_prev = np.zeros(n, dtype=float)
    beta = 0.0
    scale = max(1.0, float(np.linalg.norm(A, 2)))

    for j in range(iterations):
        q = Q[:, j]
        w = A @ q - beta * q_prev
        alpha = float(q @ w)
        w = w - alpha * q
        active_basis = Q[:, : j + 1]
        w = w - active_basis @ (active_basis.T @ w)
        beta_next = float(np.linalg.norm(w))
        if beta_next <= 64.0 * np.finfo(float).eps * scale:
            raise ValueError("the recurrence broke down before the requested iteration")
        T[j, j] = alpha
        if j > 0:
            T[j - 1, j] = beta
        T[j + 1, j] = beta_next
        Q[:, j + 1] = w / beta_next
        q_prev = q
        beta = beta_next
    return Q, T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return one-step, moderate, ill-conditioned, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5, 0.0], [0.5, 1.5, 0.25], [0.0, 0.25, 1.0]])
b = np.array([1.0, -0.5, 2.0])
iterations = 1
""",
            "call": "np.concatenate([part.ravel() for part in build_orthogonal_recurrence(A, b, iterations)])",
            "gold_call": "np.concatenate([part.ravel() for part in _oracle_build_orthogonal_recurrence(A, b, iterations)])",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(8, 1.0e4)
A, b = packed[:, :-1], packed[:, -1]
iterations = 4
""",
            "call": "np.concatenate([part.ravel() for part in build_orthogonal_recurrence(A, b, iterations)])",
            "gold_call": "np.concatenate([part.ravel() for part in _oracle_build_orthogonal_recurrence(A, b, iterations)])",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(13, 1.0e10)
A, b = packed[:, :-1], packed[:, -1]
iterations = 7
""",
            "call": "np.concatenate([part.ravel() for part in build_orthogonal_recurrence(A, b, iterations)])",
            "gold_call": "np.concatenate([part.ravel() for part in _oracle_build_orthogonal_recurrence(A, b, iterations)])",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0], [0.0, 1.0]])
b = np.ones(2)
def run_model():
    try:
        build_orthogonal_recurrence(A, b, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_orthogonal_recurrence(A, b, 1)
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
