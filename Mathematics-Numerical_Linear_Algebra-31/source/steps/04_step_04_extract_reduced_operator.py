"""
Extract the paper-specific reduced operator and reproduce Algorithm 4.1's incremental, shifted upper-banded Cholesky state.

Deleting the first row of the extended symmetric recurrence produces the square upper-triangular operator whose smallest singular value is the Krylov-space backward error. Algorithm 4.1 updates only the final column of the shifted pentadiagonal Gram matrix and its upper-banded Cholesky factor. The accumulated factor and the first rejected pivot certify the stopping dimension without a dense factorization.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray], the reduced operator, length-(k+2) certificate, and (k, k) upper-banded factor state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def extract_reduced_operator(
    T: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the reduced operator and its incremental Cholesky state.

    Delete the first row of the ``(k+1)``-by-``k`` symmetric recurrence
    matrix to obtain the paper's square upper-triangular matrix ``reduced``.
    It has upper bandwidth two. Form

    ``G = reduced.T @ reduced - threshold**2 * I``

    and update the upper-banded Cholesky factorization ``G = R.T @ R`` one
    prefix at a time. For prefix ``j``, form only the final three potentially
    nonzero entries ``G[i, j]`` for ``max(0, j-2) <= i <= j`` directly from
    column dot products of ``reduced``. Then compute only the final column of
    ``R``. Do not materialize the full Gram matrix and do not call a dense
    Cholesky factorization, eigensolver, singular-value decomposition, matrix
    inverse, or linear solver. Stop at the first diagonal pivot that is not
    greater than
    ``64 * eps * max(1, ||reduced||_inf**2, threshold**2)``.

    Return ``(reduced, certificate, R)``. The certificate has length ``k+2``:
    ``certificate[0]`` is the zero-based rejected-pivot index, or ``k`` when
    all pivots are accepted; ``certificate[1]`` is the rejected pivot, or the
    final accepted pivot when the factorization succeeds; the remaining
    entries are ``diag(R)``, with zeros from the first rejected pivot onward.
    ``R`` is the upper-triangular factor of shape ``(k, k)`` accumulated up to
    the rejection; its uncomputed entries are zero. This factor is part of the
    required state because Algorithm 4.1 updates its last column rather than
    refactorizing every prefix.

    Parameters
    ----------
    T : np.ndarray
        Finite recurrence matrix of shape ``(k+1, k)`` with ``k >= 1`` whose
        first-row deletion is upper triangular with upper bandwidth two.
    threshold : float
        Finite nonnegative singular-value threshold. Boolean values are
        invalid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The ``(k, k)`` reduced operator, the length-``k+2`` certificate, and
        the ``(k, k)`` upper-banded factor state.

    Raises
    ------
    ValueError
        If ``T`` does not have shape ``(k+1, k)`` with ``k >= 1``, if ``T``
        is not finite, if ``threshold`` is not finite and nonnegative
        (booleans included), or if the first-row deletion of ``T`` is not
        upper triangular with upper bandwidth two.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_extract_reduced_operator(
    T: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the reduced operator and structured convergence certificate."""
    np = __import__("numpy")

    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[1] < 1 or T.shape[0] != T.shape[1] + 1:
        raise ValueError("T must have shape (k+1, k) with k >= 1")
    if not np.all(np.isfinite(T)):
        raise ValueError("T must be finite")
    if isinstance(threshold, (bool, np.bool_)):
        raise ValueError("threshold must be finite and nonnegative")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be finite and nonnegative") from exc
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and nonnegative")

    reduced = T[1:, :].copy()
    k = reduced.shape[0]
    scale = max(1.0, float(np.linalg.norm(reduced, ord=np.inf)))
    structure_tolerance = 64.0 * np.finfo(float).eps * scale
    if np.any(np.abs(np.tril(reduced, k=-1)) > structure_tolerance):
        raise ValueError("the reduced operator must be upper triangular")
    if np.any(np.abs(np.triu(reduced, k=3)) > structure_tolerance):
        raise ValueError("the reduced operator must have upper bandwidth two")

    pivot_tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        scale**2,
        threshold**2,
    )
    R = np.zeros((k, k), dtype=float)
    rejected = k
    terminal_pivot = 0.0

    for j in range(k):
        for i in range(max(0, j - 2), j):
            gram_ij = float(reduced[:, i] @ reduced[:, j])
            numerator = gram_ij
            for m in range(max(0, i - 2, j - 2), i):
                numerator -= R[m, i] * R[m, j]
            numerator_scale = max(1.0, abs(gram_ij))
            if abs(R[i, i]) <= pivot_tolerance / numerator_scale:
                rejected = i
                terminal_pivot = float(R[i, i] ** 2)
                break
            R[i, j] = numerator / R[i, i]
        if rejected != k:
            break

        pivot = float(reduced[:, j] @ reduced[:, j]) - threshold**2
        for m in range(max(0, j - 2), j):
            pivot -= R[m, j] ** 2
        terminal_pivot = float(pivot)
        if not np.isfinite(pivot) or pivot <= pivot_tolerance:
            rejected = j
            break
        R[j, j] = np.sqrt(pivot)

    certificate = np.concatenate(
        ([float(rejected), terminal_pivot], np.diag(R))
    )
    return reduced, certificate, R

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return accepted, rejected, recurrence-derived, restricted, and invalid cases."""
    return [
        {
            "setup": "import numpy as np\nT = np.array([[2.0], [0.5]])\nthreshold = 0.1",
            "call": "np.concatenate([part.ravel() for part in extract_reduced_operator(T, threshold)])",
            "gold_call": "np.concatenate([part.ravel() for part in _oracle_extract_reduced_operator(T, threshold)])",
        },
        {
            "setup": """import numpy as np
T = np.array([[2.0, 0.4, 0.0], [0.4, 1.2, 0.3], [0.0, 0.3, 0.8], [0.0, 0.0, 0.2]])
threshold = 0.25
""",
            "call": "np.concatenate([part.ravel() for part in extract_reduced_operator(T, threshold)])",
            "gold_call": "np.concatenate([part.ravel() for part in _oracle_extract_reduced_operator(T, threshold)])",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(9, 1.0e6)
A, b = packed[:, :-1], packed[:, -1]
_, T = _oracle_build_orthogonal_recurrence(A, b, 5)
threshold = 1.0e-4
""",
            "call": "np.concatenate([part.ravel() for part in extract_reduced_operator(T, threshold)])",
            "gold_call": "np.concatenate([part.ravel() for part in _oracle_extract_reduced_operator(T, threshold)])",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(16, 1.0e12)
A, b = packed[:, :-1], packed[:, -1]
_, T = _oracle_build_orthogonal_recurrence(A, b, 7)
threshold = 1.0e-5
forbidden_names = ("cholesky", "svd", "eig", "eigh", "eigvals", "eigvalsh")
saved_linalg = {name: getattr(np.linalg, name) for name in forbidden_names}
def forbidden(*args, **kwargs):
    raise RuntimeError("dense factorization or spectral routine is forbidden")
def run_with_restrictions(function):
    try:
        for name in forbidden_names:
            setattr(np.linalg, name, forbidden)
        return np.concatenate([part.ravel() for part in function(T, threshold)])
    finally:
        for name, implementation in saved_linalg.items():
            setattr(np.linalg, name, implementation)
""",
            "call": "run_with_restrictions(extract_reduced_operator)",
            "gold_call": "run_with_restrictions(_oracle_extract_reduced_operator)",
        },
        {
            "setup": """import numpy as np
T = np.eye(3)
def run_model():
    try:
        extract_reduced_operator(T, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_extract_reduced_operator(T, 0.1)
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
