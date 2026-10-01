"""
Project the current sketch direction against previously generated corrections.

The stable calculation uses the least-squares operator prepared in Step 01 for the current sketch direction and projects the sketch against the accumulated historical correction space without explicitly forming an inverse Gram matrix.

Returns
-------
tuple[np.ndarray, float], containing the projected direction and historical scalar
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def project_historical_direction(
    A: np.ndarray,
    A_plus: np.ndarray,
    P: np.ndarray,
    s: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project the current direction against historical corrections.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix of shape (m, n).
    A_plus : np.ndarray
        Stable least-squares operator of shape (n, m) prepared from A.
    P : np.ndarray
        Previously generated correction directions of shape (n, r).
    s : np.ndarray
        Current sketch vector of shape (m,).

    Returns
    -------
    tuple[np.ndarray, float]
        The projected direction, and the historical correction scalar. Writing
        ``c`` for the coefficients of the least-squares fit of the sketch in the
        basis ``A @ P``, the projected direction is the refined least-squares
        direction less ``P @ c``, and the scalar is the squared Euclidean norm
        of ``A @ P @ c``. When the historical space is empty the projected
        direction is the refined least-squares direction itself and the scalar
        is 0.0.

    Raises
    ------
    ValueError
        If the inputs have incompatible shapes, contain non-finite values,
        A does not have full column rank, the historical space is rank-deficient,
        or the historical subspace has exhausted the available dimension.

    Notes
    -----
    The result is deterministic for identical inputs. The current least-squares
    direction is formed from A_plus @ s and refined once using the same operator
    on its residual before the historical projection is applied.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_project_historical_direction(
    A: np.ndarray,
    A_plus: np.ndarray,
    P: np.ndarray,
    s: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Deterministic stable reference implementation."""
    import numpy as np

    A = np.asarray(A, dtype=np.float64)
    A_plus = np.asarray(A_plus, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    if A.ndim != 2 or A_plus.ndim != 2 or P.ndim != 2 or s.ndim != 1:
        raise ValueError("inputs have incompatible dimensions")

    m, n = A.shape

    if A_plus.shape != (n, m):
        raise ValueError("A_plus has incompatible shape")

    if P.shape[0] != n:
        raise ValueError("P has incompatible shape")

    if s.shape != (m,):
        raise ValueError("s has incompatible shape")

    if not (
        np.all(np.isfinite(A))
        and np.all(np.isfinite(A_plus))
        and np.all(np.isfinite(P))
        and np.all(np.isfinite(s))
    ):
        raise ValueError("inputs must be finite")

    try:
        _, singular_values, _ = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("SVD failed while checking A") from exc

    if singular_values.size != n or not np.all(np.isfinite(singular_values)):
        raise ValueError("A must have full column rank")

    rank_tolerance = (
        max(A.shape)
        * np.finfo(np.float64).eps
        * float(singular_values[0])
    )
    if singular_values[-1] <= rank_tolerance:
        raise ValueError("A must have full column rank")

    z = A_plus @ s
    if not np.all(np.isfinite(z)):
        raise ValueError("least-squares direction is non-finite")

    # One residual refinement keeps the operator-based path numerically aligned
    # with a direct least-squares solve while still consuming Step 01's output.
    refinement = A_plus @ (s - A @ z)
    if not np.all(np.isfinite(refinement)):
        raise ValueError("least-squares refinement is non-finite")
    z = z + refinement

    if P.shape[1] == 0:
        return np.asarray(z, dtype=np.float64), 0.0

    if P.shape[1] >= n:
        raise ValueError("historical correction space has exhausted the available dimension")

    AP = A @ P
    if not np.all(np.isfinite(AP)):
        raise ValueError("A @ P is non-finite")

    c, _, rank_p, _ = np.linalg.lstsq(AP, s, rcond=None)
    if rank_p < P.shape[1]:
        raise ValueError("historical correction space is rank-deficient")
    if not np.all(np.isfinite(c)):
        raise ValueError("historical projection coefficients are non-finite")

    historical_component = AP @ c
    if not np.all(np.isfinite(historical_component)):
        raise ValueError("historical projection is non-finite")

    projected = z - P @ c
    if not np.all(np.isfinite(projected)):
        raise ValueError("projected direction is non-finite")

    delta = float(historical_component @ historical_component)
    if not np.isfinite(delta):
        raise ValueError("historical correction scalar is non-finite")

    return np.asarray(projected, dtype=np.float64), float(delta)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np

A = np.array(
    [[1.0, 2.0],
     [2.0, 1.0],
     [1.0, 0.0]],
    dtype=np.float64,
)
A_plus = np.linalg.lstsq(
    A,
    np.eye(A.shape[0], dtype=np.float64),
    rcond=None,
)[0]
P = np.array([[1.0], [0.2]], dtype=np.float64)
s = np.array([0.5, -1.0, 0.7], dtype=np.float64)
""",
            "call": "project_historical_direction(A, A_plus, P, s)",
            "gold_call": "_oracle_project_historical_direction(A, A_plus, P, s)",
        },
        {
            "setup": """import numpy as np

A = np.array([[2.0]], dtype=np.float64)
A_plus = np.array([[0.5]], dtype=np.float64)
P = np.empty((1, 0), dtype=np.float64)
s = np.array([3.0], dtype=np.float64)
""",
            "call": "project_historical_direction(A, A_plus, P, s)",
            "gold_call": "_oracle_project_historical_direction(A, A_plus, P, s)",
        },
        {
            "setup": """import numpy as np

A = np.eye(2, dtype=np.float64)
A_plus = np.eye(2, dtype=np.float64)
P = np.array(
    [[1.0, 2.0],
     [2.0, 4.0]],
    dtype=np.float64,
)
s = np.array([1.0, -1.0], dtype=np.float64)

def catches_value_error(fn):
    try:
        fn(A, A_plus, P, s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "catches_value_error(project_historical_direction)",
            "gold_call": "catches_value_error(_oracle_project_historical_direction)",
        },
    ]
