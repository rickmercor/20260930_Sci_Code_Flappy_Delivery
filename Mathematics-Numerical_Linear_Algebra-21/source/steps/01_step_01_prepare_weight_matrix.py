"""
Prepare the stable least-squares operator used by the numerical procedure.

For full-column-rank A, the least-squares operator A^+ provides the same mathematical action as W A.T without explicitly forming the inverse Gram matrix.

Returns
-------
np.ndarray, shape (n, m), containing a float64 least-squares operator A^+
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def prepare_weight_matrix(A: np.ndarray) -> np.ndarray:
    """Construct the stable least-squares operator A^+.

    Parameters
    ----------
    A : np.ndarray
        Finite matrix of shape (m, n), with m >= n and full column rank.

    Returns
    -------
    np.ndarray
        A float64 matrix with shape (n, m) representing the least-squares
        operator A^+.

    Raises
    ------
    ValueError
        If A is not two-dimensional, has fewer rows than columns, has zero
        columns, contains non-finite values, or is not full column rank.

    Notes
    -----
    The result is deterministic for identical inputs. The implementation should
    avoid forming A.T @ A and its explicit inverse.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_prepare_weight_matrix(A: np.ndarray) -> np.ndarray:
    """Deterministic stable reference implementation."""
    A = np.asarray(A, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")

    m, n = A.shape

    if m < n or n < 1:
        raise ValueError("A must satisfy m >= n >= 1")

    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain finite values")

    try:
        A_plus, _, rank, _ = np.linalg.lstsq(
            A,
            np.eye(m, dtype=np.float64),
            rcond=None,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("least-squares solve failed") from exc

    if rank < n:
        raise ValueError("A must have full column rank")

    if A_plus.shape != (n, m):
        raise ValueError("least-squares operator has incompatible shape")

    if not np.all(np.isfinite(A_plus)):
        raise ValueError("least-squares operator is non-finite")

    return np.asarray(A_plus, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
A = np.array(
    [[1.0, 2.0], [3.0, 5.0], [2.0, 1.0]],
    dtype=np.float64,
)
""",
            "call": "prepare_weight_matrix(A)",
            "gold_call": "_oracle_prepare_weight_matrix(A)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0]], dtype=np.float64)
""",
            "call": "prepare_weight_matrix(A)",
            "gold_call": "_oracle_prepare_weight_matrix(A)",
        },
        {
            "setup": """import numpy as np
A = np.array(
    [[1.0, 2.0], [2.0, 4.0]],
    dtype=np.float64,
)

def catches_value_error(fn, arg):
    try:
        fn(arg)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "catches_value_error(prepare_weight_matrix, A)",
            "gold_call": "catches_value_error(_oracle_prepare_weight_matrix, A)",
        },
    ]
