"""
Assemble the reduced generalized Sturm--Liouville matrix pencil.



For the equation ``-y'' + y = lambda q(x) y``, endpoint deletion has already

been applied to the second-derivative matrix.  The left pencil matrix is

therefore `$A = -D2 + I$` and the right matrix is the diagonal sampling of the

strictly positive profile.

Returns
-------
tuple of finite float arrays, each with shape (m, m)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_generalized_pencil(
    reduced_second_order: np.ndarray, potential: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite-dimensional pencil ``(A, Q)``.

    ``reduced_second_order`` must be a nonempty finite square matrix and
    ``potential`` a matching finite vector with strictly positive entries.
    Invalid inputs raise ``ValueError``.

    Parameters
    ----------
    reduced_second_order : np.ndarray
        Physical second-derivative matrix, shape ``(m, m)``.
    potential : np.ndarray
        Positive profile samples, shape ``(m,)``.

    Returns
    -------
    a_matrix : np.ndarray
        Left pencil matrix ``-D2 + I``, shape ``(m, m)``.
    q_matrix : np.ndarray
        Diagonal right pencil matrix, shape ``(m, m)``.
    """
    return a_matrix, q_matrix  # noqa: F821 - model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_generalized_pencil(
    reduced_second_order: np.ndarray, potential: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reference assembly of ``A = -D2 + I`` and ``Q = diag(q)``."""
    matrix = np.asarray(reduced_second_order, dtype=float)
    q = np.asarray(potential, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or q.ndim != 1
        or q.size != matrix.shape[0]
    ):
        raise ValueError("operator and potential have incompatible shapes")
    if (
        not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(q))
        or np.any(q <= 0.0)
    ):
        raise ValueError("operator must be finite and potential must be positive")

    a_matrix = -matrix + np.eye(matrix.shape[0], dtype=float)
    q_matrix = np.diag(q)
    return a_matrix, q_matrix

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return dense, singleton, nonsymmetric, and invalid-profile cases."""
    return [
        {
            "setup": """import numpy as np
reduced_second_order = np.array([[2.0, -1.5, 0.25], [0.5, -3.0, 4.0], [1.25, 0.0, 0.75]])
potential = np.array([0.9, 0.5, 0.02])
""",
            "call": "(lambda value: float(np.dot(value[0].ravel(), np.arange(1, value[0].size + 1, dtype=float)) + np.dot(value[1].ravel(), np.arange(value[0].size + 1, value[0].size + value[1].size + 1, dtype=float))))(assemble_generalized_pencil(reduced_second_order, potential))",
            "gold_call": "(lambda value: float(np.dot(value[0].ravel(), np.arange(1, value[0].size + 1, dtype=float)) + np.dot(value[1].ravel(), np.arange(value[0].size + 1, value[0].size + value[1].size + 1, dtype=float))))(_oracle_assemble_generalized_pencil(reduced_second_order, potential))",
        },
        {
            "setup": """import numpy as np
reduced_second_order = np.array([[-7.5]])
potential = np.array([0.125])
""",
            "call": "(lambda value: float(np.dot(value[0].ravel(), np.arange(1, value[0].size + 1, dtype=float)) + np.dot(value[1].ravel(), np.arange(value[0].size + 1, value[0].size + value[1].size + 1, dtype=float))))(assemble_generalized_pencil(reduced_second_order, potential))",
            "gold_call": "(lambda value: float(np.dot(value[0].ravel(), np.arange(1, value[0].size + 1, dtype=float)) + np.dot(value[1].ravel(), np.arange(value[0].size + 1, value[0].size + value[1].size + 1, dtype=float))))(_oracle_assemble_generalized_pencil(reduced_second_order, potential))",
        },
        {
            "setup": """import numpy as np
reduced_second_order = np.array([[0.0, 100.0], [-0.125, 2.0]])
potential = np.array([1e-12, 3.5])
""",
            "call": "(lambda value: float(np.dot(value[0].ravel(), np.arange(1, value[0].size + 1, dtype=float)) + np.dot(value[1].ravel(), np.arange(value[0].size + 1, value[0].size + value[1].size + 1, dtype=float))))(assemble_generalized_pencil(reduced_second_order, potential))",
            "gold_call": "(lambda value: float(np.dot(value[0].ravel(), np.arange(1, value[0].size + 1, dtype=float)) + np.dot(value[1].ravel(), np.arange(value[0].size + 1, value[0].size + value[1].size + 1, dtype=float))))(_oracle_assemble_generalized_pencil(reduced_second_order, potential))",
        },
        {
            "setup": """import numpy as np
reduced_second_order = np.eye(2)
potential = np.array([0.5, 0.0])
def run_model():
    try:
        assemble_generalized_pencil(reduced_second_order, potential)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_generalized_pencil(reduced_second_order, potential)
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
