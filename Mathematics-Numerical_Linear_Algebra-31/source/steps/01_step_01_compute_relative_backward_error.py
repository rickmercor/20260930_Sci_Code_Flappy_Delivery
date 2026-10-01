"""
Evaluate the matrix-only relative backward error of a candidate linear-system iterate.

The paper measures solver progress with the residual norm divided by the product of the matrix spectral norm and the iterate norm. This matrix-only normalization differs from backward-error definitions that also include the right-hand-side norm.

Returns
-------
float, the nonnegative matrix-only relative backward error
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_relative_backward_error(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
) -> float:
    """Return ``||A x - b||_2 / (||A||_2 ||x||_2)``.

    Parameters
    ----------
    A : np.ndarray
        Finite, nonzero square matrix of shape ``(n, n)``.
    b : np.ndarray
        Finite right-hand side of shape ``(n,)``.
    x : np.ndarray
        Finite nonzero candidate solution of shape ``(n,)``.

    Returns
    -------
    float
        The nonnegative matrix-only relative backward error.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, an input is non-finite, or ``A`` or
        ``x`` has zero 2-norm.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_relative_backward_error(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
) -> float:
    """Return the deterministic relative backward error."""
    np = __import__("numpy")

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    n = A.shape[0]
    if b.shape != (n,) or x.shape != (n,):
        raise ValueError("b and x must have shape (n,)")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b)) and np.all(np.isfinite(x))):
        raise ValueError("all inputs must be finite")
    norm_A = float(np.linalg.norm(A, 2))
    norm_x = float(np.linalg.norm(x))
    if norm_A == 0.0 or norm_x == 0.0:
        raise ValueError("A and x must have nonzero 2-norm")
    return float(np.linalg.norm(A @ x - b) / (norm_A * norm_x))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return normal, exact, scale-invariant, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
x = np.array([0.1, 0.6])
""",
            "call": "compute_relative_backward_error(A, b, x)",
            "gold_call": "_oracle_compute_relative_backward_error(A, b, x)",
        },
        {
            "setup": """import numpy as np
A = np.diag([1.0, 0.1, 0.01])
x = np.array([2.0, -3.0, 4.0])
b = A @ x
""",
            "call": "compute_relative_backward_error(A, b, x)",
            "gold_call": "_oracle_compute_relative_backward_error(A, b, x)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, -1.0], [0.5, 3.0]])
b = np.array([0.25, -1.5])
x = np.array([1.2, -0.7])
scale = 7.5
""",
            "call": "compute_relative_backward_error(scale * A, scale * b, x)",
            "gold_call": "_oracle_compute_relative_backward_error(scale * A, scale * b, x)",
        },
        {
            "setup": """import numpy as np
A = np.eye(3)
b = np.ones(3)
x = np.zeros(3)
def run_model():
    try:
        compute_relative_backward_error(A, b, x)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_relative_backward_error(A, b, x)
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
