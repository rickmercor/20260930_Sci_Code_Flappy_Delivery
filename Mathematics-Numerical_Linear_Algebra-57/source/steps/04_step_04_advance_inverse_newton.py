"""
Apply one accelerated step to the residual-bearing member of the coupled pair.



The step multiplies `$M$` by the `$p$$-th power of$$I + alpha R$`, where `$R = I - M$` is the residual carried into the step. The partner `$X$` advances by the same factor on the right over the same step, and the tie `$M = X ** p A$` is what makes that consistent, but every reported quantity of the run depends on `$M$` alone, so `$X$` is not carried here.



The factor is raised to an integer power rather than applied `$p$$times to a product, so the step remains a pure matrix-multiplication kernel and the residual after it is exactly the polynomial in$$R$` that the scalar expansion describes.

Returns
-------
np.ndarray of shape (n, n), the residual-bearing matrix after one step
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def advance_inverse_newton(M: np.ndarray, alpha: float, p: int) -> np.ndarray:
    """Advance the residual-bearing matrix by one accelerated step.

    Raises ``ValueError`` unless every one of the following holds: ``M`` is real
    rather than complex; ``M`` is a nonempty square two-dimensional array; every
    entry of ``M`` is finite; ``alpha`` is finite; and ``p`` is an integer, not a
    bool, with ``p >= 1``.

    Parameters
    ----------
    M : np.ndarray
        Finite real square matrix of shape ``(n, n)`` before the step.
    alpha : float
        Fitted coefficient for this step.
    p : int
        Root order.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n, n)`` holding the matrix after the step.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_advance_inverse_newton(M: np.ndarray, alpha: float, p: int) -> np.ndarray:
    """Reference accelerated inverse Newton step."""
    carried = np.asarray(M)
    if np.iscomplexobj(carried):
        raise ValueError("M must be real")
    carried = np.asarray(carried, dtype=float)
    if (
        carried.ndim != 2
        or carried.shape[0] != carried.shape[1]
        or carried.shape[0] == 0
    ):
        raise ValueError("M must be a nonempty square matrix")
    if not np.all(np.isfinite(carried)):
        raise ValueError("M must contain only finite values")
    alpha = float(alpha)
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
        raise ValueError("p must be an integer")
    if int(p) < 1:
        raise ValueError("p must be at least one")
    order = int(p)
    identity = np.eye(carried.shape[0], dtype=float)
    factor = identity + alpha * (identity - carried)
    return np.linalg.matrix_power(factor, order) @ carried

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return cubic, scalar, quintic, frozen-coefficient, quadratic-order, non-square, non-finite-coefficient and bool-order cases."""
    return [
        {
            "setup": """import numpy as np
M = np.array([[0.6, 0.1, 0.0], [0.1, 1.3, -0.2], [0.0, -0.2, 0.85]])
alpha, p = 0.472201531146, 3
""",
            "call": "advance_inverse_newton(M, alpha, p)",
            "gold_call": "_oracle_advance_inverse_newton(M, alpha, p)",
        },
        {
            "setup": """import numpy as np
M = np.array([[0.5]])
alpha, p = 0.75, 1
""",
            "call": "advance_inverse_newton(M, alpha, p)",
            "gold_call": "_oracle_advance_inverse_newton(M, alpha, p)",
        },
        {
            "setup": """import numpy as np
M = np.array([[0.6, 0.1, 0.0], [0.1, 1.3, -0.2], [0.0, -0.2, 0.85]])
alpha, p = 0.3, 5
""",
            "call": "advance_inverse_newton(M, alpha, p)",
            "gold_call": "_oracle_advance_inverse_newton(M, alpha, p)",
        },
        {
            "setup": """import numpy as np
M = np.array([[0.6, 0.1, 0.0], [0.1, 1.3, -0.2], [0.0, -0.2, 0.85]])
alpha, p = 0.0, 4
""",
            "call": "advance_inverse_newton(M, alpha, p)",
            "gold_call": "_oracle_advance_inverse_newton(M, alpha, p)",
        },
        {
            "setup": """import numpy as np
M = np.array([[1.4, -0.3], [-0.3, 0.2]])
alpha, p = -0.6, 2
""",
            "call": "advance_inverse_newton(M, alpha, p)",
            "gold_call": "_oracle_advance_inverse_newton(M, alpha, p)",
        },
        {
            "setup": """import numpy as np
M = np.ones((2, 3))
alpha, p = 0.5, 3
def run_model_shape():
    try:
        advance_inverse_newton(M, alpha, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_shape():
    try:
        _oracle_advance_inverse_newton(M, alpha, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_shape()",
            "gold_call": "run_oracle_shape()",
        },
        {
            "setup": """import numpy as np
M = np.array([[0.6, 0.1], [0.1, 1.3]])
alpha, p = float("inf"), 3
def run_model_inf():
    try:
        advance_inverse_newton(M, alpha, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_inf():
    try:
        _oracle_advance_inverse_newton(M, alpha, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_inf()",
            "gold_call": "run_oracle_inf()",
        },
        {
            "setup": """import numpy as np
M = np.array([[0.6, 0.1], [0.1, 1.3]])
alpha, p = 0.5, True
def run_model_bool():
    try:
        advance_inverse_newton(M, alpha, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_bool():
    try:
        _oracle_advance_inverse_newton(M, alpha, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_bool()",
            "gold_call": "run_oracle_bool()",
        },
    ]
