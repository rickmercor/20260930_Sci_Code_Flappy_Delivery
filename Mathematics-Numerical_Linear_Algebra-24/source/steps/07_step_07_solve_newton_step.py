"""
Solve the accepted Newton system and preserve the descent convention.



Once Cholesky certifies `$H_acc$` as positive definite, the step solves

`$H_acc @ delta_x = -g$`. The sign is essential because the final squared

decrement is ``-g.T @ delta_x`` and a valid step satisfies ``g.T @ delta_x < 0``.

Returns
-------
finite np.ndarray of shape (n,), solving accepted_hessian @ step = -residual
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_newton_step(
    accepted_hessian: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    """Compute the Newton step after a Cholesky acceptance check.

    Raises ``ValueError`` unless every one of the following holds:
    ``accepted_hessian`` is a square two-dimensional array; ``residual`` is
    one-dimensional with length equal to its side; every entry of both is
    finite; ``accepted_hessian`` equals its own transpose to within ``1e-12``
    absolute; and ``accepted_hessian`` is positive definite. Positive
    definiteness is a checked requirement rather than a caller guarantee, so
    an argument whose Cholesky factorization fails is rejected rather than
    solved.

    Parameters
    ----------
    accepted_hessian : np.ndarray
        Finite square matrix of shape ``(n, n)``. Symmetry and positive
        definiteness are checked requirements, not caller guarantees.
    residual : np.ndarray
        Finite assembled residual of shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Newton step of shape ``(n,)``.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solve_newton_step(accepted_hessian, residual):
    """Reference solve for the accepted positive-definite system."""
    
    hessian = np.asarray(accepted_hessian, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("accepted_hessian must be square")
    if residual.shape != (hessian.shape[0],):
        raise ValueError("residual has incompatible shape")
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(residual)):
        raise ValueError("inputs must be finite")
    if not np.allclose(hessian, hessian.T, rtol=0.0, atol=1e-12):
        raise ValueError("accepted_hessian must be symmetric")
    try:
        np.linalg.cholesky(hessian)
    except np.linalg.LinAlgError as exc:
        raise ValueError("accepted_hessian must be positive definite") from exc
    return np.linalg.solve(hessian, -residual)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative SPD, scalar, and indefinite cases."""
    return [
        {
            "setup": """
import numpy as np
H = np.array([[3.0, 0.2], [0.2, 2.0]])
g = np.array([4.0, 1.0])
""",
            "call": "solve_newton_step(H, g)",
            "gold_call": "_oracle_solve_newton_step(H, g)",
        },
        {
            "setup": """
import numpy as np
H = np.array([[2.0]])
g = np.array([4.0])
""",
            "call": "solve_newton_step(H, g)",
            "gold_call": "_oracle_solve_newton_step(H, g)",
        },
        {
            "setup": """
import numpy as np
H = np.diag([1.0, -1.0])
g = np.ones(2)

def run_model():
    try:
        solve_newton_step(H, g)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_solve_newton_step(H, g)
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
