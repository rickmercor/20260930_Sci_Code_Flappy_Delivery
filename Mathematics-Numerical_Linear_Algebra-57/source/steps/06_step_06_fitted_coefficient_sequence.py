"""
A whole run of the accelerated method is a single loop in which the state is the residual-bearing member of the coupled pair. Each pass reads the residual off the current state, contracts the post-step residual against the fixed sketch to get a scalar loss, minimises that loss over the closed coefficient interval, and advances the state by the accelerated step that the fitted coefficient defines.



This step reports the coefficient the fit selects at each pass, in order. The sketch is fixed once for the whole run rather than redrawn per iteration, so the run is deterministic and every coefficient after the first depends on all the coefficients before it; the trajectory is not a set of independent fits. Whether a given pass settles on an endpoint or on an interior stationary point is a property of the run, not something imposed in advance, and both outcomes occur on a typical trajectory.



The returned trajectory has one entry per step, so its length is the number of iterations requested and the starting state contributes no entry.



The trajectory depends on the direction and spectrum of `$A$` but not on its positive scalar magnitude: the scale is removed by the starting construction. That homogeneity and the symmetric projection used for tolerance-accepted inputs must survive composition of the earlier steps, including for finite matrices whose naive Frobenius norm is zero or infinite.

Returns
-------
np.ndarray of shape (iterations,), the fitted coefficient of each step in order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fitted_coefficient_sequence(
    A: np.ndarray,
    S: np.ndarray,
    p: int,
    lower: float,
    upper: float,
    iterations: int,
) -> np.ndarray:
    """Return the fitted coefficient selected at each step of a whole run.

    Raises ``ValueError`` unless every one of the following holds: ``A`` and
    ``S`` are real rather than complex; ``A`` is a nonempty square
    two-dimensional array that is finite, symmetric to an absolute tolerance of
    ``1e-12`` and projected to its nonzero symmetric part without a scale-unsafe
    norm; ``S`` is a finite two-dimensional
    array with at least one row and with its column count equal to the dimension
    of ``A``; ``lower`` and ``upper`` are finite with ``lower < upper``; ``p`` is
    an integer, not a bool, with ``p >= 1``; and ``iterations`` is an integer, not
    a bool, with ``iterations >= 1``.

    Parameters
    ----------
    A : np.ndarray
        Nonzero finite real symmetric matrix of shape ``(n, n)``.
    S : np.ndarray
        Fixed finite real sketch of shape ``(m, n)`` with ``m >= 1``.
    p : int
        Root order.
    lower : float
        Lower endpoint of the closed coefficient interval.
    upper : float
        Upper endpoint of the closed coefficient interval.
    iterations : int
        Number of accelerated steps to run.

    Returns
    -------
    np.ndarray
        Float64 vector of shape ``(iterations,)`` holding the fitted coefficient
        of each step in order.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_fitted_coefficient_sequence(
    A: np.ndarray,
    S: np.ndarray,
    p: int,
    lower: float,
    upper: float,
    iterations: int,
) -> np.ndarray:
    """Reference coefficient trajectory composed from the four earlier oracles."""
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)):
        raise ValueError("iterations must be an integer")
    if int(iterations) < 1:
        raise ValueError("iterations must be at least one")
    steps = int(iterations)
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("bounds must be finite and satisfy lower < upper")
    sketch = np.asarray(S)
    if np.iscomplexobj(sketch):
        raise ValueError("S must be real")
    sketch = np.asarray(sketch, dtype=float)
    if sketch.ndim != 2 or sketch.shape[0] == 0:
        raise ValueError("S must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(sketch)):
        raise ValueError("S must contain only finite values")
    carried = _oracle_scaled_starting_matrix(A, p)  # noqa: F821
    if sketch.shape[1] != carried.shape[0]:
        raise ValueError("S width must equal the dimension of A")
    order = int(p)
    identity = np.eye(carried.shape[0], dtype=float)
    selected = np.empty(steps, dtype=float)
    for index in range(steps):
        residual = identity - carried
        coefficients = _oracle_sketched_loss_coefficients(  # noqa: F821
            residual, sketch, order
        )
        alpha = _oracle_fit_bounded_coefficient(coefficients, lower, upper)  # noqa: F821
        selected[index] = alpha
        carried = _oracle_advance_inverse_newton(carried, alpha, order)  # noqa: F821
    return selected

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return ordinary, homogeneous extreme-scale, narrow-window and invalid cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5, -0.25], [0.5, 1.5, 0.125], [-0.25, 0.125, 3.0]])
S = np.array([[1.0, -0.5, 0.25], [0.2, 0.7, -0.4]])
p = 3
lower = 1.0 / 3.0
upper = 1.0
iterations = 4
""",
            "call": "fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.75, -0.5], [-0.5, 2.25]])
S = np.array([[0.6, -1.1]])
p = 2
lower = 0.2
upper = 0.8
iterations = 2
""",
            "call": "fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
v = np.array([[1.0], [-2.0], [0.5]])
A = v @ v.T + 2.0 * np.eye(3)
S = np.array([[0.5, -1.25, 0.75], [1.0, 0.75, -0.2], [-0.6, 0.2, 1.4]])
p = 5
lower = 0.1
upper = 0.4
iterations = 5
""",
            "call": "fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.25], [0.25, 1.25]])
S = np.array([[1.0, 0.0], [0.0, 1.0]])
p = 3
lower = 0.3
upper = 0.35
iterations = 1
""",
            "call": "fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5, -0.25], [0.5, 1.5, 0.125], [-0.25, 0.125, 3.0]])
S = np.array([[1.0, -0.5, 0.25], [0.2, 0.7, -0.4]])
p = 3
lower = 0.8999999999
upper = 0.9
iterations = 4
""",
            "call": "fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5, -0.25], [0.5, 1.5, 0.125], [-0.25, 0.125, 3.0]])
S = np.array([[1.0, -0.5, 0.25], [0.2, 0.7, -0.4]])
p = 3
lower = -2.0
upper = 5.0
iterations = 3
""",
            "call": "fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.5, 0.25, 0.0], [0.25, 2.5, -0.5], [0.0, -0.5, 1.0]])
S = np.array([[0.9, -0.3, 0.6]])
p = 1
lower = 0.5
upper = 1.5
iterations = 3
""",
            "call": "fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = 1e300 * np.array([[2.0, 0.5, -0.25], [0.5, 1.5, 0.125], [-0.25, 0.125, 3.0]])
S = np.array([[1.0, -0.5, 0.25], [0.2, 0.7, -0.4]])
p = 3
lower = 1.0 / 3.0
upper = 1.0
iterations = 2
""",
            "call": "fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2e-12, 9e-13], [1e-13, 1e-12]])
S = np.array([[0.6, -1.1]])
p = 3
lower = 1.0 / 3.0
upper = 1.0
iterations = 2
""",
            "call": "fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_fitted_coefficient_sequence(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5], [0.5, 1.5]])
S = np.array([[1.0, -0.5]])
p = 3
lower = 0.3
upper = 1.0
iterations = 0
def run_model_zero():
    try:
        fitted_coefficient_sequence(A, S, p, lower, upper, iterations)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_zero():
    try:
        _oracle_fitted_coefficient_sequence(A, S, p, lower, upper, iterations)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_zero()",
            "gold_call": "run_oracle_zero()",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5], [0.5, 1.5]])
S = np.array([[1.0, -0.5, 0.25]])
p = 3
lower = 0.3
upper = 1.0
iterations = 2
def run_model_width():
    try:
        fitted_coefficient_sequence(A, S, p, lower, upper, iterations)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_width():
    try:
        _oracle_fitted_coefficient_sequence(A, S, p, lower, upper, iterations)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_width()",
            "gold_call": "run_oracle_width()",
        },
    ]
