"""
Report a whole run of the accelerated method as one number: the Frobenius norm of the residual left after the requested number of steps, rounded once at the reporting stage to twelve digits after the decimal point.



Everything the number depends on has already been built. The run is driven from the scaled starting state, each step fits its own coefficient against the fixed sketch and advances the state, and the residual path records the result; this step composes those pieces and takes the last entry of the path. Rounding happens once, at the end, and never inside the loop, because rounding an iterate would change the state that the next fit sees.



The run is also certified rather than merely executed. The partner member of the coupled pair is advanced over the same coefficients and the tie `$M = X ** p A$` is checked against the state the loop ends on, so a run whose two members have drifted apart is rejected instead of reported. Both inputs are dimensionless, so the reported residual is dimensionless.



The reported residual is homogeneous of degree zero in `$A$` even though the partner used for certification is not. The orchestrator must therefore preserve the scale-safe starting constructions and the symmetric projection of an input accepted by tolerance; replacing them locally with a naive norm is not an equivalent replay of the earlier steps.

Returns
-------
float, the final Frobenius residual norm rounded to 12 digits after the decimal point
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_inverse_root_residual(
    A: np.ndarray,
    S: np.ndarray,
    p: int,
    lower: float,
    upper: float,
    iterations: int,
) -> float:
    """Return the rounded Frobenius residual left by a whole accelerated run.

    Raises ``ValueError`` unless every one of the following holds: ``A`` and
    ``S`` are real rather than complex; ``A`` is a nonempty square
    two-dimensional array that is finite, symmetric to an absolute tolerance of
    ``1e-12`` and projected to its nonzero symmetric part without a scale-unsafe
    norm; ``S`` is a finite two-dimensional
    array with at least one row and with its column count equal to the dimension
    of ``A``; ``lower`` and ``upper`` are finite with ``lower < upper``; ``p`` is
    an integer, not a bool, with ``p >= 1``; ``iterations`` is an integer, not a
    bool, with ``iterations >= 1``; and the run stays finite with its two coupled
    members still tied at the end.

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
    float
        The Frobenius residual norm after the final step, rounded to twelve
        digits after the decimal point.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_inverse_root_residual(
    A: np.ndarray,
    S: np.ndarray,
    p: int,
    lower: float,
    upper: float,
    iterations: int,
) -> float:
    """Reference end-to-end solver composed from the seven earlier oracles."""
    path = _oracle_residual_trajectory(A, S, p, lower, upper, iterations)  # noqa: F821
    selected = _oracle_fitted_coefficient_sequence(  # noqa: F821
        A, S, p, lower, upper, iterations
    )
    if selected.size + 1 != path.size:
        raise ValueError("coefficient and residual trajectories must agree in length")
    order = int(p)
    carried = _oracle_scaled_starting_matrix(A, order)  # noqa: F821
    identity = np.eye(carried.shape[0], dtype=float)
    for alpha in selected:
        residual = identity - carried
        replayed = _oracle_fit_bounded_coefficient(  # noqa: F821
            _oracle_sketched_loss_coefficients(residual, S, order),  # noqa: F821
            lower,
            upper,
        )
        if not np.isfinite(replayed):
            raise ValueError("replayed coefficient must be finite")
        carried = _oracle_advance_inverse_newton(carried, float(alpha), order)  # noqa: F821
    partner = _oracle_coupled_partner_factor(A, selected, order)  # noqa: F821
    tie = float(
        np.linalg.norm(
            carried
            - np.linalg.matrix_power(partner, order) @ np.asarray(A, dtype=float),
            ord="fro",
        )
    )
    span = max(1.0, float(np.linalg.norm(carried, ord="fro")))
    if not np.isfinite(tie) or tie > 1e-6 * span:
        raise ValueError("coupled members must stay tied through the run")
    value = float(np.linalg.norm(identity - carried, ord="fro"))
    if not np.isfinite(value):
        raise ValueError("final residual must be finite")
    return float(np.round(value, 12))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return ordinary, homogeneous extreme-scale, identity-sketch and invalid cases."""
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
            "call": "compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.75, -0.5], [-0.5, 2.25]])
S = np.array([[0.6, -1.1]])
p = 2
lower = 0.2
upper = 0.8
iterations = 3
""",
            "call": "compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
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
            "call": "compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
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
            "call": "compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
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
            "call": "compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.5, 0.25, 0.0], [0.25, 2.5, -0.5], [0.0, -0.5, 1.0]])
S = np.array([[0.9, -0.3, 0.6]])
p = 1
lower = 0.5
upper = 1.5
iterations = 4
""",
            "call": "compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5, -0.25], [0.5, 1.5, 0.125], [-0.25, 0.125, 3.0]])
S = np.eye(3)
p = 3
lower = 1.0 / 3.0
upper = 1.0
iterations = 4
""",
            "call": "compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
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
            "call": "compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_compute_inverse_root_residual(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5], [0.5, 1.5]])
S = np.array([[1.0, -0.5]])
p = 3
lower = 0.3
upper = 1.0
iterations = -1
def run_model_neg():
    try:
        compute_inverse_root_residual(A, S, p, lower, upper, iterations)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_neg():
    try:
        _oracle_compute_inverse_root_residual(A, S, p, lower, upper, iterations)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_neg()",
            "gold_call": "run_oracle_neg()",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5], [0.5, np.nan]])
S = np.array([[1.0, -0.5]])
p = 3
lower = 0.3
upper = 1.0
iterations = 2
def run_model_nan():
    try:
        compute_inverse_root_residual(A, S, p, lower, upper, iterations)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_nan():
    try:
        _oracle_compute_inverse_root_residual(A, S, p, lower, upper, iterations)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_nan()",
            "gold_call": "run_oracle_nan()",
        },
    ]
