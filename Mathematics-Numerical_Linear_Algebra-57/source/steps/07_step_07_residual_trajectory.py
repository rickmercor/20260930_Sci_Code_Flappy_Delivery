"""
The quantity the method is judged on is not the coefficient it picks but how fast the residual falls. This step reports that fall: the Frobenius norm of the residual before any step has been taken, and again after each step of a whole run.



The residual is read off the residual-bearing member of the coupled pair by the defining relation, so no partner and no matrix inverse is ever formed, and the norm is the ordinary Frobenius norm of that residual. Because the starting state contributes the first entry and every step contributes one more, the trajectory is one entry longer than the coefficient trajectory of the same run; its first entry depends on the input and the order alone, and never on the sketch, the interval or the number of steps.



A correct run is strictly decreasing on a well-scaled input, and the decrease accelerates: the published scaling places the starting residual inside a region where the accelerated step contracts, and each fitted coefficient is chosen to make the very next entry as small as the sketch can see. The last entry is what a whole-run report reduces to.



Because the starting member is homogeneous of degree zero in `$A$`, this entire residual path is unchanged by a positive scalar rescaling of `$A$`. The path must retain that invariant across the finite float64 range and must use the symmetric projection of any input admitted by the absolute symmetry tolerance.

Returns
-------
np.ndarray of shape (iterations + 1,), Frobenius residual norms before and after each step
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def residual_trajectory(
    A: np.ndarray,
    S: np.ndarray,
    p: int,
    lower: float,
    upper: float,
    iterations: int,
) -> np.ndarray:
    """Return the Frobenius residual norm before and after every step of a run.

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
        Float64 vector of shape ``(iterations + 1,)`` holding the Frobenius
        residual norm before any step and after each step.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_residual_trajectory(
    A: np.ndarray,
    S: np.ndarray,
    p: int,
    lower: float,
    upper: float,
    iterations: int,
) -> np.ndarray:
    """Reference residual path, replayed from the earlier oracles."""
    selected = _oracle_fitted_coefficient_sequence(  # noqa: F821
        A, S, p, lower, upper, iterations
    )
    carried = _oracle_scaled_starting_matrix(A, p)  # noqa: F821
    order = int(p)
    identity = np.eye(carried.shape[0], dtype=float)
    path = np.empty(selected.size + 1, dtype=float)
    path[0] = float(np.linalg.norm(identity - carried, ord="fro"))
    for index, alpha in enumerate(selected):
        carried = _oracle_advance_inverse_newton(carried, float(alpha), order)  # noqa: F821
        path[index + 1] = float(np.linalg.norm(identity - carried, ord="fro"))
    if not np.all(np.isfinite(path)):
        raise ValueError("residual trajectory must be finite")
    return path

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
            "call": "residual_trajectory(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_residual_trajectory(A, S, p, lower, upper, iterations)",
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
            "call": "residual_trajectory(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_residual_trajectory(A, S, p, lower, upper, iterations)",
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
            "call": "residual_trajectory(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_residual_trajectory(A, S, p, lower, upper, iterations)",
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
            "call": "residual_trajectory(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_residual_trajectory(A, S, p, lower, upper, iterations)",
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
            "call": "residual_trajectory(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_residual_trajectory(A, S, p, lower, upper, iterations)",
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
            "call": "residual_trajectory(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_residual_trajectory(A, S, p, lower, upper, iterations)",
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
            "call": "residual_trajectory(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_residual_trajectory(A, S, p, lower, upper, iterations)",
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
            "call": "residual_trajectory(A, S, p, lower, upper, iterations)",
            "gold_call": "_oracle_residual_trajectory(A, S, p, lower, upper, iterations)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5], [0.5, 1.5]])
S = np.array([[1.0, -0.5]])
p = 3
lower = 0.3
upper = 1.0
iterations = True
def run_model_bool():
    try:
        residual_trajectory(A, S, p, lower, upper, iterations)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_bool():
    try:
        _oracle_residual_trajectory(A, S, p, lower, upper, iterations)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_bool()",
            "gold_call": "run_oracle_bool()",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0], [0.0, 1.0]])
S = np.array([[1.0, -0.5]])
p = 3
lower = 0.3
upper = 1.0
iterations = 2
def run_model_asym():
    try:
        residual_trajectory(A, S, p, lower, upper, iterations)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_asym():
    try:
        _oracle_residual_trajectory(A, S, p, lower, upper, iterations)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_asym()",
            "gold_call": "run_oracle_asym()",
        },
    ]
