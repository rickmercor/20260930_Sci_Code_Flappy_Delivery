"""
Turn an aggregation into a transfer operator by applying one damped Jacobi smoothing step to the piecewise-constant tentative prolongator.

An aggregation defines a tentative prolongator whose columns are the indicator vectors of the aggregates: every unknown receives its coarse representative's value unchanged. That operator is a partition of unity, so it reproduces the constant vector exactly, but it is discontinuous across aggregate boundaries and its range contains large amounts of high-frequency content. Interpolating with it directly produces a coarse correction the post-smoother cannot clean up, and the resulting multigrid cycle converges slowly or not at all on operators whose coefficients vary strongly from one region to another.

Smoothed aggregation repairs this by applying a single damped Jacobi sweep to the tentative prolongator, replacing it with the tentative operator minus a damping factor times the diagonally scaled action of the level operator on it. Each column is thereby relaxed against the operator itself, which spreads the sharp indicator profile smoothly across the aggregate boundary and, crucially, drives the column towards the operator's own near-null space instead of towards a geometrically assumed one. Because the sweep is applied to the transfer operator rather than to a residual, the improvement is paid for once during setup and enjoyed at every cycle. The damping factor is what keeps the sweep stable: the diagonally scaled operator has eigenvalues reaching towards two, so undamped relaxation would amplify precisely the modes it is supposed to attenuate, and the factor in standard use is the one that minimises the spectral radius of the error propagation of Jacobi relaxation over the upper part of the spectrum.

The prolongator produced here is the interpolation operator of the level. Its transpose is used as the restriction operator, which makes the coarse operator formed in the next step a Galerkin projection and therefore keeps it symmetric whenever the fine operator is symmetric. Preserving that symmetry is not cosmetic: it is what allows the multigrid V-cycle to be a symmetric positive-definite approximate inverse, and hence what makes the spectral-equivalence constant of the later steps well defined.

Returns
-------
np.ndarray of shape (n, n_coarse), float: the smoothed-aggregation prolongator, one column per aggregate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_smoothed_prolongator(matrix: np.ndarray, aggregates: np.ndarray,
                               omega: float) -> np.ndarray:
    """Build the smoothed-aggregation prolongator of a level.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric level operator of shape (n, n) with a nonzero diagonal.
    aggregates : np.ndarray
        Integer array of shape (n,) holding consecutive aggregate indices
        starting at zero.
    omega : float
        Damping factor of the prolongator smoothing sweep, 0 < omega <= 1.

    Returns
    -------
    prolongator : np.ndarray
        Transfer operator of shape (n, n_coarse), where n_coarse is the number
        of aggregates.
    """
    return prolongator  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_smoothed_prolongator(matrix: np.ndarray, aggregates: np.ndarray,
                                       omega: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(omega, (int, float)) and np.isfinite(omega)
            and 0.0 < float(omega) <= 1.0):
        raise ValueError("omega must be a finite number in the half-open interval (0, 1]")
    level = np.asarray(matrix, dtype=float)
    if level.ndim != 2 or level.shape[0] != level.shape[1] or level.shape[0] < 1:
        raise ValueError("matrix must be a square 2D array of order n >= 1")
    if not np.all(np.isfinite(level)):
        raise ValueError("matrix must be finite")
    labels = np.asarray(aggregates).ravel()
    if labels.size != level.shape[0]:
        raise ValueError("aggregates must have one entry per unknown of matrix")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("aggregates must be an integer array")
    if labels.min() != 0 or not np.array_equal(np.unique(labels),
                                               np.arange(labels.max() + 1)):
        raise ValueError("aggregates must hold consecutive indices starting at zero")

    diagonal = np.diag(level).copy()
    if np.any(diagonal == 0.0):
        raise ValueError("matrix must have a nonzero diagonal")

    n_dof = level.shape[0]
    n_coarse = int(labels.max()) + 1

    # Piecewise-constant tentative prolongator: one indicator column per aggregate.
    tentative = np.zeros((n_dof, n_coarse), dtype=float)
    tentative[np.arange(n_dof), labels] = 1.0

    # One damped Jacobi sweep applied to every column of the tentative operator.
    prolongator = tentative - float(omega) * (level @ tentative) / diagonal[:, None]

    return prolongator

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: tridiagonal level operator with paired aggregates (normal scenario) ---
        {
            "setup": """import numpy as np
n = 12
matrix = np.diag(np.full(n, 2.0)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
aggregates = np.arange(n) // 2
omega = 2.0 / 3.0
""",
            "call": "build_smoothed_prolongator(matrix, aggregates, omega)",
            "gold_call": "_oracle_build_smoothed_prolongator(matrix, aggregates, omega)",
        },
        # --- Valid: strongly varying coefficients and uneven aggregates ---
        {
            "setup": """import numpy as np
n = 9
w = 10.0 ** np.linspace(-2.0, 2.0, n - 1)
matrix = np.zeros((n, n))
for k in range(n - 1):
    matrix[k, k] += w[k]
    matrix[k + 1, k + 1] += w[k]
    matrix[k, k + 1] -= w[k]
    matrix[k + 1, k] -= w[k]
matrix += np.eye(n) * 0.5
aggregates = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3])
omega = 2.0 / 3.0
""",
            "call": "build_smoothed_prolongator(matrix, aggregates, omega)",
            "gold_call": "_oracle_build_smoothed_prolongator(matrix, aggregates, omega)",
        },
        # --- Boundary: undamped sweep, omega at the top of its range ---
        {
            "setup": """import numpy as np
n = 8
matrix = np.diag(np.full(n, 4.0)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
aggregates = np.arange(n) // 4
omega = 1.0
""",
            "call": "build_smoothed_prolongator(matrix, aggregates, omega)",
            "gold_call": "_oracle_build_smoothed_prolongator(matrix, aggregates, omega)",
        },
        # --- Edge: every unknown its own aggregate, so no coarsening occurs ---
        {
            "setup": """import numpy as np
n = 5
matrix = np.diag(np.linspace(1.0, 3.0, n)) - np.diag(0.3 * np.ones(n - 1), 1) - np.diag(0.3 * np.ones(n - 1), -1)
aggregates = np.arange(n)
omega = 2.0 / 3.0
""",
            "call": "build_smoothed_prolongator(matrix, aggregates, omega)",
            "gold_call": "_oracle_build_smoothed_prolongator(matrix, aggregates, omega)",
        },
        # --- Invalid: aggregate labels are not consecutive from zero ---
        {
            "setup": """import numpy as np
matrix = np.eye(4) * 2.0
aggregates = np.array([0, 2, 2, 3])
def run_model():
    try:
        build_smoothed_prolongator(matrix, aggregates, 2.0 / 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_smoothed_prolongator(matrix, aggregates, 2.0 / 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: zero diagonal entry makes the Jacobi scaling undefined ---
        {
            "setup": """import numpy as np
matrix = np.diag([2.0, 0.0, 2.0])
aggregates = np.array([0, 0, 1])
def run_model():
    try:
        build_smoothed_prolongator(matrix, aggregates, 2.0 / 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_smoothed_prolongator(matrix, aggregates, 2.0 / 3.0)
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
