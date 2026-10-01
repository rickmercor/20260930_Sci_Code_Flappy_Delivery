"""
An ALS micro-step minimizes a regularized linear least-squares objective while every other tensor-train core is fixed, turning one sweep of the tensor-train fit into an ordinary ridge regression. With design matrix A, target y, and positive regularization tau, the local coefficient vector satisfies (A^T A + tau I)c = A^T y. C-order reshaping restores the three-mode core used by the tensor train.

Inputs
------
design_matrix: Float array of shape (K, p).
targets: Float array of shape (K,).
tau: Positive ridge magnitude.
core_shape: Three positive integers with product p.

Returns
-------
updated_core: Float array with shape core_shape.

Returns
-------
np.ndarray with shape core_shape, the regularized local TT core as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_ridge_core(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    tau: float,
    core_shape: tuple[int, int, int],
) -> np.ndarray:
    """Solve one regularized ALS core update.

    Parameters
    ----------
    design_matrix : np.ndarray
        Local regression matrix with shape (K, p).
    targets : np.ndarray
        Regression targets with shape (K,).
    tau : float
        Finite positive ridge magnitude.
    core_shape : tuple[int, int, int]
        Positive core dimensions whose product is p.

    Raises
    ------
    ValueError
        If matrix and target shapes disagree, an input is non-finite, `tau` is
        not positive, or `core_shape` does not contain three positive integers
        with product equal to the number of matrix columns.

    Returns
    -------
    updated_core : np.ndarray
        Updated tensor-train core with shape `core_shape`.
    """
    return updated_core  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_solve_ridge_core(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    tau: float,
    core_shape: tuple[int, int, int],
) -> np.ndarray:
    """Reference implementation."""
    matrix = np.asarray(design_matrix, dtype=float)
    y = np.asarray(targets, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("design_matrix must be non-empty and two dimensional")
    if y.shape != (matrix.shape[0],):
        raise ValueError("targets must have one entry per matrix row")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(y)):
        raise ValueError("design_matrix and targets must be finite")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    if not isinstance(core_shape, (tuple, list)) or len(core_shape) != 3:
        raise ValueError("core_shape must contain three dimensions")
    if any(not isinstance(size, (int, np.integer)) or int(size) <= 0 for size in core_shape):
        raise ValueError("core_shape dimensions must be positive integers")
    shape = tuple(int(size) for size in core_shape)
    if int(np.prod(shape)) != matrix.shape[1]:
        raise ValueError("core_shape product must match the matrix column count")

    normal_matrix = matrix.T @ matrix + float(tau) * np.eye(matrix.shape[1])
    coefficients = np.linalg.solve(normal_matrix, matrix.T @ y)
    return coefficients.reshape(shape)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[1.0, 0.2, -0.3, 0.4], [0.1, 0.8, 0.5, -0.2], [0.7, -0.4, 0.6, 0.3]])
y = np.array([0.5, -0.1, 0.8])
""",
            "call": "np.round(solve_ridge_core(A, y, 0.07, (1, 2, 2)), 12).tolist()",
            "gold_call": "np.round(_oracle_solve_ridge_core(A, y, 0.07, (1, 2, 2)), 12).tolist()",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0]])
y = np.array([1.0])
""",
            "call": "solve_ridge_core(A, y, 0.5, (1, 1, 1)).tolist()",
            "gold_call": "_oracle_solve_ridge_core(A, y, 0.5, (1, 1, 1)).tolist()",
        },
        {
            "setup": """import numpy as np
A = np.ones((2, 3))
y = np.ones(2)
def run_model():
    try:
        solve_ridge_core(A, y, 0.0, (1, 3, 1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_solve_ridge_core(A, y, 0.0, (1, 3, 1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
