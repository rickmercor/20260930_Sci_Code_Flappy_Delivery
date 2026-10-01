"""
Implement thc_grid_metric, which forms the grid metric matrix of least-squares tensor hypercontraction from a weighted collocation matrix.

In that factorization the central matrix is fixed by minimizing the squared Frobenius error between the exact four-index integral tensor and its factorized reconstruction. Collecting the discrete pair densities of the grid into the columns of one matrix turns that minimization into a linear least-squares problem, and the grid metric is the matrix over grid points that appears in its normal equations. Its entry for points P and Q measures how strongly the pair densities carried by those two points align.



The metric is therefore symmetric and positive semidefinite, and its rank can be no larger than the number of independent symmetric orbital-pair densities the basis supports. When two retained points carry nearly linearly dependent pair densities the metric becomes numerically singular, which is why it is inverted through a truncated pseudoinverse in practice.

Returns
-------
np.ndarray of shape (n_grid, n_grid), float: the symmetric positive semidefinite grid metric of the least-squares fit
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def thc_grid_metric(collocation: "np.ndarray") -> "np.ndarray":
    '''Grid metric matrix of the least-squares tensor hypercontraction fit.

    Parameters
    ----------
    collocation : np.ndarray
        Array of shape (n_ao, n_grid) holding the weighted collocation matrix X. Must have
        at least one row and at least one column.

    Returns
    -------
    metric : np.ndarray
        Symmetric positive semidefinite array of shape (n_grid, n_grid) of dtype float.

    Raises
    ------
    ValueError
        If collocation is not a two-dimensional array with at least one row and at least
        one column.
    '''
    return metric

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_thc_grid_metric(collocation: "np.ndarray") -> "np.ndarray":
    collocation = np.asarray(collocation, dtype=float)
    if collocation.ndim != 2 or collocation.shape[0] < 1 or collocation.shape[1] < 1:
        raise ValueError("collocation must have shape (n_ao, n_grid) with both sizes >= 1")
    gram = collocation.T @ collocation
    return gram * gram

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Typical: three orbitals on four points ---
        {
            "setup": """import numpy as np
collocation = np.array([[0.51, 0.22, 0.03, 0.40],
                        [0.95, 0.10, 0.00, 0.31],
                        [0.05, 0.47, 0.88, 0.12]], dtype=float)
""",
            "call": "thc_grid_metric(collocation.copy())",
            "gold_call": "_oracle_thc_grid_metric(collocation.copy())",
        },
        # --- Boundary: a single orbital gives a rank-one metric of squared products ---
        {
            "setup": """import numpy as np
collocation = np.array([[0.7, -0.2, 1.3]], dtype=float)
""",
            "call": "thc_grid_metric(collocation.copy())",
            "gold_call": "_oracle_thc_grid_metric(collocation.copy())",
        },
        # --- Edge: a zero column (a zero-weight point) gives a zero row and column ---
        {
            "setup": """import numpy as np
collocation = np.array([[0.4, 0.0, 0.9],
                        [0.2, 0.0, 0.1],
                        [0.6, 0.0, 0.3],
                        [0.1, 0.0, 0.8]], dtype=float)
""",
            "call": "thc_grid_metric(collocation.copy())",
            "gold_call": "_oracle_thc_grid_metric(collocation.copy())",
        },
        # --- Property: more points than orbital-pair densities makes the metric singular ---
        {
            "setup": """import numpy as np
collocation = np.array([[1.0, 0.2, 0.5, 0.9, 0.3],
                        [0.1, 0.8, 0.4, 0.2, 0.7]], dtype=float)
def spectrum(fn):
    metric = np.asarray(fn(collocation.copy()), dtype=float)
    return np.concatenate([metric.reshape(-1), np.linalg.eigvalsh(metric)])
""",
            "call": "spectrum(thc_grid_metric)",
            "gold_call": "spectrum(_oracle_thc_grid_metric)",
        },
        # --- Invalid: a one-dimensional collocation array ---
        {
            "setup": """import numpy as np
collocation = np.array([0.1, 0.2, 0.3], dtype=float)
def run_model():
    try:
        thc_grid_metric(collocation.copy())
        return 0.0
    except ValueError:
        return 1.0
def run_oracle():
    try:
        _oracle_thc_grid_metric(collocation.copy())
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
