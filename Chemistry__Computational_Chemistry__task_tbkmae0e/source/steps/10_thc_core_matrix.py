"""
Implement thc_core_matrix, which solves for the central matrix of least-squares tensor hypercontraction on a given weighted grid.

The factorization reconstructs the four-index integral tensor from the collocation matrix and one central matrix over pairs of grid points. With the collocation matrix fixed, the central matrix is whatever minimizes the squared Frobenius norm of the error between the exact tensor and its reconstruction, summed over all four orbital indices. That is a linear least-squares problem in the central matrix, and the grid metric is the matrix appearing in its normal equations. When the metric is singular the normal equations no longer fix the central matrix uniquely, and the minimum-norm solution is taken.

The metric of a realistic grid is rank-deficient or nearly so, and inverting its smallest singular values would amplify rounding noise without bound. The pseudoinverse is therefore truncated: the metric is decomposed into singular values and vectors, every singular value smaller than rcond times the largest one is discarded, and only the retained singular triplets are inverted. The choice of rcond is part of the definition of Z, since it decides which directions of the grid-point space enter the fit.

Returns
-------
np.ndarray of shape (n_grid, n_grid), float: the central matrix of the least-squares fit
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def thc_core_matrix(collocation: "np.ndarray", grid_metric: "np.ndarray",
                    eri_tensor: "np.ndarray", rcond: float) -> "np.ndarray":
    '''Central matrix of the least-squares tensor hypercontraction fit.

    Returns the minimum-norm matrix over pairs of grid points that minimizes the squared
    Frobenius error between the supplied integral tensor and its reconstruction from the
    supplied collocation matrix. Wherever the grid metric has to be inverted, it is
    inverted through the pseudoinverse obtained from its singular value decomposition after
    discarding every singular value smaller than rcond times the largest singular value.

    Parameters
    ----------
    collocation : np.ndarray
        Array of shape (n_ao, n_grid) holding the weighted collocation matrix X.
    grid_metric : np.ndarray
        Array of shape (n_grid, n_grid) holding the grid metric M formed from the same
        collocation matrix.
    eri_tensor : np.ndarray
        Array of shape (n_ao, n_ao, n_ao, n_ao) holding the exact repulsion integrals
        (mu nu | lambda sigma) in chemists' notation.
    rcond : float
        Non-negative relative cutoff for the singular values of the grid metric.

    Returns
    -------
    core : np.ndarray
        Array of shape (n_grid, n_grid) of dtype float holding the central matrix.

    Raises
    ------
    ValueError
        If collocation is not a two-dimensional array with at least one row and at least
        one column, if grid_metric does not have shape (n_grid, n_grid), if eri_tensor does
        not have shape (n_ao, n_ao, n_ao, n_ao), or if rcond is negative or not finite.
    '''
    return core

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _truncated_pseudoinverse(matrix, rcond):
    """Moore-Penrose pseudoinverse keeping nonzero singular values >= rcond * largest."""
    left, singular, right = np.linalg.svd(matrix)
    keep = (singular >= rcond * singular[0]) & (singular > 0.0)
    return (right[keep].T / singular[keep]) @ left[:, keep].T


def _oracle_thc_core_matrix(collocation: "np.ndarray", grid_metric: "np.ndarray",
                            eri_tensor: "np.ndarray", rcond: float) -> "np.ndarray":
    collocation = np.asarray(collocation, dtype=float)
    grid_metric = np.asarray(grid_metric, dtype=float)
    eri_tensor = np.asarray(eri_tensor, dtype=float)
    if collocation.ndim != 2 or collocation.shape[0] < 1 or collocation.shape[1] < 1:
        raise ValueError("collocation must have shape (n_ao, n_grid) with both sizes >= 1")
    n_ao, n_grid = collocation.shape
    if grid_metric.shape != (n_grid, n_grid):
        raise ValueError("grid_metric must have shape (n_grid, n_grid)")
    if eri_tensor.shape != (n_ao, n_ao, n_ao, n_ao):
        raise ValueError("eri_tensor must have shape (n_ao, n_ao, n_ao, n_ao)")
    rcond = float(rcond)
    if not np.isfinite(rcond) or rcond < 0.0:
        raise ValueError("rcond must be finite and non-negative")

    pair_density = (collocation[:, None, :] * collocation[None, :, :]).reshape(n_ao * n_ao,
                                                                              n_grid)
    projected = pair_density.T @ eri_tensor.reshape(n_ao * n_ao, n_ao * n_ao) @ pair_density
    metric_inverse = _truncated_pseudoinverse(grid_metric, rcond)
    return metric_inverse @ projected @ metric_inverse

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    system_setup = """import numpy as np
from scipy.special import erf
def s_system(ao_centers, ao_exponents, grid_points, weights):
    c = np.asarray(ao_centers, dtype=float)
    e = np.asarray(ao_exponents, dtype=float)
    g = np.asarray(grid_points, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = e.size
    norm = (2.0 * e / np.pi) ** 0.75
    values = norm[:, None] * np.exp(-e[:, None] * np.sum((g[None] - c[:, None]) ** 2, -1))
    x = values * w[None, :] ** 0.25
    metric = (x.T @ x) ** 2
    p = e[:, None] + e[None, :]
    k = np.exp(-(e[:, None] * e[None, :]) / p * np.sum((c[:, None] - c[None]) ** 2, -1))
    mid = (e[:, None, None] * c[:, None] + e[None, :, None] * c[None]) / p[:, :, None]
    pf, kf, mf = p.reshape(-1), (k * norm[:, None] * norm[None, :]).reshape(-1), mid.reshape(-1, 3)
    t = (pf[:, None] * pf[None, :]) / (pf[:, None] + pf[None, :]) \\
        * np.sum((mf[:, None] - mf[None]) ** 2, -1)
    boys = np.where(t > 1e-12, 0.5 * np.sqrt(np.pi / np.maximum(t, 1e-300))
                    * erf(np.sqrt(t)), 1.0)
    eri = (2.0 * np.pi ** 2.5 / (pf[:, None] * pf[None, :] * np.sqrt(pf[:, None] + pf[None, :]))
           * kf[:, None] * kf[None, :] * boys).reshape(n, n, n, n)
    return x, metric, eri
"""
    return [
        # --- Typical: three primitives on four points, full-rank metric ---
        {
            "setup": system_setup + """
collocation, grid_metric, eri_tensor = s_system(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], [0.3, 1.2, 0.7],
    [[0.0, 0.0, 0.0], [0.6, 0.2, -0.1], [1.5, 0.0, 0.0], [2.4, 0.9, 0.3]],
    [1.13, 4.2, 10.3, 13.05])
rcond = 1e-12
""",
            "call": "thc_core_matrix(collocation.copy(), grid_metric.copy(), eri_tensor.copy(), rcond)",
            "gold_call": "_oracle_thc_core_matrix(collocation.copy(), grid_metric.copy(), eri_tensor.copy(), rcond)",
            "tol": 1e-7,
        },
        # --- Boundary: a tensor exactly representable on the grid is reconstructed exactly ---
        {
            "setup": """import numpy as np
collocation = np.array([[0.9, 0.2, 0.4],
                        [0.1, 0.7, 0.3],
                        [0.3, 0.2, 0.8]], dtype=float)
grid_metric = (collocation.T @ collocation) ** 2
target_core = np.array([[1.0, 0.3, 0.1],
                        [0.3, 0.8, 0.2],
                        [0.1, 0.2, 0.6]], dtype=float)
pairs = (collocation[:, None, :] * collocation[None, :, :]).reshape(9, 3)
eri_tensor = (pairs @ target_core @ pairs.T).reshape(3, 3, 3, 3)
rcond = 1e-12
def reconstruction(fn):
    core = np.asarray(fn(collocation.copy(), grid_metric.copy(), eri_tensor.copy(), rcond),
                      dtype=float)
    return np.concatenate([core.reshape(-1), (pairs @ core @ pairs.T).reshape(-1)])
""",
            "call": "reconstruction(thc_core_matrix)",
            "gold_call": "reconstruction(_oracle_thc_core_matrix)",
            "tol": 1e-7,
        },
        # --- Edge: more points than pair densities, so the truncation removes null directions ---
        {
            "setup": system_setup + """
collocation, grid_metric, eri_tensor = s_system(
    [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], [0.5, 0.9],
    [[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [1.2, 0.0, 0.0], [0.3, 0.5, 0.0], [0.9, -0.4, 0.2]],
    [1.0, 2.0, 1.5, 0.7, 0.9])
rcond = 1e-12
""",
            "call": "thc_core_matrix(collocation.copy(), grid_metric.copy(), eri_tensor.copy(), rcond)",
            "gold_call": "_oracle_thc_core_matrix(collocation.copy(), grid_metric.copy(), eri_tensor.copy(), rcond)",
            "tol": 1e-7,
        },
        # --- Edge: a cutoff inside the retained spectrum changes Z, so rcond must be honored ---
        {
            "setup": system_setup + """
collocation, grid_metric, eri_tensor = s_system(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], [0.3, 1.2, 0.7],
    [[0.0, 0.0, 0.0], [0.6, 0.2, -0.1], [1.5, 0.0, 0.0], [2.4, 0.9, 0.3]],
    [1.13, 4.2, 10.3, 13.05])
rcond = 1e-2
""",
            "call": "thc_core_matrix(collocation.copy(), grid_metric.copy(), eri_tensor.copy(), rcond)",
            "gold_call": "_oracle_thc_core_matrix(collocation.copy(), grid_metric.copy(), eri_tensor.copy(), rcond)",
            "tol": 1e-7,
        },
        # --- Invalid: a negative cutoff ---
        {
            "setup": """import numpy as np
collocation = np.eye(2, dtype=float)
grid_metric = (collocation.T @ collocation) ** 2
eri_tensor = np.ones((2, 2, 2, 2), dtype=float)
rcond = -1e-12
def run_model():
    try:
        thc_core_matrix(collocation.copy(), grid_metric.copy(), eri_tensor.copy(), rcond)
        return 0.0
    except ValueError:
        return 1.0
def run_oracle():
    try:
        _oracle_thc_core_matrix(collocation.copy(), grid_metric.copy(), eri_tensor.copy(), rcond)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
