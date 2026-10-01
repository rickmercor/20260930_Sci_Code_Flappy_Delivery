"""
Implement overlap_fit_matrix, which builds the linear map from grid quadrature weights to the quadrature estimate of the atomic-orbital overlap matrix, with the orbital-pair index flattened into a single row index.

A weighted real-space grid estimates a one-electron integral by a weighted sum of integrand values over the grid points. For a fixed set of point positions that estimate is linear in the weights, so the estimate of the whole overlap matrix can be written as a single matrix acting on the weight vector, with one column per grid point and one row per ordered orbital pair.



Flattening the ordered pair (mu, nu) row-major into the compound row index mu * n_ao + nu turns the requirement that the grid reproduce the exact overlap matrix into an ordinary linear system whose right-hand side is that matrix flattened in the same order. The map depends only on the point positions, so it is built once and reused for any trial set of weights.

Returns
-------
np.ndarray of shape (n_ao * n_ao, n_grid), float: the linear map taking grid weights to the quadrature estimate of the overlap matrix, flattened row-major
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def overlap_fit_matrix(ao_centers: "np.ndarray", ao_exponents: "np.ndarray",
                       grid_points: "np.ndarray") -> "np.ndarray":
    '''Linear map from grid weights to the quadrature estimate of the overlap matrix.

    The basis functions are normalized s-type primitive Gaussians, one per supplied
    (center, exponent) pair.

    Parameters
    ----------
    ao_centers : np.ndarray
        Array of shape (n_ao, 3) holding the center of each primitive, in bohr.
        Must contain at least one row.
    ao_exponents : np.ndarray
        Array of shape (n_ao,) holding the Gaussian exponent of each primitive.
        Every exponent must be strictly positive.
    grid_points : np.ndarray
        Array of shape (n_grid, 3) holding the grid point coordinates in bohr.
        Must contain at least one row.

    Returns
    -------
    fit_matrix : np.ndarray
        Array of shape (n_ao * n_ao, n_grid) of dtype float. Multiplying it by a weight
        vector of shape (n_grid,) gives that grid's quadrature estimate of the overlap
        matrix, flattened row-major, so row mu * n_ao + nu carries the ordered pair
        (mu, nu) and column P carries grid point P.

    Raises
    ------
    ValueError
        If ao_centers is not a two-dimensional array with three columns and at least one
        row, if ao_exponents is not a one-dimensional array of matching length, if any
        exponent is not strictly positive, or if grid_points is not a two-dimensional
        array with three columns and at least one row.
    '''
    return fit_matrix

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validated_basis(ao_centers, ao_exponents):
    """Return the basis as float arrays, rejecting malformed or unphysical input."""
    centers = np.asarray(ao_centers, dtype=float)
    exponents = np.asarray(ao_exponents, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
        raise ValueError("ao_centers must have shape (n_ao, 3) with n_ao >= 1")
    if exponents.ndim != 1 or exponents.shape[0] != centers.shape[0]:
        raise ValueError("ao_exponents must have shape (n_ao,) matching ao_centers")
    if not np.all(exponents > 0.0):
        raise ValueError("every Gaussian exponent must be strictly positive")
    return centers, exponents


def _evaluate_basis(ao_centers, ao_exponents, grid_points):
    """Values of the normalized s primitives at the grid points, shape (n_ao, n_grid)."""
    centers, exponents = _validated_basis(ao_centers, ao_exponents)
    points = np.asarray(grid_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError("grid_points must have shape (n_grid, 3) with n_grid >= 1")
    norm = (2.0 * exponents / np.pi) ** 0.75
    distance2 = np.sum((points[None, :, :] - centers[:, None, :]) ** 2, axis=-1)
    return norm[:, None] * np.exp(-exponents[:, None] * distance2)


def _oracle_overlap_fit_matrix(ao_centers: "np.ndarray", ao_exponents: "np.ndarray",
                               grid_points: "np.ndarray") -> "np.ndarray":
    values = _evaluate_basis(ao_centers, ao_exponents, grid_points)
    n_ao, n_grid = values.shape
    fit_matrix = (values[:, None, :] * values[None, :, :]).reshape(n_ao * n_ao, n_grid)
    return fit_matrix

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Typical: three primitives on two centers, five scattered grid points ---
        {
            "setup": """import numpy as np
ao_centers = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [1.5, 0.0, 0.0]], dtype=float)
ao_exponents = np.array([0.3, 1.2, 0.7], dtype=float)
grid_points = np.array([[0.0, 0.0, 0.0],
                        [0.5, 0.2, -0.1],
                        [1.5, 0.0, 0.0],
                        [0.7, -0.6, 0.4],
                        [2.5, 1.0, 0.3]], dtype=float)
""",
            "call": "overlap_fit_matrix(ao_centers.copy(), ao_exponents.copy(), grid_points.copy())",
            "gold_call": "_oracle_overlap_fit_matrix(ao_centers.copy(), ao_exponents.copy(), grid_points.copy())",
        },
        # --- Boundary: points at the orbital centers, where each primitive takes its peak ---
        {
            "setup": """import numpy as np
ao_centers = np.array([[0.0, 0.0, 0.0],
                       [2.0, 0.0, 0.0]], dtype=float)
ao_exponents = np.array([0.45, 1.125], dtype=float)
grid_points = ao_centers.copy()
""",
            "call": "overlap_fit_matrix(ao_centers.copy(), ao_exponents.copy(), grid_points.copy())",
            "gold_call": "_oracle_overlap_fit_matrix(ao_centers.copy(), ao_exponents.copy(), grid_points.copy())",
        },
        # --- Edge: one primitive and one point, giving a one-by-one map ---
        {
            "setup": """import numpy as np
ao_centers = np.array([[0.3, -0.2, 0.1]], dtype=float)
ao_exponents = np.array([2.8125], dtype=float)
grid_points = np.array([[0.0, 0.0, 0.0]], dtype=float)
""",
            "call": "overlap_fit_matrix(ao_centers.copy(), ao_exponents.copy(), grid_points.copy())",
            "gold_call": "_oracle_overlap_fit_matrix(ao_centers.copy(), ao_exponents.copy(), grid_points.copy())",
        },
        # --- Edge: row ordering, via the flattened quadrature estimate for given weights ---
        {
            "setup": """import numpy as np
ao_centers = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.2, 0.0]], dtype=float)
ao_exponents = np.array([0.5, 0.9, 1.6], dtype=float)
grid_points = np.array([[0.1, 0.1, 0.0],
                        [0.9, -0.1, 0.2],
                        [-0.2, 1.1, 0.1],
                        [0.5, 0.5, 0.5]], dtype=float)
weights = np.array([0.8, 1.3, 0.6, 2.1], dtype=float)
def estimate(fn):
    fit = np.asarray(fn(ao_centers.copy(), ao_exponents.copy(), grid_points.copy()), dtype=float)
    return (fit @ weights).reshape(3, 3)
""",
            "call": "estimate(overlap_fit_matrix)",
            "gold_call": "estimate(_oracle_overlap_fit_matrix)",
        },
        # --- Invalid: grid points given as a flat array ---
        {
            "setup": """import numpy as np
ao_centers = np.zeros((1, 3), dtype=float)
ao_exponents = np.array([1.0], dtype=float)
grid_points = np.array([0.0, 0.0, 0.0], dtype=float)
def run_model():
    try:
        overlap_fit_matrix(ao_centers.copy(), ao_exponents.copy(), grid_points.copy())
        return 0.0
    except ValueError:
        return 1.0
def run_oracle():
    try:
        _oracle_overlap_fit_matrix(ao_centers.copy(), ao_exponents.copy(), grid_points.copy())
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
