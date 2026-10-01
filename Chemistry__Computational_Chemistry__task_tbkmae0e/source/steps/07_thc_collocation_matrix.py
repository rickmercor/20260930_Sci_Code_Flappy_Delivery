"""
Implement thc_collocation_matrix, which forms the weighted collocation matrix that carries the atomic orbitals in grid-based least-squares tensor hypercontraction.

That factorization writes the four-index repulsion integral tensor as a chain of collocation matrices and one central matrix, so every orbital index is carried by its own factor and the product of two collocation entries at a grid point stands for a generalized charge density there. The quadrature weight of a point is not carried by the central matrix; it is distributed over the collocation factors, each of which absorbs the same fixed fractional power of that weight, in the convention established for this factorization.



A point with zero weight therefore produces an identically zero column, which is why such points can be removed before the factorization is formed, and rescaling the weights rescales the columns without changing the space they span.

Returns
-------
np.ndarray of shape (n_ao, n_grid), float: the weighted collocation matrix of the factorization
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def thc_collocation_matrix(ao_centers: "np.ndarray", ao_exponents: "np.ndarray",
                           grid_points: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    '''Weighted collocation matrix of least-squares tensor hypercontraction.

    The basis functions are normalized s-type primitive Gaussians, one per supplied
    (center, exponent) pair. Each entry is the value of one orbital at one grid point,
    scaled by that point's quadrature weight raised to the fixed fractional power this
    factorization uses.

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
    weights : np.ndarray
        Array of shape (n_grid,) holding the non-negative quadrature weight of each point.

    Returns
    -------
    collocation : np.ndarray
        Array of shape (n_ao, n_grid) of dtype float whose entry (mu, P) belongs to
        orbital mu at grid point P.

    Raises
    ------
    ValueError
        If ao_centers is not a two-dimensional array with three columns and at least one
        row, if ao_exponents is not a one-dimensional array of matching length, if any
        exponent is not strictly positive, if grid_points is not a two-dimensional array
        with three columns and at least one row, if weights is not a one-dimensional array
        with one entry per grid point, or if any weight is negative.
    '''
    return collocation

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


def _oracle_thc_collocation_matrix(ao_centers: "np.ndarray", ao_exponents: "np.ndarray",
                                   grid_points: "np.ndarray",
                                   weights: "np.ndarray") -> "np.ndarray":
    values = _evaluate_basis(ao_centers, ao_exponents, grid_points)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != values.shape[1]:
        raise ValueError("weights must have shape (n_grid,) matching grid_points")
    if np.any(weights < 0.0):
        raise ValueError("quadrature weights must be non-negative")
    return values * weights[None, :] ** 0.25

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    basis_setup = """import numpy as np
ao_centers = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [1.5, 0.0, 0.0]], dtype=float)
ao_exponents = np.array([0.3, 1.2, 0.7], dtype=float)
grid_points = np.array([[0.0, 0.0, 0.0],
                        [0.6, 0.2, -0.1],
                        [1.5, 0.0, 0.0],
                        [2.4, 0.9, 0.3]], dtype=float)
"""
    call = ("thc_collocation_matrix(ao_centers.copy(), ao_exponents.copy(), "
            "grid_points.copy(), weights.copy())")
    gold_call = ("_oracle_thc_collocation_matrix(ao_centers.copy(), ao_exponents.copy(), "
                 "grid_points.copy(), weights.copy())")
    return [
        # --- Typical: distinct positive weights on four points ---
        {
            "setup": basis_setup + """
weights = np.array([1.13, 4.2, 10.3, 13.05], dtype=float)
""",
            "call": call,
            "gold_call": gold_call,
        },
        # --- Boundary: unit weights leave the plain orbital values unchanged ---
        {
            "setup": basis_setup + """
weights = np.ones(4, dtype=float)
""",
            "call": call,
            "gold_call": gold_call,
        },
        # --- Edge: a zero weight gives an identically zero column ---
        {
            "setup": basis_setup + """
weights = np.array([2.0, 0.0, 16.0, 0.5], dtype=float)
""",
            "call": call,
            "gold_call": gold_call,
        },
        # --- Property: two factors times the square-root weight give the quadrature overlap ---
        {
            "setup": basis_setup + """
weights = np.array([0.8, 2.5, 6.0, 9.5], dtype=float)
def quadrature_overlap(fn):
    x = np.asarray(fn(ao_centers.copy(), ao_exponents.copy(), grid_points.copy(),
                      weights.copy()), dtype=float)
    return (x * np.sqrt(weights)[None, :]) @ x.T
""",
            "call": "quadrature_overlap(thc_collocation_matrix)",
            "gold_call": "quadrature_overlap(_oracle_thc_collocation_matrix)",
        },
        # --- Invalid: a negative weight has no real fourth root ---
        {
            "setup": basis_setup + """
weights = np.array([1.0, -2.0, 1.0, 1.0], dtype=float)
def run_model():
    try:
        """ + call + """
        return 0.0
    except ValueError:
        return 1.0
def run_oracle():
    try:
        """ + gold_call + """
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
