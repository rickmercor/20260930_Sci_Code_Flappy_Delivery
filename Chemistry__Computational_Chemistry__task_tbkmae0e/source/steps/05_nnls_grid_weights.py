"""
Implement nnls_grid_weights, which refits the quadrature weights of a grid as the non-negative least-squares solution that best reproduces the exact atomic-orbital overlap matrix.

Tabulated quadrature weights, such as those from atomic partitioning schemes built for density functional integration, are not tuned to integrate the particular orbital products of a given molecule and basis set. Treating the weights instead as unknowns and demanding that the grid reproduce an exact one-electron matrix turns the reweighting into a linear least-squares problem, constrained by what a quadrature weight is allowed to be.



The fit is to be solved exactly, not stopped once a convergence tolerance is met, since where the solution sits decides which points survive the reweighting. The weights of the points the fit excludes come back as exact zeros rather than as small numbers.

Returns
-------
np.ndarray of shape (n_grid,), float: the exact non-negative least-squares weights, zero at the points the fit excludes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nnls_grid_weights(fit_matrix: "np.ndarray", overlap: "np.ndarray") -> "np.ndarray":
    '''Non-negative least-squares quadrature weights reproducing the overlap matrix.

    Returns the weight vector that minimizes, over all vectors with no negative entry, the
    Euclidean norm of the residual between the fit matrix applied to it and the overlap
    matrix flattened row-major, matching the compound row index mu * n_ao + nu of the fit
    matrix. The fit is solved exactly, not stopped at a convergence threshold. The problems
    supplied by the tests have a unique minimizer.

    Parameters
    ----------
    fit_matrix : np.ndarray
        Array of shape (n_ao * n_ao, n_grid) whose entry (mu * n_ao + nu, P) is the product
        of the values of orbitals mu and nu at grid point P.
    overlap : np.ndarray
        Square array of shape (n_ao, n_ao) holding the exact overlap matrix.

    Returns
    -------
    weights : np.ndarray
        Array of shape (n_grid,) of dtype float holding the fitted weights. Every entry is
        greater than or equal to zero, and the weights of points excluded by the fit are
        exactly zero.

    Raises
    ------
    ValueError
        If fit_matrix is not a two-dimensional array, if overlap is not a square
        two-dimensional array, or if the number of rows of fit_matrix differs from the
        number of entries of overlap.
    '''
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import nnls


def _oracle_nnls_grid_weights(fit_matrix: "np.ndarray",
                              overlap: "np.ndarray") -> "np.ndarray":
    fit_matrix = np.asarray(fit_matrix, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    if fit_matrix.ndim != 2:
        raise ValueError("fit_matrix must be a two-dimensional array")
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("overlap must be a square two-dimensional array")
    if fit_matrix.shape[0] != overlap.size:
        raise ValueError("fit_matrix must have one row per entry of the overlap matrix")
    n_grid = fit_matrix.shape[1]
    weights, _ = nnls(fit_matrix, overlap.reshape(-1), maxiter=max(1000, 50 * n_grid))
    return weights

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    system_setup = """import numpy as np
def fit_system(ao_centers, ao_exponents, grid_points):
    ao_centers = np.asarray(ao_centers, dtype=float)
    ao_exponents = np.asarray(ao_exponents, dtype=float)
    grid_points = np.asarray(grid_points, dtype=float)
    norm = (2.0 * ao_exponents / np.pi) ** 0.75
    distance2 = np.sum((grid_points[None, :, :] - ao_centers[:, None, :]) ** 2, axis=-1)
    values = norm[:, None] * np.exp(-ao_exponents[:, None] * distance2)
    n_ao = ao_exponents.size
    fit = (values[:, None, :] * values[None, :, :]).reshape(n_ao * n_ao, -1)
    separation2 = np.sum((ao_centers[:, None, :] - ao_centers[None, :, :]) ** 2, axis=-1)
    total = ao_exponents[:, None] + ao_exponents[None, :]
    product = ao_exponents[:, None] * ao_exponents[None, :]
    overlap = (2.0 * np.sqrt(product) / total) ** 1.5 * np.exp(-product / total * separation2)
    return fit, overlap
"""
    return [
        # --- Typical: the constraint binds and drives two of five weights exactly to zero ---
        {
            "setup": system_setup + """
fit_matrix, overlap = fit_system(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
    [0.3, 1.2, 0.7],
    [[0.0, 0.0, 0.0], [0.6, 0.2, -0.1], [1.5, 0.0, 0.0], [0.8, -0.7, 0.4], [2.4, 0.9, 0.3]])
""",
            "call": "nnls_grid_weights(fit_matrix.copy(), overlap.copy())",
            "gold_call": "_oracle_nnls_grid_weights(fit_matrix.copy(), overlap.copy())",
            "tol": 1e-6,
        },
        # --- Boundary: the unconstrained solution is already positive, so no point is removed ---
        {
            "setup": system_setup + """
fit_matrix, overlap = fit_system(
    [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
    [0.5, 0.8],
    [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.7, 0.0, 0.0]])
""",
            "call": "nnls_grid_weights(fit_matrix.copy(), overlap.copy())",
            "gold_call": "_oracle_nnls_grid_weights(fit_matrix.copy(), overlap.copy())",
            "tol": 1e-6,
        },
        # --- Edge: a single grid point, a one-variable non-negative fit ---
        {
            "setup": system_setup + """
fit_matrix, overlap = fit_system(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    [0.6, 1.1],
    [[0.4, 0.1, 0.0]])
""",
            "call": "nnls_grid_weights(fit_matrix.copy(), overlap.copy())",
            "gold_call": "_oracle_nnls_grid_weights(fit_matrix.copy(), overlap.copy())",
            "tol": 1e-6,
        },
        # --- Representative: 20 even-tempered primitives on four centers, 100-point grid ---
        {
            "setup": system_setup + """
centers = np.array([[0.0, 0.0, 0.0], [1.9, 0.0, 0.2],
                    [-0.2, 2.0, 0.1], [0.9, 1.1, 1.75]], dtype=float)
exponents = 0.18 * 2.5 ** np.arange(5)
ao_centers = np.repeat(centers, 5, axis=0)
ao_exponents = np.tile(exponents, 4)
directions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                       [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
radii = np.array([0.4, 0.9, 1.7, 3.0])
offsets = np.vstack([np.zeros((1, 3)),
                     (radii[:, None, None] * directions[None, :, :]).reshape(-1, 3)])
grid_points = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
fit_matrix, overlap = fit_system(ao_centers, ao_exponents, grid_points)
""",
            "call": "nnls_grid_weights(fit_matrix.copy(), overlap.copy())",
            "gold_call": "_oracle_nnls_grid_weights(fit_matrix.copy(), overlap.copy())",
            "tol": 1e-6,
        },
        # --- Invalid: the fit matrix has the wrong number of rows for the overlap matrix ---
        {
            "setup": """import numpy as np
fit_matrix = np.ones((5, 3), dtype=float)
overlap = np.eye(2, dtype=float)
def run_model():
    try:
        nnls_grid_weights(fit_matrix.copy(), overlap.copy())
        return 0.0
    except ValueError:
        return 1.0
def run_oracle():
    try:
        _oracle_nnls_grid_weights(fit_matrix.copy(), overlap.copy())
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
