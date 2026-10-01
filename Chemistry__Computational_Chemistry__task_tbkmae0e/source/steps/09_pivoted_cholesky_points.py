"""
Implement pivoted_cholesky_points, which prunes a grid by an incomplete pivoted Cholesky decomposition of its least-squares tensor hypercontraction metric and returns the indices of the selected points in the order they become pivots.

The grid metric collects the alignment between the discrete orbital-pair densities carried by the grid points, so a point whose pair densities are already spanned by other points contributes nothing new to any reconstructed repulsion integral. An incomplete pivoted Cholesky decomposition of that metric exposes this directly: each pivot is the point that adds most beyond the pivots chosen so far, and the decomposition is stopped early once the next pivot adds too little. The pivots retained at that point define the pruned grid.



How little is too little is set by a relative cutoff, in the convention of the work that introduced this pruning for tensor hypercontraction grids. The selected points keep whatever weights they carried in the input grid, since this procedure chooses points but does not refit weights.

Returns
-------
one-dimensional integer np.ndarray: the indices of the selected grid points, in pivot order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pivoted_cholesky_points(grid_metric: "np.ndarray", cutoff: float) -> "np.ndarray":
    '''Grid points selected by an incomplete pivoted Cholesky decomposition of the metric.

    The decomposition follows the standard greedy pivoted Cholesky rule, taking the
    not-yet-selected point with the largest remaining Schur-complement diagonal as the next
    pivot and breaking exact ties by the lowest index. It stops before the first pivot that
    falls below the relative cutoff, in the convention of the work that introduced this
    pruning, and also before any pivot whose remaining diagonal is not positive.

    Parameters
    ----------
    grid_metric : np.ndarray
        Symmetric positive semidefinite array of shape (n_grid, n_grid) with n_grid >= 1.
    cutoff : float
        Strictly positive relative pruning threshold, interpreted in the convention of the
        work that introduced this pruning.

    Returns
    -------
    pivots : np.ndarray
        One-dimensional integer array holding the indices of the selected grid points in
        the order in which they became pivots.

    Raises
    ------
    ValueError
        If grid_metric is not a square two-dimensional array with at least one row, or if
        cutoff is not finite and strictly positive.
    '''
    return pivots

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pivoted_cholesky_points(grid_metric: "np.ndarray", cutoff: float) -> "np.ndarray":
    metric = np.asarray(grid_metric, dtype=float)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1] or metric.shape[0] < 1:
        raise ValueError("grid_metric must be a square two-dimensional array")
    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and strictly positive")

    n_grid = metric.shape[0]
    residual = np.diag(metric).copy()
    threshold = cutoff * np.sqrt(max(residual.max(), 0.0))
    factor = np.zeros((n_grid, n_grid), dtype=float)
    selected = np.zeros(n_grid, dtype=bool)
    pivots = []
    for step in range(n_grid):
        candidates = np.where(selected, -np.inf, residual)
        pivot = int(np.argmax(candidates))
        diagonal = candidates[pivot]
        if not diagonal > 0.0 or np.sqrt(diagonal) < threshold:
            break
        column = (metric[:, pivot] - factor[:, :step] @ factor[pivot, :step]) / np.sqrt(diagonal)
        factor[:, step] = column
        residual = residual - column ** 2
        selected[pivot] = True
        pivots.append(pivot)
    return np.array(pivots, dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    metric_setup = """import numpy as np
def unit_weight_metric(ao_centers, ao_exponents, grid_points):
    c = np.asarray(ao_centers, dtype=float)
    e = np.asarray(ao_exponents, dtype=float)
    g = np.asarray(grid_points, dtype=float)
    norm = (2.0 * e / np.pi) ** 0.75
    values = norm[:, None] * np.exp(-e[:, None] * np.sum((g[None] - c[:, None]) ** 2, -1))
    return (values.T @ values) ** 2
"""
    small = metric_setup + """
grid_metric = unit_weight_metric(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.2, 0.0]], [0.3, 1.2, 0.7],
    [[0.0, 0.0, 0.0], [0.6, 0.2, -0.1], [1.5, 0.2, 0.0], [0.8, -0.7, 0.4],
     [2.4, 0.9, 0.3], [-0.9, 0.4, 0.6], [1.1, 1.3, -0.5], [0.3, -1.4, 0.2]])
"""
    invalid = """
def run_model():
    try:
        pivoted_cholesky_points(grid_metric.copy(), cutoff)
        return 0.0
    except ValueError:
        return 1.0
def run_oracle():
    try:
        _oracle_pivoted_cholesky_points(grid_metric.copy(), cutoff)
        return 0.0
    except ValueError:
        return 1.0
"""
    return [
        # --- Typical: eight asymmetric points, five survive a 1e-2 relative cutoff ---
        {
            "setup": small + """
cutoff = 1e-2
""",
            "call": "pivoted_cholesky_points(grid_metric.copy(), cutoff)",
            "gold_call": "_oracle_pivoted_cholesky_points(grid_metric.copy(), cutoff)",
        },
        # --- Boundary: cutoff one accepts only the first pivot, whose R_kk equals R_11 ---
        {
            "setup": small + """
cutoff = 1.0
""",
            "call": "pivoted_cholesky_points(grid_metric.copy(), cutoff)",
            "gold_call": "_oracle_pivoted_cholesky_points(grid_metric.copy(), cutoff)",
        },
        # --- Edge: a rank-three metric on five points stops at its numerical rank ---
        {
            "setup": metric_setup + """
grid_metric = unit_weight_metric(
    [[0.0, 0.0, 0.0], [1.2, 0.3, 0.0]], [0.5, 0.9],
    [[0.0, 0.0, 0.0], [0.6, 0.1, 0.0], [1.2, 0.3, 0.0], [0.3, 0.5, 0.2], [0.9, -0.4, 0.2]])
cutoff = 1e-4
""",
            "call": "pivoted_cholesky_points(grid_metric.copy(), cutoff)",
            "gold_call": "_oracle_pivoted_cholesky_points(grid_metric.copy(), cutoff)",
        },
        # --- Representative: unit-weight metric of a 100-point four-center grid ---
        {
            "setup": metric_setup + """
centers = np.array([[0.0, 0.0, 0.0], [1.9, 0.0, 0.2],
                    [-0.2, 2.0, 0.1], [0.9, 1.1, 1.75]], dtype=float)
exponents = 0.18 * 2.5 ** np.arange(5)
directions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                       [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
radii = np.array([0.4, 0.9, 1.7, 3.0])
offsets = np.vstack([np.zeros((1, 3)),
                     (radii[:, None, None] * directions[None, :, :]).reshape(-1, 3)])
grid_points = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
grid_metric = unit_weight_metric(np.repeat(centers, 5, axis=0), np.tile(exponents, 4),
                                 grid_points)
cutoff = 3.8e-4
""",
            "call": "pivoted_cholesky_points(grid_metric.copy(), cutoff)",
            "gold_call": "_oracle_pivoted_cholesky_points(grid_metric.copy(), cutoff)",
        },
        # --- Invalid: a cutoff of zero ---
        {
            "setup": small + """
cutoff = 0.0
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: a non-square metric ---
        {
            "setup": """import numpy as np
grid_metric = np.ones((3, 4), dtype=float)
cutoff = 1e-3
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
