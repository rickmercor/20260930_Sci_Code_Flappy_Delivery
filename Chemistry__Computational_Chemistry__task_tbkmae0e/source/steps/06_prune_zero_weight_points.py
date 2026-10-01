"""
Implement prune_zero_weight_points, which removes from a refitted quadrature grid every point whose non-negative weight is exactly zero, keeping the surviving points and weights in order.

A point whose refitted weight is zero contributes nothing to any quadrature on the grid, and in a grid-based separable factorization of the two-electron integrals it contributes nothing to the reconstructed tensor either, because every collocation factor attached to it vanishes. Such points can therefore be discarded without changing any result, and discarding them matters: the storage of the factorization and the cost of every contraction that uses it grow with the number of grid points, so the size of the retained grid sets the cost of everything downstream.



The pruning criterion is exact equality with zero rather than a magnitude threshold: a point carrying a weight that is small but positive is retained, because removing it would change the fitted quadrature. The retained points keep their original relative order, so the pruned grid is an order-preserving subsequence of the input grid.

Returns
-------
tuple (points, kept_weights) of float arrays with shapes (n_kept, 3) and (n_kept,): the points whose weight is strictly positive, in their original order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def prune_zero_weight_points(grid_points: "np.ndarray",
                             weights: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    '''Discard grid points whose non-negative quadrature weight is exactly zero.

    Parameters
    ----------
    grid_points : np.ndarray
        Array of shape (n_grid, 3) holding the grid point coordinates.
    weights : np.ndarray
        Array of shape (n_grid,) holding the non-negative weight of each point.

    Returns
    -------
    retained : tuple[np.ndarray, np.ndarray]
        A tuple (points, kept_weights) in which points has shape (n_kept, 3) and
        kept_weights has shape (n_kept,), both of dtype float, holding exactly the input
        points and weights whose weight is strictly greater than zero, in their original
        order. A weight that is positive but arbitrarily small is kept. If every weight is
        zero, both arrays have zero rows.

    Raises
    ------
    ValueError
        If grid_points is not a two-dimensional array with three columns, if weights is not
        a one-dimensional array with one entry per grid point, or if any weight is negative.
    '''
    return retained

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_prune_zero_weight_points(grid_points: "np.ndarray",
                                     weights: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    points = np.asarray(grid_points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("grid_points must have shape (n_grid, 3)")
    if weights.ndim != 1 or weights.shape[0] != points.shape[0]:
        raise ValueError("weights must have shape (n_grid,) matching grid_points")
    if np.any(weights < 0.0):
        raise ValueError("quadrature weights must be non-negative")
    keep = weights > 0.0
    return points[keep].copy(), weights[keep].copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    invalid_setup = """import numpy as np
def run_model():
    try:
        prune_zero_weight_points(grid_points.copy(), weights.copy())
        return 0.0
    except ValueError:
        return 1.0
def run_oracle():
    try:
        _oracle_prune_zero_weight_points(grid_points.copy(), weights.copy())
        return 0.0
    except ValueError:
        return 1.0
"""
    pack_setup = """import numpy as np
def packed(fn, grid_points, weights):
    result = fn(grid_points.copy(), weights.copy())
    if not isinstance(result, tuple) or len(result) != 2:
        raise AssertionError("expected a tuple (points, kept_weights)")
    points = np.asarray(result[0], dtype=float)
    kept = np.asarray(result[1], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or kept.shape != (points.shape[0],):
        raise AssertionError("expected shapes (n_kept, 3) and (n_kept,)")
    header = np.array([points.shape[0], kept.shape[0]], dtype=float)
    return np.concatenate([header, points.reshape(-1), kept])
"""
    return [
        # --- Typical: two of six weights vanish; survivors keep their order ---
        {
            "setup": pack_setup + """
grid_points = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [-0.4, 0.0, 0.0],
                        [0.0, 0.9, 0.0], [0.0, -0.9, 0.0], [0.0, 0.0, 1.7]], dtype=float)
weights = np.array([1.25, 0.0, 3.5, 0.0, 0.75, 12.0], dtype=float)
""",
            "call": "packed(prune_zero_weight_points, grid_points, weights)",
            "gold_call": "packed(_oracle_prune_zero_weight_points, grid_points, weights)",
        },
        # --- Boundary: no weight vanishes, so the grid is returned unchanged ---
        {
            "setup": pack_setup + """
grid_points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.0, 0.0, -4.0]], dtype=float)
weights = np.array([0.2, 5.0, 1.1], dtype=float)
""",
            "call": "packed(prune_zero_weight_points, grid_points, weights)",
            "gold_call": "packed(_oracle_prune_zero_weight_points, grid_points, weights)",
        },
        # --- Edge: a tiny positive weight is kept, because the criterion is exact zero ---
        {
            "setup": pack_setup + """
grid_points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=float)
weights = np.array([0.0, 1e-300, 2.0], dtype=float)
""",
            "call": "packed(prune_zero_weight_points, grid_points, weights)",
            "gold_call": "packed(_oracle_prune_zero_weight_points, grid_points, weights)",
        },
        # --- Edge: every weight vanishes, leaving arrays with zero rows ---
        {
            "setup": pack_setup + """
grid_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
weights = np.zeros(2, dtype=float)
""",
            "call": "packed(prune_zero_weight_points, grid_points, weights)",
            "gold_call": "packed(_oracle_prune_zero_weight_points, grid_points, weights)",
        },
        # --- Invalid: a negative weight ---
        {
            "setup": invalid_setup + """
grid_points = np.zeros((3, 3), dtype=float)
weights = np.array([1.0, -0.5, 2.0], dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: one weight too few for the grid ---
        {
            "setup": invalid_setup + """
grid_points = np.zeros((3, 3), dtype=float)
weights = np.array([1.0, 2.0], dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
