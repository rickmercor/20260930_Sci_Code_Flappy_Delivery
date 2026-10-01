"""
Evaluate the generalized interpolation material point basis and its spatial

gradient for every material point and grid node pair, returning the weight array

and the gradient array that carry every particle to grid transfer in the solver.

Standard material point shape functions are the finite element hat functions

evaluated at a point mass, and their gradient jumps as a point crosses a cell

boundary, which destroys the convergence of an implicit solver. The generalized

interpolation family removes that jump by giving each material point a domain of

finite length l_p instead of treating it as a point mass, and by convolving the

grid basis with the characteristic function of that domain. The resulting

one-dimensional weight is a piecewise expression in the relative coordinate

delta = X_p - X_v between the material point and the node, whose branches are

selected by comparing delta against the breakpoints built from the grid cell size

h and the particle domain length l_p; the weight is quadratic near the edges of

the support and near delta = 0, is the linear hat function in between, vanishes

once |delta| reaches the support half-width, and reduces to the hat function in

the limit l_p going to zero. The two-dimensional basis is the product of the

one-dimensional factors, S_vp = S_x S_y, so its gradient components are

(dS_x/dx) S_y and S_x (dS_y/dy), each obtained by differentiating the same

piecewise expression. Because the basis is a partition of unity, the weights of

any material point sum to one over the nodes that support it.

Returns
-------
tuple (shape_values, shape_gradients) of np.ndarray with shapes (n_points, n_nodes) and (n_points, n_nodes, 2), both float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def gimp_shape_functions(points: np.ndarray, nodes: np.ndarray, cell_size: float,
                         point_domain_length: float):
    """Evaluate the generalized interpolation basis and its gradient.

    Parameters
    ----------
    points : np.ndarray
        Material point coordinates, shape (n_points, 2).
    nodes : np.ndarray
        Grid node coordinates, shape (n_nodes, 2).
    cell_size : float
        Uniform grid cell size h.
    point_domain_length : float
        Particle domain length l_p, the same in both directions.

    Returns
    -------
    basis : tuple of np.ndarray
        The pair (shape_values, shape_gradients), holding the basis weights of
        shape (n_points, n_nodes) and the basis gradients of shape
        (n_points, n_nodes, 2).

    Raises
    ------
    ValueError
        If `points` or `nodes` is not of shape (n, 2), if `cell_size` is not
        positive, or if `point_domain_length` lies outside (0, cell_size].
    """
    return basis

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _gimp_factor(delta, cell_size, point_domain_length):
    """One-dimensional generalized interpolation weight and its derivative."""
    h = cell_size
    lp = point_domain_length
    reach = h + 0.5 * lp
    value = np.zeros_like(delta)
    slope = np.zeros_like(delta)
    left_edge = (delta > -reach) & (delta <= -h + 0.5 * lp)
    left_hat = (delta > -h + 0.5 * lp) & (delta <= -0.5 * lp)
    centre = (delta > -0.5 * lp) & (delta <= 0.5 * lp)
    right_hat = (delta > 0.5 * lp) & (delta <= h - 0.5 * lp)
    right_edge = (delta > h - 0.5 * lp) & (delta <= reach)
    value[left_edge] = (reach + delta[left_edge]) ** 2 / (2.0 * h * lp)
    slope[left_edge] = (reach + delta[left_edge]) / (h * lp)
    value[left_hat] = 1.0 + delta[left_hat] / h
    slope[left_hat] = 1.0 / h
    value[centre] = 1.0 - (delta[centre] ** 2 + 0.25 * lp ** 2) / (h * lp)
    slope[centre] = -2.0 * delta[centre] / (h * lp)
    value[right_hat] = 1.0 - delta[right_hat] / h
    slope[right_hat] = -1.0 / h
    value[right_edge] = (reach - delta[right_edge]) ** 2 / (2.0 * h * lp)
    slope[right_edge] = -(reach - delta[right_edge]) / (h * lp)
    return value, slope


def _oracle_gimp_shape_functions(points: np.ndarray, nodes: np.ndarray, cell_size: float,
                                 point_domain_length: float):
    """Reference implementation."""
    points = np.asarray(points, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    h = float(cell_size)
    lp = float(point_domain_length)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must have shape (n_nodes, 2)")
    if h <= 0.0:
        raise ValueError("cell_size must be > 0")
    if not 0.0 < lp <= h:
        raise ValueError("point_domain_length must lie in (0, cell_size]")
    value_x, slope_x = _gimp_factor(points[:, None, 0] - nodes[None, :, 0], h, lp)
    value_y, slope_y = _gimp_factor(points[:, None, 1] - nodes[None, :, 1], h, lp)
    shape_values = value_x * value_y
    shape_gradients = np.stack([slope_x * value_y, value_x * slope_y], axis=-1)
    return shape_values, shape_gradients

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

_NODES = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                   [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                   [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: points inside a cell and straddling a node ---
        {
            "setup": """import numpy as np
points = np.array([[0.1, 0.1], [0.5, 0.5], [0.7, 0.3]])
nodes = _NODES.copy()
""",
            "call": "[np.round(gimp_shape_functions(points, nodes, 0.5, 0.2)[0], 10).tolist(),"
                    " np.round(gimp_shape_functions(points, nodes, 0.5, 0.2)[1], 10).tolist()]",
            "gold_call": "[np.round(_oracle_gimp_shape_functions(points, nodes, 0.5, 0.2)[0], 10).tolist(),"
                         " np.round(_oracle_gimp_shape_functions(points, nodes, 0.5, 0.2)[1], 10).tolist()]",
        },
        # --- Boundary case: a point exactly at the edge of the support, and
        # --- the widest admissible particle domain
        {
            "setup": """import numpy as np
points = np.array([[0.6, 0.5], [0.25, 0.25]])
nodes = _NODES.copy()
""",
            "call": "[np.round(gimp_shape_functions(points, nodes, 0.5, 0.5)[0], 10).tolist(),"
                    " np.round(gimp_shape_functions(points, nodes, 0.5, 0.5)[1], 10).tolist()]",
            "gold_call": "[np.round(_oracle_gimp_shape_functions(points, nodes, 0.5, 0.5)[0], 10).tolist(),"
                         " np.round(_oracle_gimp_shape_functions(points, nodes, 0.5, 0.5)[1], 10).tolist()]",
        },
        # --- Edge case: a particle domain wider than a cell must raise ---
        {
            "setup": """import numpy as np
points = np.array([[0.1, 0.1]])
nodes = _NODES.copy()
def run_model():
    try:
        gimp_shape_functions(points, nodes, 0.5, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_gimp_shape_functions(points, nodes, 0.5, 0.8)
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
