"""
Given a resolution N, return the N + 1 Chebyshev-Lobatto nodes mapped onto the unit interval in increasing order, starting at zero and ending at one, together with the matrix that differentiates nodal values with respect to that coordinate, and the largest absolute row sum of that matrix applied to a constant, which certifies the row-sum construction.

A spectral collocation method represents a function by its values at a fixed set of nodes and differentiates it by interpolating those values with the unique polynomial of matching degree and differentiating the interpolant exactly. On the Chebyshev-Lobatto nodes, which are the extrema of a Chebyshev polynomial together with the two endpoints, the interpolant is free of the Runge oscillations that ruin equispaced interpolation, and the error decays faster than any power of the node count for a function analytic on the interval. That is the property the present problem needs, because the mode functions of the compactified total transmission problem are analytic in the interior and the whole benefit of the compactification would be lost to an algebraically convergent discretisation.

The standard construction places the nodes at the cosines of equally spaced angles on the interval from minus one to one and builds the differentiation matrix from the Lagrange cardinal polynomials in closed form, with the off-diagonal entries given by a ratio of endpoint weights over the node separation and the diagonal entries obtained from the requirement that the matrix annihilate a constant. Enforcing that requirement by subtracting the row sums, rather than by evaluating the closed form for the diagonal, matters in floating point: the closed-form diagonal loses relative accuracy through cancellation at the large entries near the endpoints, and the resulting matrix fails to differentiate a constant to machine precision, which contaminates every subsequent operator built from it.

The problem here lives on the unit interval in the compactified coordinate, so the nodes and the matrix must be mapped. Taking the coordinate to increase from zero at infinity to one at the horizon reverses the orientation of the standard grid, and the chain rule then supplies a factor of minus two on the differentiation matrix. The orientation is a convention and not a physical choice: reversing it exchanges the left and the right families of total transmission modes, which are mirror images of one another in the real frequency axis, and leaves every condition number unchanged.

Returns
-------
dict holding the array nodes, the N + 1 coordinates in increasing order; the array derivative, the (N + 1) by (N + 1) differentiation matrix in that coordinate; and the float constant_residual, the largest absolute entry of the matrix applied to a vector of ones.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def chebyshev_lobatto_operators(resolution: int) -> dict:
    """Build the Chebyshev-Lobatto nodes and differentiation matrix on the unit interval.

    Parameters
    ----------
    resolution : int
        Grid resolution N; the grid carries N + 1 nodes.

    Returns
    -------
    dict
        Under the keys nodes, derivative and constant_residual.

    Raises
    ------
    ValueError
        When the resolution is not an integer of at least four.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validate_resolution(resolution):
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)):
        raise ValueError("resolution must be an integer")
    if int(resolution) < 4:
        raise ValueError("resolution must be at least four")
    return int(resolution)


def _oracle_chebyshev_lobatto_operators(resolution: int) -> dict:
    """Reference implementation."""
    n = _validate_resolution(resolution)
    index = np.arange(n + 1)
    chebyshev = np.cos(np.pi * index / n)
    weights = np.where((index == 0) | (index == n), 2.0, 1.0) * (-1.0) ** index
    separation = chebyshev[:, None] - chebyshev[None, :]
    matrix = np.outer(weights, 1.0 / weights) / (separation + np.eye(n + 1))
    matrix = matrix - np.diag(matrix.sum(axis=1))
    nodes = (1.0 - chebyshev) / 2.0
    derivative = -2.0 * matrix
    residual = float(np.max(np.abs(derivative @ np.ones(n + 1))))
    return {"nodes": nodes, "derivative": derivative, "constant_residual": residual}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
def flat(x):
    if isinstance(x, dict):
        return flat([x[k] for k in sorted(x)])
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""

    SETUP = """
import numpy as np
def digest(out):
    return (np.round(out["nodes"], 12), np.round(out["derivative"], 9))
"""
    return [
        {
            # a normal resolution: nodes and matrix entries compared directly
            "setup": SETUP + FLAT,
            "call": "flat(digest(chebyshev_lobatto_operators(12)))",
            "gold_call": "flat(digest(_oracle_chebyshev_lobatto_operators(12)))",
        },
        {
            # spectral accuracy: the matrix must differentiate a polynomial of degree below the
            # resolution exactly, must annihilate a constant to machine precision, and must place the
            # nodes in increasing order from zero to one
            "setup": SETUP + """
def accuracy(fn):
    out = fn(24)
    nodes, D = out["nodes"], out["derivative"]
    poly = nodes ** 5 - 3.0 * nodes ** 2 + 1.0
    exact = 5.0 * nodes ** 4 - 6.0 * nodes
    ordered = int(np.all(np.diff(nodes) > 0) and abs(nodes[0]) < 1e-15 and abs(nodes[-1] - 1.0) < 1e-15)
    return (ordered,
            int(np.max(np.abs(D @ poly - exact)) < 1e-11),
            int(out["constant_residual"] < 1e-11),
            int(D.shape == (25, 25)))
""" + FLAT,
            "call": "flat(accuracy(chebyshev_lobatto_operators))",
            "gold_call": "flat(accuracy(_oracle_chebyshev_lobatto_operators))",
        },
        {
            # boundary: the smallest admissible resolution
            "setup": SETUP + FLAT,
            "call": "flat(digest(chebyshev_lobatto_operators(4)))",
            "gold_call": "flat(digest(_oracle_chebyshev_lobatto_operators(4)))",
        },
        {
            "setup": SETUP + """
def verdict(fn, value):
    try:
        fn(value)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(chebyshev_lobatto_operators, 3), "
                    "verdict(chebyshev_lobatto_operators, 12.5), "
                    "verdict(chebyshev_lobatto_operators, -8)))",
            "gold_call": "flat((verdict(_oracle_chebyshev_lobatto_operators, 3), "
                         "verdict(_oracle_chebyshev_lobatto_operators, 12.5), "
                         "verdict(_oracle_chebyshev_lobatto_operators, -8)))",
        },
    ]
