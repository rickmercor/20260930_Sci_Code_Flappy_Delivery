"""
Build a quadrature for integrals over the radial half line. Take the n_nodes Gauss-Legendre nodes and weights of the interval from minus one to one, map them onto the open unit interval by replacing each node t with (t+1)/2 and halving each weight, and send a mapped node u to the radius scale*u/(1-u). The quadrature weight belonging to that radius is the halved Gauss-Legendre weight times scale/(1-u)**2, so that the sum over nodes of weight times f(radius) approximates the integral of f from zero to infinity. Keep the nodes in the order numpy's Gauss-Legendre routine returns them, which puts the smallest radius first.

A Gauss-Legendre rule is exact for polynomials, so it has to be composed with a map that carries a finite interval onto the half line before it can integrate a decaying atomic density. The algebraic map used here concentrates nodes near the nucleus and still reaches the exponential tail, and its Jacobian is what turns the tabulated weights into radial weights.

Returns
-------
ndarray of shape (2, n_nodes): row 0 the radii in bohr, row 1 the quadrature weights in bohr.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radial_quadrature(n_nodes: int, scale: float) -> "np.ndarray":
    '''Build a quadrature for integrals over the radial half line. Take the n_nodes Gauss-Legendre nodes and weights of the interval from minus one to one, map them onto the open unit interval by replacing each node t with (t+1)/2 and halving each weight, and send a mapped node u to the radius scale*u/(1-u). The quadrature weight belonging to that radius is the halved Gauss-Legendre weight times scale/(1-u)**2, so that the sum over nodes of weight times f(radius) approximates the integral of f from zero to infinity. Keep the nodes in the order numpy's Gauss-Legendre routine returns them, which puts the smallest radius first.

    Parameters
    ----------
    n_nodes : int
        Number of Gauss-Legendre nodes, at least one.
    scale : float
        Positive length in bohr setting where the mapped nodes concentrate.

    Returns
    -------
    grid : np.ndarray
        ndarray of shape (2, n_nodes): row 0 the radii in bohr, row 1 the quadrature weights in bohr.

    Raises
    ------
    ValueError
        if n_nodes is smaller than one, or if scale is not positive.
    '''
    return grid  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _gl01(n):
    """Gauss-Legendre nodes and weights mapped from [-1, 1] onto (0, 1)."""
    t, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (t + 1.0), 0.5 * w


def _oracle_radial_quadrature(n_nodes: int, scale: float) -> "np.ndarray":
    """Gauss-Legendre radial quadrature on the half line."""
    if int(n_nodes) < 1:
        raise ValueError("n_nodes must be at least one")
    if float(scale) <= 0.0:
        raise ValueError("scale must be positive")
    u, w = _gl01(n_nodes)
    r = float(scale) * u / (1.0 - u)
    weights = w * float(scale) / (1.0 - u) ** 2
    return np.vstack((r, weights))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "radial_quadrature(24, 1.0)",
         "gold_call": "_oracle_radial_quadrature(24, 1.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "radial_quadrature(1, 1.0)",
         "gold_call": "_oracle_radial_quadrature(1, 1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "radial_quadrature(200, 3.5)",
         "gold_call": "_oracle_radial_quadrature(200, 3.5)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        radial_quadrature(0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_radial_quadrature(0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
