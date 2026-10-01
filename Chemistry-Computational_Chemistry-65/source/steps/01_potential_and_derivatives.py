"""
Evaluate the model potential and its first and second derivatives at a set of points. The potential is V(x, y) = (1/8) (x^2 - 1)^2 + (wy^2 / 2) [y + c (x^2 - 1)]^2 + a x^3, where the square bracket is squared as a whole. SPECIFICATION: points is an array of shape (n, 2) holding the x and y coordinate of each point. The result is a real array of shape (n, 6) whose six columns are, in this order, the potential, its first derivative with respect to x, its first derivative with respect to y, its second derivative with respect to x twice, its mixed second derivative, and its second derivative with respect to y twice.

A two-dimensional double well whose barrier is displaced sideways from the straight line between the minima is the standard testbed for tunnelling that cannot be reduced to one coordinate. The quartic term in x supplies two wells, the quadratic term ties the transverse coordinate to a curved channel, and the cubic term tips one well below the other. Everything downstream reads its potential, gradient and curvature through this one routine, so the packing order of the six columns is worth getting right once.

Returns
-------
numpy.ndarray of shape (n, 6) and real dtype.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def potential_and_derivatives(points: "np.ndarray", a: float, c: float, wy: float) -> "np.ndarray":
    """Evaluate the model potential and its first and second derivatives at a set of points. The potential is V(x, y) = (1/8) (x^2 - 1)^2 + (wy^2 / 2) [y + c (x^2 - 1)]^2 + a x^3, where the square bracket is squared as a whole. SPECIFICATION: points is an array of shape (n, 2) holding the x and y coordinate of each point. The result is a real array of shape (n, 6) whose six columns are, in this order, the potential, its first derivative with respect to x, its first derivative with respect to y, its second derivative with respect to x twice, its mixed second derivative, and its second derivative with respect to y twice.

    Returns
    -------
    numpy.ndarray of shape (n, 6) and real dtype.

    Raises
    ------
    ValueError: if points does not have shape (n, 2), or if any of a, c or wy is not finite.
    """
    return derivatives  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _oracle_potential_and_derivatives(points: "np.ndarray", a: float, c: float, wy: float) -> "np.ndarray":
    X = np.atleast_2d(np.asarray(points, dtype=float))
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    for v in (a, c, wy):
        if not np.isfinite(v):
            raise ValueError("a, c and wy must be finite")
    x, y = X[:, 0], X[:, 1]
    u = y + c * (x * x - 1.0)
    V = 0.125 * (x * x - 1.0) ** 2 + 0.5 * wy * wy * u * u + a * x ** 3
    gx = 0.5 * x * (x * x - 1.0) + wy * wy * u * 2.0 * c * x + 3.0 * a * x * x
    gy = wy * wy * u
    hxx = 1.5 * x * x - 0.5 + wy * wy * (2.0 * c * u + (2.0 * c * x) ** 2) + 6.0 * a * x
    hxy = wy * wy * 2.0 * c * x
    hyy = np.full_like(x, wy * wy)
    return np.stack([V, gx, gy, hxx, hxy, hyy], axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nP = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])',
         "call": 'potential_and_derivatives(P.copy(), 5e-10, 1.6, 0.55)',
         "gold_call": '_oracle_potential_and_derivatives(P.copy(), 5e-10, 1.6, 0.55)'},   # normal: valley point and both minima at the task couplings
        {"setup": 'import numpy as np\nP = np.array([[0.3, -0.2], [-0.7, 1.4], [2.0, 0.5]])',
         "call": 'potential_and_derivatives(P.copy(), 2.5e-8, 1.0, 0.8)',
         "gold_call": '_oracle_potential_and_derivatives(P.copy(), 2.5e-8, 1.0, 0.8)'},   # edge: off-valley points, reference couplings
        {"setup": 'import numpy as np\nP = np.array([[-0.5, 0.75]])',
         "call": 'potential_and_derivatives(P.copy(), 0.0, 1.0, 0.8)',
         "gold_call": '_oracle_potential_and_derivatives(P.copy(), 0.0, 1.0, 0.8)'},   # boundary: a single point, zero asymmetry
        {"setup": 'import numpy as np\ndef _c():\n    try:\n        potential_and_derivatives([[0.0, 1.0, 2.0]], 5e-10, 1.6, 0.55)\n        return 0\n    except ValueError:\n        return 1\ndef _g():\n    try:\n        _oracle_potential_and_derivatives([[0.0, 1.0, 2.0]], 5e-10, 1.6, 0.55)\n        return 0\n    except ValueError:\n        return 1',
         "call": '_c()',
         "gold_call": '_g()'},   # invalid: points not of shape (n, 2)
    ]
