"""
Reduce the canonical dual series relations to a resolving infinite system of linear algebraic equations by Minkov's method and return its kernel. Write the cap relation in the self-consistent form sum over l of X_l P_l(cos theta) = 1 + sum over m of q_m X_m P_m(cos theta) on 0 <= theta < theta0, keep the inert-part relation sum over l of (l + 1/2) X_l P_l(cos theta) = 0 on theta0 < theta <= pi, and apply to this auxiliary pair the exact solution of dual series relations of that type (the Abel-type integral representation whose inner integral of P_m(cos tau) sin tau / sqrt(cos tau - cos t) has a closed trigonometric form). The result is the system X_l - sum over m of q_m Q_lm(theta0) X_m = Q_l0(theta0) for l = 0, 1, 2, ..., in which the kernel Q_lm(theta0) depends only on the cap half-angle. Return the matrix Q_lm for l, m = 0..order, evaluated in closed form. Raise ValueError if order is not a non-negative integer or if theta0 is not in (0, pi].

For a full cap (theta0 = pi) the kernel reduces to the identity and the system decouples, giving back the spherically symmetric Berg result; for small caps the zeroth diagonal element of the kernel gives the Hill small-angle law. The kernel is symmetric, its diagonal carries the cap half-angle itself, and its entries decay only like the inverse of the index sum, which is why the regularity of the system depends on the decay of the coefficients q_m that multiply it.

Returns
-------
An (order + 1, order + 1) float64 array Q with Q[l, m] = Q_lm(theta0).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def minkov_matrix(order, theta0):
    """Reduce the canonical dual series relations to a resolving infinite system of linear
    algebraic equations by Minkov's method and return its kernel. An (order + 1, order + 1)
    float64 array Q with Q[l, m] = Q_lm(theta0)."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def _oracle_minkov_matrix(order, theta0):
    n = int(order); t = float(theta0)
    if n < 0 or n != order:
        raise ValueError("order must be a non-negative integer")
    if not (0.0 < t <= np.pi):
        raise ValueError("theta0 must lie in (0, pi]")
    l = np.arange(n + 1, dtype=float)[:, None]
    m = np.arange(n + 1, dtype=float)[None, :]
    s = l + m + 1.0
    d = l - m
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)
    return (np.sin(s * t) / s + off) / np.pi

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\norder = 10\ntheta0 = float(np.arccos(0.7))\n',
         'call': 'minkov_matrix(order, theta0)',
         'gold_call': '_oracle_minkov_matrix(order, theta0)'},
        {'setup': 'import numpy as np\norder = 150\ntheta0 = float(np.arccos(0.8))\n',
         'call': 'minkov_matrix(order, theta0)',
         'gold_call': '_oracle_minkov_matrix(order, theta0)'},
        {'setup': 'import numpy as np\norder = 60\ntheta0 = float(np.pi)\n',
         'call': 'minkov_matrix(order, theta0)',
         'gold_call': '_oracle_minkov_matrix(order, theta0)'},
        {'setup': 'import numpy as np\norder = 400\ntheta0 = float(np.pi / 2)\n',
         'call': 'minkov_matrix(order, theta0)',
         'gold_call': '_oracle_minkov_matrix(order, theta0)'},
    ]
