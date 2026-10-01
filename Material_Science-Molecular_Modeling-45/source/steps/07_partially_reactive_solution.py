"""
Solve the capture problem of a partially reactive site from the coefficient sequence q_l, the cap half-angle and the Damkohler number. On the active cap the perfect-sink condition u = 1 is replaced by the radiation condition du/dxi - Da u = -Da at xi = 1 (the dimensionless form of D dc/dr = kappa c with Da = kappa R / D), the inert part stays reflecting and the wall condition is unchanged. Keep the same expansion, the same exterior moments A_l = (1 - w_l) X_l and the same canonical unknowns X_l as in the perfect-sink problem, so that the surface value is u(1, theta) = sum over l of (1 - q_l) X_l P_l(cos theta) and the flux density into the core is -du/dxi(1, theta) = sum over l of (l + 1/2) X_l P_l(cos theta). Write the two boundary conditions on the core as one relation on the whole unit sphere between the flux density and the surface value, in which the cap is selected by its indicator function, reduce it by Galerkin projection onto the Legendre polynomials P_0..P_order with the L2 inner product on the sphere, evaluating the resulting integrals over the cap exactly (they are integrals of polynomials; a Gauss-Legendre rule with at least order + 1 nodes on the cap interval is exact), and solve the dense (order + 1)-dimensional system by direct elimination, the order being fixed by the length of q. Do not reduce the radiation problem by Minkov's method: the direct substitution of the radiation coefficient into that reduction is treated separately as a diagnostic. Raise ValueError if q is not a finite array, if theta0 is not in (0, pi], or if damkohler is not a finite positive number.

The Galerkin projection of a partly reactive boundary is the exact infinite set of linear equations whose constant-flux truncation is the Shoup-Lipari-Szabo approximation of the Solc-Stockmayer model; retaining all Legendre orders up to the truncation makes it exact in the limit and its truncation converges at the same second-order rate as the Minkov system of the perfect sink. Its solution reduces to the perfect-sink result as Da grows and to the reaction-controlled rate as Da vanishes, and for a full cap it reproduces the closed form of the isotropic radiation sphere in the corona behind the shell.

Returns
-------
An (order + 1,) float64 array X with X[l] = X_l of the partially reactive site.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def partially_reactive_solution(q, theta0, damkohler):
    """Solve the capture problem of a partially reactive site from the coefficient sequence
    q_l, the cap half-angle and the Damkohler number. An (order + 1,) float64 array X with
    X[l] = X_l of the partially reactive site."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def _legendre_rows(x, order):
    """P_l(x) for l = 0..order at the points x, shape (len(x), order + 1)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return legvander(x, int(order))

def _oracle_partially_reactive_solution(q, theta0, damkohler):
    q = np.asarray(q, dtype=float).ravel()
    t = float(theta0); da = float(damkohler)
    n = q.size - 1
    if n < 0 or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite array of length order + 1")
    if not (0.0 < t <= np.pi):
        raise ValueError("theta0 must lie in (0, pi]")
    if not (np.isfinite(da) and da > 0.0):
        raise ValueError("damkohler must be a finite positive number")
    c = float(np.cos(t))
    xg, wg = leggauss(n + 2)
    x = 0.5 * (1.0 - c) * xg + 0.5 * (1.0 + c); wq = 0.5 * (1.0 - c) * wg
    V = _legendre_rows(x, n)
    K = (V * wq[:, None]).T @ V
    b = (V * wq[:, None]).sum(axis=0)
    A = np.eye(n + 1) + da * K * (1.0 - q)[None, :]
    return np.linalg.solve(A, da * b)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ntheta0 = float(np.arccos(0.7))\neps = 0.625\nbiot = 2.5\nhindrance = 1.0\ndamkohler = 3.0\nq = _oracle_dual_series_coefficients(120, eps, biot, hindrance)[4]\n',
         'call': 'partially_reactive_solution(q, theta0, damkohler)',
         'gold_call': '_oracle_partially_reactive_solution(q, theta0, damkohler)'},
        {'setup': 'import numpy as np\ntheta0 = float(np.arccos(0.8))\neps = 0.35\nbiot = np.inf\nhindrance = 0.0\ndamkohler = 0.7\nq = _oracle_dual_series_coefficients(200, eps, biot, hindrance)[4]\n',
         'call': 'partially_reactive_solution(q, theta0, damkohler)',
         'gold_call': '_oracle_partially_reactive_solution(q, theta0, damkohler)'},
        {'setup': 'import numpy as np\ntheta0 = float(np.pi)\neps = 0.5\nbiot = 1.5\nhindrance = 0.5\ndamkohler = 3.0\nq = _oracle_dual_series_coefficients(50, eps, biot, hindrance)[4]\n',
         'call': 'partially_reactive_solution(q, theta0, damkohler)',
         'gold_call': '_oracle_partially_reactive_solution(q, theta0, damkohler)'},
        {'setup': 'import numpy as np\ntheta0 = float(np.arccos(0.4))\neps = 0.75\nbiot = 6.0\nhindrance = 0.5\ndamkohler = 12.0\nq = _oracle_dual_series_coefficients(400, eps, biot, hindrance)[4]\n',
         'call': 'partially_reactive_solution(q, theta0, damkohler)',
         'gold_call': '_oracle_partially_reactive_solution(q, theta0, damkohler)'},
    ]
