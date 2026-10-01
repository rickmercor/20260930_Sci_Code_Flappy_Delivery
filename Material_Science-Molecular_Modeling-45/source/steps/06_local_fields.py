"""
Reconstruct the trapping probability field, the cumulative capture flux and the distribution of the entry flux over the shell from a canonical solution vector. Given X_l (l = 0..n) and the (5, n + 1) coefficient array [s_plus; s_minus; eps_l; w_l; q_l] of the same order, recover the exterior moments and the interior moments and evaluate the field u(xi, theta) in the cavity at each probe point (xi, theta) of the (m, 2) array points; the field must satisfy the permeable-wall condition on the cavity wall by construction. Then evaluate the fraction F(theta_c) of the total capture flux that enters the core through the polar sub-cap 0 <= theta <= theta_c: the flux density into the core at xi = 1 is -du/dxi, the fraction is the integral of -du/dxi sin theta over 0..theta_c divided by the same integral over 0..pi, both taken exactly of the truncated series (the integrals of P_l(cos theta) sin theta over a polar cap have closed forms in P_(l-1) and P_(l+1) at cos theta_c; an exact Gauss-Legendre quadrature of the truncated series is equivalent); F(pi) = 1. Finally evaluate the fraction G of the total entry flux through the cavity wall xi = 1 / eps that enters through the hemisphere facing the site, 0 <= theta <= pi / 2, from the radial derivative of the same truncated field on the wall (the diffusivity at the wall is the same factor for every angle and cancels), again with exact integrals; G = 1/2 for an isotropic core, and G is undefined (nan) in the unbounded limit eps = 0. The same reconstruction applies to the solution vector of the partially reactive site. Raise ValueError if the coefficient array does not match X, if eps is not in [0, 1), if points is not an (m, 2) array of points inside the closed shell 1 <= xi <= 1 / eps, 0 <= theta <= pi, or if theta_c is not in (0, pi].

The source computes the local concentration maps in the main cross-section of the cavity from the same expansion and notes that raising the truncation order from 200 to 500 changes the field by about 1e-3; the flux density of a perfect-sink cap has an integrable inverse-square-root singularity at the rim of the site, so that a disproportionate share of the capture enters near the edge, while a finite reactivity removes the singularity and spreads the capture more evenly over the site. In steady state the total entry flux through the shell equals the capture at the core, but its distribution over the shell is skewed towards the side facing the site.

Returns
-------
An (m + 2,) float64 array: the m field values u at the probe points, then the capture fraction F(theta_c), then the wall entry fraction G.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def local_fields(X, coefficients, eps, points, theta_c):
    """Reconstruct the trapping probability field, the cumulative capture flux and the
    distribution of the entry flux over the shell from a canonical solution vector. An (m +
    2,) float64 array: the m field values u at the probe points, then the capture fraction
    F(theta_c), then the wall entry fraction G."""
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

def _cap_integrals(cos_theta, order):
    """b_l = integral of P_l(x) dx from cos_theta to 1, l = 0..order (closed form)."""
    n = int(order)
    c = float(cos_theta)
    P = _legendre_rows(np.array([c]), n + 1)[0]
    b = np.empty(n + 1)
    b[0] = 1.0 - c
    if n >= 1:
        l = np.arange(1, n + 1, dtype=float)
        b[1:] = (P[0:n] - P[2:n + 2]) / (2.0 * l + 1.0)
    return b

def _oracle_local_fields(X, coefficients, eps, points, theta_c):
    X = np.asarray(X, dtype=float).ravel()
    co = np.asarray(coefficients, dtype=float)
    e = float(eps)
    if X.size == 0 or co.ndim != 2 or co.shape[0] != 5 or co.shape[1] != X.size:
        raise ValueError("coefficients must be a (5, order + 1) array matching X")
    if not (0.0 <= e < 1.0):
        raise ValueError("eps must lie in [0, 1)")
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (m, 2) array of (xi, theta)")
    tc = float(theta_c)
    if not (0.0 < tc <= np.pi):
        raise ValueError("theta_c must lie in (0, pi]")
    xi = pts[:, 0]; th = pts[:, 1]
    outer = np.inf if e == 0.0 else 1.0 / e
    if np.any(xi < 1.0) or np.any(xi > outer) or np.any(th < 0.0) or np.any(th > np.pi):
        raise ValueError("probe points must lie in the shell 1 <= xi <= 1/eps, 0 <= theta <= pi")
    n = X.size - 1
    l = np.arange(n + 1, dtype=float)
    sp = co[0]; sm = co[1]; el = co[2]; w = co[3]
    A = (1.0 - w) * X                                                   # exterior moments
    P = _legendre_rows(np.cos(th), n)
    radial = xi[:, None] ** sm[None, :] - el[None, :] * xi[:, None] ** sp[None, :]
    u = (radial * A[None, :] * P).sum(axis=1)
    b = _cap_integrals(np.cos(tc), n)
    flux_fraction = float(np.sum((l + 0.5) * X * b) / X[0])
    if e == 0.0:
        wall_fraction = np.nan
    else:
        xw = 1.0 / e
        dwall = -A * (sm * xw ** (sm - 1.0) - el * sp * xw ** (sp - 1.0))          # entry flux density coefficients
        bh = _cap_integrals(0.0, n)
        wall_fraction = float(np.sum(dwall * bh) / (2.0 * dwall[0]))
    return np.concatenate([u, [flux_fraction, wall_fraction]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndef _fx_minkov_matrix(order, theta0):\n    n = int(order); t = float(theta0)\n    if n < 0 or n != order:\n        raise ValueError(\"order must be a non-negative integer\")\n    if not (0.0 < t <= np.pi):\n        raise ValueError(\"theta0 must lie in (0, pi]\")\n    l = np.arange(n + 1, dtype=float)[:, None]\n    m = np.arange(n + 1, dtype=float)[None, :]\n    s = l + m + 1.0\n    d = l - m\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)\n    return (np.sin(s * t) / s + off) / np.pi\ntheta0 = float(np.arccos(0.7))\neps = 0.625\nbiot = 2.5\nhindrance = 1.0\ndamkohler = 3.0\ncoefficients = _oracle_dual_series_coefficients(120, eps, biot, hindrance)\nq = coefficients[4]\nQ = _fx_minkov_matrix(120, theta0)\nX = _oracle_perfect_sink_solution(q, Q)\npoints = np.array([[0.5 * (1.0 + 1.0 / eps), 0.0], [0.5 * (1.0 + 1.0 / eps), np.pi], [1.25, 0.9], [1.0 / eps, 0.3]])\ntheta_c = theta0 / 2\n',
         'call': 'local_fields(X, coefficients, eps, points, theta_c)',
         'gold_call': '_oracle_local_fields(X, coefficients, eps, points, theta_c)'},
        {'setup': 'import numpy as np\ndef _fx_minkov_matrix(order, theta0):\n    n = int(order); t = float(theta0)\n    if n < 0 or n != order:\n        raise ValueError(\"order must be a non-negative integer\")\n    if not (0.0 < t <= np.pi):\n        raise ValueError(\"theta0 must lie in (0, pi]\")\n    l = np.arange(n + 1, dtype=float)[:, None]\n    m = np.arange(n + 1, dtype=float)[None, :]\n    s = l + m + 1.0\n    d = l - m\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)\n    return (np.sin(s * t) / s + off) / np.pi\ntheta0 = float(np.arccos(0.8))\neps = 0.35\nbiot = np.inf\nhindrance = 0.0\ndamkohler = 0.7\ncoefficients = _oracle_dual_series_coefficients(200, eps, biot, hindrance)\nq = coefficients[4]\nQ = _fx_minkov_matrix(200, theta0)\nX = _oracle_perfect_sink_solution(q, Q)\npoints = np.array([[1.1, 0.2], [2.0, 2.5]])\ntheta_c = float(np.pi)\n',
         'call': 'local_fields(X, coefficients, eps, points, theta_c)',
         'gold_call': '_oracle_local_fields(X, coefficients, eps, points, theta_c)'},
        {'setup': 'import numpy as np\ndef _fx_minkov_matrix(order, theta0):\n    n = int(order); t = float(theta0)\n    if n < 0 or n != order:\n        raise ValueError(\"order must be a non-negative integer\")\n    if not (0.0 < t <= np.pi):\n        raise ValueError(\"theta0 must lie in (0, pi]\")\n    l = np.arange(n + 1, dtype=float)[:, None]\n    m = np.arange(n + 1, dtype=float)[None, :]\n    s = l + m + 1.0\n    d = l - m\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)\n    return (np.sin(s * t) / s + off) / np.pi\ntheta0 = float(np.arccos(0.4))\neps = 0.75\nbiot = 6.0\nhindrance = 0.5\ndamkohler = 12.0\ncoefficients = _oracle_dual_series_coefficients(150, eps, biot, hindrance)\nq = coefficients[4]\nQ = _fx_minkov_matrix(150, theta0)\nX = _oracle_perfect_sink_solution(q, Q)\npoints = np.array([[1.05, 0.5], [1.2, theta0], [1.3, 3.0]])\ntheta_c = theta0\n',
         'call': 'local_fields(X, coefficients, eps, points, theta_c)',
         'gold_call': '_oracle_local_fields(X, coefficients, eps, points, theta_c)'},
        {'setup': 'import numpy as np\ndef _fx_minkov_matrix(order, theta0):\n    n = int(order); t = float(theta0)\n    if n < 0 or n != order:\n        raise ValueError(\"order must be a non-negative integer\")\n    if not (0.0 < t <= np.pi):\n        raise ValueError(\"theta0 must lie in (0, pi]\")\n    l = np.arange(n + 1, dtype=float)[:, None]\n    m = np.arange(n + 1, dtype=float)[None, :]\n    s = l + m + 1.0\n    d = l - m\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)\n    return (np.sin(s * t) / s + off) / np.pi\ntheta0 = float(np.pi / 2)\neps = 2.0 / 4.2\nbiot = 0.5\nhindrance = 1.5\ndamkohler = 0.5\ncoefficients = _oracle_dual_series_coefficients(400, eps, biot, hindrance)\nq = coefficients[4]\nQ = _fx_minkov_matrix(400, theta0)\nX = _oracle_perfect_sink_solution(q, Q)\npoints = np.array([[1.5, 1.0]])\ntheta_c = 0.25 * np.pi\n',
         'call': 'local_fields(X, coefficients, eps, points, theta_c)',
         'gold_call': '_oracle_local_fields(X, coefficients, eps, points, theta_c)'},
    ]
