"""
Diagnose why the radiation problem is not reduced by Minkov's method, from the coefficient sequence q_l, the Minkov kernel Q_lm of the same order and the Damkohler number. Insert the radiation condition directly into the canonical dual series relations: the cap relation then keeps the canonical form sum over l of (1 - q_l^Da) X_l P_l(cos theta) = 1 with a modified coefficient q_l^Da that follows from the surface value and the flux density of the previous step (derive it), while the inert-part relation is unchanged, so that Minkov's reduction applies formally with q_l replaced by q_l^Da. Return the sup-norm (largest row sum of absolute values) of the truncated matrix M_lm = q_m^Da Q_lm, l, m = 0..order, and the value of J = X_0 / 2 that the truncated system X_l - sum over m of M_lm X_m = Q_l0 gives when solved by direct elimination at this order (the order is fixed by the length of q). Raise ValueError if q and Q are not of consistent shape or if damkohler is not a finite positive number.

The source proves that the sup-norm of the resolving matrix being below one is sufficient for the truncation method to converge and reports that this condition already fails for thin shells; with the radiation coefficient the modified sequence grows without bound with l, the matrix entries no longer decay along a row, the sup-norm grows with the truncation order and the truncated solution converges only like the inverse of the order, an order of magnitude slower than the projected system, so that the formally exact system is useless at the truncation orders of the source.

Returns
-------
A (2,) float64 array [sup-norm of M at this order, J from the truncated naive system].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radiation_islae_regularity(q, Q, damkohler):
    """Diagnose why the radiation problem is not reduced by Minkov's method, from the
    coefficient sequence q_l, the Minkov kernel Q_lm of the same order and the Damkohler
    number. A (2,) float64 array [sup-norm of M at this order, J from the truncated naive
    system]."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def _oracle_radiation_islae_regularity(q, Q, damkohler):
    q = np.asarray(q, dtype=float).ravel(); Q = np.asarray(Q, dtype=float)
    da = float(damkohler)
    n = q.size - 1
    if n < 0 or Q.shape != (n + 1, n + 1):
        raise ValueError("q must have length order + 1 and Q must be (order + 1, order + 1)")
    if not (np.isfinite(da) and da > 0.0):
        raise ValueError("damkohler must be a finite positive number")
    l = np.arange(n + 1, dtype=float)
    qda = q - (l + 0.5) / da
    M = Q * qda[None, :]
    mnorm = float(np.max(np.sum(np.abs(M), axis=1)))
    X = np.linalg.solve(np.eye(n + 1) - M, Q[:, 0].copy())
    return np.array([mnorm, 0.5 * X[0]], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndef _fx_minkov_matrix(order, theta0):\n    n = int(order); t = float(theta0)\n    if n < 0 or n != order:\n        raise ValueError(\"order must be a non-negative integer\")\n    if not (0.0 < t <= np.pi):\n        raise ValueError(\"theta0 must lie in (0, pi]\")\n    l = np.arange(n + 1, dtype=float)[:, None]\n    m = np.arange(n + 1, dtype=float)[None, :]\n    s = l + m + 1.0\n    d = l - m\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)\n    return (np.sin(s * t) / s + off) / np.pi\ntheta0 = float(np.arccos(0.7))\neps = 0.625\nbiot = 2.5\nhindrance = 1.0\ndamkohler = 3.0\ncoefficients = _oracle_dual_series_coefficients(120, eps, biot, hindrance)\nq = coefficients[4]\nQ = _fx_minkov_matrix(120, theta0)\n',
         'call': 'radiation_islae_regularity(q, Q, damkohler)',
         'gold_call': '_oracle_radiation_islae_regularity(q, Q, damkohler)'},
        {'setup': 'import numpy as np\ndef _fx_minkov_matrix(order, theta0):\n    n = int(order); t = float(theta0)\n    if n < 0 or n != order:\n        raise ValueError(\"order must be a non-negative integer\")\n    if not (0.0 < t <= np.pi):\n        raise ValueError(\"theta0 must lie in (0, pi]\")\n    l = np.arange(n + 1, dtype=float)[:, None]\n    m = np.arange(n + 1, dtype=float)[None, :]\n    s = l + m + 1.0\n    d = l - m\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)\n    return (np.sin(s * t) / s + off) / np.pi\ntheta0 = float(np.arccos(0.8))\neps = 0.35\nbiot = np.inf\nhindrance = 0.0\ndamkohler = 0.7\ncoefficients = _oracle_dual_series_coefficients(200, eps, biot, hindrance)\nq = coefficients[4]\nQ = _fx_minkov_matrix(200, theta0)\n',
         'call': 'radiation_islae_regularity(q, Q, damkohler)',
         'gold_call': '_oracle_radiation_islae_regularity(q, Q, damkohler)'},
        {'setup': 'import numpy as np\ndef _fx_minkov_matrix(order, theta0):\n    n = int(order); t = float(theta0)\n    if n < 0 or n != order:\n        raise ValueError(\"order must be a non-negative integer\")\n    if not (0.0 < t <= np.pi):\n        raise ValueError(\"theta0 must lie in (0, pi]\")\n    l = np.arange(n + 1, dtype=float)[:, None]\n    m = np.arange(n + 1, dtype=float)[None, :]\n    s = l + m + 1.0\n    d = l - m\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)\n    return (np.sin(s * t) / s + off) / np.pi\ntheta0 = float(np.arccos(0.4))\neps = 0.75\nbiot = 6.0\nhindrance = 0.5\ndamkohler = 12.0\ncoefficients = _oracle_dual_series_coefficients(400, eps, biot, hindrance)\nq = coefficients[4]\nQ = _fx_minkov_matrix(400, theta0)\n',
         'call': 'radiation_islae_regularity(q, Q, damkohler)',
         'gold_call': '_oracle_radiation_islae_regularity(q, Q, damkohler)'},
        {'setup': 'import numpy as np\ndef _fx_minkov_matrix(order, theta0):\n    n = int(order); t = float(theta0)\n    if n < 0 or n != order:\n        raise ValueError(\"order must be a non-negative integer\")\n    if not (0.0 < t <= np.pi):\n        raise ValueError(\"theta0 must lie in (0, pi]\")\n    l = np.arange(n + 1, dtype=float)[:, None]\n    m = np.arange(n + 1, dtype=float)[None, :]\n    s = l + m + 1.0\n    d = l - m\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)\n    return (np.sin(s * t) / s + off) / np.pi\ntheta0 = float(np.pi / 2)\neps = 2.0 / 4.2\nbiot = 0.5\nhindrance = 1.5\ndamkohler = 0.5\ncoefficients = _oracle_dual_series_coefficients(60, eps, biot, hindrance)\nq = coefficients[4]\nQ = _fx_minkov_matrix(60, theta0)\n',
         'call': 'radiation_islae_regularity(q, Q, damkohler)',
         'gold_call': '_oracle_radiation_islae_regularity(q, Q, damkohler)'},
    ]
