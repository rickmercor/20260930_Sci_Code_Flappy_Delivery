"""
Solve the resolving system of the perfect-sink problem by the truncation (reduction) method. Given the coefficient sequence q_l (l = 0..order, the last row of the coefficient step) and the Minkov kernel Q_lm of the same order, keep the unknowns X_0..X_order and the first order + 1 equations, X_l - sum over m <= order of q_m Q_lm X_m = Q_l0, and solve the resulting dense linear system by direct elimination (no iteration). Return the truncated solution vector. The truncation order is fixed by the length of q and must be used exactly as given. Raise ValueError if q and Q are not finite arrays of consistent shape (order + 1,) and (order + 1, order + 1).

The source solves this system by truncation with NumPy and reports that the maximum difference between the orders 100 and 200 is of order 1e-5, so that the numerical values may be treated as exact; a sufficient condition for the truncation to converge is that the sup-norm of the matrix q_m Q_lm is below one, which fails for thin shells over a range of cap angles although the truncated solution still converges there.

Returns
-------
An (order + 1,) float64 array X with X[l] = X_l.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def perfect_sink_solution(q, Q):
    """Solve the resolving system of the perfect-sink problem by the truncation (reduction)
    method. An (order + 1,) float64 array X with X[l] = X_l."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def _oracle_perfect_sink_solution(q, Q):
    q = np.asarray(q, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float)
    n = q.size - 1
    if n < 0 or Q.ndim != 2 or Q.shape != (n + 1, n + 1):
        raise ValueError("q must have length order + 1 and Q must be (order + 1, order + 1)")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(Q)):
        raise ValueError("q and Q must be finite")
    A = np.eye(n + 1) - Q * q[None, :]
    return np.linalg.solve(A, Q[:, 0].copy())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndef _fx_minkov_matrix(order, theta0):\n    n = int(order); t = float(theta0)\n    if n < 0 or n != order:\n        raise ValueError(\"order must be a non-negative integer\")\n    if not (0.0 < t <= np.pi):\n        raise ValueError(\"theta0 must lie in (0, pi]\")\n    l = np.arange(n + 1, dtype=float)[:, None]\n    m = np.arange(n + 1, dtype=float)[None, :]\n    s = l + m + 1.0\n    d = l - m\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)\n    return (np.sin(s * t) / s + off) / np.pi\ntheta0 = float(np.arccos(0.7))\neps = 0.625\nbiot = 2.5\nhindrance = 1.0\ndamkohler = 3.0\ncoefficients = _oracle_dual_series_coefficients(120, eps, biot, hindrance)\nq = coefficients[4]\nQ = _fx_minkov_matrix(120, theta0)\n',
         'call': 'perfect_sink_solution(q, Q)',
         'gold_call': '_oracle_perfect_sink_solution(q, Q)'},
        {'setup': 'import numpy as np\ndef _fx_minkov_matrix(order, theta0):\n    n = int(order); t = float(theta0)\n    if n < 0 or n != order:\n        raise ValueError(\"order must be a non-negative integer\")\n    if not (0.0 < t <= np.pi):\n        raise ValueError(\"theta0 must lie in (0, pi]\")\n    l = np.arange(n + 1, dtype=float)[:, None]\n    m = np.arange(n + 1, dtype=float)[None, :]\n    s = l + m + 1.0\n    d = l - m\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)\n    return (np.sin(s * t) / s + off) / np.pi\ntheta0 = float(np.arccos(0.8))\neps = 0.35\nbiot = np.inf\nhindrance = 0.0\ndamkohler = 0.7\ncoefficients = _oracle_dual_series_coefficients(200, eps, biot, hindrance)\nq = coefficients[4]\nQ = _fx_minkov_matrix(200, theta0)\n',
         'call': 'perfect_sink_solution(q, Q)',
         'gold_call': '_oracle_perfect_sink_solution(q, Q)'},
        {'setup': 'import numpy as np\ndef _fx_minkov_matrix(order, theta0):\n    n = int(order); t = float(theta0)\n    if n < 0 or n != order:\n        raise ValueError(\"order must be a non-negative integer\")\n    if not (0.0 < t <= np.pi):\n        raise ValueError(\"theta0 must lie in (0, pi]\")\n    l = np.arange(n + 1, dtype=float)[:, None]\n    m = np.arange(n + 1, dtype=float)[None, :]\n    s = l + m + 1.0\n    d = l - m\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)\n    return (np.sin(s * t) / s + off) / np.pi\ntheta0 = 0.4 * np.pi\neps = 0.0\nbiot = 5.0\nhindrance = 1.0\ncoefficients = _oracle_dual_series_coefficients(80, eps, biot, hindrance)\nq = coefficients[4]\nQ = _fx_minkov_matrix(80, theta0)\n',
         'call': 'perfect_sink_solution(q, Q)',
         'gold_call': '_oracle_perfect_sink_solution(q, Q)'},
        {'setup': 'import numpy as np\ndef _fx_minkov_matrix(order, theta0):\n    n = int(order); t = float(theta0)\n    if n < 0 or n != order:\n        raise ValueError(\"order must be a non-negative integer\")\n    if not (0.0 < t <= np.pi):\n        raise ValueError(\"theta0 must lie in (0, pi]\")\n    l = np.arange(n + 1, dtype=float)[:, None]\n    m = np.arange(n + 1, dtype=float)[None, :]\n    s = l + m + 1.0\n    d = l - m\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)\n    return (np.sin(s * t) / s + off) / np.pi\ntheta0 = float(np.arccos(0.4))\neps = 0.75\nbiot = 6.0\nhindrance = 0.5\ndamkohler = 12.0\ncoefficients = _oracle_dual_series_coefficients(400, eps, biot, hindrance)\nq = coefficients[4]\nQ = _fx_minkov_matrix(400, theta0)\n',
         'call': 'perfect_sink_solution(q, Q)',
         'gold_call': '_oracle_perfect_sink_solution(q, Q)'},
    ]
