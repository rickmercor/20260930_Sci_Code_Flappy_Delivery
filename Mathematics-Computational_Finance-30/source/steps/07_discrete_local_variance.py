"""
Read the local variance off the fitted smooth surface at the queried point, using the ratio the source recalls in its Section 3 discussion of continuous local volatility. Take the strike curvature by a centred difference with the step given as a fraction of the queried strike. Take the expiry sensitivity as a one-sided difference running forward from the queried expiry to the next expiry node, the convention the task statement declares; the backward alternative gives a materially different number. Value every point through step 6. A query lying exactly on an expiry node has an empty expiry step. Return 0.0 when the expiry step is empty or the strike curvature is not strictly positive, rather than raising or returning a non-finite value.

Section 3 of the source recalls the standard local-volatility ratio and notes that its own surface is built to make that ratio well defined on a smooth grid, which a piecewise-linear interpolant would not be. The source writes the ratio with continuous derivatives and fixes no discrete stencil; the task statement declares the one to use.

Returns
-------
A single Python float: the local variance at the queried point, or 0.0 in the degenerate cases named above.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def discrete_local_variance(strikes: "np.ndarray", expiries: "np.ndarray", atm_vars: "np.ndarray", q: "np.ndarray", t_query: float, k_query: float, eta: float, h_rel: float = 0.02) -> float:
    """Read the local variance off the fitted smooth surface at the queried point by finite differences.

    Parameters
    ----------
    strikes : numpy.ndarray
        One-dimensional strike ladder, strictly increasing and strictly positive.
    expiries : numpy.ndarray
        One-dimensional, strictly increasing expiry nodes.
    atm_vars : numpy.ndarray
        Node dispersion of each expiry, same shape as expiries.
    q : numpy.ndarray
        Fitted marginals of shape (len(expiries), len(strikes)).
    t_query : float
        Query expiry; strictly positive and finite.
    k_query : float
        Query strike; strictly positive and finite.
    eta : float
        The source's scaling factor applied to every dispersion (Section 3.1).
    h_rel : float, optional
        Strike step of the centred difference as a fraction of k_query; strictly positive. Default 0.02.

    Returns
    -------
    local_var : float
        The local variance at the queried point as a native Python float, or 0.0 when the expiry step is empty (the query lies on or beyond the last expiry node, or exactly on any node) or the strike curvature is not strictly positive.

    Raises
    ------
    ValueError
        If h_rel is not strictly positive and finite, or if step 6 rejects the surface inputs or the query point.
    """
    return local_var

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.stats import norm


def _ladder(strikes):
    """Validate and return a strike ladder. Shared by several steps."""
    K = np.asarray(strikes, dtype=np.float64)
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(K)):
        raise ValueError("strikes must all be finite")
    if np.any(K <= 0.0):
        raise ValueError("strikes must be strictly positive")
    if K.size > 1 and not np.all(np.diff(K) > 0.0):
        raise ValueError("strikes must be strictly increasing")
    return K


def _oracle_discrete_local_variance(strikes: "np.ndarray", expiries: "np.ndarray", atm_vars: "np.ndarray", q: "np.ndarray", t_query: float, k_query: float, eta: float, h_rel: float = 0.02) -> float:
    """Local-variance ratio on the smooth surface: centred RELATIVE strike bump, and a
    FORWARD one-sided time difference toward the next expiry node."""
    if not np.isfinite(h_rel) or h_rel <= 0.0:
        raise ValueError("h_rel must be finite and strictly positive")
    T = np.asarray(expiries, dtype=np.float64)
    tq, kq = float(t_query), float(k_query)
    h = h_rel * kq
    f = lambda t, k: _oracle_smooth_surface_price(strikes, expiries, atm_vars, q, t, k, eta)

    j = int(np.searchsorted(T, tq))
    t_up = float(T[min(j, T.size - 1)])
    dt = t_up - tq
    if dt <= 0.0:
        return 0.0
    dCdT = (f(t_up, kq) - f(tq, kq)) / dt
    d2CdK2 = (f(tq, kq + h) - 2.0 * f(tq, kq) + f(tq, kq - h)) / (h * h)
    if d2CdK2 <= 0.0:
        return 0.0
    return float(2.0 * dCdT / (kq * kq * d2CdK2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nT = np.array([0.2, 0.5, 1.0])\nV = np.array([0.04, 0.075, 0.14])\nq = np.array([[0.05, 0.15, 0.55, 0.20, 0.05], [0.08, 0.17, 0.45, 0.22, 0.08], [0.10, 0.18, 0.40, 0.24, 0.08]])\n',
            'call': 'discrete_local_variance(K, T, V, q, 0.7, 1.05, 0.3)',
            'gold_call': '_oracle_discrete_local_variance(K, T, V, q, 0.7, 1.05, 0.3)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nT = np.array([0.2, 0.5, 1.0])\nV = np.array([0.04, 0.075, 0.14])\nq = np.array([[0.05, 0.15, 0.55, 0.20, 0.05], [0.08, 0.17, 0.45, 0.22, 0.08], [0.10, 0.18, 0.40, 0.24, 0.08]])\n',
            'call': 'discrete_local_variance(K, T, V, q, 1.0, 1.0, 0.3)',
            'gold_call': '_oracle_discrete_local_variance(K, T, V, q, 1.0, 1.0, 0.3)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nT = np.array([0.2, 0.5, 1.0])\nV = np.array([0.04, 0.075, 0.14])\nq = np.array([[0.05, 0.15, 0.55, 0.20, 0.05], [0.08, 0.17, 0.45, 0.22, 0.08], [0.10, 0.18, 0.40, 0.24, 0.08]])\n',
            'call': 'discrete_local_variance(K, T, V, q, 0.4, 0.85, 0.3)',
            'gold_call': '_oracle_discrete_local_variance(K, T, V, q, 0.4, 0.85, 0.3)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nT = np.array([0.2, 0.5, 1.0])\nV = np.array([0.04, 0.075, 0.14])\nq = np.array([[0.05, 0.15, 0.55, 0.20, 0.05], [0.08, 0.17, 0.45, 0.22, 0.08], [0.10, 0.18, 0.40, 0.24, 0.08]])\n',
            'call': 'discrete_local_variance(K, T, V, q, 0.7, 1.05, 0.3, 0.05)',
            'gold_call': '_oracle_discrete_local_variance(K, T, V, q, 0.7, 1.05, 0.3, 0.05)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nT = np.array([0.2, 0.5, 1.0])\nV = np.array([0.04, 0.075, 0.14])\nq = np.array([[0.05, 0.15, 0.55, 0.20, 0.05], [0.08, 0.17, 0.45, 0.22, 0.08], [0.10, 0.18, 0.40, 0.24, 0.08]])\n',
            'call': 'discrete_local_variance(K, T, V, q, 0.7, 1.5, 0.0)',
            'gold_call': '_oracle_discrete_local_variance(K, T, V, q, 0.7, 1.5, 0.0)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nT = np.array([0.2, 0.5, 1.0])\nV = np.array([0.04, 0.075, 0.14])\nq = np.array([[0.05, 0.15, 0.55, 0.20, 0.05], [0.08, 0.17, 0.45, 0.22, 0.08], [0.10, 0.18, 0.40, 0.24, 0.08]])\ndef run_model():\n    try:\n        discrete_local_variance(K, T, V, q, 0.7, 1.05, 0.3, 0.0)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_discrete_local_variance(K, T, V, q, 0.7, 1.05, 0.3, 0.0)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': 'run_oracle()',
        },
    ]
