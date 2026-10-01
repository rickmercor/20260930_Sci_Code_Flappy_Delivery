"""
Value a single option at an arbitrary expiry and strike from the fitted marginals, following the source's Eq. (2). Between two expiry nodes the value is the source's weight-function blend of the two bracketing marginals; the source states in Eq. (2) which dispersion each of the two terms is priced at, and the answer is not 'each at its own node'. Use the linear weight function the source gives as its example directly below Eq. (2), the convention the task statement declares. Outside the node range, fall back on the nearest node's marginal priced at that node's own dispersion. Price through step 1.

Eq. (2) of the source is written so the model value stays linear in the marginals, which is what lets step 5 stay in its program class. Immediately below Eq. (2) the source defines an interpolated dispersion and gives one concrete admissible weight function.

Returns
-------
A single Python float: the model value at the queried expiry and strike.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def smooth_surface_price(strikes: "np.ndarray", expiries: "np.ndarray", atm_vars: "np.ndarray", q: "np.ndarray", t_query: float, k_query: float, eta: float) -> float:
    """Value one option at an arbitrary expiry and strike off the fitted marginals, following the source's Eq. (2).

    Parameters
    ----------
    strikes : numpy.ndarray
        One-dimensional strike ladder, strictly increasing and strictly positive.
    expiries : numpy.ndarray
        One-dimensional, strictly increasing expiry nodes, at least one.
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

    Returns
    -------
    value : float
        The model value at the queried expiry and strike, as a native Python float.

    Raises
    ------
    ValueError
        If the ladder is invalid as in step 2, if expiries is empty or not strictly increasing, if atm_vars or q has the wrong shape, or if t_query or k_query is not strictly positive and finite.
    """
    return value

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


def _oracle_smooth_surface_price(strikes: "np.ndarray", expiries: "np.ndarray", atm_vars: "np.ndarray", q: "np.ndarray", t_query: float, k_query: float, eta: float) -> float:
    """Paper eq (2): blend the two bracketing marginals, BOTH priced at the
    INTERPOLATED dispersion V(T) = a*V_j + (1-a)*V_{j-1}, scaled by eta."""
    K = _ladder(strikes)
    T = np.asarray(expiries, dtype=np.float64)
    V = np.asarray(atm_vars, dtype=np.float64)
    Qd = np.asarray(q, dtype=np.float64)
    if T.ndim != 1 or T.size == 0 or not np.all(np.diff(T) > 0.0):
        raise ValueError("expiries must be a non-empty strictly increasing array")
    if V.shape != T.shape:
        raise ValueError("atm_vars must have one entry per expiry node")
    if Qd.shape != (T.size, K.size):
        raise ValueError(f"q must be ({T.size}, {K.size}), got {Qd.shape}")
    tq, kq, e = float(t_query), float(k_query), float(eta)
    if not (np.isfinite(tq) and np.isfinite(kq)) or kq <= 0.0 or tq <= 0.0:
        raise ValueError("t_query and k_query must be finite and strictly positive")

    j = int(np.searchsorted(T, tq))
    if j <= 0:
        return float(_oracle_bs_call_price(K, kq, e * V[0]) @ Qd[0])
    if j >= T.size:
        return float(_oracle_bs_call_price(K, kq, e * V[-1]) @ Qd[-1])
    a = (min(tq, T[j]) - T[j - 1]) / (T[j] - T[j - 1])
    vt = a * V[j] + (1.0 - a) * V[j - 1]
    px = _oracle_bs_call_price(K, kq, e * vt)
    return float(a * (px @ Qd[j]) + (1.0 - a) * (px @ Qd[j - 1]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nT = np.array([0.2, 0.5, 1.0])\nV = np.array([0.04, 0.075, 0.14])\nq = np.array([[0.05, 0.15, 0.55, 0.20, 0.05], [0.08, 0.17, 0.45, 0.22, 0.08], [0.10, 0.18, 0.40, 0.24, 0.08]])\n',
            'call': 'smooth_surface_price(K, T, V, q, 0.7, 1.05, 0.3)',
            'gold_call': '_oracle_smooth_surface_price(K, T, V, q, 0.7, 1.05, 0.3)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nT = np.array([0.2, 0.5, 1.0])\nV = np.array([0.04, 0.075, 0.14])\nq = np.array([[0.05, 0.15, 0.55, 0.20, 0.05], [0.08, 0.17, 0.45, 0.22, 0.08], [0.10, 0.18, 0.40, 0.24, 0.08]])\n',
            'call': 'smooth_surface_price(K, T, V, q, 0.5, 1.0, 0.3)',
            'gold_call': '_oracle_smooth_surface_price(K, T, V, q, 0.5, 1.0, 0.3)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nT = np.array([0.2, 0.5, 1.0])\nV = np.array([0.04, 0.075, 0.14])\nq = np.array([[0.05, 0.15, 0.55, 0.20, 0.05], [0.08, 0.17, 0.45, 0.22, 0.08], [0.10, 0.18, 0.40, 0.24, 0.08]])\n',
            'call': 'smooth_surface_price(K, T, V, q, 0.1, 0.95, 0.3)',
            'gold_call': '_oracle_smooth_surface_price(K, T, V, q, 0.1, 0.95, 0.3)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nT = np.array([0.2, 0.5, 1.0])\nV = np.array([0.04, 0.075, 0.14])\nq = np.array([[0.05, 0.15, 0.55, 0.20, 0.05], [0.08, 0.17, 0.45, 0.22, 0.08], [0.10, 0.18, 0.40, 0.24, 0.08]])\n',
            'call': 'smooth_surface_price(K, T, V, q, 5.0, 1.0, 0.3)',
            'gold_call': '_oracle_smooth_surface_price(K, T, V, q, 5.0, 1.0, 0.3)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nT = np.array([0.2, 0.5, 1.0])\nV = np.array([0.04, 0.075, 0.14])\nq = np.array([[0.05, 0.15, 0.55, 0.20, 0.05], [0.08, 0.17, 0.45, 0.22, 0.08], [0.10, 0.18, 0.40, 0.24, 0.08]])\ndef run_model():\n    try:\n        smooth_surface_price(K, T, V, q, 0.7, -1.0, 0.3)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_smooth_surface_price(K, T, V, q, 0.7, -1.0, 0.3)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': 'run_oracle()',
        },
    ]
