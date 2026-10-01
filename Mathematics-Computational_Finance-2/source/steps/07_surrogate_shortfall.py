"""
Return the surrogate value minus the delivered value.

Both are as in the problem statement, for the instance given by the arguments with M tangent points.

Returns
-------
float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def surrogate_shortfall(xmin: np.ndarray, xmax: np.ndarray, returns: np.ndarray,
                        cap: float, radius: float, M: int) -> float:
    '''Return the surrogate value minus the delivered value.

    Parameters
    ----------
    xmin : np.ndarray
        Shape (n,). Lower bounds of the return box.
    xmax : np.ndarray
        Shape (n,). Upper bounds of the return box.
    returns : np.ndarray
        Shape (N, n). Observed returns, one per row.
    cap : float
        Per-asset weight cap.
    radius : float
        Wasserstein radius.
    M : int
        Number of tangent points.

    Returns
    -------
    shortfall : float
        To floating-point accuracy.

    Raises
    ------
    ValueError
        If xmin and xmax are not finite one-dimensional arrays of a common
        length n >= 1 with -1 < xmin < xmax elementwise; if cap is not a
        finite scalar with cap > 0 and n * cap >= 1; if M is not an integer
        with M >= 2; if returns is not a finite (N, n) array with N >= 1 and
        every row inside the box; or if radius is not a finite scalar > 0.
    '''
    return shortfall  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_surrogate_shortfall(xmin: np.ndarray, xmax: np.ndarray, returns: np.ndarray,
                                cap: float, radius: float, M: int) -> float:
    """Reference implementation."""
    y_range = _oracle_attainable_return_range(xmin, xmax, cap)
    coeffs = _oracle_tangent_coefficients(y_range, M)
    weights = _oracle_surrogate_portfolio(xmin, xmax, returns, coeffs, cap, radius)
    surrogate_value = _oracle_surrogate_worst_case(weights, xmin, xmax, returns, coeffs, radius)
    delivered = _oracle_delivered_value(weights, xmin, xmax, returns, radius)
    return float(surrogate_value - delivered)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
def instance(n, N):
    i = np.arange(n)
    j = np.arange(N)
    xmin = -(150 + (37 * i) % 121) / 1000.0
    xmax = (120 + (53 * i) % 101) / 1000.0
    X = (((7 * j[:, None] + 13 * i[None, :] + 5 * j[:, None] * i[None, :]) % 181) - 90) / 1000.0
    return xmin, xmax, X
"""
    call = "surrogate_shortfall(xmin.copy(), xmax.copy(), X.copy(), cap, radius, M)"
    gold = "_oracle_surrogate_shortfall(xmin, xmax, X, cap, radius, M)"
    return [
        # --- Valid: the production instance of the problem statement ---
        {"setup": base + "xmin, xmax, X = instance(1000, 25)\ncap, radius, M = 0.01, 0.01, 9\n",
         "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Valid: six-asset, four-observation projection ---
        {"setup": base + "xmin, xmax, X = instance(6, 4)\ncap, radius, M = 0.25, 0.01, 9\n",
         "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Valid: sixty assets, eight observations, intermediate cap ---
        {"setup": base + "xmin, xmax, X = instance(60, 8)\ncap, radius, M = 0.05, 0.02, 9\n",
         "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Boundary: two tangent lines only ---
        {"setup": base + "xmin, xmax, X = instance(6, 4)\ncap, radius, M = 0.25, 0.01, 2\n",
         "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Edge: radius well beyond the l1 diameter of the box ---
        {"setup": base + "xmin, xmax, X = instance(6, 4)\ncap, radius, M = 0.25, 3.0, 9\n",
         "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Edge: a single asset at a small radius ---
        {"setup": """import numpy as np
xmin, xmax = np.array([-0.3]), np.array([0.4])
X = np.array([[0.1], [-0.2], [0.3]])
cap, radius, M = 1.0, 0.05, 5
""", "call": call, "gold_call": gold, "tol": 1e-12},
        # --- Invalid: a single tangent line ---
        {"setup": base + """xmin, xmax, X = instance(6, 4)
cap, radius, M = 0.25, 0.01, 1
def run_model():
    try:
        surrogate_shortfall(xmin.copy(), xmax.copy(), X.copy(), cap, radius, M)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_surrogate_shortfall(xmin, xmax, X, cap, radius, M)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""", "call": "run_model()", "gold_call": "run_gold()"},
    ]
