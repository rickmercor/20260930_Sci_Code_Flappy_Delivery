"""
Evaluate, on a grid of asset prices, the two parts of the drift rate mu - r G of the American put's gain process, with the local time at the strike regularised by a Gaussian kernel.

The gain of the put is G = (K - S)^+ and the underlying follows dS = r S dt + b S dW under the pricing measure. The gain is not differentiable at the strike, so its semimartingale drift mu comes from the Tanaka-Meyer formula and contains a local-time term at S = K. The drift rate needed by the additive forward representation is mu - r G, split into the part that does not involve local time and the local-time part. The local-time density is regularised as the source does: the Dirac mass at the strike is replaced by a Gaussian density of standard deviation eps in price units, and the quadratic-variation rate multiplying it is evaluated at the strike, b^2 K^2. On a grid node that coincides with the strike, any indicator of {S < K} takes its midpoint value 1/2.

Returns
-------
np.ndarray, float, shape (2, n): row 0 the part of mu - r G without local time, row 1 the local-time part, at the n grid prices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tanaka_drift_rate(s_grid: np.ndarray, strike: float, rate: float, vol: float, eps: float) -> np.ndarray:
    '''Drift rate mu - r G of the put gain, split into its two parts.

    Parameters
    ----------
    s_grid : np.ndarray
        One-dimensional array of n >= 1 finite, non-negative asset prices.
    strike : float
        Strike K > 0.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b >= 0.
    eps : float
        Standard deviation of the Gaussian kernel, in price units, eps > 0.

    Returns
    -------
    parts : np.ndarray
        Shape (2, n) float array: row 0 the part without local time, row 1
        the regularised local-time part.

    Raises
    ------
    ValueError
        If s_grid is not a finite, non-negative one-dimensional array with at
        least one entry, or if strike, rate, vol or eps is outside its domain.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs inside the function body.
    '''
    return np.zeros((2, np.atleast_1d(s_grid).size), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_tanaka_drift_rate(s_grid: np.ndarray, strike: float, rate: float, vol: float, eps: float) -> np.ndarray:
    import numpy as np

    s = np.atleast_1d(np.asarray(s_grid, dtype=float))
    if s.ndim != 1 or s.size < 1 or not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("s_grid must be a finite, non-negative one-dimensional array")
    for name, value, low in (("strike", strike, 0.0), ("eps", eps, 0.0)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= low:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(rate, bool) or not np.isfinite(float(rate)):
        raise ValueError("rate must be finite")
    if isinstance(vol, bool) or not np.isfinite(float(vol)) or float(vol) < 0.0:
        raise ValueError("vol must be finite and non-negative")
    k, r, b, e = float(strike), float(rate), float(vol), float(eps)

    # Tanaka-Meyer: d(K - S)^+ = -1{S<K} dS + (1/2) dL^K, so with dS = r S dt + ...
    # mu - r G = -r S 1{S<K} - r (K - S)^+ + (1/2) dL/dt = -r K 1{S<K} + (1/2) dL/dt.
    below = np.where(s < k, 1.0, np.where(s == k, 0.5, 0.0))
    no_local_time = -r * k * below
    # Local-time density: Gaussian kernel at the strike times b^2 K^2.
    kernel = np.exp(-(s - k) ** 2 / (2.0 * e * e)) / (np.sqrt(2.0 * np.pi) * e)
    local_time = 0.5 * b * b * k * k * kernel
    return np.vstack([no_local_time, local_time]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: at the strike the indicator is 1/2, so the part without
        # local time is -rK/2, and the kernel peaks at 1/(sqrt(2 pi) eps);
        # far below the strike the part without local time is -rK and the
        # kernel has vanished to below 1e-300.
        {
            "setup": """import numpy as np
K, r, b, eps = 100.0, 0.05, 0.2, 0.5
s = np.array([K, 50.0, 150.0])
peak = 0.5 * b * b * K * K / (np.sqrt(2.0 * np.pi) * eps)
EXPECTED = np.array([[-0.5 * r * K, -r * K, 0.0], [peak, 0.0, 0.0]])
""",
            "call": "tanaka_drift_rate(s, K, r, b, eps)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: a uniform grid around the strike.
        {
            "setup": """import numpy as np
s = np.linspace(95.0, 105.0, 41)
""",
            "call": "tanaka_drift_rate(s, 100.0, 0.05, 0.2, 0.5)",
            "gold_call": "_oracle_tanaka_drift_rate(s, 100.0, 0.05, 0.2, 0.5)",
        },
        # --- Boundary: zero volatility removes the local-time part entirely.
        {
            "setup": """import numpy as np
s = np.array([0.0, 20.0, 40.0, 60.0])
""",
            "call": "tanaka_drift_rate(s, 40.0, 0.03, 0.0, 1.0)",
            "gold_call": "_oracle_tanaka_drift_rate(s, 40.0, 0.03, 0.0, 1.0)",
        },
        # --- Edge: a very narrow kernel with a negative rate, off the grid nodes.
        {
            "setup": """import numpy as np
s = np.array([9.99, 10.0, 10.004, 10.02])
""",
            "call": "tanaka_drift_rate(s, 10.0, -0.01, 0.6, 0.005)",
            "gold_call": "_oracle_tanaka_drift_rate(s, 10.0, -0.01, 0.6, 0.005)",
        },
        # --- Invalid: non-positive kernel width ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        tanaka_drift_rate(np.array([100.0]), 100.0, 0.05, 0.2, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_tanaka_drift_rate(np.array([100.0]), 100.0, 0.05, 0.2, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative price on the grid ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        tanaka_drift_rate(np.array([-1.0, 100.0]), 100.0, 0.05, 0.2, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_tanaka_drift_rate(np.array([-1.0, 100.0]), 100.0, 0.05, 0.2, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
