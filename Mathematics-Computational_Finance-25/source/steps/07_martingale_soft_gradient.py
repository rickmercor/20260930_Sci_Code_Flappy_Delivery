"""
Return the gradient of the quadratic penalty on the martingale rows of the active states, evaluated at the current coupling. Inactive states contribute nothing.

Each active state carries one martingale row whose residual is the mass-weighted mean displacement out of that state. A quadratic penalty on those residuals has a gradient supported on the same rows.

Returns
-------
ndarray with the same shape as P, float64: the penalty gradient.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def martingale_soft_gradient(P, x, mask, penalty):
    """Return the gradient of the quadratic penalty on the martingale rows of the active states, evaluated at the current coupling. Inactive states contribute nothing.

    Returns
    -------
    ndarray with the same shape as P, float64: the penalty gradient.
    """
    return np.zeros(np.asarray(P, dtype=float).shape, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_martingale_soft_gradient(P, x, mask, penalty):
    P = np.asarray(P, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square 2-D coupling")
    if x.shape != (P.shape[0],) or mask.shape != x.shape:
        raise ValueError("x and mask must match the coupling dimension")
    penalty = float(penalty)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("penalty must be finite and non-negative")
    d = x[None, :] - x[:, None]
    resid = (P * d).sum(axis=1) * mask
    return (penalty * resid)[:, None] * d * mask[:, None]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nx = 0.25 + 0.0375*np.arange(41)\na = np.exp(-((x-1.0)**2)/(2*0.15**2)); a/=a.sum()\nb = np.exp(-((x-1.0)**2)/(2*(0.15*np.sqrt(0.70))**2)); b/=b.sum()\nP = np.outer(a, b); P/=P.sum()\nm = (a >= 0.01*a.max()).astype(float)",
            "call": "martingale_soft_gradient(P, x, m, 200.0)",
            "gold_call": "_oracle_martingale_soft_gradient(P, x, m, 200.0)",
        },
        {
            "setup": "import numpy as np\nx = np.array([0.0, 1.0, 2.0])\nP = np.full((3, 3), 1.0/9.0)\nm = np.ones(3)",
            "call": "martingale_soft_gradient(P, x, m, 200.0)",
            "gold_call": "_oracle_martingale_soft_gradient(P, x, m, 200.0)",
        },
        {
            "setup": "import numpy as np\nx = np.array([0.0, 1.0, 2.0])\nP = np.full((3, 3), 1.0/9.0)\nm = np.zeros(3)",
            "call": "martingale_soft_gradient(P, x, m, 200.0)",
            "gold_call": "_oracle_martingale_soft_gradient(P, x, m, 200.0)",
        },
    ]
