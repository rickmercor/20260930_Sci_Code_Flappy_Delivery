"""
Return the worst absolute violation of the marginal family and the worst absolute violation of the martingale family on the active states, for a given coupling.

Reporting the two families separately is what makes the priority allocation visible: a scheme that tightens one family necessarily loosens the other once the system is incompatible.

Returns
-------
ndarray of shape (2,), float64: the marginal-family violation followed by the martingale-family violation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def residual_pair(P, x, w_source, w_target, mask):
    """Return the worst absolute violation of the marginal family and the worst absolute violation of the martingale family on the active states, for a given coupling.

    Returns
    -------
    ndarray of shape (2,), float64: the marginal-family violation followed by the martingale-family violation.
    """
    return np.zeros(2, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_residual_pair(P, x, w_source, w_target, mask):
    P = np.asarray(P, dtype=float)
    x = np.asarray(x, dtype=float)
    a = np.asarray(w_source, dtype=float)
    b = np.asarray(w_target, dtype=float)
    mask = np.asarray(mask, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square 2-D coupling")
    if a.shape != (P.shape[0],) or b.shape != (P.shape[0],) or mask.shape != a.shape:
        raise ValueError("marginals, mask and grid must match the coupling dimension")
    r_marg = max(float(np.abs(P.sum(axis=1) - a).max()),
                 float(np.abs(P.sum(axis=0) - b).max()))
    d = x[None, :] - x[:, None]
    row = np.abs((P * d).sum(axis=1)) * mask
    r_cond = float(row.max()) if mask.any() else 0.0
    return np.array([r_marg, r_cond], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nx = 0.25 + 0.0375*np.arange(41)\na = np.exp(-((x-1.0)**2)/(2*0.15**2)); a/=a.sum()\nb = np.exp(-((x-1.0)**2)/(2*(0.15*np.sqrt(0.70))**2)); b/=b.sum()\nP = np.outer(a, b); P/=P.sum()\nm = (a >= 0.01*a.max()).astype(float)",
            "call": "residual_pair(P, x, a, b, m)",
            "gold_call": "_oracle_residual_pair(P, x, a, b, m)",
        },
        {
            "setup": "import numpy as np\nx = np.array([0.0, 1.0])\na = np.array([0.5, 0.5]); b = np.array([0.5, 0.5])\nP = np.array([[0.5, 0.0], [0.0, 0.5]])\nm = np.ones(2)",
            "call": "residual_pair(P, x, a, b, m)",
            "gold_call": "_oracle_residual_pair(P, x, a, b, m)",
        },
        {
            "setup": "import numpy as np\nx = np.array([0.0, 1.0, 2.0])\na = np.array([0.2, 0.5, 0.3]); b = np.array([0.3, 0.4, 0.3])\nP = np.full((3, 3), 1.0/9.0)\nm = np.zeros(3)",
            "call": "residual_pair(P, x, a, b, m)",
            "gold_call": "_oracle_residual_pair(P, x, a, b, m)",
        },
    ]
