"""
Return the slack in the conditional-Jensen inequality that any coupling meeting both prescribed marginals and the martingale rows on the active states would have to satisfy. A negative value certifies that the thresholded affine system is infeasible.

Applying conditional Jensen to the active martingale rows bounds the target dispersion below by a source-side quantity restricted to those states. Comparing the two sides decides feasibility without running any optimiser.

Returns
-------
float: target side minus source side. Negative certifies infeasibility.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def jensen_feasibility_gap(x, w_source, w_target, mask, centre):
    """Return the slack in the conditional-Jensen inequality that any coupling meeting both prescribed marginals and the martingale rows on the active states would have to satisfy. A negative value certifies that the thresholded affine system is infeasible.

    Returns
    -------
    float: target side minus source side. Negative certifies infeasibility.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_jensen_feasibility_gap(x, w_source, w_target, mask, centre):
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=float)
    if mask.shape != x.shape:
        raise ValueError("mask must match the grid shape")
    if np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("mask must be an indicator of zeros and ones")
    lhs = _oracle_discrete_second_moment(x, w_target, centre)
    w_src = np.asarray(w_source, dtype=float)
    if w_src.shape != x.shape:
        raise ValueError("w_source must match the grid shape")
    tot = float(w_src.sum())
    if not (tot > 0.0):
        raise ValueError("w_source must carry positive total mass")
    centre = float(centre)
    rhs = float(((w_src / tot) * mask * (x - centre) ** 2).sum())
    return float(lhs - rhs)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nx = 0.25 + 0.0375*np.arange(41)\na = np.exp(-((x-1.0)**2)/(2*0.15**2)); a/=a.sum()\nsy = 0.15*np.sqrt(0.70)\nb = np.exp(-((x-1.0)**2)/(2*sy**2)); b/=b.sum()\nm = (a >= 0.01*a.max()).astype(float)",
            "call": "jensen_feasibility_gap(x, a, b, m, 1.0)",
            "gold_call": "_oracle_jensen_feasibility_gap(x, a, b, m, 1.0)",
        },
        {
            "setup": "import numpy as np\nx = 0.25 + 0.0375*np.arange(41)\na = np.exp(-((x-1.0)**2)/(2*0.15**2)); a/=a.sum()\nsy = 0.15*np.sqrt(1.30)\nb = np.exp(-((x-1.0)**2)/(2*sy**2)); b/=b.sum()\nm = (a >= 0.01*a.max()).astype(float)",
            "call": "jensen_feasibility_gap(x, a, b, m, 1.0)",
            "gold_call": "_oracle_jensen_feasibility_gap(x, a, b, m, 1.0)",
        },
        {
            "setup": "import numpy as np\nx = np.array([0.0, 1.0, 2.0])\na = np.array([0.25, 0.5, 0.25]); b = np.array([0.25, 0.5, 0.25])\nm = np.zeros(3)",
            "call": "jensen_feasibility_gap(x, a, b, m, 1.0)",
            "gold_call": "_oracle_jensen_feasibility_gap(x, a, b, m, 1.0)",
        },
    ]
