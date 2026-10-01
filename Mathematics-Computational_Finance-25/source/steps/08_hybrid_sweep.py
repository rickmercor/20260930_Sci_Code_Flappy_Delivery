"""
Advance the coupling by one sweep of the priority-split calibration scheme the source introduces, in which the two marginal families and the martingale family are given different priorities. The arrangement within a sweep, and how the relaxed family is driven, are fixed by the source.

A finite-budget scheme must decide which constraint family is satisfied tightly and which absorbs the incompatibility. The scheme here treats the marginal families and the martingale family differently, and the resulting allocation is the source's central design point.

Returns
-------
ndarray with the same shape as P, float64: the coupling after one sweep, normalised to unit total mass.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hybrid_sweep(P, x, w_source, w_target, mask, hard_cap, penalty, base_step, soft_cap):
    """Advance the coupling by one sweep of the priority-split calibration scheme the source introduces, in which the two marginal families and the martingale family are given different priorities. The arrangement within a sweep, and how the relaxed family is driven, are fixed by the source.

    Returns
    -------
    ndarray with the same shape as P, float64: the coupling after one sweep, normalised to unit total mass.
    """
    return np.array(P, dtype=float, copy=True)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_hybrid_sweep(P, x, w_source, w_target, mask, hard_cap, penalty,
                         base_step, soft_cap):
    P = np.array(P, dtype=float, copy=True)
    x = np.asarray(x, dtype=float)
    a = np.asarray(w_source, dtype=float)
    b = np.asarray(w_target, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square 2-D coupling")
    if a.shape != (P.shape[0],) or b.shape != (P.shape[0],):
        raise ValueError("marginals must match the coupling dimension")
    base_step = float(base_step)
    soft_cap = float(soft_cap)
    if not (base_step > 0.0) or not (soft_cap > 0.0):
        raise ValueError("base_step and soft_cap must be strictly positive")
    n = P.shape[0]
    # L5: the source-marginal family is visited first, then the target family.
    # L4: total mass is restored after EVERY row, not once at the end of the sweep.
    for j in range(n):
        P = _oracle_capped_marginal_tilt(P, 1, j, float(a[j]), hard_cap)
        P /= float(P.sum())
    for k in range(n):
        P = _oracle_capped_marginal_tilt(P, 0, k, float(b[k]), hard_cap)
        P /= float(P.sum())
    # L2: exactly ONE batch exponentiated-gradient update follows the hard sweep.
    # No dual ascent: the multiplier stays at zero throughout.
    g = _oracle_martingale_soft_gradient(P, x, mask, penalty)
    gmax = float(np.abs(g).max())
    eta = min(base_step, soft_cap / gmax) if gmax > 0.0 else 0.0
    P *= np.exp(np.clip(-eta * g, -700.0, 700.0))
    return P / float(P.sum())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nx = 0.25 + 0.0375*np.arange(41)\na = np.exp(-((x-1.0)**2)/(2*0.15**2)); a/=a.sum()\nb = np.exp(-((x-1.0)**2)/(2*(0.15*np.sqrt(0.70))**2)); b/=b.sum()\nP = np.outer(a, b); P/=P.sum()\nm = (a >= 0.01*a.max()).astype(float)",
            "call": "hybrid_sweep(P, x, a, b, m, 0.02, 200.0, 0.5, 1.0)",
            "gold_call": "_oracle_hybrid_sweep(P, x, a, b, m, 0.02, 200.0, 0.5, 1.0)",
        },
        {
            "setup": "import numpy as np\nx = np.array([0.0, 1.0, 2.0])\na = np.array([0.2, 0.5, 0.3]); b = np.array([0.3, 0.4, 0.3])\nP = np.outer(a, b); P/=P.sum()\nm = np.ones(3)",
            "call": "hybrid_sweep(P, x, a, b, m, 5.0, 1.0, 0.5, 1.0)",
            "gold_call": "_oracle_hybrid_sweep(P, x, a, b, m, 5.0, 1.0, 0.5, 1.0)",
        },
        {
            "setup": "import numpy as np\nx = np.array([0.0, 1.0, 2.0])\na = np.array([0.2, 0.5, 0.3]); b = np.array([0.3, 0.4, 0.3])\nP = np.outer(a, b); P/=P.sum()\nm = np.zeros(3)",
            "call": "hybrid_sweep(P, x, a, b, m, 0.02, 200.0, 0.5, 1.0)",
            "gold_call": "_oracle_hybrid_sweep(P, x, a, b, m, 0.02, 200.0, 0.5, 1.0)",
        },
    ]
