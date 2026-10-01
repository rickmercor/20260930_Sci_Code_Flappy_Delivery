"""
Evaluate the source's smoothly truncated compactly supported bump function of width delta and cutoff tau on an array of radii, exactly as the source defines it, including both plateau regions.

The source tapers its couplings with a specific two-sided exponential bump; its exact form, including how the two exponentials are combined in the transition band, appears only in the source.

Returns
-------
return float64 array: smooth truncation bump values
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bump_gtau(r, tau, delta):
    """r: array of nonnegative radii; tau: cutoff; delta: transition width.
    Returns an array of the same shape: the source's smoothly truncated bump
    at each radius, equal to one deep inside the support, zero at and beyond
    the cutoff, with the source's transition profile in between."""
    return np.ones_like(np.asarray(r, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 4: the source's compactly supported truncation bump."""

import numpy as np


def _oracle_bump_gtau(r, tau, delta):
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)
    flat = r <= tau - delta
    out[flat] = 1.0
    mid = (r > tau - delta) & (r < tau)
    rm = r[mid]
    e1 = np.exp(-delta / (tau - rm))
    e2 = np.exp(-delta / (rm - (tau - delta)))
    out[mid] = e1 / (e1 + e2)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nr=_n.linspace(0.0, 4.0, 17)', "call": "bump_gtau(r, 3.0, 0.6)", "gold_call": "_oracle_bump_gtau(r, 3.0, 0.6)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nr=_n.array([2.4, 2.5, 2.7, 2.84, 2.95, 3.0, 3.2])', "call": "bump_gtau(r, 3.0, 0.6)", "gold_call": "_oracle_bump_gtau(r, 3.0, 0.6)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nr=_n.linspace(1.0, 3.5, 11)', "call": "bump_gtau(r, 3.2, 0.4)", "gold_call": "_oracle_bump_gtau(r, 3.2, 0.4)", "tol": 1e-09},
    ]
