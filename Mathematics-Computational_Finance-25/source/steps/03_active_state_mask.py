"""
Return the indicator of the grid states whose marginal mass reaches a prescribed fraction of the largest mass on the grid. Only these states carry a martingale row in the source's thresholded affine system.

Thresholding a marginal by a fraction of its peak keeps the bulk of the mass and drops the tail states, whose rows are numerically inert. The fraction is supplied by the caller.

Returns
-------
ndarray of shape (n,), float64: 1.0 on an active state, 0.0 otherwise.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def active_state_mask(w, mass_frac):
    """Return the indicator of the grid states whose marginal mass reaches a prescribed fraction of the largest mass on the grid. Only these states carry a martingale row in the source's thresholded affine system.

    Returns
    -------
    ndarray of shape (n,), float64: 1.0 on an active state, 0.0 otherwise.
    """
    return np.zeros(np.asarray(w, dtype=float).shape, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_active_state_mask(w, mass_frac):
    w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        raise ValueError("w must be 1-D")
    if np.any(w < 0.0):
        raise ValueError("w must be non-negative")
    peak = float(w.max()) if w.size else 0.0
    if not (peak > 0.0):
        raise ValueError("w has no positive peak")
    mass_frac = float(mass_frac)
    if not (0.0 <= mass_frac <= 1.0):
        raise ValueError("mass_frac must lie in [0, 1]")
    return (w >= mass_frac * peak).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nx = 0.25 + 0.0375*np.arange(41)\nw = np.exp(-((x-1.0)**2)/(2*0.15**2))\nw = w/w.sum()",
            "call": "active_state_mask(w, 0.01)",
            "gold_call": "_oracle_active_state_mask(w, 0.01)",
        },
        {
            "setup": "import numpy as np\nw = np.array([1.0, 0.5, 0.25])",
            "call": "active_state_mask(w, 1.0)",
            "gold_call": "_oracle_active_state_mask(w, 1.0)",
        },
        {
            "setup": "import numpy as np\nw = np.array([1.0, 0.5, 0.25])",
            "call": "active_state_mask(w, 0.0)",
            "gold_call": "_oracle_active_state_mask(w, 0.0)",
        },
    ]
