"""
Return the mass-weighted mean squared deviation of a grid about a given centre, using a weight vector that need not already be normalised.

The dispersion of a discrete law about a reference point is the weighted average of squared deviations. Weights are renormalised internally so that an unnormalised input gives the same result as a normalised one.

Returns
-------
float: the mass-weighted mean squared deviation about centre.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def discrete_second_moment(x, w, centre):
    """Return the mass-weighted mean squared deviation of a grid about a given centre, using a weight vector that need not already be normalised.

    Returns
    -------
    float: the mass-weighted mean squared deviation about centre.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_discrete_second_moment(x, w, centre):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.shape != w.shape:
        raise ValueError("x and w must have the same shape")
    if np.any(w < 0.0):
        raise ValueError("w must be non-negative")
    tot = float(w.sum())
    if not (tot > 0.0):
        raise ValueError("w must carry positive total mass")
    centre = float(centre)
    return float((w * (x - centre) ** 2).sum() / tot)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nx = 0.25 + 0.0375*np.arange(41)\nw = np.exp(-((x-1.0)**2)/(2*0.15**2))",
            "call": "discrete_second_moment(x, w, 1.0)",
            "gold_call": "_oracle_discrete_second_moment(x, w, 1.0)",
        },
        {
            "setup": "import numpy as np\nx = np.array([2.0, 2.0, 2.0])\nw = np.array([1.0, 1.0, 1.0])",
            "call": "discrete_second_moment(x, w, 2.0)",
            "gold_call": "_oracle_discrete_second_moment(x, w, 2.0)",
        },
        {
            "setup": "import numpy as np\nx = np.array([0.0, 10.0])\nw = np.array([1.0, 1e-12])",
            "call": "discrete_second_moment(x, w, 0.0)",
            "gold_call": "_oracle_discrete_second_moment(x, w, 0.0)",
        },
    ]
