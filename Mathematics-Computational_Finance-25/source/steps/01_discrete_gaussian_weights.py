"""
Return the normalised discrete weight vector that an unnormalised Gaussian kernel induces on a prescribed one-dimensional grid. This supplies the two prescribed marginals of the source's controlled feasibility instance.

A discrete marginal on a finite grid is obtained by evaluating a kernel at the nodes and normalising to unit total mass. The grid is supplied by the caller and is not assumed uniform.

Returns
-------
ndarray of shape (n,), float64: the normalised weights, summing to one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def discrete_gaussian_weights(x, sigma, centre):
    """Return the normalised discrete weight vector that an unnormalised Gaussian kernel induces on a prescribed one-dimensional grid. This supplies the two prescribed marginals of the source's controlled feasibility instance.

    Returns
    -------
    ndarray of shape (n,), float64: the normalised weights, summing to one.
    """
    return np.zeros(np.asarray(x, dtype=float).shape, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_discrete_gaussian_weights(x, sigma, centre):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1-D grid with at least two nodes")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    sigma = float(sigma)
    if not (sigma > 0.0) or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and strictly positive")
    centre = float(centre)
    if not np.isfinite(centre):
        raise ValueError("centre must be finite")
    w = np.exp(-((x - centre) ** 2) / (2.0 * sigma * sigma))
    s = float(w.sum())
    if not (s > 0.0):
        raise ValueError("weights underflowed to zero mass")
    return w / s

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nx = 0.25 + 0.0375*np.arange(41)",
            "call": "discrete_gaussian_weights(x, 0.15, 1.0)",
            "gold_call": "_oracle_discrete_gaussian_weights(x, 0.15, 1.0)",
        },
        {
            "setup": "import numpy as np\nx = np.array([0.0, 1.0])",
            "call": "discrete_gaussian_weights(x, 0.5, 0.5)",
            "gold_call": "_oracle_discrete_gaussian_weights(x, 0.5, 0.5)",
        },
        {
            "setup": "import numpy as np\nx = np.linspace(-3.0, 3.0, 7)",
            "call": "discrete_gaussian_weights(x, 0.02, 0.0)",
            "gold_call": "_oracle_discrete_gaussian_weights(x, 0.02, 0.0)",
        },
    ]
