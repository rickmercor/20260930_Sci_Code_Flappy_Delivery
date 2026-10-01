"""
Remove the sampling bias and place the ensembles on a common scale. Each trajectory is counted inversely to the acceptance weight it carried in a given ensemble, and the resulting column is then rescaled so that it sums back to that ensemble's own total visitation count. The second operation is not cosmetic: the combination rule that follows compares ensembles against one another, so their weights are only meaningful once fixed to a shared scale.

High-acceptance shooting moves improve decorrelation by drawing from a path distribution multiplied by an acceptance weight rather than from the physical path distribution itself. Any ensemble average taken from such a run must weight each sample by the reciprocal of that acceptance weight, otherwise the bias introduced to speed up sampling survives into the result. Path-ensemble averages are invariant under a global rescaling of these weights, which leaves the per-ensemble constant free; the reweighting fixes it by requiring the unbiased weights of an ensemble to sum to its total fractional trajectory count, so that they can be compared directly against the integer multiplicities of a synchronous simulation.

Returns
-------
np.ndarray of shape (P, n) and dtype float, the unbiased sampling weights
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def unbiased_sampling_weights(mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    '''Unbias the visitation counts and rescale them onto a common per-ensemble scale.

    Parameters
    ----------
    mu : np.ndarray
        (P, n) visitation count of each trajectory in each ensemble.
    w : np.ndarray
        (P, n) acceptance weight of each trajectory in each ensemble.

    Returns
    -------
    t : np.ndarray
        (P, n) unbiased sampling weights, each column summing to that ensemble's
        total visitation count.
    
    Raises
    ------
    ValueError
        mu must be a non-empty 2D array.
        mu and w must have the same shape.
        mu and w must be finite.
        mu entries must be non-negative.
        w must be strictly positive wherever mu is non-zero.
    '''
    return t

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_unbiased_sampling_weights(mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(w, dtype=float)
    if mu.ndim != 2 or mu.size == 0:
        raise ValueError("mu must be a non-empty 2D array")
    if mu.shape != w.shape:
        raise ValueError("mu and w must have the same shape")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(w)):
        raise ValueError("mu and w must be finite")
    if np.any(mu < 0.0):
        raise ValueError("mu entries must be non-negative")
    if np.any(w[mu > 0.0] <= 0.0):
        raise ValueError("w must be strictly positive wherever mu is non-zero")
    eta = mu.sum(axis=0)
    t = np.zeros_like(mu)
    for k in range(mu.shape[1]):
        safe = np.where(w[:, k] > 0.0, w[:, k], 1.0)
        ratio = np.where(mu[:, k] > 0.0, mu[:, k] / safe, 0.0)
        total = ratio.sum()
        if total > 0.0:
            t[:, k] = ratio * eta[k] / total
    return t

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nmu = np.array([[2.0, 0.0, 0.0],\n               [3.0, 3.0, 0.0],\n               [4.0, 2.0, 4.0],\n               [1.0, 1.0, 1.0]])\nw = np.array([[1.0, 1.0, 1.0],\n              [1.0, 2.5, 1.0],\n              [1.0, 1.75, 3.25],\n              [1.0, 1.0, 1.75]])\n",
            "call": "unbiased_sampling_weights(mu, w)",
            "gold_call": "_oracle_unbiased_sampling_weights(mu, w)"
        },
        {
            "setup": "import numpy as np\n# boundary: unit acceptance weights leave the counts untouched\nmu = np.array([[2.0, 1.0],\n               [3.0, 4.0]])\nw = np.ones((2, 2))\n",
            "call": "unbiased_sampling_weights(mu, w)",
            "gold_call": "_oracle_unbiased_sampling_weights(mu, w)"
        },
        {
            "setup": "import numpy as np\n# edge: an entirely empty ensemble column must stay at zero\nmu = np.array([[1.0, 0.0],\n               [2.0, 0.0]])\nw = np.array([[1.0, 3.0],\n              [2.0, 4.0]])\n",
            "call": "unbiased_sampling_weights(mu, w)",
            "gold_call": "_oracle_unbiased_sampling_weights(mu, w)"
        },
        {
            "setup": "import numpy as np\n# edge: a zero acceptance weight is tolerated where the count is zero\nmu = np.array([[1.0, 0.0],\n               [2.0, 3.0]])\nw = np.array([[2.0, 0.0],\n              [1.0, 1.5]])\n",
            "call": "unbiased_sampling_weights(mu, w)",
            "gold_call": "_oracle_unbiased_sampling_weights(mu, w)"
        },
        {
            "setup": "import numpy as np\nmu = np.array([[1.0, 2.0]])\nw = np.array([[1.0, 0.0]])       # zero weight where the count is non-zero\ndef run(f):\n    try:\n        f(mu, w); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(unbiased_sampling_weights)",
            "gold_call": "run(_oracle_unbiased_sampling_weights)"
        },
        {
            "setup": "import numpy as np\nmu = np.array([[1.0, 2.0]])\nw = np.array([[1.0, 2.0], [1.0, 2.0]])   # mismatched shapes\ndef run(f):\n    try:\n        f(mu, w); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(unbiased_sampling_weights)",
            "gold_call": "run(_oracle_unbiased_sampling_weights)"
        }
    ]
