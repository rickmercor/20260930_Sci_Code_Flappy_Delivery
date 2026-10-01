"""
Form the per-index normalisers of the multi-ensemble combination. The normaliser at index i is the reciprocal of a cumulative sum running over ensembles 0 through i, in which each ensemble total is divided by the crossing probability of its own interface. The lowest normaliser is therefore the reciprocal of the lowest ensemble total.

When several ensembles report on the same quantity, the weighted-histogram argument says the variance is minimised by weighting each ensemble by its sample count divided by the probability of the region it is reporting on. Applied to interface ensembles the region weight is the crossing probability of that ensemble's own interface, so an ensemble that rarely reaches its interface is upweighted relative to its raw sample count. Collecting the reciprocal of that cumulative sum into a single factor per index is what allows the final path weight to be written as one number per trajectory.

Returns
-------
np.ndarray of shape (n,) and dtype float, the cumulative normalisers
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def wham_normalisers(eta: np.ndarray, crossing_probs: np.ndarray) -> np.ndarray:
    '''Cumulative weighted-histogram normalisers, one per ensemble index.

    Parameters
    ----------
    eta : np.ndarray
        (n,) total fractional trajectory count per ensemble.
    crossing_probs : np.ndarray
        (n+1,) crossing probabilities, the first entry equal to 1.

    Returns
    -------
    norm : np.ndarray
        (n,) normaliser for each ensemble index.
    
    Raises
    ------
    ValueError
        eta must be a 1D array with at least one entry.
        crossing_probs must hold exactly one more entry than eta.
        eta and crossing_probs must be finite.
        eta entries must be non-negative.
        crossing probabilities must be strictly positive.
        cumulative normalising sum must be strictly positive.
    '''
    return norm

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_wham_normalisers(eta: np.ndarray, crossing_probs: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    eta = np.asarray(eta, dtype=float)
    probs = np.asarray(crossing_probs, dtype=float)
    if eta.ndim != 1 or eta.size < 1:
        raise ValueError("eta must be a 1D array with at least one entry")
    if probs.ndim != 1 or probs.size != eta.size + 1:
        raise ValueError("crossing_probs must hold exactly one more entry than eta")
    if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(probs)):
        raise ValueError("eta and crossing_probs must be finite")
    if np.any(eta < 0.0):
        raise ValueError("eta entries must be non-negative")
    if np.any(probs[: eta.size] <= 0.0):
        raise ValueError("crossing probabilities must be strictly positive")
    n = eta.size
    norm = np.empty(n, dtype=float)
    for i in range(n):
        total = np.sum(eta[: i + 1] / probs[: i + 1])
        if total <= 0.0:
            raise ValueError("cumulative normalising sum must be strictly positive")
        norm[i] = 1.0 / total
    return norm

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\neta = np.array([24.0, 18.0, 9.0, 4.0])\ncrossing_probs = np.array([1.0, 0.65, 0.4, 0.18, 0.05])\n",
            "call": "wham_normalisers(eta, crossing_probs)",
            "gold_call": "_oracle_wham_normalisers(eta, crossing_probs)"
        },
        {
            "setup": "import numpy as np\n# boundary: probabilities all unity reduce the normaliser to a plain running reciprocal\neta = np.array([4.0, 6.0, 10.0])\ncrossing_probs = np.ones(4)\n",
            "call": "wham_normalisers(eta, crossing_probs)",
            "gold_call": "_oracle_wham_normalisers(eta, crossing_probs)"
        },
        {
            "setup": "import numpy as np\n# edge: a single ensemble\neta = np.array([12.5])\ncrossing_probs = np.array([1.0, 0.25])\n",
            "call": "wham_normalisers(eta, crossing_probs)",
            "gold_call": "_oracle_wham_normalisers(eta, crossing_probs)"
        },
        {
            "setup": "import numpy as np\neta = np.array([10.0, 5.0])\ncrossing_probs = np.array([1.0, 0.0, 0.1])      # a vanishing crossing probability\ndef run(f):\n    try:\n        f(eta, crossing_probs); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(wham_normalisers)",
            "gold_call": "run(_oracle_wham_normalisers)"
        },
        {
            "setup": "import numpy as np\neta = np.array([10.0, 5.0])\ncrossing_probs = np.array([1.0, 0.5])           # wrong length\ndef run(f):\n    try:\n        f(eta, crossing_probs); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(wham_normalisers)",
            "gold_call": "run(_oracle_wham_normalisers)"
        }
    ]
