"""
Run the forward recursion that turns the per-interface restricted totals into the probability of advancing from the reactant boundary to each successive interface. Two features of the recursion decide the result: only the ensembles whose index lies strictly below the target interface contribute to its estimate, and the summed restricted totals are scaled by the normaliser belonging to the previous step of the recursion rather than the current one. Return the full ladder of probabilities, the first of which is unity.

Ensembles at or above the target interface contain nothing but trajectories that already passed it, so they carry no information about the probability of reaching it and are dropped. The remaining ensembles are combined with weights that are inversely proportional to their own crossing probabilities, the usual weighted-histogram choice for minimising the variance of a multi-ensemble estimate. Because the normaliser at a given index itself depends on the crossing probability at that index, the recursion would be circular if the current normaliser were used, and the iteration instead advances by one index at a time, each probability being built from quantities already known. This step consumes the restricted totals rather than recomputing them, so the membership rule lives in exactly one place.

Returns
-------
np.ndarray of shape (n+1,) and dtype float, the crossing probabilities with probs[0] equal to 1.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def crossing_probabilities(crossing_totals: np.ndarray,
                           ensemble_totals: np.ndarray) -> np.ndarray:
    '''Ladder of crossing probabilities from the per-interface restricted totals.

    Parameters
    ----------
    crossing_totals : np.ndarray
        (n, n) restricted totals. Row i-1 holds, per ensemble, the unbiased weight
        carried by the trajectories that advanced beyond interface i, for i = 1..n.
    ensemble_totals : np.ndarray
        (n,) total visitation count of each ensemble.

    Returns
    -------
    probs : np.ndarray
        (n+1,) crossing probabilities, with probs[0] equal to 1.0.
    
    Raises
    ------
    ValueError
        ensemble_totals must be a non-empty 1D array.
        crossing_totals must be square with one row per interface.
        totals must be non-negative.
        the lowest ensemble must carry non-zero weight.
        crossing probability vanished before the last ensemble.
    '''
    return probs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_crossing_probabilities(crossing_totals: np.ndarray,
                                   ensemble_totals: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    crossing_totals = np.asarray(crossing_totals, dtype=float)
    eta = np.asarray(ensemble_totals, dtype=float)
    if eta.ndim != 1 or eta.size < 1:
        raise ValueError("ensemble_totals must be a non-empty 1D array")
    n = eta.size
    if crossing_totals.ndim != 2 or crossing_totals.shape != (n, n):
        raise ValueError("crossing_totals must be square with one row per interface")
    if np.any(eta < 0.0) or np.any(crossing_totals < 0.0):
        raise ValueError("totals must be non-negative")
    if eta[0] <= 0.0:
        raise ValueError("the lowest ensemble must carry non-zero weight")
    probs = np.ones(n + 1, dtype=float)
    norm = np.zeros(n, dtype=float)
    norm[0] = 1.0 / eta[0]
    for i in range(1, n + 1):
        probs[i] = norm[i - 1] * crossing_totals[i - 1, :i].sum()
        if i < n:
            if probs[i] <= 0.0:
                raise ValueError("crossing probability vanished before the last ensemble")
            norm[i] = 1.0 / np.sum(eta[: i + 1] / probs[: i + 1])
    return probs

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ncrossing_totals = np.array([[21.0,  0.0,  0.0,  0.0],\n                            [14.5, 11.5,  0.0,  0.0],\n                            [ 7.5,  6.0,  7.5,  0.0],\n                            [ 2.5,  2.0,  2.5,  3.0]])\nensemble_totals = np.array([24.0, 18.0, 9.0, 4.0])\n",
            "call": "crossing_probabilities(crossing_totals, ensemble_totals)",
            "gold_call": "_oracle_crossing_probabilities(crossing_totals, ensemble_totals)"
        },
        {
            "setup": "import numpy as np\n# boundary: every trajectory clears every interface, so the ladder stays at one\ncrossing_totals = np.array([[4.0, 0.0, 0.0],\n                            [4.0, 3.0, 0.0],\n                            [4.0, 3.0, 2.0]])\nensemble_totals = np.array([4.0, 3.0, 2.0])\n",
            "call": "crossing_probabilities(crossing_totals, ensemble_totals)",
            "gold_call": "_oracle_crossing_probabilities(crossing_totals, ensemble_totals)"
        },
        {
            "setup": "import numpy as np\n# edge: a single ensemble, so the ladder has one non-trivial entry\ncrossing_totals = np.array([[5.0]])\nensemble_totals = np.array([8.0])\n",
            "call": "crossing_probabilities(crossing_totals, ensemble_totals)",
            "gold_call": "_oracle_crossing_probabilities(crossing_totals, ensemble_totals)"
        },
        {
            "setup": "import numpy as np\ncrossing_totals = np.zeros((3, 3))\nensemble_totals = np.array([6.0, 4.0, 2.0])\ndef run(f):\n    try:\n        f(crossing_totals, ensemble_totals); return 0   # probability vanishes early\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(crossing_probabilities)",
            "gold_call": "run(_oracle_crossing_probabilities)"
        },
        {
            "setup": "import numpy as np\ncrossing_totals = np.array([[1.0, 0.0], [1.0, 1.0]])\nensemble_totals = np.array([0.0, 2.0])\ndef run(f):\n    try:\n        f(crossing_totals, ensemble_totals); return 0   # empty lowest ensemble\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(crossing_probabilities)",
            "gold_call": "run(_oracle_crossing_probabilities)"
        },
        {
            "setup": "import numpy as np\ncrossing_totals = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])\nensemble_totals = np.array([3.0, 2.0, 1.0])\ndef run(f):\n    try:\n        f(crossing_totals, ensemble_totals); return 0   # not one row per interface\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(crossing_probabilities)",
            "gold_call": "run(_oracle_crossing_probabilities)"
        }
    ]
