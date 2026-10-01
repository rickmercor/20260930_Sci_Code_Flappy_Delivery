"""
Weight the reactant-side trajectories. This ensemble does not overlap any of the barrier-side ensembles, so no multi-ensemble combination is involved and no acceptance bias has to be removed: the weights are the visitation counts normalised to sum to unity.

The reactant-side ensemble collects trajectories confined behind the first interface. Because it shares no trajectories with the ensembles that probe the barrier, its weights decouple entirely from the weighted-histogram machinery, and only one weight per trajectory ever has to be evaluated across the whole record. Advanced high-acceptance shooting is normally not applied here, and no infinite-swap fractions arise, so the fractional visitation counts reduce to the plain multiplicities of an ordinary Markov chain.

Returns
-------
np.ndarray of shape (M,) and dtype float, weights summing to 1.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def minus_path_weights(minus_multiplicities: np.ndarray) -> np.ndarray:
    '''Normalised weights of the reactant-side trajectories.

    Parameters
    ----------
    minus_multiplicities : np.ndarray
        (M,) visitation count of each reactant-side trajectory.

    Returns
    -------
    weights : np.ndarray
        (M,) weights summing to 1.
    
    Raises
    ------
    ValueError
        minus_multiplicities must be a 1D array with at least one entry.
        minus_multiplicities must be finite.
        minus_multiplicities must be non-negative.
        the reactant-side ensemble must carry non-zero weight.
    '''
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_minus_path_weights(minus_multiplicities: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    m = np.asarray(minus_multiplicities, dtype=float)
    if m.ndim != 1 or m.size < 1:
        raise ValueError("minus_multiplicities must be a 1D array with at least one entry")
    if not np.all(np.isfinite(m)):
        raise ValueError("minus_multiplicities must be finite")
    if np.any(m < 0.0):
        raise ValueError("minus_multiplicities must be non-negative")
    total = m.sum()
    if total <= 0.0:
        raise ValueError("the reactant-side ensemble must carry non-zero weight")
    return m / total

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nminus_multiplicities = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])\n",
            "call": "minus_path_weights(minus_multiplicities)",
            "gold_call": "_oracle_minus_path_weights(minus_multiplicities)"
        },
        {
            "setup": "import numpy as np\n# boundary: equal multiplicities give a flat distribution\nminus_multiplicities = np.array([4.0, 4.0, 4.0, 4.0])\n",
            "call": "minus_path_weights(minus_multiplicities)",
            "gold_call": "_oracle_minus_path_weights(minus_multiplicities)"
        },
        {
            "setup": "import numpy as np\n# edge: a single trajectory carries the whole ensemble\nminus_multiplicities = np.array([9.0])\n",
            "call": "minus_path_weights(minus_multiplicities)",
            "gold_call": "_oracle_minus_path_weights(minus_multiplicities)"
        },
        {
            "setup": "import numpy as np\nminus_multiplicities = np.zeros(4)      # ensemble carries no weight\ndef run(f):\n    try:\n        f(minus_multiplicities); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(minus_path_weights)",
            "gold_call": "run(_oracle_minus_path_weights)"
        },
        {
            "setup": "import numpy as np\nminus_multiplicities = np.array([[1.0, 2.0]])   # 2D input\ndef run(f):\n    try:\n        f(minus_multiplicities); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(minus_path_weights)",
            "gold_call": "run(_oracle_minus_path_weights)"
        }
    ]
