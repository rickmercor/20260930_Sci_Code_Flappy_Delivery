"""
Collapse the ensemble bookkeeping into one weight per barrier-side trajectory: the normaliser evaluated at that trajectory's highest ensemble index, multiplied by the sum of its unbiased sampling weights taken over every ensemble. The sum extends over all ensembles rather than stopping at the trajectory's own index, which is harmless because the weights vanish above it.

The point of the reweighting is that all of the complications, overlapping ensembles, fractional exchange counts and biased shooting, reduce to assigning a single number to each trajectory. Once these numbers exist, any path observable is an ordinary weighted average and no ensemble bookkeeping survives into the analysis. A useful check is that the weights of the barrier-side trajectories sum to unity, which holds only when the rescaling, the ensemble labels and the normalisers are all mutually consistent.

Returns
-------
np.ndarray of shape (P,) and dtype float, the per-trajectory reweighting weights
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def plus_path_weights(t: np.ndarray, ensemble_index: np.ndarray,
                      normalisers: np.ndarray) -> np.ndarray:
    '''Single reweighting weight for each barrier-side trajectory.

    Parameters
    ----------
    t : np.ndarray
        (P, n) unbiased sampling weights.
    ensemble_index : np.ndarray
        (P,) highest ensemble index of each trajectory.
    normalisers : np.ndarray
        (n,) cumulative weighted-histogram normalisers.

    Returns
    -------
    weights : np.ndarray
        (P,) weight of each trajectory, summing to 1 for a consistent record.
    
    Raises
    ------
    ValueError
        t must be a non-empty 2D array.
        ensemble_index must carry one entry per trajectory.
        ensemble_index must hold integers.
        normalisers must carry one entry per ensemble.
        ensemble_index entries must address the normalisers.
        t and normalisers must be finite.
    '''
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_plus_path_weights(t: np.ndarray, ensemble_index: np.ndarray,
                              normalisers: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    t = np.asarray(t, dtype=float)
    index = np.asarray(ensemble_index)
    norm = np.asarray(normalisers, dtype=float)
    if t.ndim != 2 or t.size == 0:
        raise ValueError("t must be a non-empty 2D array")
    if index.ndim != 1 or index.size != t.shape[0]:
        raise ValueError("ensemble_index must carry one entry per trajectory")
    if not np.issubdtype(index.dtype, np.integer):
        raise ValueError("ensemble_index must hold integers")
    if norm.ndim != 1 or norm.size != t.shape[1]:
        raise ValueError("normalisers must carry one entry per ensemble")
    if np.any(index < 0) or np.any(index >= norm.size):
        raise ValueError("ensemble_index entries must address the normalisers")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(norm)):
        raise ValueError("t and normalisers must be finite")
    return norm[index] * t.sum(axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nt = np.array([[2.0, 0.0, 0.0, 0.0],\n              [1.0, 3.0, 0.0, 0.0],\n              [4.0, 2.0, 4.0, 0.0],\n              [1.0, 3.0, 1.0, 3.0]])\nensemble_index = np.array([0, 1, 2, 3])\nnormalisers = np.array([0.05, 0.025, 0.0125, 0.00625])\n",
            "call": "plus_path_weights(t, ensemble_index, normalisers)",
            "gold_call": "_oracle_plus_path_weights(t, ensemble_index, normalisers)"
        },
        {
            "setup": "import numpy as np\n# boundary: every trajectory sits in the lowest ensemble\nt = np.array([[2.0, 0.0],\n              [3.0, 0.0]])\nensemble_index = np.array([0, 0])\nnormalisers = np.array([0.2, 0.1])\n",
            "call": "plus_path_weights(t, ensemble_index, normalisers)",
            "gold_call": "_oracle_plus_path_weights(t, ensemble_index, normalisers)"
        },
        {
            "setup": "import numpy as np\n# edge: a trajectory carrying no weight at all\nt = np.array([[0.0, 0.0],\n              [4.0, 1.0]])\nensemble_index = np.array([0, 1])\nnormalisers = np.array([0.25, 0.125])\n",
            "call": "plus_path_weights(t, ensemble_index, normalisers)",
            "gold_call": "_oracle_plus_path_weights(t, ensemble_index, normalisers)"
        },
        {
            "setup": "import numpy as np\nt = np.ones((2, 3))\nensemble_index = np.array([0, 3])        # index beyond the last ensemble\nnormalisers = np.array([0.5, 0.25, 0.125])\ndef run(f):\n    try:\n        f(t, ensemble_index, normalisers); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(plus_path_weights)",
            "gold_call": "run(_oracle_plus_path_weights)"
        },
        {
            "setup": "import numpy as np\nt = np.ones((2, 3))\nensemble_index = np.array([0.0, 1.0])    # not integers\nnormalisers = np.array([0.5, 0.25, 0.125])\ndef run(f):\n    try:\n        f(t, ensemble_index, normalisers); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(plus_path_weights)",
            "gold_call": "run(_oracle_plus_path_weights)"
        }
    ]
