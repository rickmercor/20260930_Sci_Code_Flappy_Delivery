"""
Reduce the visitation counts to one total per ensemble. Because the counts are fractional, these totals play the role that the plain number of sampled paths plays in a synchronous simulation, and they set the scale against which every later weight is normalised.

In a conventional interface-sampling run each ensemble is advanced the same number of Monte Carlo steps, so the number of samples per ensemble is common knowledge and cancels from most expressions. Once ensembles are advanced asynchronously the sample counts differ, shorter-path ensembles accumulating more samples than long-path ones, and the totals have to be carried explicitly through the analysis. Summing the fractional visitation counts down each ensemble column gives the effective number of trajectories that ensemble contributed.

Returns
-------
np.ndarray of shape (n,) and dtype float, the column sums of the visitation counts
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ensemble_path_totals(mu: np.ndarray) -> np.ndarray:
    '''Total fractional number of trajectories contributed by each ensemble.

    Parameters
    ----------
    mu : np.ndarray
        (P, n) visitation count of each trajectory in each ensemble.

    Returns
    -------
    eta : np.ndarray
        (n,) total fractional trajectory count per ensemble.
    
    Raises
    ------
    ValueError
        mu must be a non-empty 2D array.
        mu entries must be finite.
        mu entries must be non-negative.
    '''
    return eta

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ensemble_path_totals(mu: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    mu = np.asarray(mu, dtype=float)
    if mu.ndim != 2 or mu.size == 0:
        raise ValueError("mu must be a non-empty 2D array")
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu entries must be finite")
    if np.any(mu < 0.0):
        raise ValueError("mu entries must be non-negative")
    return mu.sum(axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nmu = np.array([[2.0, 0.0, 0.0],\n               [3.0, 1.5, 0.0],\n               [4.0, 2.0, 1.25]])\n",
            "call": "ensemble_path_totals(mu)",
            "gold_call": "_oracle_ensemble_path_totals(mu)"
        },
        {
            "setup": "import numpy as np\n# boundary: an ensemble that no trajectory ever visited\nmu = np.array([[1.0, 0.0],\n               [2.0, 0.0]])\n",
            "call": "ensemble_path_totals(mu)",
            "gold_call": "_oracle_ensemble_path_totals(mu)"
        },
        {
            "setup": "import numpy as np\n# edge: one trajectory, one ensemble\nmu = np.array([[7.5]])\n",
            "call": "ensemble_path_totals(mu)",
            "gold_call": "_oracle_ensemble_path_totals(mu)"
        },
        {
            "setup": "import numpy as np\nmu = np.array([[1.0, -2.0]])     # negative visitation count\ndef run(f):\n    try:\n        f(mu); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(ensemble_path_totals)",
            "gold_call": "run(_oracle_ensemble_path_totals)"
        },
        {
            "setup": "import numpy as np\nmu = np.array([1.0, 2.0, 3.0])   # 1D instead of 2D\ndef run(f):\n    try:\n        f(mu); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(ensemble_path_totals)",
            "gold_call": "run(_oracle_ensemble_path_totals)"
        }
    ]
