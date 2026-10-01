"""
Label each trajectory with the highest-indexed ensemble it belongs to. The label is fixed by where the trajectory's furthest progress falls relative to the interfaces, using an interval that excludes its lower interface and includes its upper one, with everything beyond the last sampled interface collapsed onto that final index. Trajectories whose maximum coincides exactly with an interface therefore take the lower of the two candidate labels.

The reweighting partitions trajectories by how far they advanced, because the contribution of a trajectory to the combined estimator is governed by the last ensemble that still contains it. Assigning that label is where an off-by-one is easiest to introduce, since the natural reading of an interface condition is strict on one side and inclusive on the other. Trajectories that pass the highest sampled interface are all given that highest index, because no ensemble beyond it was simulated and the interval there has no upper bound.

Returns
-------
np.ndarray of shape (P,) and integer dtype, each entry between 0 and n-1 inclusive
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def highest_ensemble_index(lam_max: np.ndarray, interfaces: np.ndarray) -> np.ndarray:
    '''Highest ensemble index that each trajectory still belongs to.

    Parameters
    ----------
    lam_max : np.ndarray
        (P,) furthest progress reached by each trajectory.
    interfaces : np.ndarray
        (n+1,) strictly increasing interface positions.

    Returns
    -------
    index : np.ndarray
        (P,) integer ensemble index in the range 0 to n-1.
    
    Raises
    ------
    ValueError
        lam_max must be a 1D array with at least one entry.
        interfaces must be a 1D array with at least three entries.
        interfaces must be strictly increasing.
        lam_max must be finite.
        every trajectory must pass the first interface.
    '''
    return index

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_highest_ensemble_index(lam_max: np.ndarray, interfaces: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    lam_max = np.asarray(lam_max, dtype=float)
    interfaces = np.asarray(interfaces, dtype=float)
    if lam_max.ndim != 1 or lam_max.size < 1:
        raise ValueError("lam_max must be a 1D array with at least one entry")
    if interfaces.ndim != 1 or interfaces.size < 3:
        raise ValueError("interfaces must be a 1D array with at least three entries")
    if np.any(np.diff(interfaces) <= 0.0):
        raise ValueError("interfaces must be strictly increasing")
    if not np.all(np.isfinite(lam_max)):
        raise ValueError("lam_max must be finite")
    if np.any(lam_max <= interfaces[0]):
        raise ValueError("every trajectory must pass the first interface")
    n = interfaces.size - 1
    index = np.empty(lam_max.size, dtype=int)
    for j in range(lam_max.size):
        if lam_max[j] > interfaces[n - 1]:
            index[j] = n - 1
        else:
            index[j] = int(np.searchsorted(interfaces[:n], lam_max[j], side="left")) - 1
    return index

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\nlam_max = np.array([-0.95, -0.18, 0.36, 0.84, 1.12])\n",
            "call": "highest_ensemble_index(lam_max, interfaces)",
            "gold_call": "_oracle_highest_ensemble_index(lam_max, interfaces)"
        },
        {
            "setup": "import numpy as np\n# boundary: maxima sitting exactly on interfaces take the lower label\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\nlam_max = np.array([-0.4, 0.2, 0.7])\n",
            "call": "highest_ensemble_index(lam_max, interfaces)",
            "gold_call": "_oracle_highest_ensemble_index(lam_max, interfaces)"
        },
        {
            "setup": "import numpy as np\n# edge: everything past the highest sampled interface collapses onto n-1\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\nlam_max = np.array([0.70001, 5.0, 1.0])\n",
            "call": "highest_ensemble_index(lam_max, interfaces)",
            "gold_call": "_oracle_highest_ensemble_index(lam_max, interfaces)"
        },
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, 0.2, 1.0])\nlam_max = np.array([-1.0])       # sits on the reactant boundary, in no ensemble\ndef run(f):\n    try:\n        f(lam_max, interfaces); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(highest_ensemble_index)",
            "gold_call": "run(_oracle_highest_ensemble_index)"
        },
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, 0.2, 1.0])\nlam_max = np.array([np.nan])     # non-finite maximum\ndef run(f):\n    try:\n        f(lam_max, interfaces); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(highest_ensemble_index)",
            "gold_call": "run(_oracle_highest_ensemble_index)"
        }
    ]
