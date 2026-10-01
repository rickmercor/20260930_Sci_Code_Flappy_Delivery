"""
Build the per-ensemble bookkeeping of the simulation record. A trajectory belongs to the ensemble associated with interface k only when its furthest progress along the coordinate lies strictly beyond that interface, so the visitation counts form a staircase pattern that is non-zero only for the low-lying ensembles of trajectories that did not travel far. Return both the visitation counts and the acceptance weights of the biased shooting moves, stacked into one array, using the construction rules given in the problem.

Asynchronous interface sampling emulates an unlimited number of replica exchanges among the ensembles that no worker is currently occupying. A trajectory therefore accumulates a non-integer visitation count in every ensemble it could have occupied, rather than belonging to exactly one. Alongside this, advanced shooting moves sample a deliberately biased path distribution, and each trajectory carries a per-ensemble acceptance weight that has to be divided out later. Membership itself is governed by the minimal progress condition of interface sampling: a path counts towards the ensemble of interface k only if the maximum of the order parameter along it exceeds that interface.

Returns
-------
np.ndarray of shape (2, P, n) and dtype float, the visitation counts stacked on top of the acceptance weights
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def sampling_record(lam_max: np.ndarray, interfaces: np.ndarray) -> np.ndarray:
    '''Assemble visitation counts and acceptance weights for every trajectory and ensemble.

    Parameters
    ----------
    lam_max : np.ndarray
        (P,) furthest progress along the order parameter reached by each trajectory.
    interfaces : np.ndarray
        (n+1,) strictly increasing interface positions, the first being the reactant
        boundary and the last the product boundary.

    Returns
    -------
    record : np.ndarray
        (2, P, n) array whose first slab holds the visitation counts and whose second
        slab holds the acceptance weights.
    
    Raises
    ------
    ValueError
        lam_max must be a 1D array with at least one entry.
        interfaces must be a 1D array with at least three entries.
        interfaces must be strictly increasing.
        lam_max must be finite.
    '''
    return record

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sampling_record(lam_max: np.ndarray, interfaces: np.ndarray) -> np.ndarray:
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
    n = interfaces.size - 1
    npaths = lam_max.size
    mu = np.zeros((npaths, n), dtype=float)
    w = np.ones((npaths, n), dtype=float)
    for j in range(npaths):
        for k in range(n):
            if interfaces[k] < lam_max[j]:
                mu[j, k] = float(((j + 2 * k + 1) % 4) + 1)
                w[j, k] = 1.0 if k == 0 else 1.0 + 0.75 * ((j + 3 * k) % 4)
    return np.stack([mu, w])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\nlam_max = np.array([-0.95, -0.18, 0.36, 0.84, 1.12])\n",
            "call": "sampling_record(lam_max, interfaces)",
            "gold_call": "_oracle_sampling_record(lam_max, interfaces)"
        },
        {
            "setup": "import numpy as np\n# boundary: every maximum sits exactly on an interface\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\nlam_max = np.array([-0.4, 0.2, 0.7])\n",
            "call": "sampling_record(lam_max, interfaces)",
            "gold_call": "_oracle_sampling_record(lam_max, interfaces)"
        },
        {
            "setup": "import numpy as np\n# edge: a single trajectory that only just clears the first interface\ninterfaces = np.array([-1.0, -0.4, 0.2])\nlam_max = np.array([-0.999999])\n",
            "call": "sampling_record(lam_max, interfaces)",
            "gold_call": "_oracle_sampling_record(lam_max, interfaces)"
        },
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, -0.4, 1.0])   # not strictly increasing\nlam_max = np.array([0.1])\ndef run(f):\n    try:\n        f(lam_max, interfaces); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(sampling_record)",
            "gold_call": "run(_oracle_sampling_record)"
        },
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, 1.0])               # too few interfaces\nlam_max = np.array([0.1])\ndef run(f):\n    try:\n        f(lam_max, interfaces); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(sampling_record)",
            "gold_call": "run(_oracle_sampling_record)"
        }
    ]
