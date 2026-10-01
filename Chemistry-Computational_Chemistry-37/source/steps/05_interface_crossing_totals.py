"""
For one nominated interface, accumulate per ensemble the unbiased sampling weight carried by the trajectories that advanced beyond it. Trajectories that fell back before reaching the interface contribute nothing. These restricted totals are the numerators of the crossing probability that follows.

The probability of advancing from the reactant boundary to a given interface is a path-ensemble average of a step function of the furthest progress. Evaluating it from a single ensemble wastes the statistics that neighbouring ensembles hold about the same interface, so the estimator sums the restricted weight over several ensembles at once. Keeping the restricted totals separate per ensemble is what later allows them to be combined with ensemble-dependent optimal weights rather than being pooled naively.

Returns
-------
np.ndarray of shape (n,) and dtype float, the per-ensemble weight of trajectories passing the nominated interface
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def interface_crossing_totals(t: np.ndarray, lam_max: np.ndarray,
                              interfaces: np.ndarray, level: int) -> np.ndarray:
    '''Unbiased weight per ensemble carried by trajectories passing one interface.

    Parameters
    ----------
    t : np.ndarray
        (P, n) unbiased sampling weights.
    lam_max : np.ndarray
        (P,) furthest progress reached by each trajectory.
    interfaces : np.ndarray
        (n+1,) strictly increasing interface positions.
    level : int
        Index of the interface being crossed.

    Returns
    -------
    totals : np.ndarray
        (n,) restricted weight contributed by each ensemble.
    
    Raises
    ------
    ValueError
        t must be a non-empty 2D array.
        t must carry one row per trajectory.
        interfaces must be a 1D array with at least three entries.
        interfaces must be strictly increasing.
        level must be an integer.
        level must index one of the interfaces.
    '''
    return totals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_interface_crossing_totals(t: np.ndarray, lam_max: np.ndarray,
                                      interfaces: np.ndarray, level: int) -> np.ndarray:
    """Reference implementation."""
    t = np.asarray(t, dtype=float)
    lam_max = np.asarray(lam_max, dtype=float)
    interfaces = np.asarray(interfaces, dtype=float)
    if t.ndim != 2 or t.size == 0:
        raise ValueError("t must be a non-empty 2D array")
    if lam_max.ndim != 1 or t.shape[0] != lam_max.size:
        raise ValueError("t must carry one row per trajectory")
    if interfaces.ndim != 1 or interfaces.size < 3:
        raise ValueError("interfaces must be a 1D array with at least three entries")
    if np.any(np.diff(interfaces) <= 0.0):
        raise ValueError("interfaces must be strictly increasing")
    if isinstance(level, bool) or not isinstance(level, (int, np.integer)):
        raise ValueError("level must be an integer")
    if not 0 <= int(level) < interfaces.size:
        raise ValueError("level must index one of the interfaces")
    passed = (lam_max > interfaces[int(level)]).astype(float)
    return (t * passed[:, None]).sum(axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\nlam_max = np.array([-0.95, -0.18, 0.36, 0.84, 1.12])\nt = np.array([[2.0, 0.0, 0.0, 0.0],\n              [1.0, 3.0, 0.0, 0.0],\n              [4.0, 2.0, 4.0, 0.0],\n              [3.0, 1.0, 3.0, 1.0],\n              [1.0, 3.0, 1.0, 3.0]])\n",
            "call": "interface_crossing_totals(t, lam_max, interfaces, 2)",
            "gold_call": "_oracle_interface_crossing_totals(t, lam_max, interfaces, 2)"
        },
        {
            "setup": "import numpy as np\n# boundary: the first interface is passed by every trajectory in the record\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\nlam_max = np.array([-0.95, -0.18, 0.36, 0.84, 1.12])\nt = np.ones((5, 4))\n",
            "call": "interface_crossing_totals(t, lam_max, interfaces, 0)",
            "gold_call": "_oracle_interface_crossing_totals(t, lam_max, interfaces, 0)"
        },
        {
            "setup": "import numpy as np\n# edge: a maximum exactly on the interface does not count as passing it\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\nlam_max = np.array([0.2, 0.2000001])\nt = np.array([[1.0, 1.0, 1.0, 1.0],\n              [2.0, 2.0, 2.0, 2.0]])\n",
            "call": "interface_crossing_totals(t, lam_max, interfaces, 2)",
            "gold_call": "_oracle_interface_crossing_totals(t, lam_max, interfaces, 2)"
        },
        {
            "setup": "import numpy as np\n# edge: nothing reaches the product interface\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\nlam_max = np.array([0.1, 0.3])\nt = np.array([[1.0, 0.0, 0.0, 0.0],\n              [2.0, 1.0, 0.0, 0.0]])\n",
            "call": "interface_crossing_totals(t, lam_max, interfaces, 4)",
            "gold_call": "_oracle_interface_crossing_totals(t, lam_max, interfaces, 4)"
        },
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, 0.2, 1.0])\nlam_max = np.array([0.1, 0.3])\nt = np.ones((2, 3))\ndef run(f):\n    try:\n        f(t, lam_max, interfaces, 9); return 0     # level out of range\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(interface_crossing_totals)",
            "gold_call": "run(_oracle_interface_crossing_totals)"
        }
    ]
