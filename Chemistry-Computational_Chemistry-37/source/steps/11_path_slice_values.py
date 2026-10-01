"""
Place the interior phase points of one trajectory along the order parameter, following the rise-and-fall profile specified in the problem. A trajectory of length parameter L holds L+1 phase points, of which the L-1 interior ones are kept and the two end points discarded. A trajectory that reaches the product state rises monotonically instead, and a reactant-side trajectory uses the same rise-and-fall rule with the outer interface as its turning point.

Phase-space averages are obtained from interface sampling by collecting the time slices along the sampled trajectories rather than by histogramming an equilibrium simulation. The first and last slice of every path are excluded so that all contributing slices lie in one order-parameter region: the reactant region for the reactant-side ensemble, and the region between the two outermost interfaces for the barrier-side ensembles. Keeping the end points would mix slices from different regions into the same histogram and shift the resulting profile.

Returns
-------
np.ndarray of shape (path_length - 1,) and dtype float, the interior phase-point positions
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def path_slice_values(lam_max_value: float, path_length: int, interfaces: np.ndarray,
                      lower_turning_point: float = None) -> np.ndarray:
    '''Order-parameter values of the interior phase points of one trajectory.

    Parameters
    ----------
    lam_max_value : float
        Furthest progress reached by the trajectory.
    path_length : int
        Length parameter of the trajectory, which holds L+1 phase points
        in total and L-1 interior ones.
    interfaces : np.ndarray
        (n+1,) strictly increasing interface positions.
    lower_turning_point : float, optional
        Turning point for a reactant-side trajectory. When given it replaces
        lam_max_value as the extreme of the profile.

    Returns
    -------
    values : np.ndarray
        (path_length - 1,) order-parameter values of the interior phase points.
    
    Raises
    ------
    ValueError
        interfaces must be a 1D array with at least three entries.
        interfaces must be strictly increasing.
        path_length must be an integer.
        path_length must be at least 3.
        lam_max_value must be finite.
        lower_turning_point must be finite.
    '''
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_path_slice_values(lam_max_value: float, path_length: int, interfaces: np.ndarray,
                              lower_turning_point: float = None) -> np.ndarray:
    """Reference implementation."""
    interfaces = np.asarray(interfaces, dtype=float)
    if interfaces.ndim != 1 or interfaces.size < 3:
        raise ValueError("interfaces must be a 1D array with at least three entries")
    if np.any(np.diff(interfaces) <= 0.0):
        raise ValueError("interfaces must be strictly increasing")
    if isinstance(path_length, bool) or not isinstance(path_length, (int, np.integer)):
        raise ValueError("path_length must be an integer")
    if int(path_length) < 3:
        raise ValueError("path_length must be at least 3")
    if not np.isfinite(float(lam_max_value)):
        raise ValueError("lam_max_value must be finite")
    interior = int(path_length) - 1
    start = float(interfaces[0])
    if lower_turning_point is not None:
        if not np.isfinite(float(lower_turning_point)):
            raise ValueError("lower_turning_point must be finite")
        peak = float(lower_turning_point)
    elif float(lam_max_value) > float(interfaces[-1]):
        rise = (np.arange(interior, dtype=float) + 1.0) / (interior + 1.0)
        return start + (float(interfaces[-1]) - start) * rise
    else:
        peak = float(lam_max_value)
    turn = (interior - 1) // 2
    values = np.empty(interior, dtype=float)
    for m in range(interior):
        if m <= turn:
            frac = (m + 1.0) / (turn + 1.0)
        else:
            frac = (interior - m) / float(interior - turn)
        values[m] = start + (peak - start) * frac
    return values

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\n",
            "call": "path_slice_values(0.70, 7, interfaces)",
            "gold_call": "_oracle_path_slice_values(0.70, 7, interfaces)"
        },
        {
            "setup": "import numpy as np\n# boundary: a trajectory that reaches the product state rises monotonically\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\n",
            "call": "path_slice_values(1.12, 9, interfaces)",
            "gold_call": "_oracle_path_slice_values(1.12, 9, interfaces)"
        },
        {
            "setup": "import numpy as np\n# boundary: a reactant-side trajectory turning at the outer interface\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\n",
            "call": "path_slice_values(-1.0, 8, interfaces, lower_turning_point=-1.6)",
            "gold_call": "_oracle_path_slice_values(-1.0, 8, interfaces, lower_turning_point=-1.6)"
        },
        {
            "setup": "import numpy as np\n# edge: the shortest admissible trajectory\ninterfaces = np.array([-1.0, -0.4, 0.2, 1.0])\n",
            "call": "path_slice_values(0.3, 3, interfaces)",
            "gold_call": "_oracle_path_slice_values(0.3, 3, interfaces)"
        },
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, 0.2, 1.0])\ndef run(f):\n    try:\n        f(0.3, 2, interfaces); return 0        # too short to have an interior\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(path_slice_values)",
            "gold_call": "run(_oracle_path_slice_values)"
        },
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, 0.2, 1.0])\ndef run(f):\n    try:\n        f(0.3, 7.5, interfaces); return 0      # non-integer length\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(path_slice_values)",
            "gold_call": "run(_oracle_path_slice_values)"
        }
    ]
