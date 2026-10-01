"""
Compute the weighted fraction of reactant-side trajectories that terminate at the inner interface rather than at the outer one. This is a weighted average of an indicator over the reactant-side ensemble, so each trajectory enters through its own normalised weight and not through a plain head count.

For a coordinate that is bounded on the reactant side, an infinitely long trajectory produces equal numbers of segments on either side of the first interface, and the two families of trajectories can be combined without further thought. When the coordinate is unbounded, which is the situation whenever a permeant can wander arbitrarily far from the region of interest, an extra interface has to be introduced beyond the reactant basin and the reactant-side ensemble is redefined to admit trajectories starting and ending at either interface. The endpoint of such a trajectory is then no longer necessarily the starting point of a barrier-side trajectory, more segments are produced on the reactant side, and the imbalance is repaired by this fraction.

Returns
-------
float, the weighted fraction of reactant-side trajectories terminating at the inner interface
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def end_interface_fraction(minus_weights: np.ndarray,
                           ends_at_first_interface: np.ndarray) -> float:
    '''Weighted fraction of reactant-side trajectories ending at the inner interface.

    Parameters
    ----------
    minus_weights : np.ndarray
        (M,) normalised weights of the reactant-side trajectories.
    ends_at_first_interface : np.ndarray
        (M,) boolean flags, true where a trajectory terminates at the inner interface.

    Returns
    -------
    fraction : float
        Weighted fraction between 0 and 1.
    
    Raises
    ------
    ValueError
        minus_weights must be a 1D array with at least one entry.
        ends_at_first_interface must match minus_weights in length.
        minus_weights must be finite.
        minus_weights must be non-negative.
    '''
    return fraction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_end_interface_fraction(minus_weights: np.ndarray,
                                   ends_at_first_interface: np.ndarray) -> float:
    """Reference implementation."""
    weights = np.asarray(minus_weights, dtype=float)
    flags = np.asarray(ends_at_first_interface)
    if weights.ndim != 1 or weights.size < 1:
        raise ValueError("minus_weights must be a 1D array with at least one entry")
    if flags.ndim != 1 or flags.size != weights.size:
        raise ValueError("ends_at_first_interface must match minus_weights in length")
    if not np.all(np.isfinite(weights)):
        raise ValueError("minus_weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("minus_weights must be non-negative")
    flags = flags.astype(bool)
    return float(np.sum(weights * flags))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nminus_weights = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]) / 12.0\nends_at_first_interface = np.array([True, True, False, True, False, True])\n",
            "call": "end_interface_fraction(minus_weights, ends_at_first_interface)",
            "gold_call": "_oracle_end_interface_fraction(minus_weights, ends_at_first_interface)"
        },
        {
            "setup": "import numpy as np\n# boundary: every trajectory ends at the inner interface, recovering the bounded case\nminus_weights = np.array([0.25, 0.25, 0.5])\nends_at_first_interface = np.array([True, True, True])\n",
            "call": "end_interface_fraction(minus_weights, ends_at_first_interface)",
            "gold_call": "_oracle_end_interface_fraction(minus_weights, ends_at_first_interface)"
        },
        {
            "setup": "import numpy as np\n# edge: none of them do\nminus_weights = np.array([0.4, 0.6])\nends_at_first_interface = np.array([False, False])\n",
            "call": "end_interface_fraction(minus_weights, ends_at_first_interface)",
            "gold_call": "_oracle_end_interface_fraction(minus_weights, ends_at_first_interface)"
        },
        {
            "setup": "import numpy as np\nminus_weights = np.array([0.5, 0.5])\nends_at_first_interface = np.array([True])      # length mismatch\ndef run(f):\n    try:\n        f(minus_weights, ends_at_first_interface); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(end_interface_fraction)",
            "gold_call": "run(_oracle_end_interface_fraction)"
        },
        {
            "setup": "import numpy as np\nminus_weights = np.array([0.5, -0.5])           # negative weight\nends_at_first_interface = np.array([True, True])\ndef run(f):\n    try:\n        f(minus_weights, ends_at_first_interface); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(end_interface_fraction)",
            "gold_call": "run(_oracle_end_interface_fraction)"
        }
    ]
