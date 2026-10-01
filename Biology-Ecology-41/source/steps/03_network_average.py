"""
Compute the global equilibrium activity curve from a matrix of node states.

The global activity at each control value is the unweighted mean across all network nodes.

Returns
-------
return np.mean(states, axis=1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def network_average(states: "np.ndarray") -> "np.ndarray":
    '''Return the unweighted network mean for each control-parameter value.

    Parameters
    ----------
    states : np.ndarray
        Finite two-dimensional array with shape (L, N), where rows correspond
        to control values and columns correspond to network nodes.

    Returns
    -------
    mean_activity : np.ndarray
        One mean node activity for each of the L rows.

    Raises
    ------
    ValueError
        If states is not a finite two-dimensional array with at least one row
        and one node.
    '''
    return mean_activity

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_network_average(states: "np.ndarray") -> "np.ndarray":
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("states must have shape (L,N) with L,N >= 1")
    if not np.all(np.isfinite(states)):
        raise ValueError("states must be finite")
    return np.mean(states, axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative network-average cases."""
    return [
        {
            "setup": """import numpy as np
states = np.array([[1.0,2.0,3.0],[2.0,4.0,8.0]], dtype=float)
""",
            "call": "network_average(states)",
            "gold_call": "_oracle_network_average(states)",
        },
        {
            "setup": """import numpy as np
states = np.array([[5.5],[7.25]], dtype=float)
""",
            "call": "network_average(states)",
            "gold_call": "_oracle_network_average(states)",
        },
        {
            "setup": """import numpy as np
states = np.array([[0.0,1.0,0.0,1.0],[2.0,2.0,2.0,2.0],[3.0,0.0,6.0,9.0]], dtype=float)
""",
            "call": "network_average(states)",
            "gold_call": "_oracle_network_average(states)",
        },
    ]
