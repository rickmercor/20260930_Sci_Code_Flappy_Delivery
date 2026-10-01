"""
Evaluate the approximation error of a fixed sentinel node set over a complete state sweep.

The error is a dimensionless measure of the discrepancy between the selected-node estimate and the full-network activity curve.

Returns
-------
return float(np.sum((approximation - mean_activity) ** 2) / denominator)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sentinel_error(states: "np.ndarray", mean_activity: "np.ndarray", sentinel_indices: "np.ndarray") -> float:
    '''Return the dimensionless approximation error for the supplied sentinel set over the complete state sweep.

The benchmark follows the normalization used by the authors' released implementation: divide the summed squared discrepancies by the sum of full-network means. The article's printed equation includes an additional factor of L in the denominator; responses using that printed-equation form should be treated as the same convention up to the constant factor L.

    Parameters
    ----------
    states : np.ndarray
        Finite array of shape (L, N) containing node states over the sweep.
    mean_activity : np.ndarray
        Finite length-L full-network mean corresponding to the rows of states.
    sentinel_indices : np.ndarray
        One-dimensional integer node labels with no duplicates and all labels
        in [0, N-1].

    Returns
    -------
    error : float
        Dimensionless discrepancy measure between the sentinel estimate and the
        full-network activity curve.

    Raises
    ------
    ValueError
        If shapes, indices, finiteness, or the positive denominator requirement
        is violated.
    '''
    return error

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sentinel_error(states: "np.ndarray", mean_activity: "np.ndarray", sentinel_indices: "np.ndarray") -> float:
    states = np.asarray(states, dtype=float)
    mean_activity = np.asarray(mean_activity, dtype=float)
    sentinel_indices = np.asarray(sentinel_indices)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("states must have shape (L,N)")
    if mean_activity.ndim != 1 or mean_activity.shape[0] != states.shape[0]:
        raise ValueError("mean_activity must have length L")
    if sentinel_indices.ndim != 1 or sentinel_indices.size < 1:
        raise ValueError("sentinel_indices must be a nonempty one-dimensional array")
    if not np.issubdtype(sentinel_indices.dtype, np.integer):
        raise ValueError("sentinel_indices must contain integers")
    if np.any(sentinel_indices < 0) or np.any(sentinel_indices >= states.shape[1]):
        raise ValueError("sentinel index out of bounds")
    if np.unique(sentinel_indices).size != sentinel_indices.size:
        raise ValueError("sentinel_indices must not contain duplicates")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(mean_activity)):
        raise ValueError("states and mean_activity must be finite")
    denominator = float(np.sum(mean_activity))
    if denominator <= 0.0:
        raise ValueError("sum of mean_activity must be positive")
    approximation = np.mean(states[:, sentinel_indices], axis=1)
    return float(np.sum((approximation - mean_activity) ** 2) / denominator)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative sentinel-error cases."""
    return [
        {
            "setup": """import numpy as np
states = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]], dtype=float)
mean_activity = np.mean(states, axis=1)
sentinel_indices = np.array([0,2], dtype=int)
""",
            "call": "sentinel_error(states, mean_activity, sentinel_indices)",
            "gold_call": "_oracle_sentinel_error(states, mean_activity, sentinel_indices)",
        },
        {
            "setup": """import numpy as np
states = np.array([[1,2],[2,4],[3,6]], dtype=float)
mean_activity = np.array([1,2,3], dtype=float)
sentinel_indices = np.array([1], dtype=int)
""",
            "call": "sentinel_error(states, mean_activity, sentinel_indices)",
            "gold_call": "_oracle_sentinel_error(states, mean_activity, sentinel_indices)",
        },
        {
            "setup": """import numpy as np
states = np.array([[0.5,2.0,5.0],[0.75,2.5,6.0]], dtype=float)
mean_activity = np.array([2.5,3.0833333333333335], dtype=float)
sentinel_indices = np.array([0,2], dtype=int)
""",
            "call": "sentinel_error(states, mean_activity, sentinel_indices)",
            "gold_call": "_oracle_sentinel_error(states, mean_activity, sentinel_indices)",
        },
    ]
