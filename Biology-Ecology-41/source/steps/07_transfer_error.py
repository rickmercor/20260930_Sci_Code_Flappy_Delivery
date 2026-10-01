"""
Evaluate the previously selected sentinel set on a second network dynamics model.

The returned value quantifies how accurately the unchanged sentinel set reconstructs the second system's global activity curve.

Returns
-------
return float(np.sum((sentinel_mean - full_mean) ** 2) / denominator)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transfer_error(test_states: "np.ndarray", sentinel_indices: "np.ndarray") -> float:
    '''Return the approximation error of a fixed sentinel set on test dynamics.

    Parameters
    ----------
    test_states : np.ndarray
        Finite array of shape (L, N) containing terminal node states for the
        test dynamics over its control-parameter sweep.
    sentinel_indices : np.ndarray
        One-dimensional integer node labels shared with the training network.

    Returns
    -------
    error : float
        Dimensionless approximation error between the sentinel estimate and the full
        network activity curve for the test dynamics.

    Raises
    ------
    ValueError
        If the state matrix or sentinel labels are invalid or the full-network
        normalization is non-positive.
    '''
    return error

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_transfer_error(test_states: "np.ndarray", sentinel_indices: "np.ndarray") -> float:
    test_states = np.asarray(test_states, dtype=float)
    sentinel_indices = np.asarray(sentinel_indices)
    if test_states.ndim != 2 or test_states.shape[0] < 1 or test_states.shape[1] < 1:
        raise ValueError("test_states must have shape (L,N)")
    if sentinel_indices.ndim != 1 or sentinel_indices.size < 1:
        raise ValueError("sentinel_indices must be nonempty and one-dimensional")
    if not np.issubdtype(sentinel_indices.dtype, np.integer):
        raise ValueError("sentinel_indices must contain integers")
    if np.any(sentinel_indices < 0) or np.any(sentinel_indices >= test_states.shape[1]):
        raise ValueError("sentinel index out of bounds")
    if np.unique(sentinel_indices).size != sentinel_indices.size:
        raise ValueError("sentinel_indices must be unique")
    if not np.all(np.isfinite(test_states)):
        raise ValueError("test_states must be finite")

    full_mean = np.mean(test_states, axis=1)
    denominator = float(np.sum(full_mean))
    if denominator <= 0.0:
        raise ValueError("test-dynamics normalization must be positive")
    sentinel_mean = np.mean(test_states[:, sentinel_indices], axis=1)
    return float(np.sum((sentinel_mean - full_mean) ** 2) / denominator)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative transfer-error cases."""
    return [
        {
            "setup": """import numpy as np
test_states = np.array([[1,2,3],[2,3,4],[3,4,6]], dtype=float)
sentinel_indices = np.array([0,2], dtype=int)
""",
            "call": "transfer_error(test_states, sentinel_indices)",
            "gold_call": "_oracle_transfer_error(test_states, sentinel_indices)",
        },
        {
            "setup": """import numpy as np
test_states = np.array([[0.01,0.01],[0.02,0.02],[0.03,0.03]], dtype=float)
sentinel_indices = np.array([1], dtype=int)
""",
            "call": "transfer_error(test_states, sentinel_indices)",
            "gold_call": "_oracle_transfer_error(test_states, sentinel_indices)",
        },
        {
            "setup": """import numpy as np
test_states = np.array([[0.0,1.0,2.0,4.0],[0.2,1.2,2.5,5.0],[0.5,1.5,3.0,6.0]], dtype=float)
sentinel_indices = np.array([1,3], dtype=int)
""",
            "call": "transfer_error(test_states, sentinel_indices)",
            "gold_call": "_oracle_transfer_error(test_states, sentinel_indices)",
        },
    ]
