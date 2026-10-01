"""
Return the source's modified weighting factor of each resource state (its Eq. (33), denoted ed_j) from the state contributions of step 03 and n_occupied, the number of resource states that are used by at least one species in the complete resource matrix the analysis refers to (the source's r'). The factors are strictly positive for every state, including states used by no species, and sum to 1.

The classical relative weighting of resource states vanishes for unused states and for states used in identical proportions by all species, which makes niche metrics undefined or incomparable between matrices of different resolution; the source's modified factor keeps every state positive and accounts for how much of the sampled resource space is actually occupied.

 

RReturns

-------

numpy.ndarray of float64 with shape (r,): the modified weighting factors ed_j, positive and summing to 1.

Returns
-------
return weights

Returns
-------
return weights
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def modified_weights(contributions: "numpy.ndarray", n_occupied: int) -> "numpy.ndarray":
    """Return the modified resource-state weights.

    Parameters
    ----------
    contributions : numpy.ndarray
        One-dimensional array from step 03.
    n_occupied : int
        Number of occupied resource states.

    Returns
    -------
    weights : numpy.ndarray
        Strictly positive float64 weights summing to 1.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_modified_weights(contributions: "numpy.ndarray", n_occupied: int) -> "numpy.ndarray":
    """ed_j of Eq 33: exp(delta_j r'/r) normalised to unit sum, r' the occupied states of the complete matrix."""
    delta = np.asarray(contributions, dtype=np.float64)
    if delta.ndim != 1 or delta.size < 1 or not np.all(np.isfinite(delta)):
        raise ValueError("contributions must be a non-empty finite 1-D array")
    r = delta.size
    if int(n_occupied) != n_occupied or not (1 <= n_occupied <= r):
        raise ValueError("n_occupied must be an integer between 1 and the number of resource states")
    # the exponent is scaled by the occupancy fraction r'/r of the COMPLETE matrix (not of a reduced one)
    w = np.exp(delta * (float(int(n_occupied)) / r))
    return w / w.sum()

def _modified_weights_error_code(contributions: "numpy.ndarray", n_occupied: int) -> int:
    """0 if _oracle_modified_weights accepts these inputs, 1 if it raises ValueError."""
    try:
        _oracle_modified_weights(contributions, n_occupied)
        return 0
    except ValueError:
        return 1

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _oracle_use_probabilities(resource_matrix)\nentropies = _oracle_resource_entropies(resource_matrix)\ncontributions = _oracle_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = 6\n",
            "call": "modified_weights(contributions, n_occupied)",
            "gold_call": "_oracle_modified_weights(contributions, n_occupied)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nresource_matrix = np.array([[50, 30, 10], [10, 30, 50], [5, 30, 55]], dtype=float)\nuse_probs = _oracle_use_probabilities(resource_matrix)\nentropies = _oracle_resource_entropies(resource_matrix)\ncontributions = _oracle_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = 3\n",
            "call": "modified_weights(contributions, n_occupied)",
            "gold_call": "_oracle_modified_weights(contributions, n_occupied)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nresource_matrix = np.array([[30, 30, 30, 0], [10, 30, 50, 0], [5, 0, 55, 0], [30, 30, 30, 0]], dtype=float)\nuse_probs = _oracle_use_probabilities(resource_matrix)\nentropies = _oracle_resource_entropies(resource_matrix)\ncontributions = _oracle_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = 3\n",
            "call": "modified_weights(contributions, n_occupied)",
            "gold_call": "_oracle_modified_weights(contributions, n_occupied)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _oracle_use_probabilities(resource_matrix)\nentropies = _oracle_resource_entropies(resource_matrix)\ncontributions = _oracle_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = 9\ndef run_model():\n    try:\n        modified_weights(contributions, n_occupied)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_modified_weights_error_code(contributions, n_occupied)",
        },
    ]
