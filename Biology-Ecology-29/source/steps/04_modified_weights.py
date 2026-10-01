"""
Return the source's modified weighting factor of each resource state (denoted ed_j) from the state contributions of step 03 and n_occupied, the number of resource states that are used by at least one species in the complete resource matrix the analysis refers to (the source's r'). The factors are strictly positive for every state, including states used by no species, and sum to 1.

The classical relative weighting of resource states vanishes for unused states and for states used in identical proportions by all species, which makes niche metrics undefined or incomparable between matrices of different resolution; the source's modified factor keeps every state positive and accounts for how much of the sampled resource space is actually occupied.

Returns
-------
numpy.ndarray of float64 with shape (r,): the modified weighting factors ed_j, positive and summing to 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def modified_weights(contributions: "numpy.ndarray", n_occupied: int) -> "numpy.ndarray":
    """Return the source's modified weighting factor of each resource state (denoted ed_j) from the state contributions of step 03 and n_occupied, the number of resource states that are used by at least one species in the complete resource matrix the analysis refers to (the source's r'). The factors are strictly positive for every state, including states used by no species, and sum to 1. With delta_j the contribution of state j and r the number of states, ed_j = exp(delta_j n_occupied / r) / (sum over all states l of exp(delta_l n_occupied / r)).

    Parameters
    ----------
    contributions : numpy.ndarray
        Array of shape (r,) from step 03.
    n_occupied : int
        Number of resource states with a positive column total in the complete resource matrix, between 1 and r.

    Returns
    -------
    weights : numpy.ndarray
        Array of shape (r,) of strictly positive factors summing to 1 (float64).

    Raises
    ------
    ValueError
        If contributions is not a non-empty finite 1-D array, or n_occupied is not an integer between 1 and r.
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    # Inputs of the step under test are built by _fx_* copies of the upstream reference
    # arithmetic (input validation omitted), so no setup statement depends on an oracle.
    fixture = 'import numpy as np\n\ndef _fx_xlogx(a):\n    a = np.asarray(a, dtype=np.float64)\n    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)\n\ndef _fx_use_probabilities(N):\n    N = np.asarray(N, dtype=np.float64)\n    Y = N.sum(axis=1)\n    return np.where(Y[:, None] > 0.0, N / np.where(Y[:, None] > 0.0, Y[:, None], 1.0), 0.0)\n\ndef _fx_resource_entropies(N):\n    N = np.asarray(N, dtype=np.float64)\n    Z = N.sum()\n    P, Q, pi = N.sum(axis=0) / Z, N.sum(axis=1) / Z, N / Z\n    HX, HY, HXY = -_fx_xlogx(P).sum(), -_fx_xlogx(Q).sum(), -_fx_xlogx(pi).sum()\n    return np.array([HX, HY, HXY, HXY - HY])\n\ndef _fx_state_contributions(N, p, ent):\n    N = np.asarray(N, dtype=np.float64)\n    p = np.asarray(p, dtype=np.float64)\n    HX = float(np.asarray(ent, dtype=np.float64)[0])\n    Z = N.sum()\n    pi, P = N / Z, N.sum(axis=0) / Z\n    with np.errstate(divide="ignore", invalid="ignore"):\n        term = np.where(p > 0.0, pi * np.log(np.where(p > 0.0, p, 1.0)), 0.0).sum(axis=0)\n    return (term - _fx_xlogx(P)) / HX\n\ndef _raises(fn, *args):\n    try:\n        fn(*args)\n        return 0\n    except ValueError:\n        return 1\n\n'
    return [
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = 6\n',
            'call': 'modified_weights(contributions, n_occupied)',
            'gold_call': '_oracle_modified_weights(contributions, n_occupied)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[50, 30, 10], [10, 30, 50], [5, 30, 55]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = 3\n',
            'call': 'modified_weights(contributions, n_occupied)',
            'gold_call': '_oracle_modified_weights(contributions, n_occupied)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[30, 30, 30, 0], [10, 30, 50, 0], [5, 0, 55, 0], [30, 30, 30, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = 3\n',
            'call': 'modified_weights(contributions, n_occupied)',
            'gold_call': '_oracle_modified_weights(contributions, n_occupied)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = 9\ndef run_model():\n    try:\n        modified_weights(contributions, n_occupied)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': '_raises(_oracle_modified_weights, contributions, n_occupied)',
        },
    ]
