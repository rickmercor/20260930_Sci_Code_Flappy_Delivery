"""
Return the conditional resource-use probabilities for a resource matrix: resource_matrix[i, j] is the abundance (or resource-use value) of species i in resource state j, and the result holds, for every species, the probability that one of its individuals is associated with each resource state. A species whose row is entirely zero receives a row of zeros.

Niche metrics are computed from a resource matrix of species by resource states; the first quantity every metric needs is each species' own distribution of use over the states.

Returns
-------
numpy.ndarray of float64 with shape (s, r): the per-species resource-use probabilities.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def use_probabilities(resource_matrix: "numpy.ndarray") -> "numpy.ndarray":
    """Return the conditional resource-use probabilities for a resource matrix: resource_matrix[i, j] is the abundance (or resource-use value) of species i in resource state j, and the result holds, for every species, the probability that one of its individuals is associated with each resource state. A species whose row is entirely zero receives a row of zeros.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r) of finite, nonnegative abundances: s species (rows) by r resource states (columns).

    Returns
    -------
    use_probabilities : numpy.ndarray
        Array of shape (s, r); each row sums to 1 (or is all zeros for an absent species) (float64).

    Raises
    ------
    ValueError
        If resource_matrix is not two-dimensional, contains a negative or non-finite entry, or has no positive entry.
    """
    return use_probabilities

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_use_probabilities(resource_matrix: "numpy.ndarray") -> "numpy.ndarray":
    """Conditional resource-use probabilities p_ij = N_ij / Y_i (Eq 6); a species with Y_i = 0 gets a zero row."""
    N = np.asarray(resource_matrix, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("resource_matrix must be a 2-D array with at least one species and one state")
    if not np.all(np.isfinite(N)) or np.any(N < 0.0):
        raise ValueError("resource_matrix entries must be finite and nonnegative")
    if N.sum() <= 0.0:
        raise ValueError("resource_matrix must contain at least one positive entry")
    Y = N.sum(axis=1)
    return np.where(Y[:, None] > 0.0, N / np.where(Y[:, None] > 0.0, Y[:, None], 1.0), 0.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    # Inputs of the step under test are built by _fx_* copies of the upstream reference
    # arithmetic (input validation omitted), so no setup statement depends on an oracle.
    fixture = 'def _raises(fn, *args):\n    try:\n        fn(*args)\n        return 0\n    except ValueError:\n        return 1\n\n'
    return [
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\n',
            'call': 'use_probabilities(resource_matrix)',
            'gold_call': '_oracle_use_probabilities(resource_matrix)',
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[50, 30, 10], [10, 30, 50], [5, 30, 55]], dtype=float)\n',
            'call': 'use_probabilities(resource_matrix)',
            'gold_call': '_oracle_use_probabilities(resource_matrix)',
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[30, 30, 30, 0], [10, 30, 50, 0], [5, 0, 55, 0], [30, 30, 30, 0]], dtype=float)\n',
            'call': 'use_probabilities(resource_matrix)',
            'gold_call': '_oracle_use_probabilities(resource_matrix)',
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[1.0, -2.0], [3.0, 4.0]])\ndef run_model():\n    try:\n        use_probabilities(resource_matrix)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': '_raises(_oracle_use_probabilities, resource_matrix)',
        },
    ]
