"""
Return the source's adjusted resource-use probabilities p*_ij of every species in a resource matrix for the given per-state weighting factors and the source's matrix-expansion constant k: each species' abundances are divided by its weighted, k-scaled total abundance Y*_i. A species whose row is entirely zero receives a row of zeros.

Weighting the resource states changes what counts as a species' effective total abundance; the adjusted probabilities are the resource-use frequencies measured against that weighted total and are the quantities that every weighted niche metric of the source is built from.

Returns
-------
numpy.ndarray of float64 with shape (s, r): the adjusted probabilities p*_ij.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adjusted_probabilities(resource_matrix: "numpy.ndarray", weights: "numpy.ndarray",
                           k: float) -> "numpy.ndarray":
    """Return the source's adjusted resource-use probabilities p*_ij of every species in a resource matrix for the given per-state weighting factors and the source's matrix-expansion constant k: each species' abundances are divided by its weighted, k-scaled total abundance Y*_i. A species whose row is entirely zero receives a row of zeros.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r) of finite, nonnegative abundances.
    weights : numpy.ndarray
        Array of shape (r,) of finite, nonnegative per-state weighting factors.
    k : float
        Positive matrix-expansion constant of the source (10000 in its analyses).

    Returns
    -------
    adjusted : numpy.ndarray
        Array of shape (s, r) of adjusted probabilities p*_ij (float64).

    Raises
    ------
    ValueError
        If the shapes are inconsistent, an entry is negative or non-finite, k is not positive, or a species with positive abundance has zero weighted total.
    """
    return adjusted

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_matrix(resource_matrix):
    N = np.asarray(resource_matrix, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("resource_matrix must be a 2-D array with at least one species and one state")
    if not np.all(np.isfinite(N)) or np.any(N < 0.0):
        raise ValueError("resource_matrix entries must be finite and nonnegative")
    if N.sum() <= 0.0:
        raise ValueError("resource_matrix must contain at least one positive entry")
    return N


def _oracle_adjusted_probabilities(resource_matrix: "numpy.ndarray", weights: "numpy.ndarray",
                                   k: float) -> "numpy.ndarray":
    """p*_ij of Eq 19: N_ij divided by the weighted, k-scaled abundance Y*_i = sum_j w_j k N_ij."""
    N = _check_matrix(resource_matrix)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (N.shape[1],) or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be a finite nonnegative array with one entry per resource state")
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("k must be positive and finite")
    Ystar = (w[None, :] * float(k) * N).sum(axis=1)
    if np.any((N.sum(axis=1) > 0.0) & (Ystar <= 0.0)):
        raise ValueError("a species with positive abundance has zero weighted abundance: undefined p*")
    return np.where(Ystar[:, None] > 0.0, N / np.where(Ystar[:, None] > 0.0, Ystar[:, None], 1.0), 0.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    # Inputs of the step under test are built by _fx_* copies of the upstream reference
    # arithmetic (input validation omitted), so no setup statement depends on an oracle.
    fixture = 'import numpy as np\n\ndef _fx_xlogx(a):\n    a = np.asarray(a, dtype=np.float64)\n    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)\n\ndef _fx_use_probabilities(N):\n    N = np.asarray(N, dtype=np.float64)\n    Y = N.sum(axis=1)\n    return np.where(Y[:, None] > 0.0, N / np.where(Y[:, None] > 0.0, Y[:, None], 1.0), 0.0)\n\ndef _fx_resource_entropies(N):\n    N = np.asarray(N, dtype=np.float64)\n    Z = N.sum()\n    P, Q, pi = N.sum(axis=0) / Z, N.sum(axis=1) / Z, N / Z\n    HX, HY, HXY = -_fx_xlogx(P).sum(), -_fx_xlogx(Q).sum(), -_fx_xlogx(pi).sum()\n    return np.array([HX, HY, HXY, HXY - HY])\n\ndef _fx_state_contributions(N, p, ent):\n    N = np.asarray(N, dtype=np.float64)\n    p = np.asarray(p, dtype=np.float64)\n    HX = float(np.asarray(ent, dtype=np.float64)[0])\n    Z = N.sum()\n    pi, P = N / Z, N.sum(axis=0) / Z\n    with np.errstate(divide="ignore", invalid="ignore"):\n        term = np.where(p > 0.0, pi * np.log(np.where(p > 0.0, p, 1.0)), 0.0).sum(axis=0)\n    return (term - _fx_xlogx(P)) / HX\n\ndef _fx_modified_weights(delta, n_occupied):\n    delta = np.asarray(delta, dtype=np.float64)\n    w = np.exp(delta * (float(int(n_occupied)) / delta.size))\n    return w / w.sum()\n\ndef _raises(fn, *args):\n    try:\n        fn(*args)\n        return 0\n    except ValueError:\n        return 1\n\n'
    return [
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nweights = _fx_modified_weights(contributions, n_occupied)\nk = 10000.0\n',
            'call': 'adjusted_probabilities(resource_matrix, weights, k)',
            'gold_call': '_oracle_adjusted_probabilities(resource_matrix, weights, k)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[50, 30, 10], [10, 30, 50], [5, 30, 55]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nweights = _fx_modified_weights(contributions, n_occupied)\nk = 10000.0\n',
            'call': 'adjusted_probabilities(resource_matrix, weights, k)',
            'gold_call': '_oracle_adjusted_probabilities(resource_matrix, weights, k)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[10, 10, 10, 0, 0, 0], [10, 0, 10, 0, 0, 0], [0, 0, 10, 0, 0, 0], [10, 0, 12, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nweights = _fx_modified_weights(contributions, n_occupied)\nk = 1000.0\n',
            'call': 'adjusted_probabilities(resource_matrix, weights, k)',
            'gold_call': '_oracle_adjusted_probabilities(resource_matrix, weights, k)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nweights = _fx_modified_weights(contributions, n_occupied)\nk = 0.0\ndef run_model():\n    try:\n        adjusted_probabilities(resource_matrix, weights, k)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': '_raises(_oracle_adjusted_probabilities, resource_matrix, weights, k)',
        },
    ]
