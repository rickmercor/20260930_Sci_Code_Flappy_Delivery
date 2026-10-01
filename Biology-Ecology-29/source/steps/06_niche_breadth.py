"""
Return the source's standardized niche breadth beta' of one species from its row of adjusted probabilities (step 05), the per-state weighting factors used to form them, and the matrix-expansion constant k. The breadth equals 1 for a species that uses every resource state equally and is smaller for a species that concentrates its use on fewer states.

Niche breadth measures how evenly a species spreads its use over the resource states; the source's standardized form makes the value comparable across species within one analysis, the constant k fixing the scale of the adjusted probabilities.

Returns
-------
float, the standardized niche breadth beta' of one species.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def niche_breadth(adjusted_row: "numpy.ndarray", weights: "numpy.ndarray", k: float) -> float:
    """Return the source's standardized niche breadth beta' of one species from its row of adjusted probabilities (step 05), the per-state weighting factors used to form them, and the matrix-expansion constant k. The breadth equals 1 for a species that uses every resource state equally and is smaller for a species that concentrates its use on fewer states. With p*_j the species' adjusted probabilities and w_j the weighting factors, beta' = -(k / ln k) sum over j of w_j p*_j ln p*_j, with 0 ln 0 = 0.

    Parameters
    ----------
    adjusted_row : numpy.ndarray
        Array of shape (r,): the species' adjusted probabilities p*_ij.
    weights : numpy.ndarray
        Array of shape (r,) of finite, nonnegative per-state weighting factors.
    k : float
        Matrix-expansion constant of the source, greater than 1.

    Returns
    -------
    breadth : float
        The standardized niche breadth beta' of the species, as a native Python float.

    Raises
    ------
    ValueError
        If the arrays are not 1-D of the same length, an entry is negative or non-finite, k does not exceed 1, or the adjusted row has no positive entry.
    """
    return breadth

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _xlogx(a):
    """x ln x with the convention 0 ln 0 = 0 (vectorised)."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)


def _oracle_niche_breadth(adjusted_row: "numpy.ndarray", weights: "numpy.ndarray", k: float) -> float:
    """Standardized niche breadth beta' of Eq 37 for one species: -(k/ln k) sum_j w_j p*_j ln p*_j."""
    ps = np.asarray(adjusted_row, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if ps.ndim != 1 or ps.size < 1 or w.shape != ps.shape:
        raise ValueError("adjusted_row and weights must be 1-D arrays of the same length")
    if not np.all(np.isfinite(ps)) or np.any(ps < 0.0) or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("adjusted_row and weights must be finite and nonnegative")
    if not np.isfinite(k) or k <= 1.0:
        raise ValueError("k must exceed 1 (ln k must be positive)")
    if ps.sum() <= 0.0:
        raise ValueError("adjusted_row must contain a positive entry")
    return float(-(float(k) / np.log(float(k))) * (w * _xlogx(ps)).sum())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    # Inputs of the step under test are built by _fx_* copies of the upstream reference
    # arithmetic (input validation omitted), so no setup statement depends on an oracle.
    fixture = 'import numpy as np\n\ndef _fx_xlogx(a):\n    a = np.asarray(a, dtype=np.float64)\n    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)\n\ndef _fx_use_probabilities(N):\n    N = np.asarray(N, dtype=np.float64)\n    Y = N.sum(axis=1)\n    return np.where(Y[:, None] > 0.0, N / np.where(Y[:, None] > 0.0, Y[:, None], 1.0), 0.0)\n\ndef _fx_resource_entropies(N):\n    N = np.asarray(N, dtype=np.float64)\n    Z = N.sum()\n    P, Q, pi = N.sum(axis=0) / Z, N.sum(axis=1) / Z, N / Z\n    HX, HY, HXY = -_fx_xlogx(P).sum(), -_fx_xlogx(Q).sum(), -_fx_xlogx(pi).sum()\n    return np.array([HX, HY, HXY, HXY - HY])\n\ndef _fx_state_contributions(N, p, ent):\n    N = np.asarray(N, dtype=np.float64)\n    p = np.asarray(p, dtype=np.float64)\n    HX = float(np.asarray(ent, dtype=np.float64)[0])\n    Z = N.sum()\n    pi, P = N / Z, N.sum(axis=0) / Z\n    with np.errstate(divide="ignore", invalid="ignore"):\n        term = np.where(p > 0.0, pi * np.log(np.where(p > 0.0, p, 1.0)), 0.0).sum(axis=0)\n    return (term - _fx_xlogx(P)) / HX\n\ndef _fx_modified_weights(delta, n_occupied):\n    delta = np.asarray(delta, dtype=np.float64)\n    w = np.exp(delta * (float(int(n_occupied)) / delta.size))\n    return w / w.sum()\n\ndef _fx_adjusted_probabilities(N, w, k):\n    N = np.asarray(N, dtype=np.float64)\n    w = np.asarray(w, dtype=np.float64)\n    Ys = (w[None, :] * float(k) * N).sum(axis=1)\n    return np.where(Ys[:, None] > 0.0, N / np.where(Ys[:, None] > 0.0, Ys[:, None], 1.0), 0.0)\n\ndef _raises(fn, *args):\n    try:\n        fn(*args)\n        return 0\n    except ValueError:\n        return 1\n\n'
    return [
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nweights = _fx_modified_weights(contributions, n_occupied)\nk = 10000.0\nadjusted = _fx_adjusted_probabilities(resource_matrix, weights, k)\nadjusted_row = adjusted[0]\n',
            'call': 'niche_breadth(adjusted_row, weights, k)',
            'gold_call': '_oracle_niche_breadth(adjusted_row, weights, k)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nweights = _fx_modified_weights(contributions, n_occupied)\nk = 10000.0\nadjusted = _fx_adjusted_probabilities(resource_matrix, weights, k)\nadjusted_row = adjusted[3]\n',
            'call': 'niche_breadth(adjusted_row, weights, k)',
            'gold_call': '_oracle_niche_breadth(adjusted_row, weights, k)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[20, 5, 0, 15, 0], [0, 12, 18, 6, 4], [9, 9, 9, 9, 0], [0, 0, 30, 0, 0], [3, 0, 0, 0, 27]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nweights = _fx_modified_weights(contributions, n_occupied)\nk = 10000.0\nadjusted = _fx_adjusted_probabilities(resource_matrix, weights, k)\nadjusted_row = adjusted[2]\n',
            'call': 'niche_breadth(adjusted_row, weights, k)',
            'gold_call': '_oracle_niche_breadth(adjusted_row, weights, k)',
            'tol': 1e-10,
        },
        {
            # noncircular breadth of the broadest benchmark species (row 0): factors from the matrix without it
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nreduced = np.delete(resource_matrix, 0, axis=0)\np = _fx_use_probabilities(reduced)\nent = _fx_resource_entropies(reduced)\nweights = _fx_modified_weights(_fx_state_contributions(reduced, p, ent), n_occupied)\nk = 10000.0\nadjusted_row = _fx_adjusted_probabilities(resource_matrix[0:1], weights, k)[0]\n',
            "call": 'niche_breadth(adjusted_row, weights, k)',
            "gold_call": '_oracle_niche_breadth(adjusted_row, weights, k)',
            "tol": 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nweights = _fx_modified_weights(contributions, n_occupied)\nk = 10000.0\nadjusted = _fx_adjusted_probabilities(resource_matrix, weights, k)\nadjusted_row = adjusted[0]\nk = 1.0\ndef run_model():\n    try:\n        niche_breadth(adjusted_row, weights, k)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': '_raises(_oracle_niche_breadth, adjusted_row, weights, k)',
        },
    ]
