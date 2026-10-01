"""
Return the source's weighted information-theoretic niche overlap gamma' between two species from their rows of adjusted probabilities (step 05), the per-state weighting factors used to form them, and the matrix-expansion constant k. The overlap is 0 for species using disjoint resource states and 1 for species with identical adjusted distributions.

Niche overlap measures the extent to which two species share resource states; the source's weighted form credits shared use of a state in proportion to that state's weighting factor.

Returns
-------
float, the niche overlap gamma' between the two species.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def niche_overlap(adjusted_row_i: "numpy.ndarray", adjusted_row_h: "numpy.ndarray",
                  weights: "numpy.ndarray", k: float) -> float:
    """Return the source's weighted information-theoretic niche overlap gamma' between two species from their rows of adjusted probabilities (step 05), the per-state weighting factors used to form them, and the matrix-expansion constant k. The overlap is 0 for species using disjoint resource states and 1 for species with identical adjusted distributions. With a_j and b_j the two species' adjusted probabilities, w_j the weighting factors and I(x) = x ln x (0 ln 0 = 0), gamma' = -(1 / (2 ln 2)) sum over j of w_j k [I(a_j) + I(b_j) - I(a_j + b_j)].

    Parameters
    ----------
    adjusted_row_i : numpy.ndarray
        Array of shape (r,): adjusted probabilities of the first species.
    adjusted_row_h : numpy.ndarray
        Array of shape (r,): adjusted probabilities of the second species.
    weights : numpy.ndarray
        Array of shape (r,) of finite, nonnegative per-state weighting factors.
    k : float
        Positive matrix-expansion constant of the source.

    Returns
    -------
    overlap : float
        The niche overlap gamma' between the two species, as a native Python float.

    Raises
    ------
    ValueError
        If the arrays are not 1-D of the same length, an entry is negative or non-finite, k is not positive, or either adjusted row has no positive entry.
    """
    return overlap

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _xlogx(a):
    """x ln x with the convention 0 ln 0 = 0 (vectorised)."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)


def _oracle_niche_overlap(adjusted_row_i: "numpy.ndarray", adjusted_row_h: "numpy.ndarray",
                          weights: "numpy.ndarray", k: float) -> float:
    """Horn-type niche overlap gamma' of Eq 39 between two species from their adjusted probabilities."""
    a = np.asarray(adjusted_row_i, dtype=np.float64)
    b = np.asarray(adjusted_row_h, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if a.ndim != 1 or a.size < 1 or b.shape != a.shape or w.shape != a.shape:
        raise ValueError("the two adjusted rows and weights must be 1-D arrays of the same length")
    for arr in (a, b, w):
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError("adjusted rows and weights must be finite and nonnegative")
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("k must be positive and finite")
    if a.sum() <= 0.0 or b.sum() <= 0.0:
        raise ValueError("each adjusted row must contain a positive entry")
    # I(x) = x ln x; the k inside the bracket cancels against the 1/k of p*, so gamma' is k-free
    val = -(1.0 / (2.0 * np.log(2.0))) * (w * float(k) * (_xlogx(a) + _xlogx(b) - _xlogx(a + b))).sum()
    return float(val)

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
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nweights = _fx_modified_weights(contributions, n_occupied)\nk = 10000.0\nadjusted = _fx_adjusted_probabilities(resource_matrix, weights, k)\nadjusted_row_i, adjusted_row_h = adjusted[0], adjusted[1]\n',
            'call': 'niche_overlap(adjusted_row_i, adjusted_row_h, weights, k)',
            'gold_call': '_oracle_niche_overlap(adjusted_row_i, adjusted_row_h, weights, k)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nweights = _fx_modified_weights(contributions, n_occupied)\nk = 10000.0\nadjusted = _fx_adjusted_probabilities(resource_matrix, weights, k)\nadjusted_row_i, adjusted_row_h = adjusted[2], adjusted[3]\n',
            'call': 'niche_overlap(adjusted_row_i, adjusted_row_h, weights, k)',
            'gold_call': '_oracle_niche_overlap(adjusted_row_i, adjusted_row_h, weights, k)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[10, 10, 10, 0, 0, 0], [10, 0, 10, 0, 0, 0], [0, 0, 10, 0, 0, 0], [10, 0, 12, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nweights = _fx_modified_weights(contributions, n_occupied)\nk = 10000.0\nadjusted = _fx_adjusted_probabilities(resource_matrix, weights, k)\nadjusted_row_i, adjusted_row_h = adjusted[1], adjusted[3]\n',
            'call': 'niche_overlap(adjusted_row_i, adjusted_row_h, weights, k)',
            'gold_call': '_oracle_niche_overlap(adjusted_row_i, adjusted_row_h, weights, k)',
            'tol': 1e-10,
        },
        {
            # the source's Table 7, matrix H, Sp. 1 & Sp. 2 (printed 0.819): factors from the matrix without both
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[10, 10, 10, 0, 0, 0], [10, 0, 10, 0, 0, 0], [0, 0, 10, 0, 0, 0], [10, 0, 10, 0, 0, 0]], dtype=float)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nreduced = np.delete(resource_matrix, [0, 1], axis=0)\np = _fx_use_probabilities(reduced)\nent = _fx_resource_entropies(reduced)\nweights = _fx_modified_weights(_fx_state_contributions(reduced, p, ent), n_occupied)\nk = 10000.0\nadjusted = _fx_adjusted_probabilities(resource_matrix[[0, 1]], weights, k)\nadjusted_row_i, adjusted_row_h = adjusted[0], adjusted[1]\n',
            "call": 'niche_overlap(adjusted_row_i, adjusted_row_h, weights, k)',
            "gold_call": '_oracle_niche_overlap(adjusted_row_i, adjusted_row_h, weights, k)',
            "tol": 1e-10,
        },
        {
            # the source's Table 7, matrix H, Sp. 2 & Sp. 4 (printed 1.000): factors from the matrix without both
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[10, 10, 10, 0, 0, 0], [10, 0, 10, 0, 0, 0], [0, 0, 10, 0, 0, 0], [10, 0, 10, 0, 0, 0]], dtype=float)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nreduced = np.delete(resource_matrix, [1, 3], axis=0)\np = _fx_use_probabilities(reduced)\nent = _fx_resource_entropies(reduced)\nweights = _fx_modified_weights(_fx_state_contributions(reduced, p, ent), n_occupied)\nk = 10000.0\nadjusted = _fx_adjusted_probabilities(resource_matrix[[1, 3]], weights, k)\nadjusted_row_i, adjusted_row_h = adjusted[0], adjusted[1]\n',
            "call": 'niche_overlap(adjusted_row_i, adjusted_row_h, weights, k)',
            "gold_call": '_oracle_niche_overlap(adjusted_row_i, adjusted_row_h, weights, k)',
            "tol": 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ncontributions = _fx_state_contributions(resource_matrix, use_probs, entropies)\nn_occupied = int((resource_matrix.sum(axis=0) > 0).sum())\nweights = _fx_modified_weights(contributions, n_occupied)\nk = 10000.0\nadjusted = _fx_adjusted_probabilities(resource_matrix, weights, k)\nadjusted_row_i, adjusted_row_h = adjusted[0], np.zeros(7)\ndef run_model():\n    try:\n        niche_overlap(adjusted_row_i, adjusted_row_h, weights, k)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': '_raises(_oracle_niche_overlap, adjusted_row_i, adjusted_row_h, weights, k)',
        },
    ]
