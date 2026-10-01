"""
Orchestrator. For a resource matrix and the matrix-expansion constant k, return the source's noncircular niche overlap gamma' (step 07) between the two species with the largest noncircular niche breadths beta' (step 06), every weighting factor being the source's modified factor of step 04 built from steps 01-03 under the source's noncircularity rule: the factors used for one species' breadth come from the resource matrix with that species removed, and the factors used for a pair's overlap come from the matrix with both species removed, while the count of occupied states passed to step 04 is always that of the complete matrix. Adjusted probabilities (step 05) of the focal species are formed from their own rows with those factors. Raise ValueError if the two largest breadths are not uniquely determined (a tie at the first or second rank) or if fewer than three species are present. Call the earlier step functions rather than reimplementing them.

Computing a species' weighting factors from data that include the species itself makes its niche metrics partly self-referential; the source follows the classical remedy of computing the factors from the other species only. The overlap between the two broadest-niche species of a community is a natural summary of how strongly its generalists compete for the same resources.

Returns
-------
float, the noncircular niche overlap gamma' between the two species with the largest noncircular niche breadths.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generalist_pair_overlap(resource_matrix: "numpy.ndarray", k: float) -> float:
    """Orchestrator. For a resource matrix and the matrix-expansion constant k, return the source's noncircular niche overlap gamma' (step 07) between the two species with the largest noncircular niche breadths beta' (step 06), every weighting factor being the source's modified factor of step 04 built from steps 01-03 under the source's noncircularity rule: the factors used for one species' breadth come from the resource matrix with that species removed, and the factors used for a pair's overlap come from the matrix with both species removed, while the count of occupied states passed to step 04 is always that of the complete matrix. Adjusted probabilities (step 05) of the focal species are formed from their own rows with those factors. Raise ValueError if the two largest breadths are not uniquely determined (a tie at the first or second rank) or if fewer than three species are present. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r), s >= 3, of finite, nonnegative abundances.
    k : float
        Matrix-expansion constant of the source, greater than 1.

    Returns
    -------
    overlap : float
        The noncircular niche overlap gamma' of the two broadest-niche species, as a native Python float.

    Raises
    ------
    ValueError
        If resource_matrix is invalid for the earlier steps, has fewer than three species, k does not exceed 1, or the two largest noncircular breadths are tied.
    """
    return overlap

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


def _oracle_generalist_pair_overlap(resource_matrix: "numpy.ndarray", k: float) -> float:
    """ORCHESTRATOR: noncircular ed-weighted Horn overlap (Eq 39) between the two species with the largest
    noncircular niche breadths beta' (Eq 37), Sec 6 noncircularity throughout."""
    N = _check_matrix(resource_matrix)
    if not np.isfinite(k) or k <= 1.0:
        raise ValueError("k must exceed 1")
    s, r = N.shape
    if s < 3:
        raise ValueError("at least three species are needed for a noncircular pair overlap")
    n_occupied = int((N.sum(axis=0) > 0.0).sum())              # r' of the COMPLETE matrix
    # noncircular niche breadth of every species: factors from the matrix without that species
    breadth = np.empty(s)
    for i in range(s):
        reduced = np.delete(N, i, axis=0)
        p = _oracle_use_probabilities(reduced)
        ent = _oracle_resource_entropies(reduced)
        delta = _oracle_state_contributions(reduced, p, ent)
        ed = _oracle_modified_weights(delta, n_occupied)
        pstar = _oracle_adjusted_probabilities(N[i:i + 1], ed, k)  # the focal species' own row
        breadth[i] = _oracle_niche_breadth(pstar[0], ed, k)
    order = np.argsort(-breadth, kind="stable")
    if breadth[order[1]] == breadth[order[2]] or breadth[order[0]] == breadth[order[1]]:
        raise ValueError("the two broadest niches are not uniquely determined (tied breadths)")
    i, h = int(order[0]), int(order[1])
    # noncircular overlap of the pair: factors from the matrix without both species
    reduced = np.delete(N, [i, h], axis=0)
    p = _oracle_use_probabilities(reduced)
    ent = _oracle_resource_entropies(reduced)
    delta = _oracle_state_contributions(reduced, p, ent)
    ed = _oracle_modified_weights(delta, n_occupied)
    pstar = _oracle_adjusted_probabilities(N[[i, h]], ed, k)
    return _oracle_niche_overlap(pstar[0], pstar[1], ed, k)

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
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nk = 10000.0\n',
            'call': 'generalist_pair_overlap(resource_matrix, k)',
            'gold_call': '_oracle_generalist_pair_overlap(resource_matrix, k)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[20, 5, 0, 15, 0], [0, 12, 18, 6, 4], [9, 9, 9, 9, 0], [0, 0, 30, 0, 0], [3, 0, 0, 0, 27]], dtype=float)\nk = 10000.0\n',
            'call': 'generalist_pair_overlap(resource_matrix, k)',
            'gold_call': '_oracle_generalist_pair_overlap(resource_matrix, k)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[10, 10, 10, 0, 0, 0], [10, 0, 10, 0, 0, 0], [0, 0, 10, 0, 0, 0], [10, 0, 12, 0, 0, 0]], dtype=float)\nk = 1000.0\n',
            'call': 'generalist_pair_overlap(resource_matrix, k)',
            'gold_call': '_oracle_generalist_pair_overlap(resource_matrix, k)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[5.0, 1.0], [1.0, 5.0]])\nk = 10000.0\ndef run_model():\n    try:\n        generalist_pair_overlap(resource_matrix, k)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': '_raises(_oracle_generalist_pair_overlap, resource_matrix, k)',
        },
    ]
