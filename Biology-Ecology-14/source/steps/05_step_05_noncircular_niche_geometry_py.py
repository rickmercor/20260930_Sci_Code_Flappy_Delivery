"""
Build the complete noncircular niche geometry for a community

Use the recent modified factors for every leave-one-species breadth and every leave-two-species overlap.  The occupied-state count is always taken from the complete matrix.  The returned square matrix stores standardized breadths on its diagonal and pairwise information-theoretic overlaps off the diagonal.

Returns
-------
return geometry

Returns
-------
return geometry
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def noncircular_niche_geometry(resource_matrix: "numpy.ndarray", k: float) -> "numpy.ndarray":
    """Return the square noncircular niche-geometry matrix.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Species-by-resource matrix of finite nonnegative counts, with at
        least three species and two resource states.
    k : float
        Matrix-expansion constant, strictly greater than one.

    Returns
    -------
    geometry : numpy.ndarray
        Square float64 array. geometry[i,i] is species i's noncircular
        standardized breadth and geometry[i,h] is the noncircular modified-
        factor Horn overlap of species i and h. With ed_j the modified
        factors of the reduced matrix (without species i for a breadth,
        without species i and h for an overlap; step 04, using the complete
        matrix's occupied-state count), the adjusted use of a focal species
        is p*_ij = N_ij / sum_l(ed_l * k * N_il). The breadth is
        -(k / ln k) * sum_j ed_j * p*_ij * ln(p*_ij), and the overlap is
        -(1 / (2 ln 2)) * sum_j ed_j * k * [I(p*_ij) + I(p*_hj)
        - I(p*_ij + p*_hj)], with I(x) = x ln x and 0 ln 0 = 0.

    Raises
    ------
    ValueError
        If resource_matrix is not a finite nonnegative two-dimensional
        matrix with at least three nonempty species and two resource states,
        fewer than two resource states are occupied, or k is not finite and
        strictly greater than one.
    """
    return geometry

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _xlogx_geometry(a):
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)


def _adjusted_geometry(N, weights, k):
    totals = (weights[None, :] * float(k) * N).sum(axis=1)
    if np.any((N.sum(axis=1) > 0.0) & (totals <= 0.0)):
        raise ValueError("positive species abundance must have positive weighted abundance")
    return np.where(totals[:, None] > 0.0,
                    N / np.where(totals[:, None] > 0.0, totals[:, None], 1.0), 0.0)


def _breadth_geometry(row, weights, k):
    if row.sum() <= 0.0:
        raise ValueError("every focal species must use at least one resource state")
    return float(-(float(k) / np.log(float(k))) * (weights * _xlogx_geometry(row)).sum())


def _overlap_geometry(a, b, weights, k):
    if a.sum() <= 0.0 or b.sum() <= 0.0:
        raise ValueError("both focal species must use at least one resource state")
    bracket = _xlogx_geometry(a) + _xlogx_geometry(b) - _xlogx_geometry(a + b)
    return float(-(weights * float(k) * bracket).sum() / (2.0 * np.log(2.0)))


def _oracle_noncircular_niche_geometry(resource_matrix: "numpy.ndarray", k: float) -> "numpy.ndarray":
    N = np.asarray(resource_matrix, dtype=np.float64)
    if (N.ndim != 2 or N.shape[0] < 3 or N.shape[1] < 2
            or not np.all(np.isfinite(N)) or np.any(N < 0.0)
            or np.any(N.sum(axis=1) <= 0.0) or not np.isfinite(k) or k <= 1.0):
        raise ValueError("a finite nonnegative matrix with three positive species and k > 1 is required")
    occupied = int((N.sum(axis=0) > 0.0).sum())
    if occupied < 2:
        raise ValueError("at least two resource states must be occupied")
    n = N.shape[0]
    result = np.eye(n, dtype=np.float64)

    def _factors(reduced):
        p = _oracle_use_probabilities(reduced)
        ent = _oracle_resource_entropies(reduced)
        delta = _oracle_state_contributions(reduced, p, ent)
        return _oracle_modified_weights(delta, occupied)

    for i in range(n):
        weights = _factors(np.delete(N, i, axis=0))
        row = _adjusted_geometry(N[i:i + 1], weights, k)[0]
        result[i, i] = _breadth_geometry(row, weights, k)
    for i in range(n):
        for h in range(i):
            weights = _factors(np.delete(N, [h, i], axis=0))
            rows = _adjusted_geometry(N[[h, i]], weights, k)
            result[h, i] = result[i, h] = _overlap_geometry(rows[0], rows[1], weights, k)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nN=np.array([[7,39,11,9,0,6,0],[39,0,25,0,31,4,0],[0,0,0,21,37,10,0],[0,0,0,0,14,0,0],[0,0,0,37,0,0,0],[0,38,0,0,0,0,0]],float)\nk=10000.0",
            "call": "noncircular_niche_geometry(N,k)",
            "gold_call": "_oracle_noncircular_niche_geometry(N,k)",
            "tol": 1e-9,
        },
        {
            "setup": "import numpy as np\nN=np.array([[20,5,0,15,0],[0,12,18,6,4],[9,9,9,9,0],[0,0,30,0,0],[3,0,0,0,27]],float)\nk=1000.0",
            "call": "noncircular_niche_geometry(N,k)",
            "gold_call": "_oracle_noncircular_niche_geometry(N,k)",
            "tol": 1e-9,
        },
        {
            "setup": "import numpy as np\nN=np.array([[12,3,0,5],[0,8,9,1],[4,4,4,4],[0,0,11,7]],float)\nk=2500.0",
            "call": "noncircular_niche_geometry(N,k)",
            "gold_call": "_oracle_noncircular_niche_geometry(N,k)",
            "tol": 1e-9,
        },
        {
            "setup": "import numpy as np\nN=np.array([[1,0],[0,1]],float)\nk=100.0\ndef run(fn):\n    try: fn(N,k)\n    except ValueError: return 1.0\n    return 0.0",
            "call": "run(noncircular_niche_geometry)",
            "gold_call": "run(_oracle_noncircular_niche_geometry)",
            "tol": 0.0,
        },
    ]
