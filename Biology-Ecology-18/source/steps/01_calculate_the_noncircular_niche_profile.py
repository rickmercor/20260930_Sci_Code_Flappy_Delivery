"""
Calculate the noncircular niche profile.

Calculate the six species breadths and every pair overlap with the paper's noncircular exclusions. Keep the complete-table occupied-state count inside every exclusion; changing that count is a different method.  Reject misaligned, non-finite, or out-of-contract inputs with ValueError

Returns
-------
A float64 vector containing six breadths followed by the row-major 6 by 6 overlap matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def noncircular_niche_profile(resource_matrix: "np.ndarray", k: float) -> "np.ndarray":
    """Calculate the noncircular niche profile.

    Returns
    -------
    A float64 vector containing six breadths followed by the row-major 6 by 6 overlap matrix.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _xlogx(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 0.0, x * np.log(np.where(x > 0.0, x, 1.0)), 0.0)


def _niche_weights(matrix, n_occupied):
    n = np.asarray(matrix, dtype=np.float64)
    total = n.sum()
    y = n.sum(axis=1)
    x = n.sum(axis=0)
    p = np.divide(n, y[:, None], out=np.zeros_like(n), where=y[:, None] > 0.0)
    pi = n / total
    q = y / total
    P = x / total
    hx = -_xlogx(P).sum()
    hy = -_xlogx(q).sum()
    hxy = -_xlogx(pi).sum()
    conditional = hxy - hy
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(p > 0.0, pi * np.log(np.where(p > 0.0, p, 1.0)), 0.0).sum(axis=0)
    delta = (first - _xlogx(P)) / hx
    exponent = delta * float(n_occupied) / float(n.shape[1])
    z = np.exp(exponent - exponent.max())
    return z / z.sum(), np.array([hx, hy, hxy, conditional], dtype=np.float64), delta


def _oracle_noncircular_niche_profile(resource_matrix: "np.ndarray", k: float) -> "np.ndarray":
    n = np.asarray(resource_matrix, dtype=np.float64)
    if n.ndim != 2 or n.shape[0] < 3 or n.shape[1] < 2 or not np.isfinite(n).all() or np.any(n < 0.0):
        raise ValueError("resource_matrix must be finite, nonnegative, and at least 3 by 2")
    if not np.isfinite(k) or k <= 1.0 or np.any(n.sum(axis=1) <= 0.0):
        raise ValueError("each species must occur and k must exceed one")
    occupied = int(np.count_nonzero(n.sum(axis=0) > 0.0))
    s = n.shape[0]
    breadth = np.empty(s, dtype=np.float64)
    for i in range(s):
        w, _, _ = _niche_weights(np.delete(n, i, axis=0), occupied)
        ystar = float((w * k * n[i]).sum())
        ps = n[i] / ystar
        breadth[i] = -(k / np.log(k)) * float((w * _xlogx(ps)).sum())
    overlap = np.eye(s, dtype=np.float64)
    for i in range(s):
        for j in range(i):
            w, _, _ = _niche_weights(np.delete(n, [j, i], axis=0), occupied)
            pair = n[[j, i]]
            ystar = (w[None, :] * k * pair).sum(axis=1)
            ps = pair / ystar[:, None]
            val = -(w * k * (_xlogx(ps[0]) + _xlogx(ps[1]) - _xlogx(ps[0] + ps[1]))).sum() / (2.0 * np.log(2.0))
            overlap[i, j] = overlap[j, i] = float(val)
    return np.concatenate([breadth, overlap.ravel()]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    return [

        {

            "setup": 'import numpy as np\nresource_matrix=np.array([[7.0, 39.0, 11.0, 9.0, 0.0, 6.0, 0.0], [39.0, 0.0, 25.0, 0.0, 31.0, 4.0, 0.0], [0.0, 0.0, 0.0, 21.0, 37.0, 10.0, 0.0], [0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0], [0.0, 0.0, 0.0, 37.0, 0.0, 0.0, 0.0], [0.0, 38.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)\nk=10000.0',

            'call': 'noncircular_niche_profile(resource_matrix,k)',

            'gold_call': '_oracle_noncircular_niche_profile(resource_matrix,k)',

        },

        {

            "setup": 'import numpy as np\nresource_matrix=np.array([[3,1,0],[0,2,4],[1,1,2]],float)\nk=1000.0',

            'call': 'noncircular_niche_profile(resource_matrix,k)',

            'gold_call': '_oracle_noncircular_niche_profile(resource_matrix,k)',

        },

        {

            "setup": 'import numpy as np\nresource_matrix=np.array([[1,-1],[2,3],[1,0]],float)\nk=10.0\ndef candidate_code():\n    try:\n        noncircular_niche_profile(resource_matrix,k)\n    except ValueError:\n        return 1.0\n    return 0.0\ndef oracle_code():\n    try:\n        _oracle_noncircular_niche_profile(resource_matrix,k)\n    except ValueError:\n        return 1.0\n    return 0.0',

            'call': 'candidate_code()',

            'gold_call': 'oracle_code()',

            'tol': 0.0,

        },

    ]
