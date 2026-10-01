"""
Return the contribution of each resource state to the source's standardized resource heterogeneity M(X) (the quantity the source denotes delta_j), given the resource matrix, the use probabilities of step 01 and the entropies of step 02. The contributions sum to M(X); a state used by no species, or used in identical proportions by every species, contributes zero.

The source's weighting of resource states starts from the share of the shared information between species and resource that each state carries: a state whose use profile distinguishes species carries more of it than a state used alike by all.

Returns
-------
numpy.ndarray of float64 with shape (r,): the per-state contributions delta_j, summing to M(X).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def state_contributions(resource_matrix: "numpy.ndarray", use_probs: "numpy.ndarray",
                        entropies: "numpy.ndarray") -> "numpy.ndarray":
    """Return the contribution of each resource state to the source's standardized resource heterogeneity M(X) (the quantity the source denotes delta_j), given the resource matrix, the use probabilities of step 01 and the entropies of step 02. The contributions sum to M(X); a state used by no species, or used in identical proportions by every species, contributes zero.

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r) of finite, nonnegative abundances.
    use_probs : numpy.ndarray
        Array of shape (s, r) from step 01.
    entropies : numpy.ndarray
        Array of shape (4,) from step 02: [H(X), H(Y), H(XY), H_Y(X)].

    Returns
    -------
    state_contributions : numpy.ndarray
        Array of shape (r,): the contribution delta_j of every resource state (float64).

    Raises
    ------
    ValueError
        If the shapes are inconsistent, an input is negative or non-finite, or H(X) is not positive (fewer than two occupied resource states).
    """
    return state_contributions

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _xlogx(a):
    """x ln x with the convention 0 ln 0 = 0 (vectorised)."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)


def _check_matrix(resource_matrix):
    N = np.asarray(resource_matrix, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("resource_matrix must be a 2-D array with at least one species and one state")
    if not np.all(np.isfinite(N)) or np.any(N < 0.0):
        raise ValueError("resource_matrix entries must be finite and nonnegative")
    if N.sum() <= 0.0:
        raise ValueError("resource_matrix must contain at least one positive entry")
    return N


def _oracle_state_contributions(resource_matrix: "numpy.ndarray", use_probs: "numpy.ndarray",
                                entropies: "numpy.ndarray") -> "numpy.ndarray":
    """delta_j of Eq 16: the contribution of resource state j to the standardized resource heterogeneity M(X)."""
    N = _check_matrix(resource_matrix)
    p = np.asarray(use_probs, dtype=np.float64)
    ent = np.asarray(entropies, dtype=np.float64)
    if p.shape != N.shape:
        raise ValueError("use_probs must have the shape of resource_matrix")
    if ent.shape != (4,) or not np.all(np.isfinite(ent)):
        raise ValueError("entropies must be the finite array [H(X), H(Y), H(XY), H_Y(X)]")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("use_probs must be finite and nonnegative")
    HX = float(ent[0])
    if HX <= 0.0:
        raise ValueError("H(X) must be positive: the matrix needs at least two occupied resource states")
    Z = N.sum()
    pi = N / Z
    P = N.sum(axis=0) / Z
    # sum_i pi_ij ln p_ij - P_j ln P_j = sum_i pi_ij ln(p_ij / P_j): the per-state share of the shared
    # information m(X) = H(X) - H_Y(X); the states sum to M(X) = m(X)/H(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > 0.0, pi * np.log(np.where(p > 0.0, p, 1.0)), 0.0).sum(axis=0)
    return (term - _xlogx(P)) / HX

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    # Inputs of the step under test are built by _fx_* copies of the upstream reference
    # arithmetic (input validation omitted), so no setup statement depends on an oracle.
    fixture = 'import numpy as np\n\ndef _fx_xlogx(a):\n    a = np.asarray(a, dtype=np.float64)\n    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)\n\ndef _fx_use_probabilities(N):\n    N = np.asarray(N, dtype=np.float64)\n    Y = N.sum(axis=1)\n    return np.where(Y[:, None] > 0.0, N / np.where(Y[:, None] > 0.0, Y[:, None], 1.0), 0.0)\n\ndef _fx_resource_entropies(N):\n    N = np.asarray(N, dtype=np.float64)\n    Z = N.sum()\n    P, Q, pi = N.sum(axis=0) / Z, N.sum(axis=1) / Z, N / Z\n    HX, HY, HXY = -_fx_xlogx(P).sum(), -_fx_xlogx(Q).sum(), -_fx_xlogx(pi).sum()\n    return np.array([HX, HY, HXY, HXY - HY])\n\ndef _raises(fn, *args):\n    try:\n        fn(*args)\n        return 0\n    except ValueError:\n        return 1\n\n'
    return [
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\n',
            'call': 'state_contributions(resource_matrix, use_probs, entropies)',
            'gold_call': '_oracle_state_contributions(resource_matrix, use_probs, entropies)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[50, 30, 10], [10, 30, 50], [5, 30, 55]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\n',
            'call': 'state_contributions(resource_matrix, use_probs, entropies)',
            'gold_call': '_oracle_state_contributions(resource_matrix, use_probs, entropies)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[30, 30, 30, 0], [10, 30, 50, 0], [5, 0, 55, 0], [30, 30, 30, 0]], dtype=float)\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\n',
            'call': 'state_contributions(resource_matrix, use_probs, entropies)',
            'gold_call': '_oracle_state_contributions(resource_matrix, use_probs, entropies)',
            'tol': 1e-10,
        },
        {
            "setup": fixture + 'import numpy as np\nresource_matrix = np.array([[5.0, 0.0], [7.0, 0.0]])\nuse_probs = _fx_use_probabilities(resource_matrix)\nentropies = _fx_resource_entropies(resource_matrix)\ndef run_model():\n    try:\n        state_contributions(resource_matrix, use_probs, entropies)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': '_raises(_oracle_state_contributions, resource_matrix, use_probs, entropies)',
        },
    ]
