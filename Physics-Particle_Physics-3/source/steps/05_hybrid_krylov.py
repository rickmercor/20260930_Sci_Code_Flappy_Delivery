"""
Algorithm 2 of the source in its hybrid form: starting from the normalised root string, build the pruned Krylov operator basis inside the span of the strings S_codes (the codes kept by the graph pruning) by repeated commutation with H, orthonormalisation, and the source's truncation rule with threshold eps2; a candidate whose orthogonal remainder has norm below tol ends the construction, and at most kmax vectors are produced. Return the basis as a (K, 4^M) array of real Pauli-basis coefficient vectors, the root first. Raise ValueError if root is not a valid code, if it is not among S_codes, if eps2 or tol is not in (0, 1), or if kmax is not a positive integer. Pauli strings are encoded as in step 1: code = sum_s f_s 4^s with f_s = 0, 1, 2, 3 for I, X, Z, Y on site s = 0..M-1, site 1 being the least significant digit s = 0.

The source's Krylov pruning terminates the operator Lanczos iteration with a criterion that tracks how much of the generated operator survives orthogonalisation over the whole trajectory, which differs from the conventional criterion on the last remainder alone. In the hybrid scheme the iteration is confined to the operator space selected by the graph.

Returns
-------
ndarray of float64 with shape (K, 4^M), orthonormal rows, row 0 the root string.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hybrid_krylov(terms, M, root, S_codes, eps2, tol, kmax):
    """ndarray of float64 with shape (K, 4^M), orthonormal rows, row 0 the root string."""
    return np.zeros((1, 4 ** int(M)), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_hybrid_krylov(terms, M, root, S_codes, eps2, tol, kmax):
    """Algorithm 2 (hybrid), using the task's +i Liouvillian basis convention. K_0 = root string. For m = 0, 1, ...:
    O~ = i[H, K_m] projected onto span(S); n_pre = ||O~||; orthogonalise against ALL K_r;
    n_post = ||O_perp||; stop if n_post < tol; w <- w * n_post / n_pre; stop if w < eps2;
    K_{m+1} = O_perp / n_post. At most kmax vectors. Returns (K, 4^M) float64 array."""
    terms = np.asarray(terms, dtype=np.float64)
    M = int(M)
    D = 4 ** M
    if int(root) != root or not (0 <= root < D):
        raise ValueError("root must be a Pauli code in [0, 4^M)")
    if not (0.0 < eps2 < 1.0) or not (0.0 < tol < 1.0) or int(kmax) != kmax or kmax < 1:
        raise ValueError("need 0 < eps2 < 1, 0 < tol < 1 and kmax a positive integer")
    S = np.asarray(S_codes, dtype=np.float64).reshape(-1)
    mask = np.zeros(D, dtype=bool)
    mask[S.astype(np.int64)] = True
    if not mask[root]:
        raise ValueError("the root string must belong to S")
    K = [np.zeros(D)]
    K[0][root] = 1.0
    w = 1.0
    for m in range(int(kmax) - 1):
        Ot = _oracle_liouvillian_apply(terms, M, K[m])              # i[H, K_m], following the step 3 generator
        Ot = np.where(mask, Ot, 0.0)                                  # projection onto span(S)
        npre = float(np.sqrt(Ot @ Ot))
        Op = Ot.copy()
        for r in range(len(K)):                                       # full Gram-Schmidt
            Op = Op - (K[r] @ Op) * K[r]
        npost = float(np.sqrt(Op @ Op))
        if npost < tol:
            break
        w = w * npost / npre
        if w < eps2:
            break
        K.append(Op / npost)
    return np.vstack(K)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "terms = _oracle_doubled_terms(_oracle_lfxxz_terms(4, 1.0, 0.3, 1.0), 4)\nroot = 1 + 4 ** 4\nS = _oracle_graph_pruning(terms, 8, root, 0.02)\nM = 8\nS_codes = S[0]\neps2, tol, kmax = 1e-6, 1e-12, 100",
            "call": "hybrid_krylov(terms, M, root, S_codes, eps2, tol, kmax)",
            "gold_call": "_oracle_hybrid_krylov(terms, M, root, S_codes, eps2, tol, kmax)",
            "tol": 1e-08,
        },
        {
            "setup": "terms = _oracle_doubled_terms(_oracle_lfxxz_terms(4, 1.0, 0.3, 1.0), 4)\nroot = 1 + 4 ** 4\nS = _oracle_graph_pruning(terms, 8, root, 0.02)\nM = 8\nS_codes = S[0]\neps2, tol, kmax = 1e-3, 1e-12, 100",
            "call": "hybrid_krylov(terms, M, root, S_codes, eps2, tol, kmax)",
            "gold_call": "_oracle_hybrid_krylov(terms, M, root, S_codes, eps2, tol, kmax)",
            "tol": 1e-08,
        },
        {
            "setup": "terms = _oracle_doubled_terms(_oracle_lfxxz_terms(4, 1.0, 0.3, 1.0), 4)\nroot = 1 + 4 ** 4\nS = _oracle_graph_pruning(terms, 8, root, 0.02)\nM = 8\nS_codes = S[0]\neps2, tol, kmax = 1e-9, 1e-12, 20",
            "call": "hybrid_krylov(terms, M, root, S_codes, eps2, tol, kmax)",
            "gold_call": "_oracle_hybrid_krylov(terms, M, root, S_codes, eps2, tol, kmax)",
            "tol": 1e-08,
        },
    ]
