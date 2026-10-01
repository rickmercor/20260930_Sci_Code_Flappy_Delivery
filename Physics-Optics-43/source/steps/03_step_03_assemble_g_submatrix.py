"""
Assemble the exponent matrix G and extract the submatrix selected by a derivative pattern.

In the coherent-state (Glauber-Sudarshan) evaluation of a photon-number projection, the normally ordered operator is replaced by its coherent symbol, so that a_j -> alpha_j and a_j^dag -> alpha_j^*. Collecting the 2M variables as z = [alpha_1..alpha_M, alpha_1^*..alpha_M^*], the symbol of the evolved generating operator is exp( z^T G z / 2 ) with

    G = [[ A,    B   ],

         [ B^T,  A*  ]].

Taking one derivative with respect to alpha_j and one with respect to alpha_j^* therefore selects index j and index j + M of z. Differentiating a Gaussian exponential with respect to a set of variables and evaluating at zero returns the Hafnian of the principal submatrix on exactly those indices, so the index bookkeeping performed here fixes which Hafnian is computed downstream.

The index multiset is built by listing mode j exactly k_j times drawn from the annihilation block (index j), followed by mode j exactly k_j times drawn from the creation block (index j + M), visiting modes in increasing order within each block. Repeated indices are legitimate and correspond to more than one photon in the same mode, in which case the submatrix repeats the corresponding rows and columns. G is symmetric whenever A is symmetric, which the previous step guarantees.

Returns
-------
np.ndarray of shape (2K, 2K) with K = sum(k_vector), complex: the principal submatrix of [[A, B], [B.T, conj(A)]] on the index multiset described above, and a (0, 0) array when K == 0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def assemble_g_submatrix(A: np.ndarray, B: np.ndarray, k_vector: list) -> np.ndarray:
    """Assemble G and return the submatrix selected by k_vector.
 
    Parameters
    ----------
    A : np.ndarray
        Complex array of shape (M, M), the symmetric block of G.
    B : np.ndarray
        Complex array of shape (M, M), the mixed block of G.
    k_vector : list
        Sequence of M non-negative integers giving how many times each mode is
        selected from each block.
 
    Returns
    -------
    G_sub : np.ndarray
        Complex array of shape (2K, 2K) with K = sum(k_vector), the principal
        submatrix of [[A, B], [B.T, conj(A)]] on the selected index multiset.
        A (0, 0) array is returned when K == 0.
 
    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    K = int(sum(k_vector))
    return np.zeros((2 * K, 2 * K), dtype=complex)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_g_submatrix(A: np.ndarray, B: np.ndarray, k_vector: list) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
 
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if B.shape != A.shape:
        raise ValueError("B must have the same shape as A")
    M = A.shape[0]
    if len(k_vector) != M:
        raise ValueError("k_vector must have length M")
    for k in k_vector:
        if int(k) != k or int(k) < 0:
            raise ValueError("k_vector entries must be non-negative integers")
 
    G = np.block([[A, B], [B.T, A.conj()]])
    ids = [j for j, k in enumerate(k_vector) for _ in range(int(k))]
    ids += [j + M for j, k in enumerate(k_vector) for _ in range(int(k))]
    if not ids:
        return np.zeros((0, 0), dtype=complex)
    return G[np.ix_(ids, ids)]

# =============================================================================
# TEST CASES
# =============================================================================

_BASE = """import numpy as np
rng = np.random.default_rng(7)
M = 4
A = rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))
A = A + A.T
B = rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))
"""
 
 
def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: collision-free two-mode pattern (normal scenario) ---
        {
            "setup": _BASE + "k = [0, 1, 1, 0]\n",
            "call": "assemble_g_submatrix(A, B, k)",
            "gold_call": "_oracle_assemble_g_submatrix(A, B, k)",
        },
        # --- Valid: repeated index, two photons in a single mode ---
        {
            "setup": _BASE + "k = [0, 0, 0, 2]\n",
            "call": "assemble_g_submatrix(A, B, k)",
            "gold_call": "_oracle_assemble_g_submatrix(A, B, k)",
        },
        # --- Edge: empty selection returns a 0 x 0 array ---
        {
            "setup": _BASE + "k = [0, 0, 0, 0]\n",
            "call": "assemble_g_submatrix(A, B, k).shape",
            "gold_call": '_oracle_assemble_g_submatrix(A, B, k).shape',
        },
        # --- Boundary: block entries must land in the correct positions ---
        {
            "setup": _BASE + "k = [1, 0, 1, 0]\n",
            "call": (
                "(lambda S: (S.shape == (4, 4), bool(np.allclose(S, S.T)), "
                "bool(abs(S[0, 1] - A[0, 2]) < 1e-12), "
                "bool(abs(S[0, 2] - B[0, 0]) < 1e-12)))"
                "(assemble_g_submatrix(A, B, k))"
            ),
            "gold_call": '(lambda S: (S.shape == (4, 4), bool(np.allclose(S, S.T)), bool(abs(S[0, 1] - A[0, 2]) < 1e-12), bool(abs(S[0, 2] - B[0, 0]) < 1e-12)))(_oracle_assemble_g_submatrix(A, B, k))',
        },
    ]
