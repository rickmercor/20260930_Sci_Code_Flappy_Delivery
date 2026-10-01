"""
Apply the Liouvillian L[O] = i [H, O] to an operator O given as the real coefficient vector v of length 4^M over the Pauli-string basis of M sites, where H is given by its (n_terms, 2) array of [code, coefficient] rows. The Pauli strings are orthonormal under <A, B> = Tr(A^dagger B) / 2^M and the result is again a real vector of length 4^M. Raise ValueError if terms is not (n_terms, 2) or if v does not have length 4^M. Pauli strings are encoded as in step 1: code = sum_s f_s 4^s with f_s = 0, 1, 2, 3 for I, X, Z, Y on site s = 0..M-1, site 1 being the least significant digit s = 0.

The Heisenberg-picture generator acts on Pauli strings by commutation with each term of the Hamiltonian: a term either commutes with a string or maps it, up to a sign, to a single other string.

Returns
-------
ndarray of float64 with shape (4^M,), the coefficients of i [H, O].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def liouvillian_apply(terms, M, v):
    """ndarray of float64 with shape (4^M,), the coefficients of i [H, O]."""
    return np.zeros(4 ** int(M), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_liouvillian_apply(terms, M, v):
    """L[O] = i [H, O] on the Pauli-basis coefficient vector v (length 4^M). For a Hermitian
    term T and string P that anticommute, i[T, P] = 2 i T P and T P = phase * (T xor P) with
    phase in {+i, -i}, so every entry of L v is real. Commuting pairs contribute nothing."""
    terms = np.asarray(terms, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    M = int(M)
    D = 4 ** M
    if terms.ndim != 2 or terms.shape[1] != 2 or v.shape != (D,):
        raise ValueError("terms must be (n_terms, 2) and v must have length 4^M")
    codes = np.arange(D, dtype=np.int64)
    # single-site product table: O(f1) O(f2) = i^K[f1,f2] O(f1 xor f2), f = x + 2z, O = i^{xz} X^x Z^z
    K = np.zeros((4, 4), dtype=np.int64)
    ops = {0: np.eye(2), 1: np.array([[0, 1], [1, 0]]), 2: np.diag([1.0, -1.0]), 3: np.array([[0, -1j], [1j, 0]])}
    for a in range(4):
        for b in range(4):
            ph = np.trace(ops[a ^ b].conj().T @ (ops[a] @ ops[b])) / 2.0
            K[a, b] = int(round(np.angle(ph) / (np.pi / 2))) % 4
    out = np.zeros(D, dtype=np.float64)
    for code_f, coef in terms:
        T = int(code_f)
        kexp = np.zeros(D, dtype=np.int64)
        anti = np.zeros(D, dtype=np.int64)
        for s in range(M):
            t_s = (T >> (2 * s)) & 3
            if t_s == 0:
                continue
            p_s = (codes >> (2 * s)) & 3
            kexp += K[t_s, p_s]
            anti += ((p_s != 0) & (p_s != t_s)).astype(np.int64)
        mask = (anti % 2) == 1
        sign = np.real(1j ** ((kexp[mask] + 1) % 4))      # i [T, P] = 2 i T P, real
        np.add.at(out, codes[mask] ^ T, 2.0 * coef * sign * v[mask])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "terms = _oracle_doubled_terms(_oracle_lfxxz_terms(4, 1.0, 0.3, 1.0), 4)\nM = 8\nv = np.zeros(4 ** 8)\nv[1 + 4 ** 4] = 1.0",
            "call": "liouvillian_apply(terms, M, v)",
            "gold_call": "_oracle_liouvillian_apply(terms, M, v)",
        },
        {
            "setup": "terms = _oracle_doubled_terms(_oracle_lfxxz_terms(4, 1.0, 0.3, 1.0), 4)\nM = 8\nv = np.zeros(4 ** 8)\nv[2] = 0.5\nv[3 * 4 ** 5] = -1.25\nv[1 + 3 * 4 + 2 * 4 ** 6] = 2.0",
            "call": "liouvillian_apply(terms, M, v)",
            "gold_call": "_oracle_liouvillian_apply(terms, M, v)",
        },
        {
            "setup": "terms = _oracle_lfxxz_terms(4, 1.0, 0.3, 1.0)\nM = 4\nv = np.arange(256, dtype=float) / 256.0",
            "call": "liouvillian_apply(terms, M, v)",
            "gold_call": "_oracle_liouvillian_apply(terms, M, v)",
        },
    ]
