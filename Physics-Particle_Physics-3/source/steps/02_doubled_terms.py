"""
Return the Pauli-string terms of the doubled Hamiltonian that the source's Methods B protocol evolves on 2N sites, built from the terms of the N-site H: the first copy occupies sites 1..N and the second copy sites N+1..2N (site s of the chain becomes site N + s of the second copy). The doubled Hamiltonian is H_D = H (x) 1 - 1 (x) H^*: every term acts unchanged on the first copy and, complex conjugated and with a minus sign, on the second copy, where conjugating a Pauli string multiplies it by (-1)^(number of Y factors). Return the (2 n_terms, 2) array in which, for k = 0..n_terms-1, row 2k is the first-copy image of term k and row 2k+1 its second-copy image. Raise ValueError if terms is not (n_terms, 2), if N is not a positive integer, or if a term acts outside the N-site chain. Pauli strings are encoded as in step 1: code = sum_s f_s 4^s with f_s = 0, 1, 2, 3 for I, X, Z, Y on site s = 0..M-1, site 1 being the least significant digit s = 0.

The thermal out-of-time-order correlator is measured by evolving a doubled system in which a copy of the Hamiltonian acts on each half; the second copy enters conjugated and with the opposite sign so that the doubled Hamiltonian annihilates the thermofield double state.

Returns
-------
ndarray of float64 with shape (2 n_terms, 2): rows [code, coefficient] on 2N sites.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def doubled_terms(terms, N):
    """ndarray of float64 with shape (2 n_terms, 2): rows [code, coefficient] on 2N sites."""
    return np.zeros((2 * np.shape(terms)[0], 2), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_doubled_terms(terms, N):
    """Terms of the doubled Hamiltonian H_D = H (x) 1 - 1 (x) H^* on 2N sites (Methods B of the
    source): every term of H acts unchanged on the first copy (sites 1..N) and, complex
    conjugated and with a minus sign, on the second copy (sites N+1..2N). A Pauli string with
    n_Y factors of Y conjugates to (-1)^{n_Y} times itself."""
    terms = np.asarray(terms, dtype=np.float64)
    if terms.ndim != 2 or terms.shape[1] != 2 or int(N) != N or N < 1:
        raise ValueError("terms must be (n_terms, 2) and N a positive integer")
    N = int(N)
    rows = []
    for code_f, coef in terms:
        T = int(code_f)
        if T >= 4 ** N:
            raise ValueError("a term acts outside the N-site chain")
        n_y = sum(1 for s in range(N) if (T >> (2 * s)) & 3 == 3)
        rows.append([float(T), float(coef)])
        rows.append([float(T << (2 * N)), -float(coef) * (-1.0) ** n_y])
    return np.array(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "terms = _oracle_lfxxz_terms(4, 1.0, 0.3, 1.0)\nN = 4",
            "call": "doubled_terms(terms, N)",
            "gold_call": "_oracle_doubled_terms(terms, N)",
        },
        {
            "setup": "terms = _oracle_lfxxz_terms(3, 0.5, 0.1, 0.0)\nN = 3",
            "call": "doubled_terms(terms, N)",
            "gold_call": "_oracle_doubled_terms(terms, N)",
        },
        {
            "setup": "terms = np.array([[3.0, 0.7], [15.0, -0.2], [2.0, 1.0]])\nN = 2",
            "call": "doubled_terms(terms, N)",
            "gold_call": "_oracle_doubled_terms(terms, N)",
        },
    ]
