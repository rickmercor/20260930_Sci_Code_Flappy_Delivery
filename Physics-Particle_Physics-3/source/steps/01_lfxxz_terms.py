"""
Return the Pauli-string terms of the open-chain longitudinal-field XXZ Hamiltonian H = J sum_{i=1}^{N-1} (X_i X_{i+1} + Y_i Y_{i+1}) + Jz sum_{i=1}^{N-1} Z_i Z_{i+1} + hz sum_{i=1}^{N} Z_i as an (n_terms, 2) array whose rows are [code, coefficient], ordered bond by bond (XX, then YY, then ZZ for i = 1..N-1) and then the N field terms. A Pauli string on M sites is encoded by the integer code = sum_s f_s 4^s with f_s = 0, 1, 2, 3 for I, X, Z, Y on site s = 0..M-1, where site 1 of the chain is s = 0. Raise ValueError if N is not an integer of at least 2 or if a coupling is not finite.

The longitudinal-field XXZ chain with a weak ZZ coupling is the source's benchmark for higher-order correlators. Every later step works in the basis of Pauli strings labelled by this integer code.

Returns
-------
ndarray of float64 with shape (4N - 3, 2): rows [code, coefficient].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def lfxxz_terms(N, J, Jz, hz):
    """ndarray of float64 with shape (4N - 3, 2): rows [code, coefficient]."""
    return np.zeros((4 * int(N) - 3, 2), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_lfxxz_terms(N, J, Jz, hz):
    """Pauli-string terms of H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1}) + Jz sum_i Z_i Z_{i+1}
    + hz sum_i Z_i (open chain) as an (n_terms, 2) float64 array [code, coefficient], ordered
    bond by bond (XX, YY, ZZ for i = 1..N-1) and then the N field terms. Site 1 is s = 0;
    X = 1, Z = 2, Y = 3 in the base-4 digit of site s."""
    if int(N) != N or N < 2:
        raise ValueError("N must be an integer >= 2")
    for v in (J, Jz, hz):
        if not np.isfinite(v):
            raise ValueError("couplings must be finite")
    N = int(N)
    rows = []
    for s in range(N - 1):
        rows.append([1 * 4 ** s + 1 * 4 ** (s + 1), float(J)])
        rows.append([3 * 4 ** s + 3 * 4 ** (s + 1), float(J)])
        rows.append([2 * 4 ** s + 2 * 4 ** (s + 1), float(Jz)])
    for s in range(N):
        rows.append([2 * 4 ** s, float(hz)])
    return np.array(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "N, J, Jz, hz = 4, 1.0, 0.3, 1.0",
            "call": "lfxxz_terms(N, J, Jz, hz)",
            "gold_call": "_oracle_lfxxz_terms(N, J, Jz, hz)",
        },
        {
            "setup": "N, J, Jz, hz = 3, 0.5, 0.1, 0.0",
            "call": "lfxxz_terms(N, J, Jz, hz)",
            "gold_call": "_oracle_lfxxz_terms(N, J, Jz, hz)",
        },
        {
            "setup": "N, J, Jz, hz = 5, 1.0, 1.0, 0.25",
            "call": "lfxxz_terms(N, J, Jz, hz)",
            "gold_call": "_oracle_lfxxz_terms(N, J, Jz, hz)",
        },
    ]
