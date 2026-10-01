"""
Return the expectation value <psi| P |psi> of every Pauli string P on M sites, indexed by its code, for the normalised state vector psi with site 1 the leftmost tensor factor. Raise ValueError if psi does not have length 2^M. Pauli strings are encoded as in step 1: code = sum_s f_s 4^s with f_s = 0, 1, 2, 3 for I, X, Z, Y on site s = 0..M-1, site 1 being the least significant digit s = 0.

The shadow vector is initialised from the expectation values of the basis operators in the initial state; for a general state these are the Pauli-basis coefficients of its density matrix.

Returns
-------
ndarray of float64 with shape (4^M,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pauli_expectations(psi, M):
    """ndarray of float64 with shape (4^M,)."""
    return np.zeros(4 ** int(M), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pauli_expectations(psi, M):
    """<psi| P |psi> for every Pauli code P on M sites (site 1 the leftmost tensor factor),
    as a real vector of length 4^M, obtained by the site-by-site Pauli transform of
    rho = |psi><psi|."""
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    M = int(M)
    if psi.shape != (2 ** M,):
        raise ValueError("psi must have length 2^M")
    rho = np.outer(psi, psi.conj())
    out = rho.reshape([2] * M + [2] * M)
    for s in range(M):
        A = np.moveaxis(out, [0, M - s], [0, 1])       # row axis of site s+1, its column axis
        r00, r01, r10, r11 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
        # Tr(rho_s I), Tr(rho_s X), Tr(rho_s Z), Tr(rho_s Y) appended as a trailing axis
        out = np.stack([r00 + r11, r01 + r10, r00 - r11, 1j * r01 - 1j * r10], axis=-1)
    out = np.moveaxis(out, list(range(M)), list(range(M))[::-1])   # site 1 -> last (least significant) axis
    return np.real(out.reshape(-1))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "terms = _oracle_lfxxz_terms(4, 1.0, 0.3, 1.0)\ntermsD = _oracle_doubled_terms(terms, 4)\ntfd = _oracle_thermofield_state(terms, 4, 2.0)\nsgn = np.where(((np.arange(4 ** 4) >> 7) & 1) == 1, -1.0, 1.0)\npsi0 = tfd * sgn\npsi = psi0\nM = 8",
            "call": "pauli_expectations(psi, M)",
            "gold_call": "_oracle_pauli_expectations(psi, M)",
        },
        {
            "setup": "psi = np.zeros(16, dtype=complex)\npsi[0] = 0.6\npsi[5] = 0.8j\nM = 4",
            "call": "pauli_expectations(psi, M)",
            "gold_call": "_oracle_pauli_expectations(psi, M)",
        },
        {
            "setup": "psi = np.ones(4, dtype=complex) / 2.0\nM = 2",
            "call": "pauli_expectations(psi, M)",
            "gold_call": "_oracle_pauli_expectations(psi, M)",
        },
    ]
