"""
Return the thermofield double state of the N-site Hamiltonian given by terms at inverse temperature beta, as the state vector on 2N sites that the source's Methods B protocol starts from (before the perturbation is applied), with the first copy of the chain as the leftmost N tensor factors and site 1 the leftmost factor of each copy; normalise it to unit norm. Raise ValueError if terms is not (n_terms, 2), if N is not a positive integer, or if beta is negative or not finite. Pauli strings are encoded as in step 1: code = sum_s f_s 4^s with f_s = 0, 1, 2, 3 for I, X, Z, Y on site s = 0..M-1, site 1 being the least significant digit s = 0.

The finite-temperature correlator is purified into a pure state on two copies of the system whose reduced state on one copy is the Gibbs state. Its precise form, including how the eigenvectors of the second copy enter, is fixed by the protocol the source adopts.

Returns
-------
ndarray of complex128 with shape (4^N,), unit norm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def thermofield_state(terms, N, beta):
    """ndarray of complex128 with shape (4^N,), unit norm."""
    return np.zeros(4 ** int(N), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_thermofield_state(terms, N, beta):
    """Thermofield double state of Eq (18) of the source on 2N sites (first copy = sites 1..N,
    leftmost tensor factors): |TFD> = Z^{-1/2} sum_k e^{-beta E_k / 2} |psi_k> (x) |psi_k^*>,
    with E_k, |psi_k> the eigenpairs of the N-site H and Z = sum_k e^{-beta E_k}."""
    terms = np.asarray(terms, dtype=np.float64)
    N = int(N)
    if terms.ndim != 2 or terms.shape[1] != 2 or N < 1 or not (np.isfinite(beta) and beta >= 0.0):
        raise ValueError("terms must be (n_terms, 2), N positive and beta finite non-negative")
    ops = {0: np.eye(2, dtype=complex), 1: np.array([[0, 1], [1, 0]], dtype=complex),
           2: np.diag([1.0 + 0j, -1.0]), 3: np.array([[0, -1j], [1j, 0]])}
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for code_f, coef in terms:
        Mx = np.array([[1.0 + 0j]])
        for s in range(N):                            # site 1 (s = 0) is the leftmost factor
            Mx = np.kron(Mx, ops[(int(code_f) >> (2 * s)) & 3])
        H += coef * Mx
    w, V = np.linalg.eigh(H)
    psi = np.zeros(4 ** N, dtype=complex)
    for k in range(2 ** N):
        psi += np.exp(-beta * w[k] / 2.0) * np.kron(V[:, k], V[:, k].conj())
    psi /= np.sqrt(np.sum(np.exp(-beta * w)))
    return psi

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "terms = _oracle_lfxxz_terms(4, 1.0, 0.3, 1.0)\nN = 4\nbeta = 2.0",
            "call": "thermofield_state(terms, N, beta)",
            "gold_call": "_oracle_thermofield_state(terms, N, beta)",
        },
        {
            "setup": "terms = _oracle_lfxxz_terms(4, 1.0, 0.3, 1.0)\nN = 4\nbeta = 0.0",
            "call": "thermofield_state(terms, N, beta)",
            "gold_call": "_oracle_thermofield_state(terms, N, beta)",
        },
        {
            "setup": "terms = _oracle_lfxxz_terms(3, 0.5, 0.1, 0.0)\nN = 3\nbeta = 5.0",
            "call": "thermofield_state(terms, N, beta)",
            "gold_call": "_oracle_thermofield_state(terms, N, beta)",
        },
    ]
