"""
Return the exact expectation value of the Pauli string with code obs at the times ts for the state vector psi0 evolved under the M-site Hamiltonian given by terms, by dense diagonalisation; site 1 is the leftmost tensor factor. Raise ValueError if psi0 does not have length 2^M or if obs is not a valid code. Pauli strings are encoded as in step 1: code = sum_s f_s 4^s with f_s = 0, 1, 2, 3 for I, X, Z, Y on site s = 0..M-1, site 1 being the least significant digit s = 0.

For two copies of a four-site chain the exact dynamics is available by diagonalisation and serves as the reference against which the pruned simulation is scored.

Returns
-------
ndarray of float64 with shape (len(ts),).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def exact_observable_series(terms, M, psi0, obs, ts):
    """ndarray of float64 with shape (len(ts),)."""
    return np.zeros(np.size(ts), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_exact_observable_series(terms, M, psi0, obs, ts):
    """Exact <psi(t)| P_obs |psi(t)> for the state psi0 evolved under the M-site H (given by
    its terms), by dense diagonalisation; site 1 is the leftmost tensor factor."""
    terms = np.asarray(terms, dtype=np.float64)
    M = int(M)
    psi0 = np.asarray(psi0, dtype=complex).reshape(-1)
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    if psi0.shape != (2 ** M,) or int(obs) != obs or not (0 <= obs < 4 ** M):
        raise ValueError("psi0 must have length 2^M and obs must be a Pauli code")
    ops = {0: np.eye(2, dtype=complex), 1: np.array([[0, 1], [1, 0]], dtype=complex),
           2: np.diag([1.0 + 0j, -1.0]), 3: np.array([[0, -1j], [1j, 0]])}

    def matrix(code):
        Mx = np.array([[1.0 + 0j]])
        for s in range(M):
            Mx = np.kron(Mx, ops[(int(code) >> (2 * s)) & 3])
        return Mx

    H = np.zeros((2 ** M, 2 ** M), dtype=complex)
    for code_f, coef in terms:
        H += coef * matrix(int(code_f))
    w, V = np.linalg.eigh(H)
    c = V.conj().T @ psi0
    P = matrix(obs)
    out = np.empty(ts.size, dtype=np.float64)
    for k, t in enumerate(ts):
        psi = V @ (np.exp(-1j * w * t) * c)
        out[k] = float(np.real(psi.conj() @ (P @ psi)))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "terms = _oracle_doubled_terms(_oracle_lfxxz_terms(4, 1.0, 0.3, 1.0), 4)\ntfd = _oracle_thermofield_state(_oracle_lfxxz_terms(4, 1.0, 0.3, 1.0), 4, 2.0)\nsgn = np.where(((np.arange(4 ** 4) >> 7) & 1) == 1, -1.0, 1.0)\npsi0 = tfd * sgn\nM = 8\nobs = 1 + 4 ** 4\nts = 0.1 * np.arange(26)",
            "call": "exact_observable_series(terms, M, psi0, obs, ts)",
            "gold_call": "_oracle_exact_observable_series(terms, M, psi0, obs, ts)",
        },
        {
            "setup": "terms = _oracle_doubled_terms(_oracle_lfxxz_terms(4, 1.0, 0.3, 1.0), 4)\ntfd = _oracle_thermofield_state(_oracle_lfxxz_terms(4, 1.0, 0.3, 1.0), 4, 2.0)\nsgn = np.where(((np.arange(4 ** 4) >> 7) & 1) == 1, -1.0, 1.0)\npsi0 = tfd * sgn\nM = 8\nobs = 2\nts = np.array([0.0, 0.7, 2.1])",
            "call": "exact_observable_series(terms, M, psi0, obs, ts)",
            "gold_call": "_oracle_exact_observable_series(terms, M, psi0, obs, ts)",
        },
        {
            "setup": "terms = _oracle_lfxxz_terms(3, 0.5, 0.1, 0.0)\nM = 3\npsi0 = np.zeros(8, dtype=complex)\npsi0[3] = 1.0\nobs = 2 + 2 * 4\nts = np.array([0.0, 1.0, 5.0])",
            "call": "exact_observable_series(terms, M, psi0, obs, ts)",
            "gold_call": "_oracle_exact_observable_series(terms, M, psi0, obs, ts)",
        },
    ]
