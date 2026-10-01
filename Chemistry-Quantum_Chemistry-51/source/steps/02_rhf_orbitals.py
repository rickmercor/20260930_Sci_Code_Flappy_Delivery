"""
Return the restricted closed-shell Hartree-Fock orbitals C (K, K) of the Hamiltonian with one-body matrix h and density-density integrals (pq|rs) = delta_pq delta_rs V_pr, with P = n_pairs doubly occupied orbitals: solve the Roothaan equations F C = C eps, F = h + J - K/2 with J_pq = delta_pq sum_r V_pr D_rr and K_pq = V_pq D_pq for the density D = 2 C_occ C_occ^T, by damped fixed-point iteration (new density averaged 1:1 with the previous one) started from the eigenvectors of h, until the density changes by less than 1e-12 element-wise; then diagonalise the final Fock matrix once more. Columns ordered by ascending orbital energy. Column phase convention: in every column the first component with |c| > 1e-8 is positive. Raise ValueError if h or V is not a symmetric (K, K) matrix with K >= 2, if n_pairs is outside [1, K - 1], or if the iteration does not converge within 2000 cycles.

For a density-density interaction the Coulomb term is diagonal in the site basis and the exchange term is the element-wise product of the interaction matrix with the density matrix, so the Fock build needs no four-index integrals. The converged canonical orbitals are the reference for the pair coupled cluster treatment.

Returns
-------
float array (K, K) of orbital coefficients (columns), ascending orbital energy, phase-fixed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rhf_orbitals(h: "np.ndarray", V: "np.ndarray", n_pairs: int) -> "np.ndarray":
    '''Restricted Hartree-Fock orbitals of the density-density Hamiltonian.

    Parameters
    ----------
    h : np.ndarray
        Symmetric one-body matrix (K, K).
    V : np.ndarray
        Symmetric site-basis interaction matrix (K, K).
    n_pairs : int
        Number of doubly occupied orbitals P, 1 <= P < K.

    Returns
    -------
    C : np.ndarray
        Orbital coefficient matrix (K, K), column k the k-th orbital in ascending orbital-energy order, phase-fixed.

    Raises
    ------
    ValueError
        If h or V is not a symmetric (K, K) matrix with K >= 2, if n_pairs is outside [1, K - 1], or if the iteration does not converge.
    '''
    return C

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _fix_phase(C):
    """Column sign convention: first component with |c| > 1e-8 positive."""
    C = np.array(C, dtype=float)
    for k in range(C.shape[1]):
        col = C[:, k]
        nz = np.where(np.abs(col) > 1e-8)[0]
        if nz.size and col[nz[0]] < 0.0:
            C[:, k] = -col
    return C


def _oracle_rhf_orbitals(h: "np.ndarray", V: "np.ndarray", n_pairs: int) -> "np.ndarray":
    """Restricted Hartree-Fock for the density-density Hamiltonian (chemists' (pq|rs) = delta_pq delta_rs V_pr):
    F = h + diag(V D) - (1/2) V o D with D = 2 C_occ C_occ^T. Damped iteration to 1e-12 in D.
    Returns C (K, K), columns sorted by orbital energy, phase-fixed."""
    h = np.asarray(h, dtype=float); V = np.asarray(V, dtype=float); P = int(n_pairs)
    K = h.shape[0]
    if h.shape != (K, K) or V.shape != (K, K) or K < 2:
        raise ValueError("h and V must be (K, K) with K >= 2")
    if not (1 <= P < K):
        raise ValueError("need 1 <= n_pairs < K")
    if not (np.allclose(h, h.T, atol=1e-12) and np.allclose(V, V.T, atol=1e-12)):
        raise ValueError("h and V must be symmetric")
    C = np.linalg.eigh(h)[1]
    D = 2.0 * C[:, :P] @ C[:, :P].T
    for it in range(2000):
        F = h + np.diag(V @ np.diag(D)) - 0.5 * V * D
        eps, C = np.linalg.eigh(F)
        Dn = 2.0 * C[:, :P] @ C[:, :P].T
        if np.max(np.abs(Dn - D)) < 1e-12:
            D = Dn
            break
        D = 0.5 * D + 0.5 * Dn
    else:
        raise ValueError("RHF did not converge")
    F = h + np.diag(V @ np.diag(D)) - 0.5 * V * D
    eps, C = np.linalg.eigh(F)
    return _fix_phase(C)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of step test cases (normal, boundary and edge inputs, plus one invalid input)."""
    return [
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(1.0, 0.15, 6.0, 4.0, -4.0 + np.array([0.0, 0.3, -0.2, 0.1, -0.3, 0.2, 0.15, -0.25]))\nh, V, n_pairs = hV[0], hV[1], 4",
            "call": "rhf_orbitals(h, V, n_pairs)",
            "gold_call": "_oracle_rhf_orbitals(h, V, n_pairs)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(1.0, 0.1, 4.0, 3.0, -3.5 + np.array([0.0, 0.2, -0.1, 0.3, -0.2, 0.1]))\nh, V, n_pairs = hV[0], hV[1], 3",
            "call": "rhf_orbitals(h, V, n_pairs)",
            "gold_call": "_oracle_rhf_orbitals(h, V, n_pairs)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(0.8, 0.0, 5.0, 2.5, np.array([-3.0, -3.2, -2.75, -3.15]))\nh, V, n_pairs = hV[0], hV[1], 2",
            "call": "rhf_orbitals(h, V, n_pairs)",
            "gold_call": "_oracle_rhf_orbitals(h, V, n_pairs)",
            "tol": 1e-09,
        },
        {  # invalid input: the documented ValueError contract
            "setup": "import numpy as np\ndef run_model():\n    try:\n        rhf_orbitals(np.zeros((4, 4)), np.eye(4), 4)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_rhf_orbitals(np.zeros((4, 4)), np.eye(4), 4)\n        return 0\n    except ValueError:\n        return 1",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
