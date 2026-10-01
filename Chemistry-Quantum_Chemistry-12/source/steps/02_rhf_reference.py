"""
Return the restricted closed-shell Hartree-Fock reference of the density-density Hamiltonian whose two-electron integrals in the site basis are (pq|rs) = delta_pq delta_rs V_pr, with n_pairs doubly occupied orbitals: start from the eigenvectors of h, iterate the closed-shell density D = 2 C_occ C_occ^T with the damping D <- (D + D_new)/2 until max|D_new - D| < 1e-12, then diagonalise the Fock matrix built from the converged D once more. Return an array of shape (K + 1, K): row 0 holds the orbital energies in ascending order and rows 1..K the coefficient matrix C (columns = orbitals in the same order), each column phase-fixed so that its first component of magnitude above 1e-8 is positive. Raise ValueError if h and V are not symmetric K x K matrices with K >= 2, if n_pairs is not in 1..K-1, or if the iteration does not converge in 5000 steps.

Every quantity of the method is built on a mean-field reference; the self-energy corrects its orbital energies. A fixed orbital phase convention makes every downstream matrix element well defined.

Returns
-------
numpy.ndarray of float64 with shape (K + 1, K): orbital energies then the coefficient matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rhf_reference(h: "np.ndarray", V: "np.ndarray", n_pairs: int) -> "np.ndarray":
    """Return the restricted closed-shell Hartree-Fock reference of the density-density Hamiltonian whose two-electron integrals in the site basis are (pq|rs) = delta_pq delta_rs V_pr, with n_pairs doubly occupied orbitals: start from the eigenvectors of h, iterate the closed-shell density D = 2 C_occ C_occ^T with the damping D <- (D + D_new)/2 until max|D_new - D| < 1e-12, then diagonalise the Fock matrix built from the converged D once more. Return an array of shape (K + 1, K): row 0 holds the orbital energies in ascending order and rows 1..K the coefficient matrix C (columns = orbitals in the same order), each column phase-fixed so that its first component of magnitude above 1e-8 is positive. Raise ValueError if h and V are not symmetric K x K matrices with K >= 2, if n_pairs is not in 1..K-1, or if the iteration does not converge in 5000 steps.

    Parameters
    ----------
    h : numpy.ndarray
        One-electron matrix (K, K).
    V : numpy.ndarray
        Site-site interaction (K, K).
    n_pairs : int
        Number of doubly occupied orbitals.

    Returns
    -------
    ref : numpy.ndarray
        Array (K + 1, K): row 0 = orbital energies ascending, rows 1..K = C.

    Raises
    ------
    ValueError
        If the matrices are not symmetric (K, K) with K >= 2, n_pairs is out of range, or the SCF does not converge.
    """
    return ref

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _fix_phase(C):
    """Column sign convention: the first component with |c| > 1e-8 is positive."""
    C = np.array(C, dtype=float)
    for k in range(C.shape[1]):
        nz = np.where(np.abs(C[:, k]) > 1e-8)[0]
        if nz.size and C[nz[0], k] < 0.0:
            C[:, k] = -C[:, k]
    return C


def _oracle_rhf_reference(h: "np.ndarray", V: "np.ndarray", n_pairs: int) -> "np.ndarray":
    """Restricted closed-shell Hartree-Fock for the density-density Hamiltonian ((pq|rs) = delta_pq delta_rs V_pr
    in the site basis): F = h + diag(V D) - (1/2) V o D with D = 2 C_occ C_occ^T, started from the eigenvectors of
    h, damped iteration D <- (D + D_new)/2 until max|D_new - D| < 1e-12, then one final diagonalisation of F.
    Returns array (K + 1, K): row 0 = orbital energies ascending, rows 1..K = the orbital coefficient matrix C
    (columns = orbitals, phase-fixed so that the first component with |c| > 1e-8 is positive)."""
    h = np.asarray(h, dtype=float)
    V = np.asarray(V, dtype=float)
    P = int(n_pairs)
    K = h.shape[0]
    if h.shape != (K, K) or V.shape != (K, K) or K < 2:
        raise ValueError("h and V must be (K, K) with K >= 2")
    if not (1 <= P < K):
        raise ValueError("need 1 <= n_pairs < K")
    if not (np.allclose(h, h.T, atol=1e-12) and np.allclose(V, V.T, atol=1e-12)):
        raise ValueError("h and V must be symmetric")
    C = np.linalg.eigh(h)[1]
    D = 2.0 * C[:, :P] @ C[:, :P].T
    for _ in range(5000):
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
    return np.vstack([eps[None, :], _fix_phase(C)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "eps_site = np.array([0.4, -0.3, 0.2, -0.1, 0.3, -0.5, 0.1, -0.2]) - 1.5\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 4\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\n",
            "call": "rhf_reference(h_m, V_m, n_pairs)",
            "gold_call": "_oracle_rhf_reference(h_g, V_g, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([0.2, -0.4, 0.1, 0.3, -0.2, -0.1]) - 1.0\nt, delta, U, kappa, n_pairs = 1.0, 0.1, 3.0, 2.0, 3\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\n",
            "call": "rhf_reference(h_m, V_m, n_pairs)",
            "gold_call": "_oracle_rhf_reference(h_g, V_g, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\n",
            "call": "rhf_reference(h_m, V_m, n_pairs)",
            "gold_call": "_oracle_rhf_reference(h_g, V_g, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([-0.5, -0.9])\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 1\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\n",
            "call": "rhf_reference(h_m, V_m, n_pairs)",
            "gold_call": "_oracle_rhf_reference(h_g, V_g, n_pairs)",
        },
        {'setup': 'eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nn_pairs = 4\ndef _fx_exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n', 'call': '_fx_exception_code(rhf_reference, h_m, V_m, n_pairs)', 'gold_call': '(hV_g := _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site), h_g := hV_g[0], V_g := hV_g[1], _fx_exception_code(_oracle_rhf_reference, h_g, V_g, n_pairs))[-1]'},
    ]
