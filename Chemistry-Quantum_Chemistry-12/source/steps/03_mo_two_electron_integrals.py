"""
Return the chemists' two-electron integrals of the density-density Hamiltonian in the orbital basis C: (pq|rs) = sum_{ij} C_ip C_iq V_ij C_jr C_js, as an array of shape (K, K, K, K) indexed [p, q, r, s]. Raise ValueError if V or C is not K x K.

The site-diagonal interaction becomes a full four-index tensor in the molecular-orbital basis; every later step reads it in chemists' notation, with the first pair (pq) on one electron and the second pair (rs) on the other.

Returns
-------
numpy.ndarray of float64 with shape (K, K, K, K): (pq|rs).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mo_two_electron_integrals(V: "np.ndarray", C: "np.ndarray") -> "np.ndarray":
    """Return the chemists' two-electron integrals of the density-density Hamiltonian in the orbital basis C: (pq|rs) = sum_{ij} C_ip C_iq V_ij C_jr C_js, as an array of shape (K, K, K, K) indexed [p, q, r, s]. Raise ValueError if V or C is not K x K.

    Parameters
    ----------
    V : numpy.ndarray
        Site-site interaction (K, K).
    C : numpy.ndarray
        Orbital coefficients (K, K), columns = orbitals.

    Returns
    -------
    eri : numpy.ndarray
        Array (K, K, K, K) with eri[p, q, r, s] = (pq|rs).

    Raises
    ------
    ValueError
        If V or C is not (K, K).
    """
    return eri

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_mo_two_electron_integrals(V: "np.ndarray", C: "np.ndarray") -> "np.ndarray":
    """Chemists' integrals (pq|rs) = sum_ij C_ip C_iq V_ij C_jr C_js in the orbital basis C. Returns (K, K, K, K)."""
    V = np.asarray(V, dtype=float)
    C = np.asarray(C, dtype=float)
    K = V.shape[0]
    if V.shape != (K, K) or C.shape != (K, K):
        raise ValueError("V and C must be (K, K)")
    X = np.einsum('ip,iq->ipq', C, C)
    return np.einsum('ipq,ij,jrs->pqrs', X, V, X, optimize=True)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "eps_site = np.array([0.4, -0.3, 0.2, -0.1, 0.3, -0.5, 0.1, -0.2]) - 1.5\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 4\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nV_m = hV_m[1]\nV_g = hV_g[1]\nC_m = rhf_reference(hV_m[0], V_m, n_pairs)[1:]\nC_g = _oracle_rhf_reference(hV_g[0], V_g, n_pairs)[1:]\n",
            "call": "mo_two_electron_integrals(V_m, C_m)",
            "gold_call": "_oracle_mo_two_electron_integrals(V_g, C_g)",
        },
        {
            "setup": "eps_site = np.array([0.2, -0.4, 0.1, 0.3, -0.2, -0.1]) - 1.0\nt, delta, U, kappa, n_pairs = 1.0, 0.1, 3.0, 2.0, 3\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nV_m = hV_m[1]\nV_g = hV_g[1]\nC_m = rhf_reference(hV_m[0], V_m, n_pairs)[1:]\nC_g = _oracle_rhf_reference(hV_g[0], V_g, n_pairs)[1:]\n",
            "call": "mo_two_electron_integrals(V_m, C_m)",
            "gold_call": "_oracle_mo_two_electron_integrals(V_g, C_g)",
        },
        {
            "setup": "eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nV_m = hV_m[1]\nV_g = hV_g[1]\nC_m = rhf_reference(hV_m[0], V_m, n_pairs)[1:]\nC_g = _oracle_rhf_reference(hV_g[0], V_g, n_pairs)[1:]\n",
            "call": "mo_two_electron_integrals(V_m, C_m)",
            "gold_call": "_oracle_mo_two_electron_integrals(V_g, C_g)",
        },
        {
            "setup": "eps_site = np.array([-0.5, -0.9])\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 1\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nV_m = hV_m[1]\nV_g = hV_g[1]\nC_m = rhf_reference(hV_m[0], V_m, n_pairs)[1:]\nC_g = _oracle_rhf_reference(hV_g[0], V_g, n_pairs)[1:]\n",
            "call": "mo_two_electron_integrals(V_m, C_m)",
            "gold_call": "_oracle_mo_two_electron_integrals(V_g, C_g)",
        },
        {'setup': 'eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nV_m = hV_m[1]\nC_m = rhf_reference(hV_m[0], V_m, n_pairs)[1:][:, :3]\ndef _fx_exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n', 'call': '_fx_exception_code(mo_two_electron_integrals, V_m, C_m)', 'gold_call': '(hV_g := _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site), V_g := hV_g[1], C_g := _oracle_rhf_reference(hV_g[0], V_g, n_pairs)[1:][:, :3], _fx_exception_code(_oracle_mo_two_electron_integrals, V_g, C_g))[-1]'},
    ]
