"""
Return the statically screened interaction W0 - the zero-frequency screened interaction of the direct random-phase approximation (spin-summed, real orbitals), the screening used by GW and by the source's Bethe-Salpeter kernel - in the orbital basis, as an array with the same index layout as the integrals, W0[p, q, r, s] = (pq|W0|rs) in chemists' notation; the occupied-virtual pairs (i, a) run over i = 0..n_occ-1 (occupied) and a = n_occ..K-1 (virtual). Raise ValueError if eri is not (K, K, K, K) for K = len(eps), if n_occ is not in 1..K-1, or if any occupied-virtual gap e_a - e_i is not positive.

Screening replaces the bare Coulomb interaction by one weakened by the polarisation of the electrons; at zero frequency it is the interaction that enters the exchange terms of the Bethe-Salpeter kernel and the source's second half-diagram, while the bare interaction stays in the direct terms.

Returns
-------
numpy.ndarray of float64 with shape (K, K, K, K): the statically screened interaction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def static_screened_interaction(eps: "np.ndarray", eri: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Return the statically screened interaction W0 - the zero-frequency screened interaction of the direct random-phase approximation (spin-summed, real orbitals), the screening used by GW and by the source's Bethe-Salpeter kernel - in the orbital basis, as an array with the same index layout as the integrals, W0[p, q, r, s] = (pq|W0|rs) in chemists' notation; the occupied-virtual pairs (i, a) run over i = 0..n_occ-1 (occupied) and a = n_occ..K-1 (virtual). Raise ValueError if eri is not (K, K, K, K) for K = len(eps), if n_occ is not in 1..K-1, or if any occupied-virtual gap e_a - e_i is not positive.

    Parameters
    ----------
    eps : numpy.ndarray
        Reference orbital energies, length K, ascending.
    eri : numpy.ndarray
        Chemists' integrals (K, K, K, K).
    n_occ : int
        Number of occupied orbitals.

    Returns
    -------
    W0 : numpy.ndarray
        Array (K, K, K, K) with W0[p, q, r, s] = (pq|W0|rs).

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n_occ is out of range, or an occupied-virtual gap is not positive.
    """
    return W0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_static_screened_interaction(eps: "np.ndarray", eri: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Statically screened interaction of the direct RPA in the orbital basis:
    W0_{pq,rs} = (pq|rs) + sum_{ia,jb} (pq|ia) chi_{ia,jb} (jb|rs), chi = (1 - chi0 J)^{-1} chi0 with
    chi0_{ia,jb} = -4 delta_ij delta_ab / (e_a - e_i) (spin-summed, real orbitals) and J_{ia,jb} = (ia|jb).
    Returns W0 with the same index layout as eri, (K, K, K, K)."""
    eps = np.asarray(eps, dtype=float).ravel()
    eri = np.asarray(eri, dtype=float)
    n_occ = int(n_occ)
    K = eps.size
    if eri.shape != (K, K, K, K) or not (1 <= n_occ < K):
        raise ValueError("eri must be (K, K, K, K) and 1 <= n_occ < K")
    pairs = [(i, a) for i in range(n_occ) for a in range(n_occ, K)]
    D = np.array([eps[a] - eps[i] for i, a in pairs])
    if np.any(D <= 0.0):
        raise ValueError("the reference must have a positive occupied-virtual gap")
    J = np.array([[eri[i, a, j, b] for (j, b) in pairs] for (i, a) in pairs])
    chi0 = np.diag(-4.0 / D)
    chi = np.linalg.solve(np.eye(len(pairs)) - chi0 @ J, chi0)
    vph = np.array([[eri[p, q, i, a] for (i, a) in pairs] for p in range(K) for q in range(K)]).reshape(K, K, len(pairs))
    return eri + np.einsum('pqm,mn,rsn->pqrs', vph, chi, vph, optimize=True)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "eps_site = np.array([0.4, -0.3, 0.2, -0.1, 0.3, -0.5, 0.1, -0.2]) - 1.5\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 4\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\n",
            "call": "static_screened_interaction(eps_m, eri_m, n_pairs)",
            "gold_call": "_oracle_static_screened_interaction(eps_g, eri_g, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([0.2, -0.4, 0.1, 0.3, -0.2, -0.1]) - 1.0\nt, delta, U, kappa, n_pairs = 1.0, 0.1, 3.0, 2.0, 3\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\n",
            "call": "static_screened_interaction(eps_m, eri_m, n_pairs)",
            "gold_call": "_oracle_static_screened_interaction(eps_g, eri_g, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\n",
            "call": "static_screened_interaction(eps_m, eri_m, n_pairs)",
            "gold_call": "_oracle_static_screened_interaction(eps_g, eri_g, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([-0.5, -0.9])\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 1\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\n",
            "call": "static_screened_interaction(eps_m, eri_m, n_pairs)",
            "gold_call": "_oracle_static_screened_interaction(eps_g, eri_g, n_pairs)",
        },
        {'setup': 'eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\n_n_pairs_orig = n_pairs\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nref_m = rhf_reference(h_m, V_m, _n_pairs_orig)\neps_m, C_m = ref_m[0], ref_m[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\nn_pairs = 4\ndef _fx_exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n', 'call': '_fx_exception_code(static_screened_interaction, eps_m, eri_m, n_pairs)', 'gold_call': '(hV_g := _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site), h_g := hV_g[0], V_g := hV_g[1], ref_g := _oracle_rhf_reference(h_g, V_g, _n_pairs_orig), eps_g := ref_g[0], C_g := ref_g[1:], eri_g := _oracle_mo_two_electron_integrals(V_g, C_g), _fx_exception_code(_oracle_static_screened_interaction, eps_g, eri_g, n_pairs))[-1]'},
    ]
