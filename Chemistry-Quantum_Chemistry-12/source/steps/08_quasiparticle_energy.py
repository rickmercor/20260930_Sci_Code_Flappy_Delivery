"""
Return the one-shot quasiparticle energy of orbital p (0-based) on the Hartree-Fock reference: the solution of omega = e_p + Sigma_pp(omega) with the self-energy of step 07, found graphically as the root of f(omega) = omega - e_p - Sigma_pp(omega) nearest e_p by bisection: start from the bracket [e_p - 0.5, e_p + 0.5], widen both ends by 0.25 (at most 40 times) until f changes sign across it, then bisect keeping the sub-bracket in which f changes sign until its width is below 1e-12 and return its midpoint. No linearisation. Raise ValueError if p is out of range or no root is bracketed.

With a Hartree-Fock reference the exchange is already in the orbital energy, so the correlation self-energy alone shifts it; solving the quasiparticle equation graphically rather than linearising it is the source's one-shot protocol.

Returns
-------
float: the one-shot PSD-I quasiparticle energy of orbital p.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def quasiparticle_energy(eps: "np.ndarray", Om: "np.ndarray", res: "np.ndarray", n_occ: int, p: int) -> float:
    """Return the one-shot quasiparticle energy of orbital p (0-based) on the Hartree-Fock reference: the solution of omega = e_p + Sigma_pp(omega) with the self-energy of step 07, found graphically as the root of f(omega) = omega - e_p - Sigma_pp(omega) nearest e_p by bisection: start from the bracket [e_p - 0.5, e_p + 0.5], widen both ends by 0.25 (at most 40 times) until f changes sign across it, then bisect keeping the sub-bracket in which f changes sign until its width is below 1e-12 and return its midpoint. No linearisation. Raise ValueError if p is out of range or no root is bracketed.

    Parameters
    ----------
    eps : numpy.ndarray
        Reference orbital energies, length K.
    Om : numpy.ndarray
        Singlet excitation energies, length n.
    res : numpy.ndarray
        Residue tensor (n, K, K, K) from step 06.
    n_occ : int
        Number of occupied orbitals.
    p : int
        Orbital index, 0-based.

    Returns
    -------
    e_qp : float
        The quasiparticle energy of orbital p.

    Raises
    ------
    ValueError
        If p is out of range or no root is bracketed.
    """
    return e_qp

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_quasiparticle_energy(eps: "np.ndarray", Om: "np.ndarray", res: "np.ndarray", n_occ: int, p: int) -> float:
    """One-shot graphical solution of omega = e_p + Sigma_pp(omega) for orbital p (0-based) with the PSD-I
    self-energy of step 07 on the HF reference: the root nearest e_p, located by bisection on
    f(omega) = omega - e_p - Sigma_pp(omega) in the bracket [e_p - w, e_p + w] with w = 0.5 widened by 0.25 until
    f changes sign, to |bracket| < 1e-12.  Returns the quasiparticle energy."""
    eps = np.asarray(eps, dtype=float).ravel()
    p = int(p)
    K = eps.size
    if not (0 <= p < K):
        raise ValueError("p out of range")
    e0 = eps[p]

    def _f(w):
        return w - e0 - _oracle_psd_self_energy(eps, Om, res, n_occ, w)[p, p]

    lo, hi = e0 - 0.5, e0 + 0.5
    for _ in range(40):
        if _f(lo) * _f(hi) <= 0.0:
            break
        lo -= 0.25
        hi += 0.25
    else:
        raise ValueError("no quasiparticle root bracketed")
    flo = _f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = _f(mid)
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
        if hi - lo < 1e-12:
            break
    return float(0.5 * (lo + hi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "eps_site = np.array([0.4, -0.3, 0.2, -0.1, 0.3, -0.5, 0.1, -0.2]) - 1.5\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 4\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_g = _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0)\nres_m = self_energy_residues(eri_m, W0_m, bse_m, n_pairs)\nres_g = _oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs)\nOm_m = bse_m[0]\nOm_g = bse_g[0]\np = n_pairs - 1\n",
            "call": "quasiparticle_energy(eps_m, Om_m, res_m, n_pairs, p)",
            "gold_call": "_oracle_quasiparticle_energy(eps_g, Om_g, res_g, n_pairs, p)",
        },
        {
            "setup": "eps_site = np.array([0.2, -0.4, 0.1, 0.3, -0.2, -0.1]) - 1.0\nt, delta, U, kappa, n_pairs = 1.0, 0.1, 3.0, 2.0, 3\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_g = _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0)\nres_m = self_energy_residues(eri_m, W0_m, bse_m, n_pairs)\nres_g = _oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs)\nOm_m = bse_m[0]\nOm_g = bse_g[0]\np = n_pairs\n",
            "call": "quasiparticle_energy(eps_m, Om_m, res_m, n_pairs, p)",
            "gold_call": "_oracle_quasiparticle_energy(eps_g, Om_g, res_g, n_pairs, p)",
        },
        {
            "setup": "eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_g = _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0)\nres_m = self_energy_residues(eri_m, W0_m, bse_m, n_pairs)\nres_g = _oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs)\nOm_m = bse_m[0]\nOm_g = bse_g[0]\np = 0\n",
            "call": "quasiparticle_energy(eps_m, Om_m, res_m, n_pairs, p)",
            "gold_call": "_oracle_quasiparticle_energy(eps_g, Om_g, res_g, n_pairs, p)",
        },
        {
            "setup": "eps_site = np.array([-0.5, -0.9])\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 1\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_g = _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0)\nres_m = self_energy_residues(eri_m, W0_m, bse_m, n_pairs)\nres_g = _oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs)\nOm_m = bse_m[0]\nOm_g = bse_g[0]\np = 1\n",
            "call": "quasiparticle_energy(eps_m, Om_m, res_m, n_pairs, p)",
            "gold_call": "_oracle_quasiparticle_energy(eps_g, Om_g, res_g, n_pairs, p)",
        },
        {'setup': 'eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nres_m = self_energy_residues(eri_m, W0_m, bse_m, n_pairs)\nOm_m = bse_m[0]\np = 4\ndef _fx_exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n', 'call': '_fx_exception_code(quasiparticle_energy, eps_m, Om_m, res_m, n_pairs, p)', 'gold_call': '(hV_g := _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site), h_g := hV_g[0], V_g := hV_g[1], ref_g := _oracle_rhf_reference(h_g, V_g, n_pairs), eps_g := ref_g[0], C_g := ref_g[1:], eri_g := _oracle_mo_two_electron_integrals(V_g, C_g), W0_g := _oracle_static_screened_interaction(eps_g, eri_g, n_pairs), bse_g := _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0), res_g := _oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs), Om_g := bse_g[0], _fx_exception_code(_oracle_quasiparticle_energy, eps_g, Om_g, res_g, n_pairs, p))[-1]'},
    ]
