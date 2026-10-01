"""
Return the source's minimal positive-semidefinite self-energy at a real frequency omega as the real symmetric (K, K) matrix: the residue matrices R of step 06 summed over the singlet excitations Omega_nu of step 05 and over the intermediate orbitals, each divided by omega minus its pole (singlets only; no imaginary broadening). Raise ValueError if the shapes are inconsistent, n_occ is out of range, or omega lies within 1e-12 of a pole.

A self-energy built from the excited states of the Bethe-Salpeter equation replaces the random-phase screening of GW; the source's positive-semidefinite form guarantees a non-negative spectral function, which the original BSE-based self-energy does not.

Returns
-------
numpy.ndarray of float64 with shape (K, K): Sigma_PSD-I(omega).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def psd_self_energy(eps: "np.ndarray", Om: "np.ndarray", res: "np.ndarray", n_occ: int, omega: float) -> "np.ndarray":
    """Return the source's minimal positive-semidefinite self-energy at a real frequency omega as the real symmetric (K, K) matrix: the residue matrices R of step 06 summed over the singlet excitations Omega_nu of step 05 and over the intermediate orbitals, each divided by omega minus its pole (singlets only; no imaginary broadening). Raise ValueError if the shapes are inconsistent, n_occ is out of range, or omega lies within 1e-12 of a pole.

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
    omega : float
        Real frequency.

    Returns
    -------
    sigma : numpy.ndarray
        The (K, K) self-energy matrix at omega.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n_occ is out of range, or omega coincides with a pole.
    """
    return sigma

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_psd_self_energy(eps: "np.ndarray", Om: "np.ndarray", res: "np.ndarray", n_occ: int, omega: float) -> "np.ndarray":
    """The source's minimal PSD self-energy (Eq 11/13) at real frequency omega (eta -> 0):
    Sigma_pq(omega) = sum_nu [ sum_{k occ} R[nu, k, p, q] / (omega - e_k + Omega_nu)
                             + sum_{c vir} R[nu, c, p, q] / (omega - e_c - Omega_nu) ],
    with the residues R = c c^T of step 06, singlet excitations only.  Returns the real symmetric (K, K) matrix."""
    eps = np.asarray(eps, dtype=float).ravel()
    Om = np.asarray(Om, dtype=float).ravel()
    res = np.asarray(res, dtype=float)
    n_occ, omega = int(n_occ), float(omega)
    K = eps.size
    if res.shape != (Om.size, K, K, K) or not (1 <= n_occ < K) or not np.isfinite(omega):
        raise ValueError("inconsistent shapes")
    den = np.empty((Om.size, K))
    den[:, :n_occ] = omega - eps[None, :n_occ] + Om[:, None]
    den[:, n_occ:] = omega - eps[None, n_occ:] - Om[:, None]
    if np.any(np.abs(den) < 1e-12):
        raise ValueError("omega coincides with a pole")
    return np.einsum('nkpq,nk->pq', res, 1.0 / den, optimize=True)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "eps_site = np.array([0.4, -0.3, 0.2, -0.1, 0.3, -0.5, 0.1, -0.2]) - 1.5\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 4\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_g = _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0)\nres_m = self_energy_residues(eri_m, W0_m, bse_m, n_pairs)\nres_g = _oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs)\nOm_m = bse_m[0]\nOm_g = bse_g[0]\nomega_m = eps_m[n_pairs - 1] - 0.2\nomega_g = eps_g[n_pairs - 1] - 0.2\n",
            "call": "psd_self_energy(eps_m, Om_m, res_m, n_pairs, omega_m)",
            "gold_call": "_oracle_psd_self_energy(eps_g, Om_g, res_g, n_pairs, omega_g)",
        },
        {
            "setup": "eps_site = np.array([0.2, -0.4, 0.1, 0.3, -0.2, -0.1]) - 1.0\nt, delta, U, kappa, n_pairs = 1.0, 0.1, 3.0, 2.0, 3\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_g = _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0)\nres_m = self_energy_residues(eri_m, W0_m, bse_m, n_pairs)\nres_g = _oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs)\nOm_m = bse_m[0]\nOm_g = bse_g[0]\nomega_m = eps_m[n_pairs] + 0.3\nomega_g = eps_g[n_pairs] + 0.3\n",
            "call": "psd_self_energy(eps_m, Om_m, res_m, n_pairs, omega_m)",
            "gold_call": "_oracle_psd_self_energy(eps_g, Om_g, res_g, n_pairs, omega_g)",
        },
        {
            "setup": "eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_g = _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0)\nres_m = self_energy_residues(eri_m, W0_m, bse_m, n_pairs)\nres_g = _oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs)\nOm_m = bse_m[0]\nOm_g = bse_g[0]\nomega_m = -0.5\nomega_g = -0.5\n",
            "call": "psd_self_energy(eps_m, Om_m, res_m, n_pairs, omega_m)",
            "gold_call": "_oracle_psd_self_energy(eps_g, Om_g, res_g, n_pairs, omega_g)",
        },
        {
            "setup": "eps_site = np.array([-0.5, -0.9])\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 1\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_g = _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0)\nres_m = self_energy_residues(eri_m, W0_m, bse_m, n_pairs)\nres_g = _oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs)\nOm_m = bse_m[0]\nOm_g = bse_g[0]\nomega_m = eps_m[n_pairs - 1] - 0.1\nomega_g = eps_g[n_pairs - 1] - 0.1\n",
            "call": "psd_self_energy(eps_m, Om_m, res_m, n_pairs, omega_m)",
            "gold_call": "_oracle_psd_self_energy(eps_g, Om_g, res_g, n_pairs, omega_g)",
        },
        {'setup': 'eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nres_m = self_energy_residues(eri_m, W0_m, bse_m, n_pairs)\nOm_m = bse_m[0]\nomega_m = eps_m[n_pairs - 1]\nres_m = res_m[:, :, :, :2]\ndef _fx_exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n', 'call': '_fx_exception_code(psd_self_energy, eps_m, Om_m, res_m, n_pairs, omega_m)', 'gold_call': '(hV_g := _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site), h_g := hV_g[0], V_g := hV_g[1], ref_g := _oracle_rhf_reference(h_g, V_g, n_pairs), eps_g := ref_g[0], C_g := ref_g[1:], eri_g := _oracle_mo_two_electron_integrals(V_g, C_g), W0_g := _oracle_static_screened_interaction(eps_g, eri_g, n_pairs), bse_g := _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0), res_g := _oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs), Om_g := bse_g[0], omega_g := eps_g[n_pairs - 1], res_g := res_g[:, :, :, :2], _fx_exception_code(_oracle_psd_self_energy, eps_g, Om_g, res_g, n_pairs, omega_g))[-1]'},
    ]
