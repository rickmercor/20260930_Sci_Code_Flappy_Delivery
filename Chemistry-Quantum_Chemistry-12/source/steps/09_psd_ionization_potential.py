"""
Orchestrator. Return the first ionisation potential of the PPP chain at the source's PSD-I level, IP = -e_HOMO^QP, by calling the earlier steps rather than re-implementing them: the Hamiltonian (step 01), the Hartree-Fock reference (02), the orbital-basis integrals (03), the static RPA screening (04), the singlet Bethe-Salpeter solution (05; also solve the triplet manifold and raise ValueError if either manifold has a non-positive excitation energy), the self-energy residues (06), the self-energy (07) and the graphical quasiparticle energy (08) of the highest occupied orbital p = n_pairs - 1; verify that the returned energy satisfies e = e_p + Sigma_pp(e) to 1e-9 with step 07 and raise ValueError otherwise. Raise ValueError for invalid model parameters as in the earlier steps.

The ionisation potential is the quasiparticle energy of the highest occupied orbital with its sign reversed; at the PSD-I level it carries the vertex correction beyond GW while keeping a positive spectral function.

Returns
-------
float: the first ionisation potential -e_HOMO^QP at the PSD-I level.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def psd_ionization_potential(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray", n_pairs: int) -> float:
    """Orchestrator. Return the first ionisation potential of the PPP chain at the source's PSD-I level, IP = -e_HOMO^QP, by calling the earlier steps rather than re-implementing them: the Hamiltonian (step 01), the Hartree-Fock reference (02), the orbital-basis integrals (03), the static RPA screening (04), the singlet Bethe-Salpeter solution (05; also solve the triplet manifold and raise ValueError if either manifold has a non-positive excitation energy), the self-energy residues (06), the self-energy (07) and the graphical quasiparticle energy (08) of the highest occupied orbital p = n_pairs - 1; verify that the returned energy satisfies e = e_p + Sigma_pp(e) to 1e-9 with step 07 and raise ValueError otherwise. Raise ValueError for invalid model parameters as in the earlier steps.

    Parameters
    ----------
    t : float
        Nearest-neighbour hopping (positive).
    delta : float
        Bond alternation, |delta| < 1.
    U : float
        On-site repulsion (positive).
    kappa : float
        Ohno screening length (positive).
    eps_site : numpy.ndarray
        Site energies, length K >= 2.
    n_pairs : int
        Number of doubly occupied orbitals.

    Returns
    -------
    IP : float
        The PSD-I first ionisation potential.

    Raises
    ------
    ValueError
        If the model parameters are invalid, the reference is unstable, or the quasiparticle equation is not satisfied.
    """
    return IP

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_psd_ionization_potential(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray", n_pairs: int) -> float:
    """ORCHESTRATOR: the first ionisation potential of the PPP chain at the PSD-I level, IP = -e_HOMO^QP, with
    the chain of steps 01-08 (singlet BSE excitations; the triplet manifold of step 05 is evaluated as a
    stability check; step 06 supplies the self-energy residues)."""
    n_pairs = int(n_pairs)
    hV = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)
    h, V = hV[0], hV[1]
    ref = _oracle_rhf_reference(h, V, n_pairs)
    eps, C = ref[0], ref[1:]
    eri = _oracle_mo_two_electron_integrals(V, C)
    W0 = _oracle_static_screened_interaction(eps, eri, n_pairs)
    bse_s = _oracle_bse_excitations(eps, eri, W0, n_pairs, 0)
    bse_t = _oracle_bse_excitations(eps, eri, W0, n_pairs, 1)
    if bse_t[0].min() <= 0.0 or bse_s[0].min() <= 0.0:
        raise ValueError("unstable reference")
    res = _oracle_self_energy_residues(eri, W0, bse_s, n_pairs)
    e_qp = _oracle_quasiparticle_energy(eps, bse_s[0], res, n_pairs, n_pairs - 1)
    sig = _oracle_psd_self_energy(eps, bse_s[0], res, n_pairs, e_qp)
    if abs(e_qp - eps[n_pairs - 1] - sig[n_pairs - 1, n_pairs - 1]) > 1e-9:
        raise ValueError("quasiparticle equation not satisfied")
    return float(-e_qp)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "eps_site = np.array([0.4, -0.3, 0.2, -0.1, 0.3, -0.5, 0.1, -0.2]) - 1.5\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 4\n",
            "call": "psd_ionization_potential(t, delta, U, kappa, eps_site, n_pairs)",
            "gold_call": "_oracle_psd_ionization_potential(t, delta, U, kappa, eps_site, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([0.2, -0.4, 0.1, 0.3, -0.2, -0.1]) - 1.0\nt, delta, U, kappa, n_pairs = 1.0, 0.1, 3.0, 2.0, 3\n",
            "call": "psd_ionization_potential(t, delta, U, kappa, eps_site, n_pairs)",
            "gold_call": "_oracle_psd_ionization_potential(t, delta, U, kappa, eps_site, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\n",
            "call": "psd_ionization_potential(t, delta, U, kappa, eps_site, n_pairs)",
            "gold_call": "_oracle_psd_ionization_potential(t, delta, U, kappa, eps_site, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([-0.5, -0.9])\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 1\n",
            "call": "psd_ionization_potential(t, delta, U, kappa, eps_site, n_pairs)",
            "gold_call": "_oracle_psd_ionization_potential(t, delta, U, kappa, eps_site, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([0.4, -0.3, 0.2, -0.1, 0.3, -0.5, 0.1, -0.2]) - 1.5\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 4\nn_pairs = 8\ndef run_model():\n    try:\n        psd_ionization_potential(t, delta, U, kappa, eps_site, n_pairs)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_psd_ionization_potential(t, delta, U, kappa, eps_site, n_pairs)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
