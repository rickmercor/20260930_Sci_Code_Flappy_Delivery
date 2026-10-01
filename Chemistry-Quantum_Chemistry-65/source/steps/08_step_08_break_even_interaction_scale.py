"""
Find the interaction scale at which the density-corrected G0W0 ionization energy of a polyene stops being more accurate than plain G0W0 (orchestrator).

One-shot G0W0 on Hartree-Fock orbitals overestimates the first ionization energy of weakly correlated pi systems and
underestimates it once the electron repulsion is strong. The static correction from the linearized GW density matrix
always lowers the ionization energy, so it helps at weak coupling and overshoots at strong coupling. At the break-even
scale lambda_be the two approximations lie equally far from the exact full configuration interaction value on
opposite sides, IP_G0W0 + IP_G0W0+gamma = 2 IP_FCI. For every scale the pipeline builds the PPP Hamiltonian, solves the
restricted Hartree-Fock equations, transforms the site repulsion to the orbital basis, obtains the plain and the
density-corrected quasiparticle ionization energies from the extended coupled-cluster effective Hamiltonian, and
computes the exact ionization energy; the break-even condition is then solved for the scale. The condition also holds
trivially at lambda = 0, where all three methods agree, so the root is sought inside a bracket of positive scales.

Returns
-------
numpy.ndarray of shape (4,): [break-even scale lambda_be, IP_G0W0, IP_G0W0+gamma, IP_FCI in eV at lambda_be]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def break_even_interaction_scale(n_sites: int, params: "np.ndarray", scale_low: float, scale_high: float) -> "np.ndarray":
    '''Break-even interaction scale of density-corrected versus plain G0W0@HF ionization energies of a PPP polyene.

    Parameters
    ----------
    n_sites : int
        Number of carbon atoms of the all-trans polyene, even and at least 2; n_sites / 2 orbitals are doubly occupied.
    params : np.ndarray
        Shape (6,), [W, U, t_short, t_long, r_short, r_long] of the PPP model in eV and Angstrom, as in
        ppp_polyene_hamiltonian.
    scale_low, scale_high : float
        Positive bracket 0 < scale_low < scale_high of the interaction scale lambda.

    Returns
    -------
    result : np.ndarray
        Shape (4,), [lambda_be, IP_G0W0, IP_G0W0+gamma, IP_FCI], the last three in eV at lambda_be. IP_G0W0 is the
        highest-occupied quasiparticle ionization energy with the Hartree-Fock Fock matrix, IP_G0W0+gamma the same with
        the Fock block augmented by the static correction of the linearized GW density matrix, and IP_FCI the full
        configuration interaction value. lambda_be is the root of IP_G0W0 + IP_G0W0+gamma - 2 IP_FCI inside the
        bracket, located to 1e-7.

    Raises
    ------
    ValueError
        If the bracket is not positive and increasing, or the break-even function has the same sign at both ends.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _oracle_break_even_interaction_scale(n_sites: int, params: "np.ndarray", scale_low: float,
                                         scale_high: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    from scipy.optimize import brentq
    if not 0.0 < scale_low < scale_high:
        raise ValueError("the bracket must satisfy 0 < scale_low < scale_high")
    occ = int(n_sites) // 2

    def _ionization_energies(lam):
        hg = _oracle_ppp_polyene_hamiltonian(n_sites, lam, params)
        h, g = hg[0], hg[1]
        scf = _oracle_restricted_hartree_fock(h, g, occ)
        e, c = scf[0], scf[1:]
        pair = np.einsum('mp,mq->mpq', c, c)
        eri = np.einsum('mpq,mn,nrs->pqrs', pair, g, pair, optimize=True)
        ip_plain = _oracle_ecc_quasiparticle_ionization(e, eri, occ, np.diag(e))[0]
        dens = _oracle_linearized_gw_density_matrix(e, eri, occ)
        sigma = _oracle_static_self_energy_correction(eri, dens, occ)
        ip_corr = _oracle_ecc_quasiparticle_ionization(e, eri, occ, np.diag(e) + sigma)[0]
        ip_exact = _oracle_fci_ionization_energy(h, g, occ)
        return ip_plain, ip_corr, ip_exact

    def _balance(lam):
        a, b, c_ = _ionization_energies(lam)
        return a + b - 2.0 * c_

    f_low, f_high = _balance(scale_low), _balance(scale_high)
    if f_low * f_high > 0.0:
        raise ValueError("the break-even function does not change sign inside the bracket")
    lam_be = brentq(_balance, scale_low, scale_high, xtol=1e-9, rtol=1e-12)
    ips = _ionization_energies(lam_be)
    return np.array([lam_be, ips[0], ips[1], ips[2]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "params = np.array([11.16, 11.13, -2.60, -2.20, 1.35, 1.46])\n")
    return [
        # --- Normal: hexatriene with the standard parameters ---
        {
            "setup": base,
            "call": "break_even_interaction_scale(6, params.copy(), 0.5, 2.0)",
            "gold_call": "_oracle_break_even_interaction_scale(6, params, 0.5, 2.0)",
            "tol": 1e-6,
        },
        # --- Boundary: butadiene, the shortest chain with a nontrivial screening problem ---
        {
            "setup": base,
            "call": "break_even_interaction_scale(4, params.copy(), 0.5, 2.0)",
            "gold_call": "_oracle_break_even_interaction_scale(4, params, 0.5, 2.0)",
            "tol": 1e-6,
        },
        # --- Edge: hexatriene with weaker bond alternation, longer bonds and a smaller on-site repulsion ---
        {
            "setup": "import numpy as np\nparams = np.array([11.16, 9.5, -2.45, -2.30, 1.38, 1.44])\n",
            "call": "break_even_interaction_scale(6, params.copy(), 0.4, 2.5)",
            "gold_call": "_oracle_break_even_interaction_scale(6, params, 0.4, 2.5)",
            "tol": 1e-6,
        },
        # --- Invalid: a bracket that holds no sign change must raise ValueError ---
        {
            "setup": base + "def run(fn):\n"
                            "    try:\n"
                            "        fn(4, params, 1.2, 2.0)\n"
                            "        return 0\n"
                            "    except ValueError:\n"
                            "        return 1\n",
            "call": "run(break_even_interaction_scale)",
            "gold_call": "run(_oracle_break_even_interaction_scale)",
        },
    ]
