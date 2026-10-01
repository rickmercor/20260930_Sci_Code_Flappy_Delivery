"""
Orchestrator. Return the first ionisation potential of the half-filled open PPP chain (K = len(eps_site) sites, K electrons, P = K/2 pairs) from the extended Koopmans' theorem built on orbital-optimised pCCD: build the model of step 1, take the RHF orbitals, optimise the orbitals (step 8, started from the RHF orbitals), and in the resulting natural-orbital basis obtain the amplitudes, the multipliers, the response density matrices and the generalised Fock matrix F of the earlier steps; discard every natural orbital whose per-spin occupation gamma_p is below cutoff; symmetrise F on the kept orbitals, F_s = (F + F^T)/2, transform F' = gamma^{-1/2} F_s gamma^{-1/2} with gamma the diagonal 1-RDM on the kept orbitals, and diagonalise F'. The ionisation potentials are the negatives of those eigenvalues whose unit-norm eigenvectors of F' carry more than half of their squared norm on the P most strongly occupied natural orbitals; return the smallest of them. Call the earlier step functions rather than reimplementing them. Raise ValueError if the number of sites is odd or below 4, if cutoff is outside [0, 1), or if no eigenvector of occupied character exists.

The extended Koopmans' theorem turns the one- and two-particle density matrices of a correlated wavefunction into a generalised eigenvalue problem for the ionisation energies; with the variationally optimised pair coupled cluster wavefunction the required matrices are the natural by-products of the orbital optimisation, so correlated ionisation potentials follow at mean-field cost.

Returns
-------
float, the first ionisation potential in units of t.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def first_ionization_potential(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray", cutoff: float) -> float:
    '''First ionisation potential of the half-filled PPP chain from the extended Koopmans' theorem on orbital-optimised pCCD.

    Parameters
    ----------
    t : float
        Hopping scale, t > 0.
    delta : float
        Bond alternation, |delta| < 1.
    U : float
        On-site repulsion, U > 0.
    kappa : float
        Ohno screening parameter, kappa > 0.
    eps_site : np.ndarray
        Site energies, 1-D array of even length K >= 4.
    cutoff : float
        Natural orbitals with per-spin occupation below this value are discarded, 0 <= cutoff < 1.

    Returns
    -------
    ip : float
        The first (smallest) ionisation potential in units of t, as a Python float.

    Raises
    ------
    ValueError
        If the number of sites is odd or below 4, cutoff is outside [0, 1), or no eigenvector of occupied character exists.
    '''
    return ip

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_first_ionization_potential(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray", cutoff: float) -> float:
    """EKT(oo-pCCD) first ionisation potential of the PPP chain at half filling (K sites, K electrons, K even).
    Steps: model -> RHF -> oo-pCCD natural orbitals -> amplitudes, Lambda, response RDMs, generalised Fock
    -> drop natural orbitals with occupation below cutoff -> F' = gamma^{-1/2} F gamma^{-1/2} (F symmetrised)
    -> eigenvalues e; the ionisation potentials are -e for eigenvectors with more than half their weight on
    the P strongly occupied natural orbitals; return the smallest of them."""
    eps_site = np.asarray(eps_site, dtype=float); K = eps_site.size
    if K % 2 or K < 4:
        raise ValueError("need an even number of sites, at least 4")
    cutoff = float(cutoff)
    if not (0.0 <= cutoff < 1.0):
        raise ValueError("cutoff must lie in [0, 1)")
    P = K // 2
    hV = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site); h, V = hV[0], hV[1]
    C0 = _oracle_rhf_orbitals(h, V, P)
    C = _oracle_oo_pccd_orbitals(h, V, C0, P)
    hm = C.T @ h @ C
    Vm = _oracle_mo_two_electron_integrals(V, C)
    c = _oracle_pccd_amplitudes(hm, Vm, P)
    lam = _oracle_pccd_lambda_amplitudes(hm, Vm, c)
    rd = _oracle_pccd_response_rdms(c, lam)
    F = _oracle_generalized_fock(hm, Vm, rd)
    gam = np.diag(rd[0])
    keep = np.where(gam > cutoff)[0]
    Fk = 0.5 * (F[np.ix_(keep, keep)] + F[np.ix_(keep, keep)].T); gk = gam[keep]
    Fp = Fk / np.sqrt(gk)[:, None] / np.sqrt(gk)[None, :]
    e, Cp = np.linalg.eigh(Fp)
    occ_idx = np.where(keep < P)[0]
    weight = (Cp[occ_idx, :] ** 2).sum(axis=0)
    phys = e[weight > 0.5]
    if phys.size == 0:
        raise ValueError("no eigenvector with dominant occupied character")
    return float(-phys.max())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of step test cases (normal, boundary and edge inputs, plus one invalid input)."""
    return [
        {
            "setup": "import numpy as np\nt, delta, U, kappa, cutoff = 1.0, 0.15, 6.0, 4.0, 5e-5\neps_site = -4.0 + np.array([0.0, 0.3, -0.2, 0.1, -0.3, 0.2, 0.15, -0.25])",
            "call": "first_ionization_potential(t, delta, U, kappa, eps_site, cutoff)",
            "gold_call": "_oracle_first_ionization_potential(t, delta, U, kappa, eps_site, cutoff)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nt, delta, U, kappa, cutoff = 1.0, 0.1, 4.0, 3.0, 5e-5\neps_site = -3.5 + np.array([0.0, 0.2, -0.1, 0.3, -0.2, 0.1])",
            "call": "first_ionization_potential(t, delta, U, kappa, eps_site, cutoff)",
            "gold_call": "_oracle_first_ionization_potential(t, delta, U, kappa, eps_site, cutoff)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nt, delta, U, kappa, cutoff = 0.8, 0.0, 5.0, 2.5, 1e-3\neps_site = np.array([-3.0, -3.2, -2.75, -3.15])",
            "call": "first_ionization_potential(t, delta, U, kappa, eps_site, cutoff)",
            "gold_call": "_oracle_first_ionization_potential(t, delta, U, kappa, eps_site, cutoff)",
            "tol": 1e-08,
        },
        {  # invalid input: the documented ValueError contract
            "setup": "import numpy as np\ndef run_model():\n    try:\n        first_ionization_potential(1.0, 0.1, 5.0, 3.0, np.array([-3.0, -3.1, -2.9]), 5e-5)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_first_ionization_potential(1.0, 0.1, 5.0, 3.0, np.array([-3.0, -3.1, -2.9]), 5e-5)\n        return 0\n    except ValueError:\n        return 1",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
