"""
Orchestrator. For the chain of step 01 with the given parameters and the active spatial orbitals active (indices into the ascending canonical orbitals of step 01), run the whole downfolding: solve the mean-field problem with step 01, transform the integrals with step 02 (and the one-body matrix of the chain to the same basis), evaluate the RPA correlation energy of the full chain with step 03 on its complete spin-orbital particle-hole space, build the constrained particle-hole space with step 05, its static kernel with step 04 and the screened interaction with step 06, evaluate with step 03 the RPA correlation energy of the active orbitals alone with the screened interaction (the source's prescription for that term), build the effective one-body matrix with step 07 and the constant shift with step 08, solve the active space with step 09 using the number of electrons that occupy the active orbitals in the mean-field reference, and return the correlation energy E_corr = E_tot - E_HF, where E_tot is the total energy of the downfolded Hamiltonian (constant term plus active-space energy) and E_HF the restricted Hartree-Fock energy of the full chain from step 01 (the constant of the model cancels). The value is returned in the energy units in which t, U and kappa are given (it scales linearly with a common scaling of the three); for t = 1 this is the value in units of t. Call the earlier step functions rather than reimplementing them.

The source characterises its downfolding methods by the total ground-state energies they yield when the effective Hamiltonian is solved exactly, compared with the correlation energy of the full system; for a model whose full solution is tractable the correlation energy recovered by the downfolded Hamiltonian is the single number that summarises the method.

Returns
-------
float, the correlation energy of the downfolded Hamiltonian in the energy units of the inputs (units of t for t = 1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def downfolded_correlation_energy(n_sites: int, t: float, delta: float, U: float, kappa: float, active: "list[int]") -> float:
    """Orchestrator. For the chain of step 01 with the given parameters and the active spatial orbitals active (indices into the ascending canonical orbitals of step 01), run the whole downfolding: solve the mean-field problem with step 01, transform the integrals with step 02 (and the one-body matrix of the chain to the same basis), evaluate the RPA correlation energy of the full chain with step 03 on its complete spin-orbital particle-hole space, build the constrained particle-hole space with step 05, its static kernel with step 04 and the screened interaction with step 06, evaluate with step 03 the RPA correlation energy of the active orbitals alone with the screened interaction (the source's prescription for that term), build the effective one-body matrix with step 07 and the constant shift with step 08, solve the active space with step 09 using the number of electrons that occupy the active orbitals in the mean-field reference, and return the correlation energy E_corr = E_tot - E_HF, where E_tot is the total energy of the downfolded Hamiltonian (constant term plus active-space energy) and E_HF the restricted Hartree-Fock energy of the full chain from step 01 (the constant of the model cancels). The value is returned in the energy units in which t, U and kappa are given (it scales linearly with a common scaling of the three); for t = 1 this is the value in units of t. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    n_sites : int
        Even number of sites (at least 2).
    t : float
        Positive hopping scale.
    delta : float
        Bond alternation, |delta| < 1.
    U : float
        Positive on-site repulsion.
    kappa : float
        Positive Ohno range parameter.
    active : list[int]
        Distinct indices in [0, n_sites - 1] of the active canonical orbitals (non-empty).

    Returns
    -------
    e_corr : float
        E_corr = E_tot - E_HF in the energy units of t, U and kappa (native Python float).

    Raises
    ------
    ValueError
        If n_sites is odd or below 2, active is invalid, |delta| >= 1, or t, U or kappa is not positive.
    """
    return e_corr

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_scalar(x, name, positive=False, nonneg=False):
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("%s must be finite" % name)
    if positive and x <= 0:
        raise ValueError("%s must be positive" % name)
    if nonneg and x < 0:
        raise ValueError("%s must be non-negative" % name)
    return x


def _check_int(n, name, minimum=0):
    if int(n) != n or n < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(n)


def _check_active(active, n_orb):
    act = [int(p) for p in np.asarray(active).ravel()]
    if len(act) == 0 or len(set(act)) != len(act) or min(act) < 0 or max(act) >= n_orb:
        raise ValueError("active must be a non-empty list of distinct orbital indices in range")
    return sorted(act)


def _ppp_site_hamiltonian(n_sites, t, delta, U, kappa):
    """PPP chain in the neutral form: H = sum_k t_k (a+_k a_k+1 + h.c.) + U sum_i n_i,up n_i,dn
    + sum_{i<j} V_ij (n_i - 1)(n_j - 1); t_k = -t (1 + delta (-1)^k), Ohno V_ij = U / sqrt(1 + (U |i-j| / kappa)^2).
    Returns the one-body site matrix (with the -sum_j V_ij on-site shift), the site-pair interaction V,
    and the constant sum_{i<j} V_ij."""
    h = np.zeros((n_sites, n_sites))
    for k in range(n_sites - 1):
        h[k, k + 1] = h[k + 1, k] = -t * (1.0 + delta * (-1) ** k)
    idx = np.arange(n_sites)
    V = U / np.sqrt(1.0 + (U * np.abs(idx[:, None] - idx[None, :]) / kappa) ** 2)
    for i in range(n_sites):
        h[i, i] = -(np.sum(V[i]) - V[i, i])
    E_const = 0.5 * (np.sum(V) - np.trace(V))
    return h, V, E_const


def _ph_pairs(n_occ, n_orb):
    """same-spin particle-hole pairs of spin-orbitals (2p = alpha, 2p+1 = beta), i outer, a inner."""
    occ = range(2 * n_occ)
    virt = range(2 * n_occ, 2 * n_orb)
    return [(i, a) for i in occ for a in virt if i % 2 == a % 2]


def _pair_matrices(eps, eri, pairs):
    """delta_eps (n,) and v_ph (n, n) = (ia|jb) over the given spin-orbital pairs; eps, eri spatial."""
    n = len(pairs)
    d = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs]).reshape(n)
    v = np.array([[eri[i // 2, a // 2, j // 2, b // 2] for (j, b) in pairs] for (i, a) in pairs]).reshape(n, n)
    return d, v


def _oracle_downfolded_correlation_energy(n_sites: int, t: float, delta: float, U: float, kappa: float, active: "list[int]") -> float:
    n_sites = _check_int(n_sites, "n_sites", 2)
    t = _check_scalar(t, "t", positive=True)
    delta = _check_scalar(delta, "delta")
    U = _check_scalar(U, "U", positive=True)
    kappa = _check_scalar(kappa, "kappa", positive=True)
    if n_sites % 2:
        raise ValueError("n_sites must be even")
    act = _check_active(active, n_sites)
    n_occ = n_sites // 2
    h, V, E_const = _ppp_site_hamiltonian(n_sites, t, delta, U, kappa)
    out = _oracle_ppp_rhf(n_sites, t, delta, U, kappa)
    eps, C = out[0], out[1:]
    h_mo = C.T @ h @ C
    eri = _oracle_mo_integrals(C, U, kappa)
    E_hf = sum(h_mo[i, i] + eps[i] for i in range(n_occ))          # closed-shell RHF energy (electronic part)
    # RPA correlation energy of the full system: all same-spin spin-orbital particle-hole pairs, bare v
    d_full, v_full = _pair_matrices(eps, eri, _ph_pairs(n_occ, n_sites))
    e_rpa_full = _oracle_rpa_correlation_energy(d_full, v_full)
    # constrained RPA: reduced pair space, its static kernel, and the screened interaction on the active block
    pairs_red = _oracle_constrained_ph_pairs(n_sites, n_occ, act)
    d_red, v_red = _pair_matrices(eps, eri, [(int(i), int(a)) for i, a in pairs_red])
    kernel = _oracle_rpa_static_kernel(d_red, v_red)
    veff = _oracle_screened_interaction(eri, pairs_red, kernel, act)
    # RPA correlation energy of the active orbitals alone with the screened (ia|jb) and the HF orbital energies
    loc = {p: k for k, p in enumerate(act)}
    pairs_act = [(i, a) for (i, a) in _ph_pairs(n_occ, n_sites) if i // 2 in loc and a // 2 in loc]
    d_act = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs_act])
    v_act = np.array([[veff[loc[i // 2], loc[a // 2], loc[j // 2], loc[b // 2]] for (j, b) in pairs_act] for (i, a) in pairs_act]).reshape(len(pairs_act), len(pairs_act))
    e_rpa_active = _oracle_rpa_correlation_energy(d_act, v_act)
    t_eff = _oracle_effective_one_body(h_mo, eri, veff, n_occ, act)
    E_E = _oracle_energy_shift(h_mo, eri, veff, n_occ, act, e_rpa_full, e_rpa_active)
    n_elec = 2 * sum(1 for p in act if p < n_occ)
    E_cas = _oracle_active_space_fci(t_eff, veff, n_elec)
    # E_tot = E_const + E_E + E_cas ; the constant cancels in E_tot - E_HF
    return float(E_E + E_cas - E_hf)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "n_sites, t, delta, U, kappa, active = 10, 1.0, 0.07, 4.0, 6.0, [3, 4, 5, 6]\n",
            "call": "downfolded_correlation_energy(n_sites, t, delta, U, kappa, active)",
            "gold_call": "_oracle_downfolded_correlation_energy(n_sites, t, delta, U, kappa, active)",
            "tol": 1e-07,
        },
        {
            "setup": "n_sites, t, delta, U, kappa, active = 8, 1.0, 0.07, 4.0, 4.0, [2, 3, 4, 5]\n",
            "call": "downfolded_correlation_energy(n_sites, t, delta, U, kappa, active)",
            "gold_call": "_oracle_downfolded_correlation_energy(n_sites, t, delta, U, kappa, active)",
            "tol": 1e-07,
        },
        {
            "setup": "n_sites, t, delta, U, kappa, active = 6, 1.0, 0.1, 3.0, 4.0, [2, 3]\n",
            "call": "downfolded_correlation_energy(n_sites, t, delta, U, kappa, active)",
            "gold_call": "_oracle_downfolded_correlation_energy(n_sites, t, delta, U, kappa, active)",
            "tol": 1e-07,
        },
        {
            "setup": "n_sites, t, delta, U, kappa, active = 6, 1.0, 0.1, 3.0, 4.0, [0, 1, 2, 3, 4, 5]\n",
            "call": "downfolded_correlation_energy(n_sites, t, delta, U, kappa, active)",
            "gold_call": "_oracle_downfolded_correlation_energy(n_sites, t, delta, U, kappa, active)",
            "tol": 1e-07,
        },
        {
            "setup": "n_sites, t, delta, U, kappa, active = 10, 1.0, 0.07, 4.0, 6.0, [3, 4, 5, 10]\ndef run_model():\n    try:\n        downfolded_correlation_energy(n_sites, t, delta, U, kappa, active)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_downfolded_correlation_energy(n_sites, t, delta, U, kappa, active)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
