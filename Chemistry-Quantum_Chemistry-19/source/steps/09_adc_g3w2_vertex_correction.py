"""
The eigenvalues below the Fermi level are ionization energies (with a minus sign): each eigenvector has a component on every orbital, and the principal ionization potential of the highest occupied orbital is minus the eigenvalue whose eigenvector carries the largest squared weight on that orbital. The quantity requested by the task is the vertex correction to this principal ionization potential, the ADC-G3W2 value minus the ADC-GW value. It isolates the effect of the second-order screened exchange and of its ADC completion (the cubic couplings and the three-body configurations); the intermediate schemes that stop at the screened-exchange couplings or at the configuration-coupling block move the ionization potential in the opposite direction to the complete scheme, so the sign of this correction is a sharp test of the full construction.

All pieces are now in place to assemble the two effective Hamiltonians of interest and to read off the principal ionization potential from each. The ADC-GW Hamiltonian is the upfolded GW problem: the Fock matrix (diagonal in the canonical basis) in the one-particle block, the GW effective integrals as the coupling to the 2h1p and 2p1h configurations, and the dressed energies on their diagonals; the hole and particle sectors do not couple to each other directly. The ADC-G3W2 Hamiltonian adds the screened-exchange and cubic corrections to the couplings, the configuration-coupling block inside each two-body sector, and the three-body sectors with their couplings to the orbitals and to the two-body configurations of the same branch; the layout is given in Eq. (22) of the source paper. Both matrices are real symmetric, so their eigenvalues are real and the associated spectral functions are positive, which is the whole point of the ADC resummation compared with the raw G3W2 self-energy.

Returns
-------
float: the vertex correction IP(ADC-G3W2) - IP(ADC-GW) to the principal ionization potential of the highest occupied orbital, in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adc_g3w2_vertex_correction(bond_lengths, hoppings, site_energies, hubbard_u, n_occ):
    '''Vertex correction to the principal ionization potential of a PPP chain.

    Parameters
    ----------
    bond_lengths : array_like of float, shape (n - 1,)
        Distances in Angstrom between consecutive centres along the line.
    hoppings : array_like of float, shape (n - 1,)
        Hopping integrals in eV between consecutive centres.
    site_energies : array_like of float, shape (n,)
        Site energies in eV of the n centres (before the background shift).
    hubbard_u : float
        On-site repulsion U in eV.
    n_occ : int
        Number of doubly occupied orbitals, between 1 and n - 1.

    Returns
    -------
    delta : float
        IP(ADC-G3W2) minus IP(ADC-GW) for the principal ionization of the
        highest occupied orbital (the eigenvalue of each effective
        Hamiltonian whose eigenvector has the largest squared weight on
        that orbital), in eV.
        Raises ValueError for inconsistent or non-physical input.
    '''
    delta = 0.0
    return delta

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _principal_ip(H, orbital):
    w, v = np.linalg.eigh(H)
    k = int(np.argmax(v[orbital, :] ** 2))
    return -w[k]


def _assemble(f, blocks):
    """blocks[branch] = (U1, D1, U2, C21, K2); returns the symmetric Hamiltonian."""
    n = f.shape[0]
    Um, Dm, U2m, C21m, K2m = blocks["hole"]
    Up, Dp, U2p, C21p, K2p = blocks["particle"]
    dm, d2m, dp, d2p = Dm.shape[0], K2m.shape[0], Dp.shape[0], K2p.shape[0]
    o = np.cumsum([0, n, dm, d2m, dp, d2p])
    H = np.zeros((o[-1], o[-1]))
    H[o[0]:o[1], o[0]:o[1]] = f
    H[o[1]:o[2], o[0]:o[1]] = Um
    H[o[1]:o[2], o[1]:o[2]] = Dm
    H[o[2]:o[3], o[0]:o[1]] = U2m
    H[o[2]:o[3], o[1]:o[2]] = C21m
    H[o[2]:o[3], o[2]:o[3]] = K2m
    H[o[3]:o[4], o[0]:o[1]] = Up
    H[o[3]:o[4], o[3]:o[4]] = Dp
    H[o[4]:o[5], o[0]:o[1]] = U2p
    H[o[4]:o[5], o[3]:o[4]] = C21p
    H[o[4]:o[5], o[4]:o[5]] = K2p
    return np.tril(H) + np.tril(H, -1).T


def _oracle_adc_g3w2_vertex_correction(bond_lengths, hoppings, site_energies, hubbard_u, n_occ):
    eps_site = np.asarray(site_energies, dtype=float)
    n = eps_site.shape[0]
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    n_occ = int(n_occ)
    lengths = np.asarray(bond_lengths, dtype=float)
    if lengths.shape != (n - 1,) or np.any(lengths <= 0.0):
        raise ValueError("bond_lengths must have n - 1 positive entries")
    e2 = 14.397
    pos = np.concatenate([[0.0], np.cumsum(lengths)])
    r = np.abs(pos[:, None] - pos[None, :])
    V = e2 / np.sqrt((e2 / float(hubbard_u)) ** 2 + r ** 2)
    F = _oracle_ppp_fock_matrix(bond_lengths, hoppings, site_energies, hubbard_u, n_occ)
    eps = np.linalg.eigvalsh(F)
    eri = _oracle_mo_eri_tensor(F, V)
    Om = _oracle_drpa_excitation_energies(eps, eri, n_occ)
    M = _oracle_gw_effective_integrals(eps, eri, n_occ)
    nov = Om.shape[0]
    f = np.diag(eps)
    gw = {}
    full = {}
    for branch in ("hole", "particle"):
        rows = list(range(n_occ)) if branch == "hole" else list(range(n_occ, n))
        sgn = -1.0 if branch == "hole" else 1.0
        K1 = np.diag([eps[p] + sgn * Om[v] for p in rows for v in range(nov)])
        if branch == "hole":
            U1 = np.array([[M[q, i, v] for q in range(n)] for i in rows for v in range(nov)])
        else:
            U1 = np.array([[M[a, q, v] for q in range(n)] for a in rows for v in range(nov)])
        d1 = K1.shape[0]
        gw[branch] = (U1, K1, np.zeros((0, n)), np.zeros((0, d1)), np.zeros((0, 0)))
        U1_full = (U1 + _oracle_sosex_coupling_block(eps, eri, M, Om, n_occ, branch)
                   + _oracle_third_order_coupling_block(eps, M, Om, n_occ, branch))
        D1 = K1 + _oracle_adc3_diagonal_block(eps, M, Om, n_occ, branch)
        row = _oracle_three_body_row_block(eps, M, Om, n_occ, branch)
        U2 = row[:, :n]
        C21 = row[:, n:n + d1]
        K2 = row[:, n + d1:]
        full[branch] = (U1_full, D1, U2, C21, K2)
    homo = n_occ - 1
    ip_gw = _principal_ip(_assemble(f, gw), homo)
    ip_full = _principal_ip(_assemble(f, full), homo)
    return float(ip_full - ip_gw)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
lengths = [1.34, 1.47, 1.36, 1.46, 1.38]
hops = [-2.23, -1.62, -2.02, -2.05, -2.10]
site = [-14.2, -11.2, -10.5, -11.9, -10.4, -10.4]
"""
    return [
        # --- Normal: the reference chain of the task ---
        {
            "setup": setup,
            "call": "round(adc_g3w2_vertex_correction(lengths, hops, site, 14.0, 3), 7)",
            "gold_call": "round(_oracle_adc_g3w2_vertex_correction(lengths, hops, site, 14.0, 3), 7)",
        },
        # --- Normal: a four-centre chain ---
        {
            "setup": setup,
            "call": "round(adc_g3w2_vertex_correction([1.35, 1.45, 1.37], [-2.4, -1.9, -2.3], [-12.6, -11.0, -11.5, -10.2], 11.0, 2), 7)",
            "gold_call": "round(_oracle_adc_g3w2_vertex_correction([1.35, 1.45, 1.37], [-2.4, -1.9, -2.3], [-12.6, -11.0, -11.5, -10.2], 11.0, 2), 7)",
        },
        # --- Boundary: weaker interaction on the reference chain ---
        {
            "setup": setup,
            "call": "round(adc_g3w2_vertex_correction(lengths, hops, site, 8.0, 3), 7)",
            "gold_call": "round(_oracle_adc_g3w2_vertex_correction(lengths, hops, site, 8.0, 3), 7)",
        },
        # --- Edge: a two-centre chain (one occupied and one virtual orbital, one mode) ---
        {
            "setup": setup,
            "call": "round(adc_g3w2_vertex_correction([1.40], [-2.5], [-11.0, -12.4], 10.0, 1), 7)",
            "gold_call": "round(_oracle_adc_g3w2_vertex_correction([1.40], [-2.5], [-11.0, -12.4], 10.0, 1), 7)",
        },
        # --- Invalid: n_occ leaves no virtual orbital ---
        {
            "setup": setup + """
def run_model():
    try:
        adc_g3w2_vertex_correction(lengths, hops, site, 14.0, 6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_adc_g3w2_vertex_correction(lengths, hops, site, 14.0, 6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
