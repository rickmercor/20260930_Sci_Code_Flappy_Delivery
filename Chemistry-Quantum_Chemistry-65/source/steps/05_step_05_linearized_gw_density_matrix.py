"""
Compute the linearized GW one-body density matrix of a closed-shell reference with direct-ring screening.

The Hartree-Fock density matrix is idempotent: occupied orbitals hold exactly one electron per spin and virtual
orbitals none. The linearized GW density matrix is the first correction to it from the G0W0 self-energy expanded
around the Hartree-Fock Green's function. Removing electrons from occupied orbitals and placing them in virtual
orbitals through the virtual emission and absorption of screening bosons lowers the occupied occupations and raises
the virtual ones by the same total amount, so the electron count is conserved. Formally the correction is the
frequency integral of G0 Sigma_c G0, the first-order change of the Green's function in the correlation self-energy.
An extended coupled-cluster perturbation treatment of the doubly similarity-transformed electron-boson Hamiltonian reproduces the same matrix: its
occupied and virtual blocks arise from the second-order energy and its occupied-virtual block from a third-order term,
and the resulting matrix is symmetrized.

The screening bosons are the positive-frequency direct random-phase modes, with excitation energies Omega_nu and
eigenvectors normalized as X^T X - Y^T Y = 1. For a spin-restricted reference each singlet mode couples to an electron
moving between orbitals p and q with M_{pq,nu} = sqrt(2) sum_{ia} (pq|ia) (X + Y)_{ia,nu}.

Returns
-------
numpy.ndarray of shape (n, n): symmetric per-spin linearized GW density matrix in the Hartree-Fock orbital basis
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def linearized_gw_density_matrix(orbital_energies: "np.ndarray", eri: "np.ndarray", n_occ: int) -> "np.ndarray":
    '''Per-spin linearized GW one-body density matrix in the canonical Hartree-Fock orbital basis.

    Parameters
    ----------
    orbital_energies : np.ndarray
        Shape (n,), canonical Hartree-Fock orbital energies in ascending order (eV); orbitals 0 to n_occ-1 are doubly
        occupied.
    eri : np.ndarray
        Shape (n, n, n, n), real two-electron integrals (pq|rs) in chemists' notation in the same orbital basis (eV).
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ < n.

    Returns
    -------
    density : np.ndarray
        Shape (n, n), symmetric, normalized per spin so that the Hartree-Fock matrix would be 1 on each occupied
        diagonal element and its trace is n_occ. With indices i, j occupied, a, b virtual and nu the direct-RPA modes,
        gamma_ij = delta_ij - sum_{a,nu} M_{ia,nu} M_{ja,nu} / ((e_i - e_a - Omega_nu)(e_j - e_a - Omega_nu)),
        gamma_ab = sum_{i,nu} M_{ai,nu} M_{bi,nu} / ((e_i - e_a - Omega_nu)(e_i - e_b - Omega_nu)),
        gamma_ia = gamma_ai = [sum_{b,nu} M_{ib,nu} M_{ab,nu} / (e_i - e_b - Omega_nu)
                    - sum_{j,nu} M_{ij,nu} M_{aj,nu} / (e_j - e_a - Omega_nu)] / (e_i - e_a).
        These blocks are the frequency integral of G0 Sigma_c G0 with the Hartree-Fock Green's function G0.

    Raises
    ------
    ValueError
        If n_occ is outside 1 <= n_occ < n or eri does not have shape (n, n, n, n).
    '''
    return density

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_linearized_gw_density_matrix(orbital_energies: "np.ndarray", eri: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    e = np.asarray(orbital_energies, dtype=float)
    v4 = np.asarray(eri, dtype=float)
    n = e.shape[0]
    o = int(n_occ)
    if not 1 <= o < n:
        raise ValueError("n_occ must satisfy 1 <= n_occ < n")
    if v4.shape != (n, n, n, n):
        raise ValueError("eri must have shape (n, n, n, n)")
    v = n - o
    gap = (e[o:][None, :] - e[:o][:, None]).reshape(-1)
    k = v4[:o, o:, :o, o:].reshape(o * v, o * v)
    root = np.sqrt(gap)
    omega2, vec = np.linalg.eigh(root[:, None] * (np.diag(gap) + 4.0 * k) * root[None, :])
    omega = np.sqrt(omega2)
    x_plus_y = (root[:, None] * vec) / np.sqrt(omega)[None, :]
    mcoup = np.sqrt(2.0) * np.einsum('pqk,kv->pqv', v4[:, :, :o, o:].reshape(n, n, o * v), x_plus_y)
    ei, ea = e[:o], e[o:]
    den = ei[:, None, None] - ea[None, :, None] - omega[None, None, :]          # (i, a, nu)
    m_ov = mcoup[:o, o:, :] / den
    g_oo = np.eye(o) - np.einsum('iav,jav->ij', m_ov, m_ov)
    g_vv = np.einsum('iav,ibv->ab', m_ov, m_ov)
    term_b = np.einsum('ibv,abv->ia', m_ov, mcoup[o:, o:, :])
    term_j = np.einsum('ijv,ajv->ia', mcoup[:o, :o, :], (mcoup[o:, :o, :] / den.transpose(1, 0, 2)))
    g_ov = (term_b - term_j) / (ei[:, None] - ea[None, :])   # hole term enters with a minus sign
    dens = np.zeros((n, n))
    dens[:o, :o] = g_oo
    dens[o:, o:] = g_vv
    dens[:o, o:] = g_ov
    dens[o:, :o] = g_ov.T
    return 0.5 * (dens + dens.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    make = ("import numpy as np\n"
            "def mo_system(n, o, seed, scale):\n"
            "    rng = np.random.default_rng(seed)\n"
            "    q, _ = np.linalg.qr(rng.normal(size=(n, n)))\n"
            "    pos = np.cumsum(np.full(n, 1.4))\n"
            "    g = scale * 11.13 / np.sqrt(1.0 + (11.13 * np.abs(pos[:, None] - pos[None]) / 14.397) ** 2)\n"
            "    eri = np.einsum('mp,mq,mn,nr,ns->pqrs', q, q, g, q, q)\n"
            "    e = np.sort(rng.uniform(-12.0, -8.0, o)).tolist() + np.sort(rng.uniform(-2.0, 1.0, n - o)).tolist()\n"
            "    return np.array(e), eri\n")
    return [
        # --- Normal: five orbitals, two occupied ---
        {
            "setup": make + "e, eri = mo_system(5, 2, 4, 1.0)\n",
            "call": "linearized_gw_density_matrix(e.copy(), eri.copy(), 2)",
            "gold_call": "_oracle_linearized_gw_density_matrix(e, eri, 2)",
            "tol": 1e-8,
        },
        # --- Boundary: two orbitals, where the occupied-virtual block reduces to a single element ---
        {
            "setup": make + "e, eri = mo_system(2, 1, 9, 1.0)\n",
            "call": "linearized_gw_density_matrix(e.copy(), eri.copy(), 1)",
            "gold_call": "_oracle_linearized_gw_density_matrix(e, eri, 1)",
            "tol": 1e-8,
        },
        # --- Edge: four occupied and two virtual orbitals with strong coupling ---
        {
            "setup": make + "e, eri = mo_system(6, 4, 17, 2.4)\n",
            "call": "linearized_gw_density_matrix(e.copy(), eri.copy(), 4)",
            "gold_call": "_oracle_linearized_gw_density_matrix(e, eri, 4)",
            "tol": 1e-8,
        },
    ]
