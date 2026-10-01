"""
Orchestrator. Build the chain (step 01), converge its scaled-exchange reference (step 02), form the reference operators and integrals in the orbital basis (steps 03-04), expand to spin-orbitals with the interleaved index of step 05 (orbital energies repeated for the two spins, integrals carried over for equal spins within each pair), form the static exchange-correlation difference of the reference from the exchange operator of step 03 and the fraction alpha, then check the screening is stable (step 05), obtain the GW correction (step 06) and the iterated-Dyson correction (step 07); add the correction to the reference density matrix (unit occupations of the nelec lowest spin-orbitals) and return the electronic dipole D = -sum_i x_i n_i with x_i = i - (N - 1)/2. Call the earlier step functions rather than reimplementing them.

The source assesses its density matrices through one-body observables such as the electronic dipole (its Eq. (26) and Fig. 2); for the task's asymmetric chain the dipole of the iterated-Dyson GW density matrix is the single number that summarises the whole procedure.

Returns
-------
float, the electronic dipole D of the iterated-Dyson GW density matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def idgw_dipole(N: int, t: float, U: float, V: float, eps: "np.ndarray", nelec: int, alpha: float) -> float:
    '''Electronic dipole of the task's chain from the source's iterated-Dyson GW density matrix (orchestrator).

    Parameters
    ----------
    N : int
        Number of sites.
    t : float
        Hopping amplitude.
    U : float
        On-site interaction.
    V : float
        Nearest-neighbour interaction.
    eps : np.ndarray
        One-dimensional array of the N site energies.
    nelec : int
        Number of electrons (even).
    alpha : float
        Exchange fraction of the reference mean field.

    Returns
    -------
    D : float
        Electronic dipole D = -sum_i x_i n_i of the iterated-Dyson GW density matrix, with site
        positions x_i = i - (N - 1)/2 for i = 0, ..., N - 1 and n_i the site occupations (both
        spins), as a native Python float.

    Raises
    ------
    ValueError
        If N < 2, nelec is odd or exceeds 2N, alpha is negative, or any earlier step raises.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.linalg import eigh, solve


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
    if isinstance(n, bool) or int(n) != n or int(n) < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(n)

def _spin_orbital_integrals(eri_spatial):
    """(PQ|RS) = (pq|rs) delta_{sP sQ} delta_{sR sS} with the interleaved spin-orbital index P = 2p + sigma"""
    n = eri_spatial.shape[0]
    eri = np.zeros((2 * n, 2 * n, 2 * n, 2 * n))
    for s1 in range(2):
        for s2 in range(2):
            eri[s1::2, s1::2, s2::2, s2::2] = eri_spatial
    return eri

def _spin_orbital_matrix(m_spatial):
    """same-spin block expansion of a one-body MO matrix, interleaved index P = 2p + sigma"""
    n = m_spatial.shape[0]
    m = np.zeros((2 * n, 2 * n))
    m[0::2, 0::2] = m_spatial
    m[1::2, 1::2] = m_spatial
    return m

def _site_positions(N):
    return np.arange(N, dtype=np.float64) - (N - 1) / 2.0


def _oracle_idgw_dipole(N: int, t: float, U: float, V: float, eps: "np.ndarray", nelec: int, alpha: float) -> float:
    N = _check_int(N, "N", 2)
    nelec = _check_int(nelec, "nelec", 2)
    alpha = _check_scalar(alpha, "alpha", nonneg=True)
    hW = _oracle_chain_hamiltonian(N, t, U, V, eps)
    h, W = hW[0], hW[1]
    C = _oracle_scaled_exchange_orbitals(h, W, nelec, alpha)
    FK = _oracle_mean_field_matrices(h, W, C, nelec, alpha)
    e_mo = np.diag(FK[0])
    eri_sp = _oracle_mo_coulomb_integrals(C, W)
    # spin-orbital quantities (interleaved index P = 2p + sigma)
    eps_so = np.repeat(e_mo, 2)
    eri = _spin_orbital_integrals(eri_sp)
    sxv = -(1.0 - alpha) * _spin_orbital_matrix(FK[1])       # <p|Sigma_x[gamma_gKS] - v_xc[gamma_gKS]|q> = -(1 - alpha) K_pq
    nocc = nelec                                              # number of occupied spin-orbitals
    Om = _oracle_rpa_excitation_energies(eps_so, eri, nocc)
    if Om[0] <= 0:
        raise ValueError("unstable screening")
    dg_gw = _oracle_gw_density_correction(eps_so, eri, sxv, nocc)
    dg = _oracle_iterated_dyson_correction(eps_so, eri, dg_gw, nocc)
    gamma = np.diag(np.r_[np.ones(nocc), np.zeros(2 * N - nocc)]) + dg
    x_mo = _spin_orbital_matrix(C.T @ np.diag(_site_positions(N)) @ C)
    D = -float(np.sum(gamma * x_mo))                          # D = -sum_i x_i n_i  (electron charge -1)
    if not np.isfinite(D):
        raise ValueError("non-finite dipole")
    return D

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nN, t, U, V, eps, nelec, alpha = 6, 1.0, 1.5, 1.5, np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 0.0]), 6, 0.75",
            "call": "idgw_dipole(N, t, U, V, eps.copy(), nelec, alpha)",
            "gold_call": "_oracle_idgw_dipole(N, t, U, V, eps.copy(), nelec, alpha)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nN, t, U, V, eps, nelec, alpha = 6, 1.0, 1.5, 1.5, np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 0.0]), 6, 1.0",
            "call": "idgw_dipole(N, t, U, V, eps.copy(), nelec, alpha)",
            "gold_call": "_oracle_idgw_dipole(N, t, U, V, eps.copy(), nelec, alpha)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nN, t, U, V, eps, nelec, alpha = 4, 0.8, 2.0, 0.7, np.array([-0.6, 0.0, 0.0, 0.3]), 4, 0.5",
            "call": "idgw_dipole(N, t, U, V, eps.copy(), nelec, alpha)",
            "gold_call": "_oracle_idgw_dipole(N, t, U, V, eps.copy(), nelec, alpha)",
            "tol": 1e-07,
        },
    ]
