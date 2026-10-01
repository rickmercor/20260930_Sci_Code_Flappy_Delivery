"""
Return, in increasing order, the excitation energies of the source's screening problem (its Eq. (10), the random-phase approximation that defines the GW screened interaction) for the spin-orbital energies eps and integrals eri of a closed-shell reference with nocc occupied spin-orbitals.

Within GW the polarisable part of the screened interaction is obtained from the random-phase approximation; the source diagonalises the corresponding Casida-like problem and uses its excitation energies and transition vectors to build the dynamical self-energy (its Eqs. (8)-(11)); in the spin-orbital formulation the spin-flip and triplet-like combinations appear as uncoupled modes at the bare orbital-energy differences.

Returns
-------
ndarray of float64 with shape (nov,), increasing.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rpa_excitation_energies(eps: "np.ndarray", eri: "np.ndarray", nocc: int) -> "np.ndarray":
    '''Excitation energies of the source's RPA screening problem (its Eq. (10)) in the spin-orbital basis.

    Parameters
    ----------
    eps : np.ndarray
        Spin-orbital energies, shape (M,), M = 2N, interleaved spin-orbital index P = 2p + sigma so
        that eps[2p] = eps[2p + 1] is the energy of spatial orbital p; occupied spin-orbitals first.
    eri : np.ndarray
        Spin-orbital two-electron integrals (PQ|RS), shape (M, M, M, M): the spatial integrals of
        step 04 carried over to spin-orbitals of the same spin in each pair and zero otherwise.
    nocc : int
        Number of occupied spin-orbitals (1 <= nocc < M).

    Returns
    -------
    Om : np.ndarray
        The nov = nocc (M - nocc) excitation energies Omega_s of the screening problem in increasing
        order, shape (nov,).

    Raises
    ------
    ValueError
        If the shapes are inconsistent, nocc is not in [1, M - 1], or the screening problem is
        unstable.
    '''
    return [0.0]

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.linalg import eigh, solve


def _check_int(n, name, minimum=0):
    if isinstance(n, bool) or int(n) != n or int(n) < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(n)

def _check_array(a, name, ndim=None, shape=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    return a

def _spin_orbital_integrals(eri_spatial):
    """(PQ|RS) = (pq|rs) delta_{sP sQ} delta_{sR sS} with the interleaved spin-orbital index P = 2p + sigma"""
    n = eri_spatial.shape[0]
    eri = np.zeros((2 * n, 2 * n, 2 * n, 2 * n))
    for s1 in range(2):
        for s2 in range(2):
            eri[s1::2, s1::2, s2::2, s2::2] = eri_spatial
    return eri

def _ov_pairs(nocc, M):
    occ = np.arange(nocc); vir = np.arange(nocc, M)
    I, A = np.meshgrid(occ, vir, indexing="ij")
    return I.ravel(), A.ravel()                             # pair index I = i * nvir + (a - nocc)

def _rpa_casida(eps, eri, nocc):
    """direct-RPA Casida problem in the spin-orbital basis: A = delta(e_a - e_i) + (ia|jb), B = (ia|jb);
    returns Omega (ascending) and X + Y with the normalisation X^T X - Y^T Y = 1"""
    M = eps.shape[0]
    io, av = _ov_pairs(nocc, M)
    Kmat = eri[io[:, None], av[:, None], io[None, :], av[None, :]]          # (ia|jb)
    de = eps[av] - eps[io]
    A = np.diag(de) + Kmat
    B = Kmat.copy()
    w, Uv = eigh(A - B)
    if w.min() <= 0:
        raise ValueError("A - B is not positive definite")
    S = (Uv * np.sqrt(w)) @ Uv.T                            # (A - B)^{1/2}
    om2, Z = eigh(S @ (A + B) @ S)
    if om2.min() <= 0:
        raise ValueError("RPA instability: non-positive excitation energy")
    Om = np.sqrt(om2)
    XpY = (S @ Z) / np.sqrt(Om)                             # X^T X - Y^T Y = 1  <=>  (X+Y)^T (A-B)^{-1} (X+Y) = 1
    return Om, XpY


def _oracle_rpa_excitation_energies(eps: "np.ndarray", eri: "np.ndarray", nocc: int) -> "np.ndarray":
    eps = _check_array(eps, "eps", ndim=1); M = eps.shape[0]
    eri = _check_array(eri, "eri", shape=(M, M, M, M))
    nocc = _check_int(nocc, "nocc", 1)
    if nocc >= M:
        raise ValueError("nocc must be smaller than the number of spin-orbitals")
    Om, XpY = _rpa_casida(eps, eri, nocc)
    return np.sort(Om)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(6, 1.0, 1.5, 1.5, np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 0.0]))\nC = _oracle_scaled_exchange_orbitals(hW[0], hW[1], 6, 0.75)\nFK = _oracle_mean_field_matrices(hW[0], hW[1], C, 6, 0.75)\neps = np.repeat(np.diag(FK[0]), 2)\neri = _spin_orbital_integrals(_oracle_mo_coulomb_integrals(C, hW[1]))\nnocc = 6",
            "call": "rpa_excitation_energies(eps.copy(), eri.copy(), nocc)",
            "gold_call": "_oracle_rpa_excitation_energies(eps.copy(), eri.copy(), nocc)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(4, 0.8, 2.0, 0.7, np.array([-0.6, 0.0, 0.0, 0.3]))\nC = _oracle_scaled_exchange_orbitals(hW[0], hW[1], 4, 1.0)\nFK = _oracle_mean_field_matrices(hW[0], hW[1], C, 4, 1.0)\neps = np.repeat(np.diag(FK[0]), 2)\neri = _spin_orbital_integrals(_oracle_mo_coulomb_integrals(C, hW[1]))\nnocc = 4",
            "call": "rpa_excitation_energies(eps.copy(), eri.copy(), nocc)",
            "gold_call": "_oracle_rpa_excitation_energies(eps.copy(), eri.copy(), nocc)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(5, 1.2, 1.0, 0.5, np.array([0.5, -0.2, 0.0, 0.1, -0.4]))\nC = _oracle_scaled_exchange_orbitals(hW[0], hW[1], 4, 0.5)\nFK = _oracle_mean_field_matrices(hW[0], hW[1], C, 4, 0.5)\neps = np.repeat(np.diag(FK[0]), 2)\neri = _spin_orbital_integrals(_oracle_mo_coulomb_integrals(C, hW[1]))\nnocc = 4",
            "call": "rpa_excitation_energies(eps.copy(), eri.copy(), nocc)",
            "gold_call": "_oracle_rpa_excitation_energies(eps.copy(), eri.copy(), nocc)",
            "tol": 1e-08,
        },
    ]
