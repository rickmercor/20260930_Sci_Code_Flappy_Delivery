"""
Return the source's iterated-Dyson GW correction Delta gamma^idGW (its Sec. III): the Dyson equation is iterated with the static part of the self-energy updated to the final density matrix, which leaves the occupied-occupied and virtual-virtual blocks of step 06 unchanged (its Eqs. (18a)-(18b)) and turns the occupied-virtual block into the solution of its Eq. (21); solve that equation exactly (the source's Sec. III B) rather than by iteration.

Iterating the Dyson equation with an updated static self-energy, an idea the source takes from the algebraic diagrammatic construction, makes the Hartree and exchange potentials depend on the final density matrix; because the static part is frequency independent only the occupied-virtual block responds, which the source relates to orbital relaxation.

Returns
-------
ndarray of float64 with shape (M, M), symmetric.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def iterated_dyson_correction(eps: "np.ndarray", eri: "np.ndarray", dg_gw: "np.ndarray", nocc: int) -> "np.ndarray":
    '''Iterated-Dyson GW density-matrix correction of the source (its Eq. (21), solved as in its Sec. III B).

    Parameters
    ----------
    eps : np.ndarray
        Spin-orbital energies, shape (M,), occupied first.
    eri : np.ndarray
        Spin-orbital two-electron integrals (PQ|RS), shape (M, M, M, M).
    dg_gw : np.ndarray
        GW correction Delta gamma^GW of step 06, shape (M, M), symmetric.
    nocc : int
        Number of occupied spin-orbitals.

    Returns
    -------
    dg : np.ndarray
        Symmetric correction Delta gamma^idGW of shape (M, M): the occupied-occupied and virtual-
        virtual blocks of dg_gw unchanged, the occupied-virtual block replaced by the source's
        iterated-Dyson solution.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, dg_gw is not symmetric, or nocc is not in [1, M - 1].
    '''
    return [[0.0]]

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

def _spin_orbital_matrix(m_spatial):
    """same-spin block expansion of a one-body MO matrix, interleaved index P = 2p + sigma"""
    n = m_spatial.shape[0]
    m = np.zeros((2 * n, 2 * n))
    m[0::2, 0::2] = m_spatial
    m[1::2, 1::2] = m_spatial
    return m

def _ov_pairs(nocc, M):
    occ = np.arange(nocc); vir = np.arange(nocc, M)
    I, A = np.meshgrid(occ, vir, indexing="ij")
    return I.ravel(), A.ravel()                             # pair index I = i * nvir + (a - nocc)


def _oracle_iterated_dyson_correction(eps: "np.ndarray", eri: "np.ndarray", dg_gw: "np.ndarray", nocc: int) -> "np.ndarray":
    eps = _check_array(eps, "eps", ndim=1); M = eps.shape[0]
    eri = _check_array(eri, "eri", shape=(M, M, M, M))
    dg_gw = _check_array(dg_gw, "dg_gw", shape=(M, M))
    nocc = _check_int(nocc, "nocc", 1)
    if nocc >= M:
        raise ValueError("nocc must be smaller than the number of spin-orbitals")
    if not np.allclose(dg_gw, dg_gw.T, atol=1e-10):
        raise ValueError("dg_gw must be symmetric")
    occ = np.arange(nocc); vir = np.arange(nocc, M)
    io, av = _ov_pairs(nocc, M)
    de = eps[av] - eps[io]
    # <pm||qn> = (pq|mn) - (pn|mq)
    anti = lambda p, m, q, n: eri[p, q, m, n] - eri[p, n, m, q]
    I, J = np.meshgrid(np.arange(len(io)), np.arange(len(io)), indexing="ij")
    i, a, j, b = io[I], av[I], io[J], av[J]
    Amat = np.diag(de) + anti(i, j, a, b) + anti(i, b, a, j)                      # Eq. (23a): HF electronic Hessian
    # Eq. (23c): sources from the occupied-occupied and virtual-virtual GW blocks
    oo = dg_gw[np.ix_(occ, occ)]; vv = dg_gw[np.ix_(vir, vir)]
    Y = np.empty(len(io))
    for n_, (ii, aa) in enumerate(zip(io, av)):
        Y[n_] = -np.sum(anti(ii, occ[:, None], aa, occ[None, :]) * oo) \
                - np.sum(anti(ii, vir[:, None], aa, vir[None, :]) * vv) \
                + de[n_] * dg_gw[ii, aa]
    X = solve(Amat, Y)
    dg = dg_gw.copy()
    dg[io, av] = X
    dg[av, io] = X
    return dg

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(6, 1.0, 1.5, 1.5, np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 0.0]))\nC = _oracle_scaled_exchange_orbitals(hW[0], hW[1], 6, 0.75)\nFK = _oracle_mean_field_matrices(hW[0], hW[1], C, 6, 0.75)\neps = np.repeat(np.diag(FK[0]), 2)\neri = _spin_orbital_integrals(_oracle_mo_coulomb_integrals(C, hW[1]))\nnocc = 6\ndg_gw = _oracle_gw_density_correction(eps, eri, -(1.0 - 0.75) * _spin_orbital_matrix(FK[1]), nocc)",
            "call": "iterated_dyson_correction(eps.copy(), eri.copy(), dg_gw.copy(), nocc)",
            "gold_call": "_oracle_iterated_dyson_correction(eps.copy(), eri.copy(), dg_gw.copy(), nocc)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(4, 0.8, 2.0, 0.7, np.array([-0.6, 0.0, 0.0, 0.3]))\nC = _oracle_scaled_exchange_orbitals(hW[0], hW[1], 4, 1.0)\nFK = _oracle_mean_field_matrices(hW[0], hW[1], C, 4, 1.0)\neps = np.repeat(np.diag(FK[0]), 2)\neri = _spin_orbital_integrals(_oracle_mo_coulomb_integrals(C, hW[1]))\nnocc = 4\ndg_gw = _oracle_gw_density_correction(eps, eri, np.zeros((8, 8)), nocc)",
            "call": "iterated_dyson_correction(eps.copy(), eri.copy(), dg_gw.copy(), nocc)",
            "gold_call": "_oracle_iterated_dyson_correction(eps.copy(), eri.copy(), dg_gw.copy(), nocc)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(5, 1.2, 1.0, 0.5, np.array([0.5, -0.2, 0.0, 0.1, -0.4]))\nC = _oracle_scaled_exchange_orbitals(hW[0], hW[1], 4, 0.5)\nFK = _oracle_mean_field_matrices(hW[0], hW[1], C, 4, 0.5)\neps = np.repeat(np.diag(FK[0]), 2)\neri = _spin_orbital_integrals(_oracle_mo_coulomb_integrals(C, hW[1]))\nnocc = 4\ndg_gw = _oracle_gw_density_correction(eps, eri, -(1.0 - 0.5) * _spin_orbital_matrix(FK[1]), nocc)",
            "call": "iterated_dyson_correction(eps.copy(), eri.copy(), dg_gw.copy(), nocc)",
            "gold_call": "_oracle_iterated_dyson_correction(eps.copy(), eri.copy(), dg_gw.copy(), nocc)",
            "tol": 1e-08,
        },
    ]
