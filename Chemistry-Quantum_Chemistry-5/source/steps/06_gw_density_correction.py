"""
Return the source's GW correction to the density matrix of the reference, Delta gamma^GW, obtained from the linearized Dyson equation with the one-shot GW self-energy (its Eqs. (4)-(6) evaluated analytically as its Eqs. (9)-(12)): build the screening of step 05 together with its transition vectors, form the source's coupling vectors (its Eq. (11)) and evaluate every block of the correction with the static exchange-correlation difference sxv of the reference; return the full spin-orbital matrix.

The source obtains the density matrix as the equal-time limit of the Green's function, i.e. a contour integral of the linearized Dyson equation over the upper half plane; with the GW self-energy in its pole representation the integral is done analytically and gives closed-form blocks of the correction (its Eqs. (12a)-(12c)).

Returns
-------
ndarray of float64 with shape (M, M), symmetric.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gw_density_correction(eps: "np.ndarray", eri: "np.ndarray", sxv: "np.ndarray", nocc: int) -> "np.ndarray":
    '''GW correction to the one-body density matrix from the source's linearized Dyson equation (its Eqs. (8)-(12)).

    Parameters
    ----------
    eps : np.ndarray
        Spin-orbital energies, shape (M,), occupied first (interleaved index of step 05).
    eri : np.ndarray
        Spin-orbital two-electron integrals (PQ|RS), shape (M, M, M, M).
    sxv : np.ndarray
        Matrix elements <p|Sigma_x[gamma_ref] - v_xc[gamma_ref]|q> of the reference in the spin-
        orbital basis, shape (M, M) (zero for a Hartree-Fock reference).
    nocc : int
        Number of occupied spin-orbitals.

    Returns
    -------
    dg : np.ndarray
        Symmetric correction Delta gamma^GW = gamma^GW - gamma_ref of shape (M, M) in the spin-
        orbital basis, all three blocks (occupied-occupied, virtual-virtual, occupied-virtual)
        filled.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, sxv is not symmetric, nocc is not in [1, M - 1], or the
        screening problem is unstable.
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


def _oracle_gw_density_correction(eps: "np.ndarray", eri: "np.ndarray", sxv: "np.ndarray", nocc: int) -> "np.ndarray":
    eps = _check_array(eps, "eps", ndim=1); M = eps.shape[0]
    eri = _check_array(eri, "eri", shape=(M, M, M, M))
    sxv = _check_array(sxv, "sxv", shape=(M, M))
    nocc = _check_int(nocc, "nocc", 1)
    if nocc >= M:
        raise ValueError("nocc must be smaller than the number of spin-orbitals")
    if not np.allclose(sxv, sxv.T, atol=1e-10):
        raise ValueError("sxv must be symmetric")
    io, av = _ov_pairs(nocc, M)
    Om, XpY = _rpa_casida(eps, eri, nocc)                    # screening, Eq. (10)
    # w^s_{kp} = sum_{ia} (kp|ia) (X+Y)^s_{ia}                                     Eq. (11)
    w = np.einsum("kpI,Is->skp", eri[:, :, io, av], XpY)
    occ = np.arange(nocc); vir = np.arange(nocc, M)
    e_o = eps[occ]; e_v = eps[vir]
    # occupied-occupied block, Eq. (12a): -sum_{sc} w_ic w_jc / ((e_i - e_c - Om)(e_j - e_c - Om))
    den_oc = e_o[:, None, None] - e_v[None, :, None] - Om[None, None, :]         # (i, c, s)
    t_oc = w[:, occ][:, :, vir].transpose(1, 2, 0) / den_oc                        # w_ic / den  (i, c, s)
    dg = np.zeros((M, M))
    dg[np.ix_(occ, occ)] = -np.einsum("ics,jcs->ij", t_oc, t_oc)
    # virtual-virtual block, Eq. (12b): sum_{sk} w_ak w_bk / ((e_a - e_k + Om)(e_b - e_k + Om))
    den_vk = e_v[:, None, None] - e_o[None, :, None] + Om[None, None, :]         # (a, k, s)
    t_vk = w[:, vir][:, :, occ].transpose(1, 2, 0) / den_vk                        # (a, k, s)
    dg[np.ix_(vir, vir)] = np.einsum("aks,bks->ab", t_vk, t_vk)
    # occupied-virtual block, Eq. (12c)
    w_ok = w[:, occ][:, :, occ].transpose(1, 2, 0)                                 # w_ik  (i, k, s)
    w_ic = w[:, occ][:, :, vir].transpose(1, 2, 0)                                 # w_ic  (i, c, s)
    w_bk = w[:, vir][:, :, occ].transpose(1, 2, 0)                                 # w_bk  (b, k, s)
    w_bc = w[:, vir][:, :, vir].transpose(1, 2, 0)                                 # w_bc  (b, c, s)
    term1 = np.einsum("iks,bks->ib", w_ok, w_bk / den_vk)                          # sum_{sk} w_ik w_bk / (e_b - e_k + Om)
    term2 = np.einsum("ics,bcs->ib", w_ic / den_oc, w_bc)                          # sum_{sc} w_ic w_bc / (e_i - e_c - Om)
    ov = (term1 + term2 + sxv[np.ix_(occ, vir)]) / (e_o[:, None] - e_v[None, :])
    dg[np.ix_(occ, vir)] = ov
    dg[np.ix_(vir, occ)] = ov.T
    return dg

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(6, 1.0, 1.5, 1.5, np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 0.0]))\nC = _oracle_scaled_exchange_orbitals(hW[0], hW[1], 6, 0.75)\nFK = _oracle_mean_field_matrices(hW[0], hW[1], C, 6, 0.75)\neps = np.repeat(np.diag(FK[0]), 2)\neri = _spin_orbital_integrals(_oracle_mo_coulomb_integrals(C, hW[1]))\nnocc = 6\nsxv = -(1.0 - 0.75) * _spin_orbital_matrix(FK[1])",
            "call": "gw_density_correction(eps.copy(), eri.copy(), sxv.copy(), nocc)",
            "gold_call": "_oracle_gw_density_correction(eps.copy(), eri.copy(), sxv.copy(), nocc)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(4, 0.8, 2.0, 0.7, np.array([-0.6, 0.0, 0.0, 0.3]))\nC = _oracle_scaled_exchange_orbitals(hW[0], hW[1], 4, 1.0)\nFK = _oracle_mean_field_matrices(hW[0], hW[1], C, 4, 1.0)\neps = np.repeat(np.diag(FK[0]), 2)\neri = _spin_orbital_integrals(_oracle_mo_coulomb_integrals(C, hW[1]))\nnocc = 4\nsxv = np.zeros((8, 8))",
            "call": "gw_density_correction(eps.copy(), eri.copy(), sxv.copy(), nocc)",
            "gold_call": "_oracle_gw_density_correction(eps.copy(), eri.copy(), sxv.copy(), nocc)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(5, 1.2, 1.0, 0.5, np.array([0.5, -0.2, 0.0, 0.1, -0.4]))\nC = _oracle_scaled_exchange_orbitals(hW[0], hW[1], 4, 0.5)\nFK = _oracle_mean_field_matrices(hW[0], hW[1], C, 4, 0.5)\neps = np.repeat(np.diag(FK[0]), 2)\neri = _spin_orbital_integrals(_oracle_mo_coulomb_integrals(C, hW[1]))\nnocc = 4\nsxv = -(1.0 - 0.5) * _spin_orbital_matrix(FK[1])",
            "call": "gw_density_correction(eps.copy(), eri.copy(), sxv.copy(), nocc)",
            "gold_call": "_oracle_gw_density_correction(eps.copy(), eri.copy(), sxv.copy(), nocc)",
            "tol": 1e-08,
        },
    ]
