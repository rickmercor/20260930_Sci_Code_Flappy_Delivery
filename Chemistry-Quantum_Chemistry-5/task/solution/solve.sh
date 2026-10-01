#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

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

def _check_array(a, name, ndim=None, shape=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    return a


def chain_hamiltonian(N: int, t: float, U: float, V: float, eps: "np.ndarray") -> "np.ndarray":
    N = _check_int(N, "N", 2)
    t = _check_scalar(t, "t"); U = _check_scalar(U, "U"); V = _check_scalar(V, "V")
    eps = _check_array(eps, "eps", shape=(N,))
    if t == 0.0:
        raise ValueError("the hopping amplitude t must be non-zero (the sites would not be connected)")
    h = np.diag(eps)
    idx = np.arange(N - 1)
    h[idx, idx + 1] = -t
    h[idx + 1, idx] = -t
    W = U * np.eye(N)
    W[idx, idx + 1] = V
    W[idx + 1, idx] = V
    return np.stack([h, W])

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


def _check_array(a, name, ndim=None, shape=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    return a


def _fix_signs(C):
    """make the largest-magnitude component of every column positive (removes the eigenvector sign freedom)"""
    C = np.array(C, dtype=np.float64, copy=True)
    for k in range(C.shape[1]):
        col = C[:, k]
        if col[np.argmax(np.abs(col))] < 0:
            C[:, k] = -col
    return C


def scaled_exchange_orbitals(h: "np.ndarray", W: "np.ndarray", nelec: int, alpha: float) -> "np.ndarray":
    h = _check_array(h, "h", ndim=2); N = h.shape[0]
    W = _check_array(W, "W", shape=(N, N))
    nelec = _check_int(nelec, "nelec", 2)
    if nelec % 2 or nelec > 2 * N:
        raise ValueError("nelec must be even and at most 2N")
    alpha = _check_scalar(alpha, "alpha", nonneg=True)
    if not (np.allclose(h, h.T) and np.allclose(W, W.T)):
        raise ValueError("h and W must be symmetric")
    nocc = nelec // 2
    e, C = eigh(h)
    P = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    damp = 0.3
    for it in range(20000):
        J = np.diag(W @ np.diag(P)); K = 0.5 * W * P
        e, C = eigh(h + J - alpha * K)
        Pn = 2.0 * C[:, :nocc] @ C[:, :nocc].T
        if np.max(np.abs(Pn - P)) < 1e-13:
            P = Pn
            break
        P = (1.0 - damp) * Pn + damp * P
    else:
        raise ValueError("mean field did not converge")
    J = np.diag(W @ np.diag(P)); K = 0.5 * W * P
    e, C = eigh(h + J - alpha * K)                          # canonical orbitals of the converged operator
    if nocc < N and e[nocc] - e[nocc - 1] < 1e-8:
        raise ValueError("closed-shell reference is degenerate at the Fermi level")
    return _fix_signs(C)

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

def _check_array(a, name, ndim=None, shape=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    return a

def _mean_field_pieces(h, W, C, nocc, alpha):
    P = 2.0 * C[:, :nocc] @ C[:, :nocc].T                 # spatial density matrix (both spins)
    J = np.diag(W @ np.diag(P))                            # Hartree: J_ii = sum_j W_ij n_j
    K = 0.5 * W * P                                        # exchange: K_ij = (1/2) W_ij P_ij
    F = h + J - alpha * K
    return P, J, K, F


def mean_field_matrices(h: "np.ndarray", W: "np.ndarray", C: "np.ndarray", nelec: int, alpha: float) -> "np.ndarray":
    h = _check_array(h, "h", ndim=2); N = h.shape[0]
    W = _check_array(W, "W", shape=(N, N)); C = _check_array(C, "C", shape=(N, N))
    nelec = _check_int(nelec, "nelec", 2)
    if nelec % 2 or nelec > 2 * N:
        raise ValueError("nelec must be even and at most 2N")
    alpha = _check_scalar(alpha, "alpha", nonneg=True)
    if not np.allclose(C.T @ C, np.eye(N), atol=1e-8):
        raise ValueError("C must be orthonormal")
    P, J, K, F = _mean_field_pieces(h, W, C, nelec // 2, alpha)
    return np.stack([C.T @ F @ C, C.T @ K @ C])

import numpy as np
from numpy.linalg import eigh, solve


def _check_array(a, name, ndim=None, shape=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    return a


def mo_coulomb_integrals(C: "np.ndarray", W: "np.ndarray") -> "np.ndarray":
    C = _check_array(C, "C", ndim=2); N = C.shape[0]
    if C.shape[1] != N:
        raise ValueError("C must be a square matrix of orbital coefficients")
    W = _check_array(W, "W", shape=(N, N))
    D = np.einsum("ip,iq->ipq", C, C)                      # site charge density of the orbital pair (p, q)
    return np.einsum("ipq,ij,jrs->pqrs", D, W, D)           # (pq|rs) = sum_ij C_ip C_iq W_ij C_jr C_js

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


def rpa_excitation_energies(eps: "np.ndarray", eri: "np.ndarray", nocc: int) -> "np.ndarray":
    eps = _check_array(eps, "eps", ndim=1); M = eps.shape[0]
    eri = _check_array(eri, "eri", shape=(M, M, M, M))
    nocc = _check_int(nocc, "nocc", 1)
    if nocc >= M:
        raise ValueError("nocc must be smaller than the number of spin-orbitals")
    Om, XpY = _rpa_casida(eps, eri, nocc)
    return np.sort(Om)

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


def gw_density_correction(eps: "np.ndarray", eri: "np.ndarray", sxv: "np.ndarray", nocc: int) -> "np.ndarray":
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


def iterated_dyson_correction(eps: "np.ndarray", eri: "np.ndarray", dg_gw: "np.ndarray", nocc: int) -> "np.ndarray":
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


def idgw_dipole(N: int, t: float, U: float, V: float, eps: "np.ndarray", nelec: int, alpha: float) -> float:
    N = _check_int(N, "N", 2)
    nelec = _check_int(nelec, "nelec", 2)
    alpha = _check_scalar(alpha, "alpha", nonneg=True)
    hW = chain_hamiltonian(N, t, U, V, eps)
    h, W = hW[0], hW[1]
    C = scaled_exchange_orbitals(h, W, nelec, alpha)
    FK = mean_field_matrices(h, W, C, nelec, alpha)
    e_mo = np.diag(FK[0])
    eri_sp = mo_coulomb_integrals(C, W)
    # spin-orbital quantities (interleaved index P = 2p + sigma)
    eps_so = np.repeat(e_mo, 2)
    eri = _spin_orbital_integrals(eri_sp)
    sxv = -(1.0 - alpha) * _spin_orbital_matrix(FK[1])       # <p|Sigma_x[gamma_gKS] - v_xc[gamma_gKS]|q> = -(1 - alpha) K_pq
    nocc = nelec                                              # number of occupied spin-orbitals
    Om = rpa_excitation_energies(eps_so, eri, nocc)
    if Om[0] <= 0:
        raise ValueError("unstable screening")
    dg_gw = gw_density_correction(eps_so, eri, sxv, nocc)
    dg = iterated_dyson_correction(eps_so, eri, dg_gw, nocc)
    gamma = np.diag(np.r_[np.ones(nocc), np.zeros(2 * N - nocc)]) + dg
    x_mo = _spin_orbital_matrix(C.T @ np.diag(_site_positions(N)) @ C)
    D = -float(np.sum(gamma * x_mo))                          # D = -sum_i x_i n_i  (electron charge -1)
    if not np.isfinite(D):
        raise ValueError("non-finite dipole")
    return D
SCICODE_GOLD_EOF
