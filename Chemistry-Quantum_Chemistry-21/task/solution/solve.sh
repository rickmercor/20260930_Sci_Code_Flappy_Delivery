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


def ppp_rhf(n_sites: int, t: float, delta: float, U: float, kappa: float) -> "np.ndarray":
    n_sites = _check_int(n_sites, "n_sites", 2)
    t = _check_scalar(t, "t", positive=True)
    delta = _check_scalar(delta, "delta")
    U = _check_scalar(U, "U", positive=True)
    kappa = _check_scalar(kappa, "kappa", positive=True)
    if n_sites % 2 or abs(delta) >= 1:
        raise ValueError("n_sites must be even and |delta| < 1")
    h, V, _ = _ppp_site_hamiltonian(n_sites, t, delta, U, kappa)
    n_occ = n_sites // 2
    eps, C = np.linalg.eigh(h)                       # Hueckel guess
    D_old = None
    for _ in range(2000):
        Cocc = C[:, :n_occ]
        D = Cocc @ Cocc.T                            # per-spin density in the site basis
        J = np.diag(V @ np.diag(D))                  # (pq|rs) = d_pq d_rs V_pr in the site basis
        K = V * D
        F = h + 2.0 * J - K
        if D_old is not None:
            F = 0.7 * F + 0.3 * F_old                # mild damping for robustness
        eps, C = np.linalg.eigh(F)
        if D_old is not None and np.max(np.abs(D - D_old)) < 1e-13:
            break
        D_old, F_old = D, F
    else:
        raise ValueError("SCF did not converge")
    Cocc = C[:, :n_occ]
    D = Cocc @ Cocc.T
    F = h + 2.0 * np.diag(V @ np.diag(D)) - V * D
    eps, C = np.linalg.eigh(F)
    for p in range(n_sites):                         # phase: coefficient on site 0 positive
        if C[0, p] < 0:
            C[:, p] *= -1.0
    return np.vstack([eps[None, :], C])

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


def _check_array(a, name, shape=None, ndim=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must be %d-dimensional" % (name, ndim))
    return a


def mo_integrals(C: "np.ndarray", U: float, kappa: float) -> "np.ndarray":
    C = _check_array(C, "C", ndim=2)
    n = C.shape[0]
    if C.shape[1] != n:
        raise ValueError("C must be square")
    U = _check_scalar(U, "U", positive=True)
    kappa = _check_scalar(kappa, "kappa", positive=True)
    idx = np.arange(n)
    V = U / np.sqrt(1.0 + (U * np.abs(idx[:, None] - idx[None, :]) / kappa) ** 2)
    # (pq|rs) = sum_ij C_ip C_iq V_ij C_jr C_js   (site-diagonal PPP interaction)
    Q = np.einsum('ip,iq->ipq', C, C)                # charge distributions of orbital pairs
    return np.einsum('ipq,ij,jrs->pqrs', Q, V, Q, optimize=True)

import numpy as np


def _check_array(a, name, shape=None, ndim=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must be %d-dimensional" % (name, ndim))
    return a


def _rpa_solve(delta_eps, v_ph):
    """direct RPA in the recast form (A-B)^1/2 (A+B) (A-B)^1/2 Z = Z Omega^2 with A = diag(delta_eps) + v_ph,
    B = v_ph; returns Omega (ascending) and X+Y (columns = modes, same order), Eq. (2.9)-(2.10)."""
    d = np.asarray(delta_eps, dtype=np.float64)
    v = np.asarray(v_ph, dtype=np.float64)
    if np.any(d <= 0):
        raise ValueError("delta_eps must be positive")
    sq = np.sqrt(d)
    M = sq[:, None] * (np.diag(d) + 2.0 * v) * sq[None, :]
    M = 0.5 * (M + M.T)
    w2, Z = np.linalg.eigh(M)
    if np.any(w2 <= 0):
        raise ValueError("RPA instability: non-positive squared excitation energy")
    om = np.sqrt(w2)
    xpy = (sq[:, None] * Z) / np.sqrt(om)[None, :]          # (A-B)^1/2 Z Omega^-1/2
    return om, xpy


def rpa_correlation_energy(delta_eps: "np.ndarray", v_ph: "np.ndarray") -> float:
    d = _check_array(delta_eps, "delta_eps", ndim=1)
    n = d.shape[0]
    v = _check_array(v_ph, "v_ph", shape=(n, n))
    if n == 0:
        return 0.0
    if np.any(d <= 0) or not np.allclose(v, v.T, atol=1e-10):
        raise ValueError("delta_eps must be positive and v_ph symmetric")
    om, _ = _rpa_solve(d, v)
    # Klein functional, Eq. (2.22) = 1/2 Tr[Omega - A], Eq. (B.9), with A = diag(delta_eps) + v_ph
    return float(0.5 * (np.sum(om) - np.sum(d) - np.trace(v)))

import numpy as np


def _check_array(a, name, shape=None, ndim=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must be %d-dimensional" % (name, ndim))
    return a


def _rpa_solve(delta_eps, v_ph):
    """direct RPA in the recast form (A-B)^1/2 (A+B) (A-B)^1/2 Z = Z Omega^2 with A = diag(delta_eps) + v_ph,
    B = v_ph; returns Omega (ascending) and X+Y (columns = modes, same order), Eq. (2.9)-(2.10)."""
    d = np.asarray(delta_eps, dtype=np.float64)
    v = np.asarray(v_ph, dtype=np.float64)
    if np.any(d <= 0):
        raise ValueError("delta_eps must be positive")
    sq = np.sqrt(d)
    M = sq[:, None] * (np.diag(d) + 2.0 * v) * sq[None, :]
    M = 0.5 * (M + M.T)
    w2, Z = np.linalg.eigh(M)
    if np.any(w2 <= 0):
        raise ValueError("RPA instability: non-positive squared excitation energy")
    om = np.sqrt(w2)
    xpy = (sq[:, None] * Z) / np.sqrt(om)[None, :]          # (A-B)^1/2 Z Omega^-1/2
    return om, xpy


def rpa_static_kernel(delta_eps: "np.ndarray", v_ph: "np.ndarray") -> "np.ndarray":
    d = _check_array(delta_eps, "delta_eps", ndim=1)
    n = d.shape[0]
    v = _check_array(v_ph, "v_ph", shape=(n, n))
    if n == 0:
        return np.zeros((0, 0))
    if np.any(d <= 0) or not np.allclose(v, v.T, atol=1e-10):
        raise ValueError("delta_eps must be positive and v_ph symmetric")
    om, xpy = _rpa_solve(d, v)
    # sum_nu (X+Y)^nu (X+Y)^nu^T / Omega_nu  (= (A+B)^-1): the omega = 0 limit of the pole sum in Eq. (2.12)
    return (xpy / om[None, :]) @ xpy.T

import numpy as np


def _check_int(n, name, minimum=0):
    if int(n) != n or n < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(n)


def _check_active(active, n_orb):
    act = [int(p) for p in np.asarray(active).ravel()]
    if len(act) == 0 or len(set(act)) != len(act) or min(act) < 0 or max(act) >= n_orb:
        raise ValueError("active must be a non-empty list of distinct orbital indices in range")
    return sorted(act)


def _ph_pairs(n_occ, n_orb):
    """same-spin particle-hole pairs of spin-orbitals (2p = alpha, 2p+1 = beta), i outer, a inner."""
    occ = range(2 * n_occ)
    virt = range(2 * n_occ, 2 * n_orb)
    return [(i, a) for i in occ for a in virt if i % 2 == a % 2]


def constrained_ph_pairs(n_orb: int, n_occ: int, active: "list[int]") -> "np.ndarray":
    n_orb = _check_int(n_orb, "n_orb", 2)
    n_occ = _check_int(n_occ, "n_occ", 1)
    if n_occ >= n_orb:
        raise ValueError("n_occ must be smaller than n_orb")
    act = set(_check_active(active, n_orb))
    # cRPA, Eq. (2.14): every same-spin spin-orbital particle-hole pair except those with both orbitals
    # in the active space (their rows and columns of A +/- B are zero and the modes drop out of W)
    pairs = [(i, a) for (i, a) in _ph_pairs(n_occ, n_orb) if not (i // 2 in act and a // 2 in act)]
    return np.array(pairs, dtype=np.int64).reshape(len(pairs), 2)

import numpy as np


def _check_array(a, name, shape=None, ndim=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must be %d-dimensional" % (name, ndim))
    return a


def _check_active(active, n_orb):
    act = [int(p) for p in np.asarray(active).ravel()]
    if len(act) == 0 or len(set(act)) != len(act) or min(act) < 0 or max(act) >= n_orb:
        raise ValueError("active must be a non-empty list of distinct orbital indices in range")
    return sorted(act)


def screened_interaction(eri: "np.ndarray", pairs: "np.ndarray", kernel: "np.ndarray", active: "list[int]") -> "np.ndarray":
    eri = _check_array(eri, "eri", ndim=4)
    n = eri.shape[0]
    if eri.shape != (n, n, n, n):
        raise ValueError("eri must have shape (n, n, n, n)")
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or (pairs.size and (pairs.min() < 0 or pairs.max() >= 2 * n)):
        raise ValueError("pairs must be an (m, 2) array of spin-orbital indices")
    pairs = [(int(i), int(a)) for i, a in pairs]
    m = len(pairs)
    K = _check_array(kernel, "kernel", shape=(m, m))
    act = _check_active(active, n)
    # W(omega = 0), Eq. (2.12)-(2.13): W = v + sum_nu w w [1/(0 - Omega) - 1/(0 + Omega)] = v - 2 v K v,
    # with w^nu_pq = sum_ia v_pq,ia (X+Y)^nu_ia summed over the (spin-orbital) pairs of the reduced space
    vpq = np.array([[[eri[p, q, i // 2, a // 2] for (i, a) in pairs] for q in act] for p in act]).reshape(len(act), len(act), m)
    return eri[np.ix_(act, act, act, act)] - 2.0 * np.einsum('pqI,IJ,rsJ->pqrs', vpq, K, vpq, optimize=True)

import numpy as np


def _check_int(n, name, minimum=0):
    if int(n) != n or n < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(n)


def _check_array(a, name, shape=None, ndim=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must be %d-dimensional" % (name, ndim))
    return a


def _check_active(active, n_orb):
    act = [int(p) for p in np.asarray(active).ravel()]
    if len(act) == 0 or len(set(act)) != len(act) or min(act) < 0 or max(act) >= n_orb:
        raise ValueError("active must be a non-empty list of distinct orbital indices in range")
    return sorted(act)


def effective_one_body(h_mo: "np.ndarray", eri: "np.ndarray", veff: "np.ndarray", n_occ: int, active: "list[int]") -> "np.ndarray":
    h_mo = _check_array(h_mo, "h_mo", ndim=2)
    n = h_mo.shape[0]
    if h_mo.shape != (n, n):
        raise ValueError("h_mo must be square")
    eri = _check_array(eri, "eri", shape=(n, n, n, n))
    n_occ = _check_int(n_occ, "n_occ", 1)
    act = _check_active(active, n)
    nA = len(act)
    veff = _check_array(veff, "veff", shape=(nA, nA, nA, nA))
    env_occ = [i for i in range(n_occ) if i not in act]
    act_occ = [k for k, p in enumerate(act) if p < n_occ]
    # f_tu, Eq. (2.5): bare one-body term plus the mean field of the (doubly occupied) environment orbitals
    f = h_mo[np.ix_(act, act)].copy()
    for i in env_occ:
        f += 2.0 * eri[np.ix_(act, act)][:, :, i, i] - eri[np.ix_(act)][:, i, i, :][:, act]
    # double counting, Eq. (2.19), with the Hartree-Fock density (rho_vw = delta_vw on the occupied active orbitals)
    vt = veff - eri[np.ix_(act, act, act, act)]
    tdc = np.zeros((nA, nA))
    for v in act_occ:
        tdc += 2.0 * vt[:, :, v, v] - vt[:, v, v, :]
    return f - tdc

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


def _check_array(a, name, shape=None, ndim=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must be %d-dimensional" % (name, ndim))
    return a


def _check_active(active, n_orb):
    act = [int(p) for p in np.asarray(active).ravel()]
    if len(act) == 0 or len(set(act)) != len(act) or min(act) < 0 or max(act) >= n_orb:
        raise ValueError("active must be a non-empty list of distinct orbital indices in range")
    return sorted(act)


def energy_shift(h_mo: "np.ndarray", eri: "np.ndarray", veff: "np.ndarray", n_occ: int, active: "list[int]", e_rpa_full: float, e_rpa_active: float) -> float:
    h_mo = _check_array(h_mo, "h_mo", ndim=2)
    n = h_mo.shape[0]
    if h_mo.shape != (n, n):
        raise ValueError("h_mo must be square")
    eri = _check_array(eri, "eri", shape=(n, n, n, n))
    n_occ = _check_int(n_occ, "n_occ", 1)
    act = _check_active(active, n)
    nA = len(act)
    veff = _check_array(veff, "veff", shape=(nA, nA, nA, nA))
    e_rpa_full = _check_scalar(e_rpa_full, "e_rpa_full")
    e_rpa_active = _check_scalar(e_rpa_active, "e_rpa_active")
    env_occ = [i for i in range(n_occ) if i not in act]
    act_occ = [k for k, p in enumerate(act) if p < n_occ]
    vt = veff - eri[np.ix_(act, act, act, act)]
    # Eq. (2.6): core and electron-electron Hartree-Fock energy of the environment orbitals
    E_core = 2.0 * sum(h_mo[i, i] for i in env_occ)
    E_hf_env = sum(2.0 * eri[i, i, j, j] - eri[i, j, j, i] for i in env_occ for j in env_occ)
    # Eq. (2.21), third term: Hartree-Fock energy of the occupied active orbitals with veff - v (cRPA only non-zero)
    E_hf_act = sum(2.0 * vt[t, t, u, u] - vt[t, u, u, t] for t in act_occ for u in act_occ)
    # plus the RPA correlation energy of the full system (bare v) minus that of the active orbitals with veff
    return float(E_core + E_hf_env + E_hf_act + e_rpa_full - e_rpa_active)

import itertools
import numpy as np


def _check_int(n, name, minimum=0):
    if int(n) != n or n < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(n)


def _check_array(a, name, shape=None, ndim=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must be %d-dimensional" % (name, ndim))
    return a


def _sgn(bit, p):
    """fermionic sign (-1)^(number of occupied spin-orbitals below p) in the occupation string bit."""
    return -1.0 if bin(bit & ((1 << p) - 1)).count('1') % 2 else 1.0


def active_space_fci(t_eff: "np.ndarray", veff: "np.ndarray", n_elec: int) -> float:
    t_eff = _check_array(t_eff, "t_eff", ndim=2)
    nA = t_eff.shape[0]
    veff = _check_array(veff, "veff", shape=(nA, nA, nA, nA))
    n_elec = _check_int(n_elec, "n_elec", 0)
    if t_eff.shape[1] != nA or n_elec % 2 or n_elec > 2 * nA:
        raise ValueError("t_eff must be square, n_elec even and at most 2 * n_orbitals")
    if n_elec == 0:
        return 0.0
    n_so = 2 * nA
    hso = np.zeros((n_so, n_so))
    hso[0::2, 0::2] = t_eff
    hso[1::2, 1::2] = t_eff
    vso = np.zeros((n_so,) * 4)
    for s in (0, 1):
        for s2 in (0, 1):
            vso[s::2, s::2, s2::2, s2::2] = veff
    # antisymmetrised <pq||rs> = (pr|qs) - (ps|qr)
    g = np.einsum('prqs->pqrs', vso) - np.einsum('psqr->pqrs', vso)
    na = n_elec // 2
    alphas = list(itertools.combinations(range(0, n_so, 2), na))
    betas = list(itertools.combinations(range(1, n_so, 2), na))
    dets = [tuple(sorted(a + b)) for a in alphas for b in betas]
    bits = [sum(1 << p for p in d) for d in dets]
    nd = len(dets)

    H = np.zeros((nd, nd))
    for I, dI in enumerate(dets):
        bI = bits[I]
        H[I, I] = sum(hso[p, p] for p in dI) + 0.5 * sum(g[p, q, p, q] for p in dI for q in dI)
        for J in range(I + 1, nd):
            bJ = bits[J]
            diff = bI ^ bJ
            nd_ = bin(diff).count('1')
            if nd_ == 2:
                p = (bI & diff).bit_length() - 1
                q = (bJ & diff).bit_length() - 1
                val = hso[p, q] + sum(g[p, r, q, r] for r in dI if r != p)
                H[I, J] = H[J, I] = _sgn(bI, p) * _sgn(bJ, q) * val
            elif nd_ == 4:
                p1, p2 = [k for k in range(n_so) if (bI & diff) >> k & 1]
                q1, q2 = [k for k in range(n_so) if (bJ & diff) >> k & 1]
                # <I| a+_p1 a+_p2 a_q2 a_q1 |J>: act on J right to left, tracking the occupation string
                bt = bJ
                s = _sgn(bt, q1); bt ^= 1 << q1
                s *= _sgn(bt, q2); bt ^= 1 << q2
                s *= _sgn(bt, p2); bt ^= 1 << p2
                s *= _sgn(bt, p1)
                H[I, J] = H[J, I] = s * g[p1, p2, q1, q2]
    return float(np.linalg.eigvalsh(H)[0])

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


def downfolded_correlation_energy(n_sites: int, t: float, delta: float, U: float, kappa: float, active: "list[int]") -> float:
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
    out = ppp_rhf(n_sites, t, delta, U, kappa)
    eps, C = out[0], out[1:]
    h_mo = C.T @ h @ C
    eri = mo_integrals(C, U, kappa)
    E_hf = sum(h_mo[i, i] + eps[i] for i in range(n_occ))          # closed-shell RHF energy (electronic part)
    # RPA correlation energy of the full system: all same-spin spin-orbital particle-hole pairs, bare v
    d_full, v_full = _pair_matrices(eps, eri, _ph_pairs(n_occ, n_sites))
    e_rpa_full = rpa_correlation_energy(d_full, v_full)
    # constrained RPA: reduced pair space, its static kernel, and the screened interaction on the active block
    pairs_red = constrained_ph_pairs(n_sites, n_occ, act)
    d_red, v_red = _pair_matrices(eps, eri, [(int(i), int(a)) for i, a in pairs_red])
    kernel = rpa_static_kernel(d_red, v_red)
    veff = screened_interaction(eri, pairs_red, kernel, act)
    # RPA correlation energy of the active orbitals alone with the screened (ia|jb) and the HF orbital energies
    loc = {p: k for k, p in enumerate(act)}
    pairs_act = [(i, a) for (i, a) in _ph_pairs(n_occ, n_sites) if i // 2 in loc and a // 2 in loc]
    d_act = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs_act])
    v_act = np.array([[veff[loc[i // 2], loc[a // 2], loc[j // 2], loc[b // 2]] for (j, b) in pairs_act] for (i, a) in pairs_act]).reshape(len(pairs_act), len(pairs_act))
    e_rpa_active = rpa_correlation_energy(d_act, v_act)
    t_eff = effective_one_body(h_mo, eri, veff, n_occ, act)
    E_E = energy_shift(h_mo, eri, veff, n_occ, act, e_rpa_full, e_rpa_active)
    n_elec = 2 * sum(1 for p in act if p < n_occ)
    E_cas = active_space_fci(t_eff, veff, n_elec)
    # E_tot = E_const + E_E + E_cas ; the constant cancels in E_tot - E_HF
    return float(E_E + E_cas - E_hf)
SCICODE_GOLD_EOF
