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

def _check_params(params):
    p = np.asarray(params, float).ravel()
    if p.size != 20:
        raise ValueError("params must be a length-20 vector in the stated order")
    if not np.all(np.isfinite(p)):
        raise ValueError("params must be finite")
    if p[0] <= 0.0:
        raise ValueError("r_e must be positive")
    if p[2] <= 0.0 or p[3] <= 0.0:
        raise ValueError("rho_e and rho_s must be positive")
    return p

def eam_pair_and_density(r, params, r_cut=6.0):
    p = _check_params(params)
    r = np.asarray(r, float)
    if np.any(r < 0.0):
        raise ValueError("interatomic distances must be non-negative")
    if not np.isfinite(r_cut) or r_cut <= 0.0:
        raise ValueError("r_cut must be a finite positive scalar")
    re, fe, alpha, beta = p[0], p[1], p[4], p[5]
    A, B, kap, lam = p[6], p[7], p[8], p[9]
    x = r / re
    er, ea = np.exp(-alpha * (x - 1.0)), np.exp(-beta * (x - 1.0))
    dr, da = 1.0 + (x - kap) ** 20, 1.0 + (x - lam) ** 20
    rep, att, fd = A * er / dr, B * ea / da, fe * ea / da

    def dterm(val, k, s):
        return (-k * val - val * 20.0 * (x - s) ** 19 / (1.0 + (x - s) ** 20)) / re

    phi = rep - att
    dphi = dterm(rep, alpha, kap) - dterm(att, beta, lam)
    dfd = dterm(fd, beta, lam)
    m = r < r_cut
    z = np.zeros_like(x)
    return np.stack([np.where(m, phi, z), np.where(m, fd, z),
                     np.where(m, dphi, z), np.where(m, dfd, z)])

import numpy as np

def eam_embedding(rho, params):
    p = _check_params(params)
    rho = np.asarray(rho, float)
    if np.any(rho < 0.0):
        raise ValueError("the electron density must be non-negative")
    rhoe, rhos, eta, Fe = p[2], p[3], p[18], p[19]
    Fn, Fi = p[10:14], p[14:18]
    rn, r0 = 0.85 * rhoe, 1.15 * rhoe
    val = np.zeros(rho.shape, float)
    der = np.zeros(rho.shape, float)
    m1, m2, m3 = rho < rn, (rho >= rn) & (rho < r0), rho >= r0
    if m1.any():
        t = rho[m1] / rn - 1.0
        val[m1] = sum(Fn[i] * t ** i for i in range(4))
        der[m1] = sum(i * Fn[i] * t ** (i - 1) for i in range(1, 4)) / rn
    if m2.any():
        t = rho[m2] / rhoe - 1.0
        val[m2] = sum(Fi[i] * t ** i for i in range(4))
        der[m2] = sum(i * Fi[i] * t ** (i - 1) for i in range(1, 4)) / rhoe
    if m3.any():
        t = rho[m3] / rhos
        val[m3] = Fe * (1.0 - eta * np.log(t)) * t ** eta
        der[m3] = -Fe * eta ** 2 * np.log(t) * t ** (eta - 1.0) / rhos
    return np.stack([val, der])

import numpy as np

def eam_energy_forces(positions, cell, params, r_cut=6.0):
    p = _check_params(params)
    pos = np.asarray(positions, float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if pos.shape[0] < 1:
        raise ValueError("at least one atom is required")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must be finite")
    L = np.atleast_1d(np.asarray(cell, float))
    if L.size not in (1, 3):
        raise ValueError("cell must be a scalar or a length-3 vector of box lengths")
    L = np.broadcast_to(L, (3,)) if L.size == 3 else np.full(3, float(L[0]))
    if np.any(L <= 0.0) or not np.all(np.isfinite(L)):
        raise ValueError("all box lengths must be finite and positive")
    if not np.isfinite(r_cut) or r_cut <= 0.0:
        raise ValueError("r_cut must be a finite positive scalar")
    nmax = [int(np.ceil(r_cut / L[a])) for a in range(3)]
    g = np.meshgrid(*[np.arange(-n, n + 1) for n in nmax], indexing="ij")
    S = np.stack([g[a].ravel() * L[a] for a in range(3)], axis=1)
    d = pos[:, None, :] - pos[None, :, :]
    d = d - L * np.round(d / L)
    d = d[:, :, None, :] + S[None, None, :, :]
    r = np.sqrt((d ** 2).sum(-1))
    good = (r > 1e-10) & (r < r_cut)
    rs = np.where(good, r, 1.0)
    phi, fd, dphi, dfd = eam_pair_and_density(rs, p, r_cut)
    phi = np.where(good, phi, 0.0)
    fd = np.where(good, fd, 0.0)
    dphi = np.where(good, dphi, 0.0)
    dfd = np.where(good, dfd, 0.0)
    rho = fd.sum(axis=(1, 2))
    Fval, Fder = eam_embedding(rho, p)
    E = 0.5 * phi.sum() + Fval.sum()
    coef = 0.5 * dphi + 0.5 * (Fder[:, None, None] + Fder[None, :, None]) * dfd
    frc = -2.0 * (coef / rs)[..., None] * d
    frc = np.where(good[..., None], frc, 0.0).sum(axis=(1, 2))
    return np.concatenate([[E], frc.ravel()])

import numpy as np

EV_A3_TO_GPA = 160.21766208

def _bcc_shells(a, r_cut):
    nmax = int(np.ceil(r_cut / a)) + 1
    n = np.arange(-nmax, nmax + 1)
    I, J, K = np.meshgrid(n, n, n, indexing="ij")
    b = np.stack([I, J, K], -1).reshape(-1, 3).astype(float)
    return np.concatenate([b, b + 0.5])

def _bcc_energy(a, params, r_cut, eps=None):
    d = _bcc_shells(a, r_cut) * a
    if eps is not None:
        d = d @ (np.eye(3) + np.asarray(eps, float)).T
    r = np.sqrt((d ** 2).sum(-1))
    r = r[(r > 1e-10) & (r < r_cut)]
    phi, fd = eam_pair_and_density(r, params, r_cut)[:2]
    return 0.5 * phi.sum() + eam_embedding(np.array([fd.sum()]), params)[0, 0]

def _bcc_dEda(a, params, r_cut):
    c = np.linalg.norm(_bcc_shells(a, r_cut), axis=1)
    c = c[(c > 1e-10) & (c * a < r_cut)]
    r = c * a
    phi, fd, dphi, dfd = eam_pair_and_density(r, params, r_cut)
    Fder = eam_embedding(np.array([fd.sum()]), params)[1, 0]
    return 0.5 * (dphi * c).sum() + Fder * (dfd * c).sum()

def bcc_indicator_properties(params, r_cut=6.0, a_start=3.30, h_a=1e-4,
                                     n_newton=8, h_strain=1e-2):
    p = _check_params(params)
    if not np.isfinite(a_start) or a_start <= 0.0:
        raise ValueError("a_start must be a finite positive scalar")
    if not (isinstance(n_newton, (int, np.integer)) or float(n_newton).is_integer()):
        raise ValueError("n_newton must be an integer")
    if int(n_newton) < 1:
        raise ValueError("n_newton must be a positive integer")
    if h_a <= 0.0 or h_strain <= 0.0:
        raise ValueError("h_a and h_strain must be positive")
    a = float(a_start)
    for _ in range(int(n_newton)):
        d1 = _bcc_dEda(a, p, r_cut)
        d2 = (_bcc_dEda(a + h_a, p, r_cut) - _bcc_dEda(a - h_a, p, r_cut)) / (2.0 * h_a)
        if not np.isfinite(d2) or d2 == 0.0:
            raise ValueError("the BCC energy has no isolated minimum for these parameters")
        a = a - d1 / d2
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError("the lattice-constant iteration left the physical range")
    e0 = _bcc_energy(a, p, r_cut)
    V0 = a ** 3 / 2.0

    def second(mk):
        return (_bcc_energy(a, p, r_cut, mk(h_strain)) - 2.0 * e0
                + _bcc_energy(a, p, r_cut, mk(-h_strain))) / h_strain ** 2

    c11 = second(lambda e: np.diag([e, 0.0, 0.0])) / V0 * EV_A3_TO_GPA
    c11p12 = second(lambda e: np.diag([e, e, 0.0])) / (2.0 * V0) * EV_A3_TO_GPA

    def m44(e):
        m = np.zeros((3, 3))
        m[1, 2] = m[2, 1] = 0.5 * e
        return m

    c44 = second(m44) / V0 * EV_A3_TO_GPA
    return np.array([a, -e0, c11, c11p12 - c11, c44])

import numpy as np

MASTER_IDX = (0, 5, 6, 7, 8, 18, 19)

def log_parameter_jacobian(func, params, master_idx=MASTER_IDX, h=5e-3):
    p = np.asarray(params, float).ravel().copy()
    idx = [int(j) for j in np.atleast_1d(np.asarray(master_idx)).ravel()]
    if len(idx) < 1:
        raise ValueError("master_idx must contain at least one index")
    if any(j < 0 or j >= p.size for j in idx):
        raise ValueError("every master index must address an entry of params")
    if len(set(idx)) != len(idx):
        raise ValueError("master_idx must not repeat an index")
    if np.any(p[idx] == 0.0):
        raise ValueError("a master parameter of zero has no logarithm")
    if np.ndim(h) != 0 or not np.isfinite(float(h)) or float(h) <= 0.0:
        raise ValueError("h must be a finite positive scalar")
    h = float(h)
    cols = []
    for j in idx:
        pp = p.copy(); pp[j] = p[j] * np.exp(h)
        pm = p.copy(); pm[j] = p[j] * np.exp(-h)
        cols.append((np.atleast_1d(np.asarray(func(pp), float)).ravel()
                     - np.atleast_1d(np.asarray(func(pm), float)).ravel()) / (2.0 * h))
    return np.stack(cols, axis=1)

import numpy as np

MASTER_IDX = (0, 5, 6, 7, 8, 18, 19)

def data_fisher_matrices(positions_list, cells, params,
                                 master_idx=MASTER_IDX, h=5e-3, r_cut=6.0):
    p = _check_params(params)
    plist = list(positions_list)
    if len(plist) < 1:
        raise ValueError("at least one configuration is required")
    cl = list(cells) if not np.isscalar(cells) else [cells] * len(plist)
    if len(cl) != len(plist):
        raise ValueError("positions_list and cells must have the same length")
    n = len(list(np.atleast_1d(np.asarray(master_idx)).ravel()))
    IE = np.zeros((n, n))
    IF = np.zeros((n, n))
    for pos, cell in zip(plist, cl):
        pos = np.asarray(pos, float)
        Jm = log_parameter_jacobian(
            lambda q, _p=pos, _c=cell: eam_energy_forces(_p, _c, q, r_cut),
            p, master_idx, h)
        IE += np.outer(Jm[0], Jm[0])
        for i in range(pos.shape[0]):
            G = Jm[1 + 3 * i:4 + 3 * i]
            IF += G.T @ G
    return np.stack([IE, IF])

import numpy as np

def minimal_information_scale(I_pool, qoi_jacobian, delta):
    A = np.asarray(I_pool, float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("I_pool must be a square matrix")
    if not np.allclose(A, A.T, rtol=0.0, atol=1e-8 * max(1.0, np.abs(A).max())):
        raise ValueError("I_pool must be symmetric")
    H = np.atleast_2d(np.asarray(qoi_jacobian, float))
    if H.shape[1] != A.shape[0]:
        raise ValueError("qoi_jacobian and I_pool disagree in the parameter dimension")
    d = np.atleast_1d(np.asarray(delta, float)).ravel()
    if d.size != H.shape[0]:
        raise ValueError("delta must supply one target uncertainty per QoI")
    if np.any(d <= 0.0) or not np.all(np.isfinite(d)):
        raise ValueError("every target uncertainty must be finite and positive")
    J = (H / d[:, None]).T @ (H / d[:, None])
    ev = np.linalg.eigvalsh(A)
    if ev.min() <= 1e-12 * max(ev.max(), 1.0):
        raise ValueError("I_pool is singular: the candidate data cannot match any "
                         "target with a finite common weight")
    Li = np.linalg.inv(np.linalg.cholesky(A))
    return float(max(np.linalg.eigvalsh(Li @ J @ Li.T).max(), 0.0))

import numpy as np

MASTER_IDX = (0, 5, 6, 7, 8, 18, 19)

TA_PARAMS = np.array([
    2.860082, 3.086341, 33.787168, 33.787168, 8.489528, 4.527748,
    0.611679, 1.032101, 0.176977, 0.353954,
    -5.103845, -0.405524, 1.112997, -3.585325,
    -5.14, 0.0, 1.640098, 0.221375, 0.848843, -5.141526])

DELTA_QOI = (0.0075, 1.0869, 9.6143, 5.9992, 6.4602)

def _alim_configuration(m, n_cell=2):
    a = 3.05 + 0.06 * (m % 8)
    L = a * n_cell
    amp = 0.08 + 0.04 * (m % 4)
    c = np.arange(n_cell, dtype=float)
    I, J, K = np.meshgrid(c, c, c, indexing="ij")
    b = np.stack([I, J, K], -1).reshape(-1, 3)
    R = np.concatenate([b, b + 0.5]) * a
    i = np.arange(R.shape[0], dtype=float)
    u = amp * np.stack([np.sin(1.7 * i + 0.9 * m + 0.3),
                        np.sin(2.3 * i + 1.4 * m + 1.1),
                        np.sin(3.1 * i + 2.2 * m + 1.9)], axis=1)
    return (R + u) % L, np.full(3, L)

def alim_uncertainty_ratio_sum(params=None, n_conf=12, n_cell=2, data="EF",
                                       w_ratio=1.0e-2, r_cut=6.0, h=5.0e-3,
                                       delta=None, master_idx=MASTER_IDX,
                                       a_start=3.30, h_a=1.0e-4, n_newton=8,
                                       h_strain=1.0e-2):
    p = _check_params(TA_PARAMS if params is None else params)
    if not (isinstance(n_conf, (int, np.integer)) or float(n_conf).is_integer()):
        raise ValueError("n_conf must be an integer")
    if int(n_conf) < 1:
        raise ValueError("n_conf must be a positive integer")
    if int(n_cell) < 1:
        raise ValueError("n_cell must be a positive integer")
    if str(data) not in ("E", "F", "EF"):
        raise ValueError("data must be one of 'E', 'F', 'EF'")
    if np.ndim(w_ratio) != 0 or not np.isfinite(float(w_ratio)) or float(w_ratio) < 0.0:
        raise ValueError("w_ratio must be a finite non-negative scalar")
    delta = DELTA_QOI if delta is None else delta
    confs = [_alim_configuration(m, int(n_cell)) for m in range(int(n_conf))]
    IE, IF = data_fisher_matrices([c[0] for c in confs], [c[1] for c in confs],
                                          p, master_idx, h, r_cut)
    H = log_parameter_jacobian(
        lambda q: bcc_indicator_properties(q, r_cut, a_start, h_a,
                                                   n_newton, h_strain),
        p, master_idx, h)
    if str(data) == "E":
        Ipool = IE
    elif str(data) == "F":
        Ipool = IF
    else:
        Ipool = float(w_ratio) * IE + IF
    t = minimal_information_scale(Ipool, H, delta)
    if t <= 0.0:
        raise ValueError("the target information is empty; the matching scale vanishes")
    Iinv = np.linalg.inv(t * Ipool)
    d = np.atleast_1d(np.asarray(delta, float)).ravel()
    sig = np.sqrt(np.einsum("ni,ij,nj->n", H, Iinv, H))
    return float(np.sum(sig / d))
SCICODE_GOLD_EOF
