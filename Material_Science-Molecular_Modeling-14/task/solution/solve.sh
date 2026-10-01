#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Step 1: rotated layer geometry with the source's base conventions."""

import numpy as np

_A_CONST = 1.42 * np.sqrt(3.0)


def _base_geometry():
    a = _A_CONST
    A = (a / 2.0) * np.array([[2.0, 1.0], [0.0, np.sqrt(3.0)]])
    B = (2.0 * np.pi / (3.0 * a)) * np.array([[3.0, 0.0], [-np.sqrt(3.0), 2.0 * np.sqrt(3.0)]])
    K = (4.0 * np.pi / (3.0 * a)) * np.array([1.0, 0.0])
    Kp = (2.0 * np.pi / (3.0 * a)) * np.array([1.0, np.sqrt(3.0)])
    tauB = (a / 2.0) * np.array([1.0, np.sqrt(3.0) / 3.0])
    return A, B, K, Kp, tauB


def layer_geometry(thetas):
    thetas = np.asarray(thetas, dtype=np.float64)
    if thetas.shape != (3,) or not np.all(np.isfinite(thetas)):
        raise ValueError("invalid angle triple")
    A0, B0, K0, Kp0, tau0 = _base_geometry()
    out = np.empty((3, 14))
    for j in range(3):
        c, s = np.cos(float(thetas[j])), np.sin(float(thetas[j]))
        R = np.array([[c, -s], [s, c]])
        out[j, 0:4] = (R @ A0).reshape(-1)
        out[j, 4:8] = (R @ B0).reshape(-1)
        out[j, 8:10] = R @ K0
        out[j, 10:12] = R @ Kp0
        out[j, 12:14] = R @ tau0
    return out

"""Step 2: reduction into the layer reciprocal cell."""

import numpy as np


def cell_reduce(X, B):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    B = np.asarray(B, dtype=np.float64).reshape(2, 2)
    Binv = np.linalg.inv(B)
    n = np.floor(X @ Binv.T)
    return X - n @ B.T

"""Step 3: WL-truncated reciprocal degrees of freedom."""

import numpy as np


def wl_dof(q, geom, W, L, nmax):
    q = np.asarray(q, dtype=np.float64)
    geom = np.asarray(geom, dtype=np.float64)
    if not (np.isfinite(W) and np.isfinite(L)) or W <= 0 or L <= 0:
        raise ValueError("nonpositive truncation radius")
    rows = []
    rng = range(-int(nmax), int(nmax) + 1)
    for j in range(3):
        others = [t for t in range(3) if t != j]
        k, l = others[0], others[1]
        Bj = geom[j, 4:8]
        Bk = geom[k, 4:8].reshape(2, 2)
        Bl = geom[l, 4:8].reshape(2, 2)
        Kj = geom[j, 8:10]
        Kpj = geom[j, 10:12]
        qred = cell_reduce(q[None, :], Bj)[0]
        Kt = Kj if np.linalg.norm(qred - Kj) <= np.linalg.norm(qred - Kpj) else Kpj
        for nk1 in rng:
            for nk2 in rng:
                Gk = Bk @ np.array([nk1, nk2], dtype=np.float64)
                if np.linalg.norm(Gk) >= L:
                    continue
                for nl1 in rng:
                    for nl2 in rng:
                        Gl = Bl @ np.array([nl1, nl2], dtype=np.float64)
                        if np.linalg.norm(Gl) >= L:
                            continue
                        shifted = cell_reduce((q + Gk + Gl)[None, :], Bj)[0]
                        if np.linalg.norm(shifted - Kt) < W:
                            rows.append([j, nk1, nk2, nl1, nl2])
    return np.asarray(rows, dtype=np.int64)

"""Step 4: the source's compactly supported truncation bump."""

import numpy as np


def bump_gtau(r, tau, delta):
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)
    flat = r <= tau - delta
    out[flat] = 1.0
    mid = (r > tau - delta) & (r < tau)
    rm = r[mid]
    e1 = np.exp(-delta / (tau - rm))
    e2 = np.exp(-delta / (rm - (tau - delta)))
    out[mid] = e1 / (e1 + e2)
    return out

"""Step 5: truncated intralayer Bloch block."""

import numpy as np

_T1, _T2, _T3, _T4 = -1.0, 0.15, 0.08, 0.05


def _hop_list(Aj, tauB):
    a1 = Aj[:, 0]
    a2 = Aj[:, 1]
    hops = []
    for (n1, n2) in ((0, 0), (-1, 0), (-1, 1)):
        R = n1 * a1 + n2 * a2
        hops.append((R, 0, 1, _T1))
        hops.append((-R, 1, 0, _T1))
    for (n1, n2) in ((1, 0), (0, 1), (1, -1)):
        R = n1 * a1 + n2 * a2
        for al in (0, 1):
            hops.append((R, al, al, _T2))
            hops.append((-R, al, al, _T2))
    for (n1, n2) in ((0, 1), (-2, 1), (0, -1)):
        R = n1 * a1 + n2 * a2
        hops.append((R, 0, 1, _T3))
        hops.append((-R, 1, 0, _T3))
    for (n1, n2) in ((2, -1), (-2, 0), (1, 1)):
        R = n1 * a1 + n2 * a2
        hops.append((R, 0, 1, _T4))
        hops.append((-R, 1, 0, _T4))
    return hops


def intralayer_block(qeff, Aj, tauB, tau):
    qeff = np.asarray(qeff, dtype=np.float64)
    Aj = np.asarray(Aj, dtype=np.float64).reshape(2, 2)
    tauB = np.asarray(tauB, dtype=np.float64)
    taus = [np.zeros(2), tauB]
    H = np.zeros((2, 2), dtype=np.complex128)
    for (R, al, be, t) in _hop_list(Aj, tauB):
        if np.linalg.norm(R) > tau:
            continue
        H[al, be] += t * np.exp(-1j * (qeff @ (R + taus[al] - taus[be])))
    H = (H + H.conj().T) / 2.0
    out = np.empty((2, 2, 2))
    out[:, :, 0] = H.real
    out[:, :, 1] = H.imag
    return out

"""Step 6: assembly of the truncated reciprocal Hamiltonian."""

import numpy as np

_W0_INTER = 0.045
_ETA_INTER = 0.1
_TAU_TR = 3.0
_DELTA_TR = 0.6


def _hhat_inter(xi):
    xi = np.asarray(xi, dtype=np.float64)
    return _W0_INTER * np.exp(-_ETA_INTER * float(xi @ xi))


def assemble_hamiltonian(q, dofs, geom):
    q = np.asarray(q, dtype=np.float64)
    dofs = np.asarray(dofs, dtype=np.int64)
    geom = np.asarray(geom, dtype=np.float64)
    nd = dofs.shape[0]
    dim = 2 * nd
    H = np.zeros((dim, dim), dtype=np.complex128)
    detB = abs(np.linalg.det(geom[0, 4:8].reshape(2, 2)))
    cstar2 = detB
    cache = []
    for r in range(nd):
        j = int(dofs[r, 0])
        others = [t for t in range(3) if t != j]
        k, l = others[0], others[1]
        Gk = geom[k, 4:8].reshape(2, 2) @ dofs[r, 1:3].astype(np.float64)
        Gl = geom[l, 4:8].reshape(2, 2) @ dofs[r, 3:5].astype(np.float64)
        cache.append((j, k, l, Gk, Gl))
    for r in range(nd):
        j, k, l, Gk, Gl = cache[r]
        blk = intralayer_block(q + Gk + Gl, geom[j, 0:4], geom[j, 12:14], _TAU_TR)
        H[2 * r:2 * r + 2, 2 * r:2 * r + 2] = blk[:, :, 0] + 1j * blk[:, :, 1]
    for r in range(nd):
        j, kj, lj, Gkj, Glj = cache[r]
        for r2 in range(nd):
            j2, kj2, lj2, Gkj2, Glj2 = cache[r2]
            if abs(j - j2) != 1:
                continue
            shared = [t for t in range(3) if t != j and t != j2][0]
            Gj_sh = Gkj if kj == shared else Glj
            Gj2_sh = Gkj2 if kj2 == shared else Glj2
            if np.linalg.norm(Gj_sh - Gj2_sh) > 1e-9:
                continue
            Gj2_at_j = Gkj if kj == j2 else Glj
            Gj_at_j2 = Gkj2 if kj2 == j else Glj2
            xi = q + Gj_at_j2 + Gj2_at_j + Gj_sh
            amp = _hhat_inter(xi) * float(bump_gtau(np.array([np.linalg.norm(xi)]), _TAU_TR, _DELTA_TR)[0]) * cstar2
            taus_j = [np.zeros(2), geom[j, 12:14]]
            taus_j2 = [np.zeros(2), geom[j2, 12:14]]
            for al in range(2):
                for be in range(2):
                    phase = np.exp(1j * (Gj_at_j2 @ taus_j[al] - Gj2_at_j @ taus_j2[be]))
                    H[2 * r + al, 2 * r2 + be] += amp * phase
    H = (H + H.conj().T) / 2.0
    out = np.empty((dim, dim, 2))
    out[:, :, 0] = H.real
    out[:, :, 1] = H.imag
    return out

"""Step 7: Jackson-damped Chebyshev momentum LDoS."""

import numpy as np


def _jackson_g(P):
    n = np.arange(P)
    return (2.0 - (n == 0)) * ((P - n + 1) * np.cos(np.pi * n / (P + 1)) +
                               np.sin(np.pi * n / (P + 1)) / np.tan(np.pi / (P + 1))) / (P + 1)


def kpm_ldos(Hri, dofs, Elist, P, s):
    if isinstance(P, bool) or not isinstance(P, (int, np.integer)) or P < 2:
        raise ValueError("invalid polynomial order")
    Hri = np.asarray(Hri, dtype=np.float64)
    H = Hri[:, :, 0] + 1j * Hri[:, :, 1]
    dofs = np.asarray(dofs, dtype=np.int64)
    Elist = np.asarray(Elist, dtype=np.float64)
    dim = H.shape[0]
    Hs = s * H
    g = _jackson_g(int(P))
    idx = []
    for r in range(dofs.shape[0]):
        if np.all(dofs[r, 1:5] == 0):
            idx.extend([2 * r, 2 * r + 1])
    V = np.zeros((dim, len(idx)), dtype=np.complex128)
    for c, i in enumerate(idx):
        V[i, c] = 1.0
    mu = np.zeros((int(P), len(idx)))
    Tm2 = V.copy()
    Tm1 = Hs @ V
    for n in range(int(P)):
        if n == 0:
            Tn = Tm2
        elif n == 1:
            Tn = Tm1
        else:
            Tn = 2.0 * (Hs @ Tm1) - Tm2
            Tm2 = Tm1
            Tm1 = Tn
        for c, i in enumerate(idx):
            mu[n, c] = float(Tn[i, c].real)
    out = np.empty(Elist.shape[0])
    for ei, E in enumerate(Elist):
        sE = s * float(E)
        acc = np.zeros(len(idx))
        for n in range(int(P)):
            acc += g[n] * mu[n, :] * np.cos(n * np.arccos(np.clip(sE, -1.0, 1.0)))
        vals = acc / (np.pi * np.sqrt(max(1.0 - sE * sE, 1e-15)))
        out[ei] = float(np.sum(vals)) / 6.0
    return out

"""Step 8 (final orchestrator): trilayer audit over the two configurations."""

import numpy as np

_VARIANTS = [(-0.06, 0.0, 0.11, 1.15, 5.0), (-0.09, 0.0, 0.07, 1.05, 4.6)]
_Q_OFF = np.array([0.03, 0.02])
_E_LIST = np.array([-0.5, 0.0, 0.5])
_NMAX = 3
_S_KPM = 0.25


def ttg_audit(P):
    if isinstance(P, bool) or not isinstance(P, (int, np.integer)) or P < 2:
        raise ValueError("invalid polynomial order")
    rows = []
    for variant, (t1, t2, t3, W, L) in enumerate(_VARIANTS):
        geom = layer_geometry(np.array([t1, t2, t3]))
        qb = (geom[1, 8:10] if variant == 0 else geom[1, 10:12]) + _Q_OFF
        dofs = wl_dof(qb, geom, W, L, _NMAX)
        Hri = assemble_hamiltonian(qb, dofs, geom)
        Hc = Hri[:, :, 0] + 1j * Hri[:, :, 1]
        herm = float(np.linalg.norm(Hc - Hc.conj().T))
        lam = np.linalg.eigvalsh(Hc)
        ld = kpm_ldos(Hri, dofs, _E_LIST, P, _S_KPM)
        rows.append([float(dofs.shape[0]), float(Hc.shape[0]), herm,
                     float(ld[0]), float(ld[1]), float(ld[2]), float(lam.max())])
    return np.asarray(rows, dtype=np.float64)
SCICODE_GOLD_EOF
