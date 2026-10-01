#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Step 1: total potential energy of a flexible A-B-A cluster."""

import numpy as np

_PAR = {
    "kb": 800.0, "r0": 1.0, "kth": 120.0, "th0": 1.9106332362490186,
    "epsB": 1.2, "sigB": 2.6, "epsA": 0.05, "sigA": 1.4,
    "mA": 1.0, "mB": 16.0,
}
_PAIRS = ((1, 1, "epsB", "sigB"), (0, 0, "epsA", "sigA"), (0, 2, "epsA", "sigA"),
          (2, 0, "epsA", "sigA"), (2, 2, "epsA", "sigA"))




def _validate_coords(coords):
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 3 or c.shape[1:] != (3, 3) or c.shape[0] < 1 or not np.all(np.isfinite(c)):
        raise ValueError("invalid coordinates")
    return c


def cluster_energy(coords):
    r = np.asarray(coords, dtype=np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or r.shape[0] < 1 or not np.all(np.isfinite(r)):
        raise ValueError("invalid coordinates")
    I = r.shape[0]
    kb, r0 = _PAR["kb"], _PAR["r0"]
    kth, th0 = _PAR["kth"], _PAR["th0"]
    V = 0.0
    for i in range(I):
        A1, B, A2 = r[i]
        for a in (A1, A2):
            V += 0.5 * kb * (np.linalg.norm(a - B) - r0) ** 2
        u, w = A1 - B, A2 - B
        c = float(np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)))
        c = max(-1.0, min(1.0, c))
        V += 0.5 * kth * (np.arccos(c) - th0) ** 2
    for i in range(I):
        for j in range(i + 1, I):
            for (ai, aj, ek, sk) in _PAIRS:
                eps, sig = _PAR[ek], _PAR[sk]
                d = r[i, ai] - r[j, aj]
                s6 = (sig * sig / float(np.dot(d, d))) ** 3
                V += 4.0 * eps * (s6 * s6 - s6)
    return float(V)

"""Step 2: analytic gradient of the cluster potential."""

import numpy as np

_PAR = {
    "kb": 800.0, "r0": 1.0, "kth": 120.0, "th0": 1.9106332362490186,
    "epsB": 1.2, "sigB": 2.6, "epsA": 0.05, "sigA": 1.4,
    "mA": 1.0, "mB": 16.0,
}
_PAIRS = ((1, 1, "epsB", "sigB"), (0, 0, "epsA", "sigA"), (0, 2, "epsA", "sigA"),
          (2, 0, "epsA", "sigA"), (2, 2, "epsA", "sigA"))




def cluster_gradient(coords):
    r = np.asarray(coords, dtype=np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or r.shape[0] < 1 or not np.all(np.isfinite(r)):
        raise ValueError("invalid coordinates")
    I = r.shape[0]
    kb, r0 = _PAR["kb"], _PAR["r0"]
    kth, th0 = _PAR["kth"], _PAR["th0"]
    g = np.zeros_like(r)
    for i in range(I):
        A1, B, A2 = r[i]
        for (ai, a) in ((0, A1), (2, A2)):
            d = a - B
            L = np.linalg.norm(d)
            gd = kb * (L - r0) * d / L
            g[i, ai] += gd
            g[i, 1] -= gd
        u, w = A1 - B, A2 - B
        lu, lw = np.linalg.norm(u), np.linalg.norm(w)
        c = float(np.dot(u, w) / (lu * lw))
        c = max(-1.0, min(1.0, c))
        th = np.arccos(c)
        s = np.sqrt(max(1e-15, 1.0 - c * c))
        pref = kth * (th - th0) * (-1.0 / s)
        dc_du = w / (lu * lw) - c * u / (lu * lu)
        dc_dw = u / (lu * lw) - c * w / (lw * lw)
        g[i, 0] += pref * dc_du
        g[i, 2] += pref * dc_dw
        g[i, 1] -= pref * (dc_du + dc_dw)
    for i in range(I):
        for j in range(i + 1, I):
            for (ai, aj, ek, sk) in _PAIRS:
                eps, sig = _PAR[ek], _PAR[sk]
                d = r[i, ai] - r[j, aj]
                L2 = float(np.dot(d, d))
                s6 = (sig * sig / L2) ** 3
                f = 24.0 * eps * (2.0 * s6 * s6 - s6) / L2
                g[i, ai] -= f * d
                g[j, aj] += f * d
    return g.astype(np.float64, copy=False)

"""Step 3: per-monomer mass-scaled Hessian blocks, exact analytic form."""

import numpy as np

_KB, _R0 = 800.0, 1.0
_KTH, _TH0 = 120.0, 1.9106332362490186
_EPS_B, _SIG_B = 1.2, 2.6
_EPS_A, _SIG_A = 0.05, 1.4
_MA, _MB = 1.0, 16.0


I3 = np.eye(3)


def _radial_hessian(d, fp, fpp):
    """Hessian of f(|d|) wrt the first endpoint: fpp*dh dh^T + (fp/r)(I - dh dh^T)."""
    r = np.linalg.norm(d)
    dh = d / r
    outer = np.outer(dh, dh)
    return fpp * outer + (fp / r) * (I3 - outer)


def _bond_hessian(A, B, kb, r0):
    """6x6 Hessian of kb/2 (|A-B| - r0)^2 over (A, B)."""
    d = A - B
    r = np.linalg.norm(d)
    Haa = _radial_hessian(d, kb * (r - r0), kb)
    H = np.zeros((6, 6))
    H[0:3, 0:3] = Haa
    H[3:6, 3:6] = Haa
    H[0:3, 3:6] = -Haa
    H[3:6, 0:3] = -Haa
    return H


def _lj_hessian_self(a, b, eps, sig):
    """3x3 Hessian of the LJ pair energy wrt the first atom only."""
    d = a - b
    r = np.linalg.norm(d)
    s6 = (sig / r) ** 6
    fp = 4.0 * eps * (-12.0 * s6 * s6 + 6.0 * s6) / r
    fpp = 4.0 * eps * (156.0 * s6 * s6 - 42.0 * s6) / (r * r)
    return _radial_hessian(d, fp, fpp)


def _angle_hessian(A1, B, A2, kth, th0):
    """9x9 Hessian of kth/2 (theta - th0)^2 over (A1, B, A2).

    Exact vector calculus of theta = acos(uh . wh): cosine gradients wrt the
    two arm tips, their derivative blocks, then the B rows and columns from
    translational invariance (the energy depends on A1-B and A2-B only)."""
    u = A1 - B
    w = A2 - B
    nu = np.linalg.norm(u)
    nw = np.linalg.norm(w)
    uh = u / nu
    wh = w / nw
    c = float(np.dot(uh, wh))
    s = np.sqrt(1.0 - c * c)
    th = np.arccos(c)

    dcd1 = (wh - c * uh) / nu
    dcd2 = (uh - c * wh) / nw
    g1 = -dcd1 / s
    g2 = -dcd2 / s

    Pu = (I3 - np.outer(uh, uh)) / nu
    Pw = (I3 - np.outer(wh, wh)) / nw

    D11 = (-(np.outer(uh, dcd1)) - c * Pu) / nu - np.outer(wh - c * uh, uh) / (nu * nu)
    D12 = (Pw - np.outer(uh, dcd2)) / nu
    D22 = (-(np.outer(wh, dcd2)) - c * Pw) / nw - np.outer(uh - c * wh, wh) / (nw * nw)

    def theta_block(Ddc, dca, dcb):
        return -Ddc / s - (c / (s ** 3)) * np.outer(dca, dcb)

    T11 = theta_block(D11, dcd1, dcd1)
    T12 = theta_block(D12, dcd1, dcd2)
    T22 = theta_block(D22, dcd2, dcd2)

    dl = th - th0
    H11 = kth * (dl * T11 + np.outer(g1, g1))
    H12 = kth * (dl * T12 + np.outer(g1, g2))
    H22 = kth * (dl * T22 + np.outer(g2, g2))
    H21 = H12.T

    H = np.zeros((9, 9))
    H[0:3, 0:3] = H11
    H[0:3, 6:9] = H12
    H[6:9, 0:3] = H21
    H[6:9, 6:9] = H22
    H[0:3, 3:6] = -(H11 + H12)
    H[6:9, 3:6] = -(H21 + H22)
    H[3:6, 0:3] = H[0:3, 3:6].T
    H[3:6, 6:9] = H[6:9, 3:6].T
    H[3:6, 3:6] = H11 + H12 + H21 + H22
    return H


def block_mass_scaled_hessian(coords):
    r = np.asarray(coords, dtype=np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError("invalid coordinates")
    I = r.shape[0]
    H = np.zeros((I, 9, 9))
    for i in range(I):
        A1, B, A2 = r[i]
        hb1 = _bond_hessian(A1, B, _KB, _R0)
        H[i, 0:3, 0:3] += hb1[0:3, 0:3]; H[i, 0:3, 3:6] += hb1[0:3, 3:6]
        H[i, 3:6, 0:3] += hb1[3:6, 0:3]; H[i, 3:6, 3:6] += hb1[3:6, 3:6]
        hb2 = _bond_hessian(A2, B, _KB, _R0)
        H[i, 6:9, 6:9] += hb2[0:3, 0:3]; H[i, 6:9, 3:6] += hb2[0:3, 3:6]
        H[i, 3:6, 6:9] += hb2[3:6, 0:3]; H[i, 3:6, 3:6] += hb2[3:6, 3:6]
        H[i] += _angle_hessian(A1, B, A2, _KTH, _TH0)
    # intermonomer LJ curvature enters each block through the diagonal
    # atom terms of the pair Hessians
    for i in range(I):
        for j in range(I):
            if i == j:
                continue
            H[i, 3:6, 3:6] += _lj_hessian_self(r[i, 1], r[j, 1], _EPS_B, _SIG_B)
            for ai, sl in ((0, 0), (2, 6)):
                for aj in (0, 2):
                    H[i, sl:sl+3, sl:sl+3] += _lj_hessian_self(r[i, ai], r[j, aj], _EPS_A, _SIG_A)
    m = np.repeat([_MA, _MB, _MA], 3) ** -0.5
    S = np.outer(m, m)
    for i in range(I):
        H[i] = ((H[i] + H[i].T) / 2.0) * S
    return H

"""Step 5: map a cluster configuration onto the reference manifold."""

import numpy as np

_R0 = 1.0
_TH0 = 1.9106332362490186
_SIN_TOL = 1e-8



def reference_configuration(coords):
    r = np.asarray(coords, dtype=np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError("invalid coordinates")
    I = r.shape[0]
    out = np.empty_like(r)
    for i in range(I):
        A1, B, A2 = r[i]
        u, w = A1 - B, A2 - B
        nu, nw = np.linalg.norm(u), np.linalg.norm(w)
        if nu <= 0.0 or nw <= 0.0 or not (np.isfinite(nu) and np.isfinite(nw)):
            raise ValueError("degenerate monomer geometry")
        bis = u / nu + w / nw
        nb = np.linalg.norm(bis)
        if nb <= _SIN_TOL or not np.isfinite(nb):
            raise ValueError("degenerate monomer geometry")
        x = bis / nb
        zc = np.cross(u, w)
        nz = np.linalg.norm(zc)
        if nz / (nu * nw) < _SIN_TOL:
            k = int(np.argmin(np.abs(x)))
            e = np.zeros(3)
            e[k] = 1.0
            z = e - np.dot(e, x) * x
            z = z / np.linalg.norm(z)
        else:
            z = zc / nz
        y = np.cross(z, x)
        ca, sa = np.cos(_TH0 / 2.0), np.sin(_TH0 / 2.0)
        out[i, 1] = B
        out[i, 0] = B + _R0 * (ca * x + sa * y)
        out[i, 2] = B + _R0 * (ca * x - sa * y)
    return out.astype(np.float64, copy=False)

"""Step 6: subspace harmonic relaxation by Newton-Raphson iterations."""

import numpy as np

_MASSES = (1.0, 16.0, 1.0)
_LI = 3




def _stiff_pseudoinverse(blocks):
    K = np.asarray(blocks, dtype=np.float64)
    if K.ndim != 3 or K.shape[1:] != (9, 9) or not np.all(np.isfinite(K)):
        raise ValueError("invalid blocks")
    I = K.shape[0]
    Kt = np.zeros_like(K)
    for i in range(I):
        lam, U = np.linalg.eigh(K[i])
        order = np.argsort(lam)[::-1][:_LI]
        if np.any(lam[order] <= 0.0):
            raise ValueError("nonpositive selected eigenvalue")
        for l in order:
            Kt[i] += (1.0 / lam[l]) * np.outer(U[:, l], U[:, l])
    return Kt.astype(np.float64, copy=False)


def shr_relax(coords, num_iters):
    r = np.asarray(coords, dtype=np.float64).copy()
    if r.ndim != 3 or r.shape[1:] != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError("invalid coordinates")
    if isinstance(num_iters, (bool, np.bool_)) or not isinstance(num_iters, (int, np.integer)) or num_iters < 0:
        raise ValueError("invalid iteration count")
    I = r.shape[0]
    sm = np.repeat(np.asarray(_MASSES, dtype=np.float64), 3) ** -0.5
    Kt = _stiff_pseudoinverse(block_mass_scaled_hessian(r))
    for _ in range(int(num_iters)):
        g = cluster_gradient(r)
        for i in range(I):
            step = sm * (Kt[i] @ (sm * g[i].reshape(9)))
            r[i] = r[i] - step.reshape(3, 3)
    return r.astype(np.float64, copy=False)

"""Step 8: harmonic thermodynamic quantities from stiff-mode frequencies."""

import numpy as np

def harmonic_thermo(V, omegas, T):
    om = np.asarray(omegas, dtype=np.float64)
    if om.ndim != 1 or om.size < 1 or not np.all(np.isfinite(om)) or np.any(om <= 0.0):
        raise ValueError("invalid frequencies")
    if not np.isfinite(V) or not np.isfinite(T) or T <= 0.0:
        raise ValueError("invalid scalar input")
    beta = 1.0 / float(T)
    F_cl = float(V) + float(np.sum(np.log(beta * om))) / beta
    F_qm = float(V) + float(np.sum(om / 2.0 + np.log(1.0 - np.exp(-beta * om)) / beta))
    v_cl = float(np.sum(1.0 / (beta * om * om)))
    v_qm = float(np.sum((1.0 / (2.0 * om)) / np.tanh(beta * om / 2.0)))
    return np.asarray([F_cl, F_qm, v_cl, v_qm], dtype=np.float64)

"""Step 8 gold: twenty-column SHR audit with per-monomer sums."""

import numpy as np

_LI = 3
_MASSES = (1.0, 16.0, 1.0)


def _audit_frequencies(coords):
    blocks = block_mass_scaled_hessian(coords)
    I = blocks.shape[0]
    oms = []
    for i in range(I):
        lam = np.linalg.eigvalsh(blocks[i])
        top = np.sort(lam)[-_LI:]
        if np.any(top <= 0.0) or not np.all(np.isfinite(top)):
            raise ValueError("nonpositive selected eigenvalue")
        oms.extend(np.sqrt(top))
    return np.sort(np.asarray(oms, dtype=np.float64))


def _frame_of(mono):
    """Local frame of one monomer per the frame contract, including the
    near-collinear fallback."""
    A1, B, A2 = mono
    u, w = A1 - B, A2 - B
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    uh, wh = u / nu, w / nw
    bis = uh + wh
    x = bis / np.linalg.norm(bis)
    zc = np.cross(u, w)
    nz = np.linalg.norm(zc)
    if nz / (nu * nw) < 1e-8:
        k = int(np.argmin(np.abs(x)))
        e = np.zeros(3)
        e[k] = 1.0
        z = e - np.dot(e, x) * x
        z = z / np.linalg.norm(z)
    else:
        z = zc / nz
    y = np.cross(z, x)
    return x, y, z


def _cov_audit(coords, T):
    """Backmapping covariance audit summed over every monomer, each using
    its own B atom at the given (relaxed) configuration: [lam_max sum,
    anisotropy sum, signed alignment sum, quantum/classical trace ratio
    sum]."""
    beta = 1.0 / T
    blocks = block_mass_scaled_hessian(coords)
    sm = np.repeat(np.asarray(_MASSES, dtype=np.float64), 3) ** -0.5
    out = np.zeros(4)
    for i in range(blocks.shape[0]):
        lam, U = np.linalg.eigh(blocks[i])
        Cq = np.zeros((3, 3))
        Cc = np.zeros((3, 3))
        for l in np.argsort(lam)[-_LI:]:
            om = np.sqrt(lam[l])
            wvec = (sm * U[:, l])[3:6]
            Cq += ((1.0 / (2.0 * om)) / np.tanh(beta * om / 2.0)) * np.outer(wvec, wvec)
            Cc += (1.0 / (beta * om * om)) * np.outer(wvec, wvec)
        ev, EV = np.linalg.eigh(Cq)
        tr = float(np.trace(Cq))
        v = EV[:, -1]
        x, y, z = _frame_of(coords[i])
        for comp in (float(np.dot(v, x)), float(np.dot(v, y)), float(np.dot(v, z))):
            if abs(comp) > 1e-8:
                if comp < 0.0:
                    v = -v
                break
        out += [float(ev[-1]), (float(ev[-1]) - float(ev[0])) / tr,
                float(np.dot(v, z)), tr / float(np.trace(Cc))]
    return list(out)


def _frame_twist(r0c, rP):
    """Signed frame twist summed over every monomer: for each monomer the
    frames at the reference and relaxed points stack x, y, z as rows, the
    relative rotation is F_P times F_0 transpose, the axis comes from the
    antisymmetric part, and the sign from its dot product with that
    monomer's reference-point z axis."""
    total = 0.0
    for i in range(r0c.shape[0]):
        x0, y0, z0 = _frame_of(r0c[i])
        xP, yP, zP = _frame_of(rP[i])
        F0 = np.vstack([x0, y0, z0])
        FP = np.vstack([xP, yP, zP])
        R = FP @ F0.T
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
        th = np.arctan2(np.linalg.norm(w), (np.trace(R) - 1.0) / 2.0)
        total += float(th * (1.0 if np.dot(w, z0) >= 0.0 else -1.0))
    return total


def _mem_residual(coords, blocks=None, g=None):
    """Eq 6 minimum-energy-manifold residual, summed over every monomer: the
    stiff-subspace pseudoinverse is rebuilt at this configuration and applied
    once to the mass-scaled gradient."""
    if blocks is None:
        blocks = block_mass_scaled_hessian(coords)
    if g is None:
        g = cluster_gradient(coords)
    sm = np.repeat(np.asarray(_MASSES, dtype=np.float64), 3) ** -0.5
    total = 0.0
    for i in range(blocks.shape[0]):
        lam, U = np.linalg.eigh(blocks[i])
        order = np.argsort(lam)[::-1][:_LI]
        if np.any(lam[order] <= 0.0):
            raise ValueError("nonpositive selected eigenvalue")
        Kt = np.zeros((9, 9), dtype=np.float64)
        for l in order:
            Kt += (1.0 / lam[l]) * np.outer(U[:, l], U[:, l])
        total += float(np.linalg.norm(Kt @ (sm * g[i].reshape(9))))
    return total


def shr_audit(configs, num_iters, T):
    if isinstance(num_iters, (bool, np.bool_)) or not isinstance(num_iters, (int, np.integer)) or num_iters < 0:
        raise ValueError("invalid iteration count")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("invalid temperature")
    rows = []
    for cfg in configs:
        r_ref = np.asarray(cfg, dtype=np.float64)
        r0c = reference_configuration(r_ref)
        V0 = cluster_energy(r0c)
        traj = [r0c]
        for p in range(1, int(num_iters) + 1):
            traj.append(shr_relax(r0c, p))
        rP = traj[-1]
        VP = cluster_energy(rP)
        om = _audit_frequencies(rP)
        th = harmonic_thermo(VP, om, T)
        om_fixed = _audit_frequencies(r0c)
        th_fixed = harmonic_thermo(V0, om_fixed, T)
        d1 = float(np.linalg.norm(traj[1] - traj[0])) if len(traj) > 1 else 0.0
        d2 = float(np.linalg.norm(traj[2] - traj[1])) if len(traj) > 2 else 0.0
        cov = _cov_audit(rP, T)
        tw = _frame_twist(r0c, rP)
        blocks_P = block_mass_scaled_hessian(rP)
        grad_P = cluster_gradient(rP)
        mem = _mem_residual(rP, blocks_P, grad_P)
        rows.append([
            V0, VP, th[0], th[1], th_fixed[0], th[0] - th_fixed[0],
            float(om[0]), float(om[-1]), th[2], th[3], d1, d2,
            1.0 if VP < V0 else 0.0,
            1.0 if th[0] < th_fixed[0] else 0.0,
            cov[0], cov[1], cov[2], cov[3], tw, mem,
        ])
    return np.asarray(rows, dtype=np.float64)
SCICODE_GOLD_EOF
