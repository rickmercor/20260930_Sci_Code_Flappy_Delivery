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


def plane_wave_grid(n_grid, cell_length, coulomb_scale):
    ng = int(n_grid)
    if ng != n_grid or ng < 2:
        raise ValueError("n_grid must be an integer >= 2")
    L = float(cell_length)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("cell_length must be finite and positive")
    cs = float(coulomb_scale)
    if not np.isfinite(cs):
        raise ValueError("coulomb_scale must be finite")
    idx = np.arange(ng)
    kn = np.where(idx < ng // 2, idx, idx - ng).astype(float)
    G = 2.0 * np.pi * kn / L
    Gsq = G ** 2
    nz = Gsq > 1e-12
    v = np.zeros(ng, dtype=float)
    v[nz] = 4.0 * np.pi / Gsq[nz]
    v *= cs / ng
    out = np.zeros((ng, 3), dtype=float)
    out[:, 0] = G
    out[:, 1] = Gsq
    out[:, 2] = v
    return out

import numpy as np

def ground_state_orbitals(hamiltonian, mu, beta):
    H = np.asarray(hamiltonian)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 1:
        raise ValueError("hamiltonian must be a square 2-D array")
    if not np.all(np.isfinite(H)):
        raise ValueError("hamiltonian must be finite")
    if not np.isfinite(float(mu)):
        raise ValueError("mu must be finite")
    if not np.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be finite and positive")
    H = np.asarray(H, dtype=complex)
    w, U = np.linalg.eigh((H + H.conj().T) / 2.0)
    f = 1.0 / (1.0 + np.exp(np.clip(float(beta) * (w - float(mu)), -700.0, 700.0)))
    n = U.shape[0]
    packed = np.zeros((n, n + 2), dtype=complex)
    packed[:, :n] = U
    packed[:, n] = w.astype(complex)
    packed[:, n + 1] = f.astype(complex)
    return packed

import numpy as np

def sternheimer_solve(hamiltonian, orbitals, eigenvalues, band_index, delta_V,
                              tol=1e-10, max_iter=200):
    H = np.asarray(hamiltonian)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 1:
        raise ValueError("hamiltonian must be a square 2-D array")
    if not np.all(np.isfinite(H)):
        raise ValueError("hamiltonian must be finite")
    H = np.asarray(H, dtype=complex)
    H = (H + H.conj().T) / 2.0
    nb = H.shape[0]
    Phi = np.asarray(orbitals, dtype=complex)
    if Phi.ndim != 2 or Phi.shape[0] != nb or Phi.shape[1] < 1 or Phi.shape[1] > nb:
        raise ValueError("orbitals must have shape (Nb, Nocc) with 1 <= Nocc <= Nb")
    if not np.all(np.isfinite(Phi)):
        raise ValueError("orbitals must be finite")
    w = np.asarray(eigenvalues, dtype=float)
    if w.ndim != 1 or w.shape[0] != Phi.shape[1] or not np.all(np.isfinite(w)):
        raise ValueError("eigenvalues must be a finite 1-D array of length Nocc")
    n = int(band_index)
    if n != band_index or n < 0 or n >= Phi.shape[1]:
        raise ValueError("band_index must be an integer in [0, Nocc)")
    dV = np.asarray(delta_V, dtype=float)
    if dV.ndim != 1 or dV.shape[0] != nb or not np.all(np.isfinite(dV)):
        raise ValueError("delta_V must be a finite 1-D array of length Nb")
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be finite and positive")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer")

    def Qop(v):
        return v - Phi @ (Phi.conj().T @ v)

    def Aop(v):
        v = Qop(v)
        return Qop(H @ v - w[n] * v)

    b = -Qop(dV * Phi[:, n])
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = np.real(np.vdot(r, r))
    it = 0
    while np.sqrt(rs) > float(tol) and it < int(max_iter):
        Ap = Aop(p)
        denom = np.real(np.vdot(p, Ap))
        if abs(denom) < 1e-300:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.real(np.vdot(r, r))
        p = r + (rs_new / rs) * p
        rs = rs_new
        it += 1
    return Qop(x)

import numpy as np

def apply_chi0(hamiltonian, orbitals, eigenvalues, occupations, delta_V, beta,
                       tol=1e-10, max_iter=200):
    H = np.asarray(hamiltonian)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 1:
        raise ValueError("hamiltonian must be a square 2-D array")
    if not np.all(np.isfinite(H)):
        raise ValueError("hamiltonian must be finite")
    nb = H.shape[0]
    Phi = np.asarray(orbitals, dtype=complex)
    if Phi.ndim != 2 or Phi.shape[0] != nb or Phi.shape[1] < 1 or Phi.shape[1] > nb:
        raise ValueError("orbitals must have shape (Nb, Nocc) with 1 <= Nocc <= Nb")
    nocc = Phi.shape[1]
    w = np.asarray(eigenvalues, dtype=float)
    if w.ndim != 1 or w.shape[0] != nocc or not np.all(np.isfinite(w)):
        raise ValueError("eigenvalues must be a finite 1-D array of length Nocc")
    f = np.asarray(occupations, dtype=float)
    if f.ndim != 1 or f.shape[0] != nocc or not np.all(np.isfinite(f)):
        raise ValueError("occupations must be a finite 1-D array of length Nocc")
    if np.any(f < -1e-12) or np.any(f > 1.0 + 1e-12):
        raise ValueError("occupations must lie in [0, 1]")
    dV = np.asarray(delta_V, dtype=float)
    if dV.ndim != 1 or dV.shape[0] != nb or not np.all(np.isfinite(dV)):
        raise ValueError("delta_V must be a finite 1-D array of length Nb")
    if not np.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be finite and positive")
    tarr = np.atleast_1d(np.asarray(tol, dtype=float))
    if tarr.size not in (1, nocc):
        raise ValueError("tol must be a scalar or an array of length Nocc")
    if np.any(np.isnan(tarr)) or np.any(tarr <= 0.0):
        raise ValueError("every tolerance must be positive")
    tols = np.broadcast_to(tarr, (nocc,))

    beta = float(beta)
    fp = -beta * f * (1.0 - f)
    B = Phi.conj().T @ (dV[:, None] * Phi)
    drho = np.zeros(nb, dtype=float)

    for n in range(nocc):
        for m in range(nocc):
            if m == n:
                continue
            de = w[n] - w[m]
            wt = fp[n] if abs(de) < 1e-10 else (f[n] - f[m]) / de
            drho += wt * np.real(Phi[:, m] * np.conj(Phi[:, n]) * B[m, n])

    for n in range(nocc):
        if abs(f[n]) < 1e-14:
            continue
        dphi = sternheimer_solve(hamiltonian, Phi, w, n, dV, tols[n], max_iter)
        drho += 2.0 * f[n] * np.real(np.conj(Phi[:, n]) * dphi)

    diagB = np.real(np.diag(B))
    denom = float(np.sum(fp))
    if denom == 0.0:
        df = np.zeros(nocc, dtype=float)      # zero-compressibility limit
    else:
        df = fp * (diagB - float(np.sum(fp * diagB)) / denom)
    drho += (np.abs(Phi) ** 2) @ df
    return drho

import numpy as np


def kerker_precondition(vector, g_squared, alpha):
    v = np.asarray(vector, dtype=float)
    if v.ndim != 1 or v.shape[0] < 1 or not np.all(np.isfinite(v)):
        raise ValueError("vector must be a finite 1-D array")
    Gsq = np.asarray(g_squared, dtype=float)
    if Gsq.shape != v.shape or not np.all(np.isfinite(Gsq)) or np.any(Gsq < 0.0):
        raise ValueError("g_squared must be finite, non-negative and the shape of vector")
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("alpha must be finite and positive")
    ng = v.shape[0]
    j = np.arange(ng)
    W = np.exp(-2j * np.pi * np.outer(j, j) / ng) / np.sqrt(ng)
    D = Gsq / (Gsq + a * a)
    D = np.where(Gsq <= 1e-12, 1.0, D)
    return np.real(W.conj().T @ (D * (W @ v.astype(complex))))

import numpy as np


def adaptive_cg_tolerance(target_tol, residual_norm, sigma_min, iteration,
                                  restart_size, occupations, orbitals, kernel_vector,
                                  cell_volume, n_grid, n_occ):
    tau = float(target_tol)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("target_tol must be finite and positive")
    f = np.asarray(occupations, dtype=float)
    if f.ndim != 1 or f.shape[0] < 1 or not np.all(np.isfinite(f)):
        raise ValueError("occupations must be a finite 1-D array")
    if np.any(f < -1e-12) or np.any(f > 1.0 + 1e-12):
        raise ValueError("occupations must lie in [0, 1]")
    Phi = np.asarray(orbitals)
    if Phi.ndim != 2 or Phi.shape[1] != f.shape[0]:
        raise ValueError("orbitals must have shape (Ng, Nocc) matching occupations")
    Kv = np.asarray(kernel_vector, dtype=float)
    if Kv.ndim != 1 or Kv.shape[0] < 1 or not np.all(np.isfinite(Kv)):
        raise ValueError("kernel_vector must be a finite 1-D array")
    vol = float(cell_volume)
    if not np.isfinite(vol) or vol <= 0.0:
        raise ValueError("cell_volume must be finite and positive")
    if int(n_grid) < 1 or int(n_occ) < 1:
        raise ValueError("n_grid and n_occ must be positive integers")
    it = int(iteration)
    if it < 0:
        raise ValueError("iteration must be a non-negative integer")
    m = int(restart_size)
    if m < 1:
        raise ValueError("restart_size must be a positive integer")

    nrmK = float(np.linalg.norm(Kv))
    if nrmK <= 0.0:
        raise ValueError("kernel_vector must have positive norm")
    orb = float(np.max(np.linalg.norm(np.real(Phi), axis=0)))
    if orb <= 0.0:
        raise ValueError("orbitals must have a non-zero real part")
    C = np.empty_like(f)
    small = f <= 1e-14
    C[~small] = np.sqrt(vol) / (2.0 * f[~small] * nrmK * orb
                                * np.sqrt(float(n_grid) * float(n_occ)))
    C[small] = np.inf
    if it == 0:
        return C * (tau / 3.0)
    rn = float(residual_norm)
    if not np.isfinite(rn) or rn <= 0.0:
        raise ValueError("residual_norm must be finite and positive")
    sm = float(sigma_min)
    if not np.isfinite(sm) or sm < 0.0:
        raise ValueError("sigma_min must be finite and non-negative")
    return C * (sm / (3.0 * m)) * (tau / rn)

import numpy as np


def preconditioned_gmres(hamiltonian, orbitals, eigenvalues, occupations, rhs,
                                 coulomb_gspace, xc_local, g_squared, alpha, beta,
                                 target_tol, cell_volume, restart_size=5,
                                 max_cycles=12, cg_max_iter=400):
    b = np.asarray(rhs, dtype=float)
    if b.ndim != 1 or b.shape[0] < 1 or not np.all(np.isfinite(b)):
        raise ValueError("rhs must be a finite 1-D array")
    tau = float(target_tol)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("target_tol must be finite and positive")
    vol = float(cell_volume)
    if not np.isfinite(vol) or vol <= 0.0:
        raise ValueError("cell_volume must be finite and positive")
    m = int(restart_size)
    if m < 1 or int(max_cycles) < 1 or int(cg_max_iter) < 1:
        raise ValueError("restart_size, max_cycles and cg_max_iter must be positive")
    ng = b.shape[0]
    Phi = np.asarray(orbitals)
    nocc = Phi.shape[1]
    f = np.asarray(occupations, dtype=float)

    vGa = np.asarray(coulomb_gspace, dtype=float)
    if vGa.shape != b.shape or not np.all(np.isfinite(vGa)):
        raise ValueError("coulomb_gspace must be finite and the shape of rhs")
    if not np.isfinite(float(xc_local)):
        raise ValueError("xc_local must be finite")
    jgrid = np.arange(ng)
    Wf = np.exp(-2j * np.pi * np.outer(jgrid, jgrid) / ng) / np.sqrt(ng)

    def _kernel(d):
        d = np.asarray(d, dtype=float)
        return np.real(Wf.conj().T @ (vGa * (Wf @ d.astype(complex)))) + float(xc_local) * d

    Tb = kerker_precondition(b, g_squared, alpha)
    x0 = np.zeros(ng, dtype=float)
    s = np.inf
    for _cycle in range(int(max_cycles)):
        if np.any(x0):
            Ex0 = x0 - apply_chi0(
                hamiltonian, orbitals, eigenvalues, occupations,
                _kernel(x0), beta,
                tau / 3.0, cg_max_iter)
            r0 = Tb - kerker_precondition(Ex0, g_squared, alpha)
        else:
            r0 = Tb
        beta0 = float(np.linalg.norm(r0))
        if beta0 == 0.0:
            return x0
        Q = np.zeros((ng, m + 1), dtype=float)
        Hh = np.zeros((m + 1, m), dtype=float)
        Q[:, 0] = r0 / beta0
        res = beta0
        y = np.zeros(1)
        kk = 0
        for k in range(m):
            kk = k
            sig = 1.0 if not np.isfinite(s) else s
            Kv = _kernel(Q[:, k])
            nk = float(np.linalg.norm(Kv))
            if nk == 0.0:
                chi = np.zeros(ng, dtype=float)
            else:
                t = adaptive_cg_tolerance(tau, res, sig, k, m, f, Phi, Kv,
                                                  vol, ng, nocc)
                chi = nk * apply_chi0(hamiltonian, orbitals, eigenvalues,
                                              occupations, Kv / nk, beta, t, cg_max_iter)
            v = kerker_precondition(Q[:, k] - chi, g_squared, alpha)
            for i in range(k + 1):
                Hh[i, k] = v @ Q[:, i]
                v = v - Hh[i, k] * Q[:, i]
            Hh[k + 1, k] = float(np.linalg.norm(v))
            if Hh[k + 1, k] > 1e-14:
                Q[:, k + 1] = v / Hh[k + 1, k]
            e1 = np.zeros(k + 2)
            e1[0] = beta0
            y, *_ = np.linalg.lstsq(Hh[:k + 2, :k + 1], e1, rcond=None)
            res = float(np.linalg.norm(Hh[:k + 2, :k + 1] @ y - e1))
            if res <= tau / 3.0:
                break
        sig_i = float(np.linalg.svd(Hh[:kk + 2, :kk + 1], compute_uv=False)[-1])
        x = x0 + Q[:, :kk + 1] @ y
        if res <= tau / 3.0 and s <= sig_i:
            return x
        s = min(s, sig_i)
        x0 = x
    return x0

import numpy as np


def response_observable(hamiltonian, delta_V_ext, observable, mu, beta, n_occ,
                                n_grid, cell_length, coulomb_scale, xc_local, alpha,
                                target_tol, restart_size=5):
    H = np.asarray(hamiltonian)
    nb = H.shape[0]
    O = np.asarray(observable, dtype=float)
    if O.ndim != 1 or O.shape[0] != nb:
        raise ValueError("observable must be a 1-D array of length Ng")
    no = int(n_occ)
    if no < 1 or no > nb:
        raise ValueError("n_occ must be an integer in [1, Ng]")
    grid = plane_wave_grid(n_grid, cell_length, coulomb_scale)
    Gsq = grid[:, 1]
    vG = grid[:, 2]
    packed = ground_state_orbitals(hamiltonian, mu, beta)
    U = packed[:, :nb]
    w = packed[:, nb].real
    f = packed[:, nb + 1].real
    Phi = U[:, :no]
    b = apply_chi0(hamiltonian, Phi, w[:no], f[:no], delta_V_ext, beta, 1e-12, 400)
    dn = preconditioned_gmres(hamiltonian, Phi, w[:no], f[:no], b, vG, xc_local,
                                      Gsq, alpha, beta, target_tol, float(cell_length),
                                      restart_size)
    return float(O @ dn)
SCICODE_GOLD_EOF
