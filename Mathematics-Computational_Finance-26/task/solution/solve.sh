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


def shifted_legendre_basis_table(S, N, smax):
    def _legendre(nmax, x):
        P = np.zeros((nmax + 1, x.size))
        dP = np.zeros((nmax + 1, x.size))
        d2P = np.zeros((nmax + 1, x.size))
        P[0] = 1.0
        if nmax >= 1:
            P[1] = x
            dP[1] = 1.0
        for n in range(1, nmax):
            P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - n * P[n - 1]) / (n + 1.0)
            dP[n + 1] = ((2.0 * n + 1.0) * (P[n] + x * dP[n])
                         - n * dP[n - 1]) / (n + 1.0)
            d2P[n + 1] = ((2.0 * n + 1.0) * (2.0 * dP[n] + x * d2P[n])
                          - n * d2P[n - 1]) / (n + 1.0)
        return P, dP, d2P

    S = np.asarray(S, dtype=float).reshape(-1)
    N = int(N)
    smax = float(smax)
    if N < 0:
        raise ValueError("N must be non-negative")
    if smax <= 0.0:
        raise ValueError("smax must be positive")
    if S.size == 0:
        raise ValueError("S must contain at least one point")
    if np.any(S < 0.0) or np.any(S > smax):
        raise ValueError("every S must lie in [0, smax]")
    x = 2.0 * S / smax - 1.0
    P, dP, d2P = _legendre(N, x)
    nrm = np.sqrt((2.0 * np.arange(N + 1) + 1.0) / smax).reshape(-1, 1)
    lv = nrm * P
    l1 = nrm * dP * (2.0 / smax)
    l2 = nrm * d2P * (2.0 / smax) ** 2
    return np.concatenate((lv.reshape(-1), l1.reshape(-1), l2.reshape(-1)))

import numpy as np


def observed_price_profile(ns, smax, strikes, T, sigma0, eta, sref, r,
                                   delta, seed):
    def _vol(t, s, sig0, curv, ref, hor):
        return sig0 * np.sqrt(1.0 + curv * np.exp(-t / hor)
                              * ((s - ref) / ref) ** 2)

    def _march(grid, payoff, hor, sig0, curv, ref, rate):
        nt = max(1, int(round(hor / 5.0e-4)))
        dt = hor / nt
        ds = grid[1] - grid[0]
        s_in = grid[1:-1]
        u = payoff.copy()
        for n in range(nt, 0, -1):
            sig = _vol(n * dt, s_in, sig0, curv, ref, hor)
            new = np.empty_like(u)
            new[1:-1] = (u[1:-1]
                         + dt * 0.5 * sig ** 2 * s_in ** 2
                         * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / ds ** 2
                         + dt * rate * s_in * (u[2:] - u[1:-1]) / ds
                         - rate * dt * u[1:-1])
            new[0] = 2.0 * new[1] - new[2]
            new[-1] = 2.0 * new[-2] - new[-3]
            u = new
        return u

    ns = int(ns)
    smax = float(smax)
    K = np.asarray(strikes, dtype=float).reshape(-1)
    T = float(T)
    sigma0 = float(sigma0)
    eta = float(eta)
    sref = float(sref)
    r = float(r)
    delta = float(delta)
    if ns < 2:
        raise ValueError("ns must be at least 2")
    if smax <= 0.0:
        raise ValueError("smax must be positive")
    if K.size != 3 or not (0.0 < K[0] < K[1] < K[2]):
        raise ValueError("strikes must be three increasing positive values")
    if T <= 0.0 or sigma0 <= 0.0 or sref <= 0.0:
        raise ValueError("T, sigma0 and sref must be positive")
    if eta < 0.0 or delta < 0.0:
        raise ValueError("eta and delta must be non-negative")
    grid = np.linspace(0.0, smax, ns + 1)
    payoff = (np.maximum(grid - K[0], 0.0) - 2.0 * np.maximum(grid - K[1], 0.0)
              + np.maximum(grid - K[2], 0.0))
    clean = _march(grid, payoff, T, sigma0, eta, sref, r)
    xi = np.random.default_rng(int(seed)).uniform(-1.0, 1.0, size=ns + 1)
    noisy = clean * (1.0 + delta * xi)
    return np.concatenate((clean, noisy))

import numpy as np


def project_profile_onto_basis(S, profile, N, smax):
    def _values(nmax, pts, right):
        x = 2.0 * pts / right - 1.0
        P = np.zeros((nmax + 1, x.size))
        P[0] = 1.0
        if nmax >= 1:
            P[1] = x
        for n in range(1, nmax):
            P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - n * P[n - 1]) / (n + 1.0)
        nrm = np.sqrt((2.0 * np.arange(nmax + 1) + 1.0) / right).reshape(-1, 1)
        return nrm * P

    S = np.asarray(S, dtype=float).reshape(-1)
    profile = np.asarray(profile, dtype=float).reshape(-1)
    N = int(N)
    smax = float(smax)
    if S.size != profile.size:
        raise ValueError("S and profile must have the same length")
    if S.size < 2:
        raise ValueError("at least two grid points are required")
    if N < 0:
        raise ValueError("N must be non-negative")
    if smax <= 0.0:
        raise ValueError("smax must be positive")
    L = _values(N, S, smax)
    integ = L * profile.reshape(1, -1)
    dS = (S[1:] - S[:-1]).reshape(1, -1)
    return np.sum(0.5 * (integ[:, 1:] + integ[:, :-1]) * dS, axis=1)

import numpy as np


def reduced_convection_matrix(N, smax, nq):
    def _tab(nmax, pts, right):
        x = 2.0 * pts / right - 1.0
        P = np.zeros((nmax + 1, x.size))
        dP = np.zeros((nmax + 1, x.size))
        P[0] = 1.0
        if nmax >= 1:
            P[1] = x
            dP[1] = 1.0
        for n in range(1, nmax):
            P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - n * P[n - 1]) / (n + 1.0)
            dP[n + 1] = ((2.0 * n + 1.0) * (P[n] + x * dP[n])
                         - n * dP[n - 1]) / (n + 1.0)
        nrm = np.sqrt((2.0 * np.arange(nmax + 1) + 1.0) / right).reshape(-1, 1)
        return nrm * P, nrm * dP * (2.0 / right)

    N = int(N)
    smax = float(smax)
    nq = int(nq)
    if N < 0:
        raise ValueError("N must be non-negative")
    if smax <= 0.0:
        raise ValueError("smax must be positive")
    if nq < 1:
        raise ValueError("nq must be at least 1")
    xg, wg = np.polynomial.legendre.leggauss(nq)
    Sq = 0.5 * smax * (xg + 1.0)
    wq = 0.5 * smax * wg
    L, dL = _tab(N, Sq, smax)
    B = (L * (wq * Sq).reshape(1, -1)) @ dL.T
    return B.reshape(-1)

import numpy as np


def reduced_diffusion_matrix(t, N, smax, nq, sigma0, eta, sref, T):
    def _tab(nmax, pts, right):
        x = 2.0 * pts / right - 1.0
        P = np.zeros((nmax + 1, x.size))
        dP = np.zeros((nmax + 1, x.size))
        d2P = np.zeros((nmax + 1, x.size))
        P[0] = 1.0
        if nmax >= 1:
            P[1] = x
            dP[1] = 1.0
        for n in range(1, nmax):
            P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - n * P[n - 1]) / (n + 1.0)
            dP[n + 1] = ((2.0 * n + 1.0) * (P[n] + x * dP[n])
                         - n * dP[n - 1]) / (n + 1.0)
            d2P[n + 1] = ((2.0 * n + 1.0) * (2.0 * dP[n] + x * d2P[n])
                          - n * d2P[n - 1]) / (n + 1.0)
        nrm = np.sqrt((2.0 * np.arange(nmax + 1) + 1.0) / right).reshape(-1, 1)
        return nrm * P, nrm * d2P * (2.0 / right) ** 2

    t = float(t)
    N = int(N)
    smax = float(smax)
    nq = int(nq)
    sigma0 = float(sigma0)
    eta = float(eta)
    sref = float(sref)
    T = float(T)
    if N < 0:
        raise ValueError("N must be non-negative")
    if smax <= 0.0 or sref <= 0.0 or T <= 0.0:
        raise ValueError("smax, sref and T must be positive")
    if nq < 1:
        raise ValueError("nq must be at least 1")
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    xg, wg = np.polynomial.legendre.leggauss(nq)
    Sq = 0.5 * smax * (xg + 1.0)
    wq = 0.5 * smax * wg
    L, d2L = _tab(N, Sq, smax)
    s2 = sigma0 ** 2 * (1.0 + eta * np.exp(-t / T) * ((Sq - sref) / sref) ** 2)
    A = (L * (wq * s2 * Sq ** 2).reshape(1, -1)) @ d2L.T
    return A.reshape(-1)

import numpy as np


def reduced_coefficient_matrix(a_flat, b_flat, r):
    def _square(v):
        v = np.asarray(v, dtype=float).reshape(-1)
        k = int(round(float(np.sqrt(v.size))))
        if k * k != v.size or v.size == 0:
            raise ValueError("flattened input is not a square matrix")
        return v.reshape(k, k)

    A = _square(a_flat)
    B = _square(b_flat)
    r = float(r)
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    C = -0.5 * A - r * B + r * np.eye(A.shape[0])
    return C.reshape(-1)

import numpy as np


def assemble_tikhonov_system(u0d, c_stack, T, alpha):
    def _build(nt, m, dt, alpha, C, data):
        nvar = (nt + 1) * m
        nrow = m * (4 * nt + 1)
        M = np.zeros((nrow, nvar))
        rhs = np.zeros(nrow)
        eye = np.eye(m)
        sdt = float(np.sqrt(dt))
        sa = float(np.sqrt(alpha * dt))
        p = 0
        for k in range(nt):
            M[p:p + m, k * m:(k + 1) * m] = sdt * (-eye / dt - 0.5 * C[k])
            M[p:p + m, (k + 1) * m:(k + 2) * m] = sdt * (eye / dt - 0.5 * C[k])
            p += m
        M[p:p + m, 0:m] = eye
        rhs[p:p + m] = data
        p += m
        for k in range(nt + 1):
            M[p:p + m, k * m:(k + 1) * m] = sa * eye
            p += m
        for k in range(nt):
            M[p:p + m, k * m:(k + 1) * m] = -sa * eye / dt
            M[p:p + m, (k + 1) * m:(k + 2) * m] = sa * eye / dt
            p += m
        for k in range(1, nt):
            M[p:p + m, (k - 1) * m:k * m] = sa * eye / dt ** 2
            M[p:p + m, k * m:(k + 1) * m] = -2.0 * sa * eye / dt ** 2
            M[p:p + m, (k + 1) * m:(k + 2) * m] = sa * eye / dt ** 2
            p += m
        return M, rhs

    u0d = np.asarray(u0d, dtype=float).reshape(-1)
    c_stack = np.asarray(c_stack, dtype=float).reshape(-1)
    T = float(T)
    alpha = float(alpha)
    m = u0d.size
    if m < 1:
        raise ValueError("u0d must be non-empty")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if c_stack.size % (m * m) != 0 or c_stack.size == 0:
        raise ValueError("c_stack length is not a multiple of (N+1)**2")
    nt = c_stack.size // (m * m)
    if nt < 2:
        raise ValueError("at least two time intervals are required")
    C = c_stack.reshape(nt, m, m)
    M, rhs = _build(nt, m, T / nt, alpha, C, u0d)
    return np.concatenate((M.reshape(-1), rhs))

import numpy as np


def solve_tikhonov_system(sys_flat, ncol):
    def _unpack(v, nrow, ncol):
        return v[:nrow * ncol].reshape(nrow, ncol), v[nrow * ncol:]

    sys_flat = np.asarray(sys_flat, dtype=float).reshape(-1)
    ncol = int(ncol)
    if ncol < 1:
        raise ValueError("ncol must be at least 1")
    if sys_flat.size == 0 or sys_flat.size % (ncol + 1) != 0:
        raise ValueError("sys_flat length is not a multiple of ncol + 1")
    nrow = sys_flat.size // (ncol + 1)
    if nrow < ncol:
        raise ValueError("the implied system is not overdetermined")
    M, rhs = _unpack(sys_flat, nrow, ncol)
    sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
    return np.asarray(sol, dtype=float).reshape(-1)

import numpy as np


def lcurve_corner_index(residual_quantity, regularisation_norm):
    def _curvature(lx, ly, j):
        x1 = 0.5 * (lx[j + 1] - lx[j - 1])
        y1 = 0.5 * (ly[j + 1] - ly[j - 1])
        x2 = lx[j + 1] - 2.0 * lx[j] + lx[j - 1]
        y2 = ly[j + 1] - 2.0 * ly[j] + ly[j - 1]
        den = (x1 * x1 + y1 * y1) ** 1.5
        if den == 0.0:
            return 0.0
        return (x1 * y2 - x2 * y1) / den

    a = np.asarray(residual_quantity, dtype=float).reshape(-1)
    b = np.asarray(regularisation_norm, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError("the two quantities must have the same length")
    if a.size < 3:
        raise ValueError("at least three candidates are required")
    if np.any(a <= 0.0) or np.any(b <= 0.0):
        raise ValueError("both quantities must be strictly positive")
    lx = np.log(a)
    ly = np.log(b)
    best = -np.inf
    idx = 1
    for j in range(1, a.size - 1):
        kap = _curvature(lx, ly, j)
        if kap > best:
            best = kap
            idx = j
    return float(idx)

import numpy as np


def run_legendre_tikhonov_reconstruction(market, numerics):
    def _lcurve_quantities(sol, data, c_blocks, dt, nt, m):
        V = sol.reshape(nt + 1, m)
        res2 = 0.0
        for k in range(nt):
            Ck = c_blocks[k * m * m:(k + 1) * m * m].reshape(m, m)
            d = (V[k + 1] - V[k]) / dt - Ck @ (0.5 * (V[k] + V[k + 1]))
            res2 += dt * float(d @ d)
        e = V[0] - data
        res2 += float(e @ e)
        d1 = (V[1:] - V[:-1]) / dt
        d2 = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / dt ** 2
        q2 = dt * (float(np.sum(V * V)) + float(np.sum(d1 * d1))
                   + float(np.sum(d2 * d2)))
        return np.sqrt(res2), np.sqrt(q2)

    market = np.asarray(market, dtype=float).reshape(-1)
    numerics = np.asarray(numerics, dtype=float).reshape(-1)
    if market.size != 9 or numerics.size != 9:
        raise ValueError("market and numerics must both have size 9")
    smax, r, sigma0, eta, sref, T, K1, K2, K3 = market
    N, nt, delta, seed, ns, nq, sstar, lo, nalpha = numerics
    N = int(N)
    nt = int(nt)
    ns = int(ns)
    nq = int(nq)
    seed = int(seed)
    nalpha = int(nalpha)
    if smax <= 0.0:
        raise ValueError("smax must be positive")
    if not (0.0 <= sstar <= smax):
        raise ValueError("sstar must lie in [0, smax]")
    if nalpha < 3:
        raise ValueError("at least three candidate weights are required")

    prof = observed_price_profile(ns, smax, np.array([K1, K2, K3]), T,
                                          sigma0, eta, sref, r, delta, seed)
    noisy = prof[ns + 1:]
    grid = np.linspace(0.0, smax, ns + 1)
    u0d = project_profile_onto_basis(grid, noisy, N, smax)

    b_flat = reduced_convection_matrix(N, smax, nq)
    dt = T / nt
    blocks = []
    for k in range(nt):
        a_flat = reduced_diffusion_matrix((k + 0.5) * dt, N, smax, nq,
                                                  sigma0, eta, sref, T)
        blocks.append(reduced_coefficient_matrix(a_flat, b_flat, r))
    c_stack = np.concatenate(blocks)

    m = N + 1
    ncol = (nt + 1) * m
    alphas = np.array([10.0 ** (lo + 1.0 * j) for j in range(nalpha)])
    sols = []
    resq = np.zeros(nalpha)
    regn = np.zeros(nalpha)
    for j in range(nalpha):
        sys_flat = assemble_tikhonov_system(u0d, c_stack, T,
                                                    float(alphas[j]))
        sol = solve_tikhonov_system(sys_flat, ncol)
        sols.append(sol)
        resq[j], regn[j] = _lcurve_quantities(sol, u0d, c_stack, dt, nt, m)

    jstar = int(lcurve_corner_index(resq, regn))
    vT = sols[jstar].reshape(nt + 1, m)[-1]
    tab = shifted_legendre_basis_table(np.array([sstar]), N, smax)
    return float(np.dot(vT, tab[:m]))
SCICODE_GOLD_EOF
