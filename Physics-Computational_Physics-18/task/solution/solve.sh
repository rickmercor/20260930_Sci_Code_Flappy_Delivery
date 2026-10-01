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

# ----------------------------------------------------------------------------
# shared fixtures
# ----------------------------------------------------------------------------

PFC_PARAMS = dict(eps=0.1, M_phi=1.0, M_m=1.0e-2, eta=10.0, gamma12=0.5,
                  omega0=1.0, theta=1.0e-9, theta1=1.0e-9, theta2=1.0e-9,
                  alpha=1.0, beta=10.0, S_phi=10.0, S_m=10.0,
                  a1=1.0, a2=1.0, a12=1.2, B=1.0e7,
                  gamma1=0.01, gamma2=0.01, eta1=-0.1, eta2=-0.1)

_KEYS = ("eps", "M_phi", "M_m", "eta", "gamma12", "omega0", "theta", "theta1",
         "theta2", "alpha", "beta", "S_phi", "S_m", "a1", "a2", "a12", "B",
         "gamma1", "gamma2", "eta1", "eta2")


def _check_params(params):
    if params is None:
        return dict(PFC_PARAMS)
    p = dict(PFC_PARAMS)
    for k in params:
        if k not in _KEYS:
            raise ValueError("unknown model parameter %r" % (k,))
    p.update({k: float(v) for k, v in params.items()})
    if not all(np.isfinite(v) for v in p.values()):
        raise ValueError("model parameters must be finite")
    if p["B"] <= 0.0:
        raise ValueError("B must be positive")
    if p["M_phi"] <= 0.0 or p["M_m"] <= 0.0:
        raise ValueError("the mobilities must be positive")
    if p["omega0"] < 0.0 or p["beta"] < 0.0:
        raise ValueError("omega0 and beta must be non-negative")
    return p


def _check_cell(cell, shape):
    L = np.atleast_1d(np.asarray(cell, float)).ravel()
    if L.size == 1:
        L = np.full(len(shape), float(L[0]))
    if L.size != len(shape):
        raise ValueError("cell must be a scalar or match the field rank")
    if not np.all(np.isfinite(L)) or np.any(L <= 0.0):
        raise ValueError("cell edge lengths must be finite and positive")
    return L


def _wavenumbers(shape, L):
    return [2.0 * np.pi * np.fft.fftfreq(n, d=Ln / n) for n, Ln in zip(shape, L)]


def _k2(shape, L):
    ks = _wavenumbers(shape, L)
    return sum(k.reshape([-1 if i == j else 1 for j in range(len(shape))]) ** 2
               for i, k in enumerate(ks))


def _ip(f, g, L):
    """Discrete L2 inner product on a uniform periodic grid."""
    dv = np.prod(L) / np.prod(f.shape[-len(L):])
    return float(np.sum(f * g) * dv)


def pfc_spectral_derivatives(u, cell, a=0.0):
    u = np.asarray(u, float)
    if u.ndim != 2:
        raise ValueError("u must be a two-dimensional periodic field")
    if not np.all(np.isfinite(u)):
        raise ValueError("u must be finite")
    L = _check_cell(cell, u.shape)
    a = float(a)
    if not np.isfinite(a):
        raise ValueError("a must be finite")
    kx, ky = _wavenumbers(u.shape, L)
    KX, KY = kx[:, None], ky[None, :]
    k2 = KX ** 2 + KY ** 2
    uh = np.fft.fft2(u)
    ux = np.real(np.fft.ifft2(1j * KX * uh))
    uy = np.real(np.fft.ifft2(1j * KY * uh))
    lap = np.real(np.fft.ifft2(-k2 * uh))
    bih = np.real(np.fft.ifft2(k2 ** 2 * uh))
    shf = np.real(np.fft.ifft2((a - k2) ** 2 * uh))
    return np.stack([ux, uy, lap, bih, shf])

import numpy as np

def pfc_free_energy(phi1, phi2, M, cell, params=None, H=None):
    p = _check_params(params)
    phi1 = np.asarray(phi1, float)
    phi2 = np.asarray(phi2, float)
    M = np.asarray(M, float)
    if phi1.ndim != 2 or phi2.shape != phi1.shape:
        raise ValueError("phi1 and phi2 must be two-dimensional fields of equal shape")
    if M.shape != (2,) + phi1.shape:
        raise ValueError("M must have shape (2,) + phi1.shape")
    L = _check_cell(cell, phi1.shape)
    H = np.zeros_like(M) if H is None else np.asarray(H, float)
    if H.shape != M.shape:
        raise ValueError("H must have the same shape as M")
    if not all(np.all(np.isfinite(v)) for v in (phi1, phi2, M, H)):
        raise ValueError("phi1, phi2, M and H must be finite")

    d1 = pfc_spectral_derivatives(phi1, L, p["a1"])
    d2 = pfc_spectral_derivatives(phi2, L, p["a2"])
    c12a = pfc_spectral_derivatives(phi2, L, p["a12"])[4]
    g1 = d1[:2]
    g2 = d2[:2]
    m2 = M[0] ** 2 + M[1] ** 2
    mg1 = M[0] * g1[0] + M[1] * g1[1]
    mg2 = M[0] * g2[0] + M[1] * g2[1]
    gm2 = sum(pfc_spectral_derivatives(M[j], L)[i] ** 2
              for j in range(2) for i in range(2))

    FB = (0.5 * phi1 * d1[4] + 0.5 * phi2 * d2[4] + 0.5 * phi1 * c12a
          + 0.25 * phi1 ** 4 - 0.5 * p["eps"] * phi1 ** 2
          + 0.25 * phi2 ** 4 - 0.5 * p["eps"] * phi2 ** 2
          + p["eta"] / 3.0 * (np.abs(phi1) ** 3 + np.abs(phi2) ** 3
                              - phi1 ** 3 - phi2 ** 3)
          + 0.5 * p["gamma12"] * phi1 ** 2 * phi2 ** 2)
    FGL = (0.5 * p["omega0"] * gm2 - 0.5 * p["alpha"] * m2
           + 0.25 * p["beta"] * m2 ** 2 - (M[0] * H[0] + M[1] * H[1])
           - p["gamma1"] * m2 * phi1 - p["gamma2"] * m2 * phi2
           - 0.5 * p["eta1"] * mg1 ** 2 - 0.5 * p["eta2"] * mg2 ** 2)
    FR = (p["theta"] / 6.0 * m2 ** 3
          + 0.25 * p["theta1"] * (g1[0] ** 2 + g1[1] ** 2) ** 2
          + 0.25 * p["theta2"] * (g2[0] ** 2 + g2[1] ** 2) ** 2)
    dv = np.prod(L) / phi1.size
    return float(np.sum(FB + FGL + FR) * dv)

import numpy as np

def pfc_nonlinear_terms(phi1, phi2, M, cell, params=None, H=None):
    p = _check_params(params)
    phi1 = np.asarray(phi1, float)
    phi2 = np.asarray(phi2, float)
    M = np.asarray(M, float)
    if phi1.ndim != 2 or phi2.shape != phi1.shape:
        raise ValueError("phi1 and phi2 must be two-dimensional fields of equal shape")
    if M.shape != (2,) + phi1.shape:
        raise ValueError("M must have shape (2,) + phi1.shape")
    L = _check_cell(cell, phi1.shape)
    H = np.zeros_like(M) if H is None else np.asarray(H, float)
    if H.shape != M.shape:
        raise ValueError("H must have the same shape as M")
    if not all(np.all(np.isfinite(v)) for v in (phi1, phi2, M, H)):
        raise ValueError("phi1, phi2, M and H must be finite")

    g1 = pfc_spectral_derivatives(phi1, L)[:2]
    g2 = pfc_spectral_derivatives(phi2, L)[:2]
    m2 = M[0] ** 2 + M[1] ** 2
    mg1 = M[0] * g1[0] + M[1] * g1[1]
    mg2 = M[0] * g2[0] + M[1] * g2[1]
    s1 = g1[0] ** 2 + g1[1] ** 2
    s2 = g2[0] ** 2 + g2[1] ** 2

    def div(vx, vy):
        return (pfc_spectral_derivatives(vx, L)[0]
                + pfc_spectral_derivatives(vy, L)[1])

    N = (0.25 * phi1 ** 4 - 0.5 * p["eps"] * phi1 ** 2
         + 0.25 * phi2 ** 4 - 0.5 * p["eps"] * phi2 ** 2
         + p["eta"] / 3.0 * (np.abs(phi1) ** 3 + np.abs(phi2) ** 3
                             - phi1 ** 3 - phi2 ** 3)
         + 0.5 * p["gamma12"] * phi1 ** 2 * phi2 ** 2
         - 0.5 * p["alpha"] * m2 + 0.25 * p["beta"] * m2 ** 2
         - (M[0] * H[0] + M[1] * H[1])
         - p["gamma1"] * m2 * phi1 - p["gamma2"] * m2 * phi2
         - 0.5 * p["eta1"] * mg1 ** 2 - 0.5 * p["eta2"] * mg2 ** 2
         + p["theta"] / 6.0 * m2 ** 3
         + 0.25 * p["theta1"] * s1 ** 2 + 0.25 * p["theta2"] * s2 ** 2)

    N1 = (phi1 ** 3 - p["eps"] * phi1
          + p["eta"] * (np.abs(phi1) - phi1) * phi1
          + p["gamma12"] * phi1 * phi2 ** 2
          + p["eta1"] * div(mg1 * M[0], mg1 * M[1])
          - p["gamma1"] * m2
          - p["theta1"] * div(s1 * g1[0], s1 * g1[1]))
    N2 = (phi2 ** 3 - p["eps"] * phi2
          + p["eta"] * (np.abs(phi2) - phi2) * phi2
          + p["gamma12"] * phi1 ** 2 * phi2
          + p["eta2"] * div(mg2 * M[0], mg2 * M[1])
          - p["gamma2"] * m2
          - p["theta2"] * div(s2 * g2[0], s2 * g2[1]))
    N3 = (-p["alpha"] * M + p["beta"] * m2 * M - H
          - 2.0 * p["gamma1"] * M * phi1 - 2.0 * p["gamma2"] * M * phi2
          - p["eta1"] * mg1 * g1 - p["eta2"] * mg2 * g2
          + p["theta"] * m2 ** 2 * M)
    return np.stack([N, N1, N2, N3[0], N3[1]])

import numpy as np

def pfc_ieq_coefficients(phi1, phi2, M, cell, params=None, H=None):
    p = _check_params(params)
    phi1 = np.asarray(phi1, float)
    phi2 = np.asarray(phi2, float)
    M = np.asarray(M, float)
    L = _check_cell(cell, phi1.shape)
    T = pfc_nonlinear_terms(phi1, phi2, M, L, params, H)
    N, N1, N2 = T[0], T[1], T[2]
    N3 = T[3:5]
    m2 = M[0] ** 2 + M[1] ** 2
    rad = (0.5 * p["a1"] * phi1 ** 2 + 0.5 * p["a2"] * phi2 ** 2 + N
           - 0.5 * p["S_phi"] * phi1 ** 2 - 0.5 * p["S_phi"] * phi2 ** 2
           - 0.5 * p["S_m"] * m2 + p["B"])
    if not np.all(np.isfinite(rad)) or np.any(rad <= 0.0):
        raise ValueError("the IEQ radicand must be finite and strictly positive")
    U = np.sqrt(rad)
    H1 = (p["a1"] * phi1 - p["S_phi"] * phi1 + N1) / U
    H2 = (p["a2"] * phi2 - p["S_phi"] * phi2 + N2) / U
    R = (-p["S_m"] * M + N3) / U
    return np.stack([U, H1, H2, R[0], R[1]])


def _pfc_initial_fields(grid, cell):
    Nx, Ny = int(grid[0]), int(grid[1])
    Lx, Ly = float(cell[0]), float(cell[1])
    x = np.arange(Nx) * (Lx / Nx)
    y = np.arange(Ny) * (Ly / Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx, ky = 2.0 * np.pi / Lx, 2.0 * np.pi / Ly
    phi1 = np.cos(8.0 * kx * X) * np.sin(8.0 * ky * Y)
    phi2 = np.cos(8.0 * kx * X) * np.cos(8.0 * ky * Y)
    M = np.stack([np.sin(2.0 * kx * X) * np.sin(2.0 * ky * Y),
                  np.cos(2.0 * kx * X) * np.cos(2.0 * ky * Y)])
    Hf = np.stack([np.sin(2.0 * kx * X) * np.cos(ky * Y),
                   np.cos(ky * Y)])
    return phi1, phi2, M, Hf


def _pfc_initial_state(grid, cell, dt, params=None):
    """Initial layer, the two backward ghost layers and Q^0 = 1."""
    p = _check_params(params)
    L = _check_cell(cell, (int(grid[0]), int(grid[1])))
    phi1, phi2, M, Hf = _pfc_initial_fields(grid, L)
    C = pfc_ieq_coefficients(phi1, phi2, M, L, params, Hf)
    U, H1, H2 = C[0], C[1], C[2]
    R = C[3:5]
    d1 = pfc_spectral_derivatives(phi1, L, p["a12"])
    d2 = pfc_spectral_derivatives(phi2, L, p["a12"])
    lapM = np.stack([pfc_spectral_derivatives(M[j], L)[2] for j in range(2)])
    mu1 = d1[3] + 2.0 * p["a1"] * d1[2] + 0.5 * d2[4] + p["S_phi"] * phi1 + H1 * U
    mu2 = d2[3] + 2.0 * p["a2"] * d2[2] + 0.5 * d1[4] + p["S_phi"] * phi2 + H2 * U
    mu3 = -p["omega0"] * lapM + p["S_m"] * M + R * U
    v1 = -p["M_phi"] * (mu1 - np.mean(mu1))
    v2 = -p["M_phi"] * (mu2 - np.mean(mu2))
    vM = -p["M_m"] * mu3
    vU = 0.5 * (H1 * v1 + H2 * v2 + R[0] * vM[0] + R[1] * vM[1])
    layer = np.stack([phi1, phi2, M[0], M[1], mu1, mu2, mu3[0], mu3[1], U])
    z = np.zeros_like(phi1)
    vel = np.stack([v1, v2, vM[0], vM[1], z, z, z, z, vU])
    return np.stack([layer, layer - dt * vel, layer - 2.0 * dt * vel]), Hf

import numpy as np

def pfc_split_solve(G13, G24, sign, cell, dt, params=None):
    p = _check_params(params)
    G13 = np.asarray(G13, float)
    G24 = np.asarray(G24, float)
    if G13.ndim != 2 or G24.shape != G13.shape:
        raise ValueError("G13 and G24 must be two-dimensional fields of equal shape")
    if not (np.all(np.isfinite(G13)) and np.all(np.isfinite(G24))):
        raise ValueError("the right-hand sides must be finite")
    s = float(sign)
    if s not in (1.0, -1.0):
        raise ValueError("sign must be +1 or -1")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    L = _check_cell(cell, G13.shape)
    a12, Sp, Mp = p["a12"], p["S_phi"], p["M_phi"]
    k2 = _k2(G13.shape, L)
    n = G13.size

    mean13 = float(np.sum(G13)) / n
    mean24 = float(np.sum(G24)) / n
    pbar = Mp * dt * mean13
    rhs = (G13 - 0.5 * (G24 - mean24)
           + 0.5 * (Sp + 0.5 * s * a12 ** 2) * pbar)
    A = 1.0 / (Mp * dt) + 0.5 * k2 ** 2 + 0.5 * Sp + 0.25 * s * (a12 - k2) ** 2
    if np.any(np.abs(A) < 1.0e-12 * np.max(np.abs(A))):
        raise ValueError("the split operator is singular for this parameter set")
    psi = np.real(np.fft.ifft2(np.fft.fft2(rhs) / A))
    d = pfc_spectral_derivatives(psi, L, a12)
    mu = d[3] + Sp * psi + 0.5 * s * d[4] + G24
    return np.stack([psi, mu])

import numpy as np

def pfc_time_step(state, q_n, cell, dt, params=None, H=None):
    p = _check_params(params)
    S = np.asarray(state, float)
    if S.ndim != 4 or S.shape[0] != 3 or S.shape[1] != 9:
        raise ValueError("state must have shape (3, 9, Nx, Ny)")
    if not np.all(np.isfinite(S)):
        raise ValueError("state must be finite")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    q_n = float(q_n)
    if not np.isfinite(q_n):
        raise ValueError("q_n must be a finite scalar")
    grid = S.shape[2:]
    L = _check_cell(cell, grid)
    H = np.zeros((2,) + grid) if H is None else np.asarray(H, float)
    if H.shape != (2,) + grid:
        raise ValueError("H must have shape (2,) + grid")

    Mp, Mm = p["M_phi"], p["M_m"]
    Sp, Sm, a1, a2, a12, w0 = (p["S_phi"], p["S_m"], p["a1"], p["a2"],
                               p["a12"], p["omega0"])

    n0, n1, n2 = S[0], S[1], S[2]          # layers n, n-1, n-2
    p1n, p2n = n0[0], n0[1]
    Mn = n0[2:4]
    mu1n, mu2n = n0[4], n0[5]
    mu3n = n0[6:8]
    Un = n0[8]

    # second-order extrapolations
    p1s = 1.5 * p1n - 0.5 * n1[0]
    p2s = 1.5 * p2n - 0.5 * n1[1]
    Ms = 1.5 * Mn - 0.5 * n1[2:4]
    Us = 1.5 * Un - 0.5 * n1[8]
    p1ts = (2.0 * p1n - 3.0 * n1[0] + n2[0]) / dt
    p2ts = (2.0 * p2n - 3.0 * n1[1] + n2[1]) / dt
    Mts = (2.0 * Mn - 3.0 * n1[2:4] + n2[2:4]) / dt

    C = pfc_ieq_coefficients(p1s, p2s, Ms, L, params, H)
    H1s, H2s = C[1], C[2]
    Rs = C[3:5]

    d1n = pfc_spectral_derivatives(p1n, L, a12)
    d2n = pfc_spectral_derivatives(p2n, L, a12)
    lap_p1s = pfc_spectral_derivatives(p1s, L)[2]
    lap_p2s = pfc_spectral_derivatives(p2s, L)[2]
    lapM = np.stack([pfc_spectral_derivatives(Mn[j], L)[2]
                     for j in range(2)])

    nsite = p1n.size
    G1 = p1n / (Mp * dt) - 0.5 * (mu1n - np.sum(mu1n) / nsite)
    G3 = p2n / (Mp * dt) - 0.5 * (mu2n - np.sum(mu2n) / nsite)
    G2 = -mu1n + d1n[3] + 0.5 * d2n[4] + Sp * p1n + 4.0 * a1 * lap_p1s
    G4 = -mu2n + d2n[3] + 0.5 * d1n[4] + Sp * p2n + 4.0 * a2 * lap_p2s
    G5 = Mn / (Mm * dt) - 0.5 * mu3n
    G6 = -mu3n - w0 * lapM + Sm * Mn

    a = pfc_split_solve(G1 + G3, G2 + G4, +1.0, L, dt, params)
    b = pfc_split_solve(G1 - G3, G2 - G4, -1.0, L, dt, params)
    p11, mu11 = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
    p21, mu21 = 0.5 * (a[0] - b[0]), 0.5 * (a[1] - b[1])

    zero = np.zeros(grid)
    c = pfc_split_solve(zero, 2.0 * (H1s + H2s) * Us, +1.0, L, dt, params)
    d = pfc_split_solve(zero, 2.0 * (H1s - H2s) * Us, -1.0, L, dt, params)
    p12, mu12 = 0.5 * (c[0] + d[0]), 0.5 * (c[1] + d[1])
    p22, mu22 = 0.5 * (c[0] - d[0]), 0.5 * (c[1] - d[1])

    k2 = _k2(grid, L)
    AM = 1.0 / (Mm * dt) + 0.5 * (w0 * k2 + Sm)

    def msolve(rhs):
        return np.stack([np.real(np.fft.ifft2(np.fft.fft2(rhs[j]) / AM))
                         for j in range(2)])

    M1 = msolve(G5 - 0.5 * G6)
    M2 = msolve(-Rs * Us)
    lapM1 = np.stack([pfc_spectral_derivatives(M1[j], L)[2] for j in range(2)])
    lapM2 = np.stack([pfc_spectral_derivatives(M2[j], L)[2] for j in range(2)])
    mu31 = -w0 * lapM1 + Sm * M1 + G6
    mu32 = -w0 * lapM2 + Sm * M2 + 2.0 * Rs * Us

    U1 = Un
    U2 = dt * 0.5 * (H1s * p1ts + H2s * p2ts + Rs[0] * Mts[0] + Rs[1] * Mts[1])

    RM = Rs[0] * Mts[0] + Rs[1] * Mts[1]
    xi1 = (_ip(H1s * Us, (p11 - p1n) / dt, L) - _ip(H1s * p1ts, Un, L)
           + _ip(H2s * Us, (p21 - p2n) / dt, L) - _ip(H2s * p2ts, Un, L)
           + _ip(Rs * Us, (M1 - Mn) / dt, L) - _ip(RM, Un, L))
    xi2 = (_ip(H1s * Us, p12 / dt, L) - _ip(H1s * p1ts, 0.5 * U2, L)
           + _ip(H2s * Us, p22 / dt, L) - _ip(H2s * p2ts, 0.5 * U2, L)
           + _ip(Rs * Us, M2 / dt, L) - _ip(RM, 0.5 * U2, L))

    den = 2.0 / dt - xi2
    if den == 0.0:
        raise ValueError("the ZEC scalar equation is singular")
    qbar = (2.0 / dt * q_n + xi1) / den
    q_new = 2.0 * qbar - q_n

    out = np.empty((10,) + grid)
    out[0] = p11 + qbar * p12
    out[1] = p21 + qbar * p22
    out[2:4] = M1 + qbar * M2
    out[4] = mu11 + qbar * mu12
    out[5] = mu21 + qbar * mu22
    out[6:8] = mu31 + qbar * mu32
    out[8] = U1 + qbar * U2
    out[9] = q_new
    return out

import numpy as np

def pfc_modified_energy(phi1, phi2, M, U, q, cell, params=None,
                                phi1_prev=None, phi2_prev=None):
    p = _check_params(params)
    phi1 = np.asarray(phi1, float)
    phi2 = np.asarray(phi2, float)
    M = np.asarray(M, float)
    U = np.asarray(U, float)
    if phi1.ndim != 2 or phi2.shape != phi1.shape or U.shape != phi1.shape:
        raise ValueError("phi1, phi2 and U must be two-dimensional fields of equal shape")
    if M.shape != (2,) + phi1.shape:
        raise ValueError("M must have shape (2,) + phi1.shape")
    L = _check_cell(cell, phi1.shape)
    try:
        if np.ndim(q) != 0:
            raise ValueError("q must be a finite scalar")
        q = float(q)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("q must be a finite scalar") from exc
    if not np.isfinite(q):
        raise ValueError("q must be a finite scalar")
    p1p = phi1 if phi1_prev is None else np.asarray(phi1_prev, float)
    p2p = phi2 if phi2_prev is None else np.asarray(phi2_prev, float)
    if p1p.shape != phi1.shape or p2p.shape != phi1.shape:
        raise ValueError("the previous-layer fields must match phi1.shape")
    if not all(np.all(np.isfinite(v)) for v in (phi1, phi2, M, U, p1p, p2p)):
        raise ValueError("all current and previous fields must be finite")

    def gsq(u):
        """Squared gradient norm in the Galerkin sense, -(u, Laplacian u)."""
        return -_ip(u, pfc_spectral_derivatives(u, L)[2], L)

    d1 = pfc_spectral_derivatives(phi1, L, p["a12"])
    d2 = pfc_spectral_derivatives(phi2, L, p["a12"])

    E = (0.5 * _ip(d1[2], d1[2], L) + 0.5 * _ip(d2[2], d2[2], L)
         + 0.5 * _ip(d1[4], phi2, L)
         - p["a1"] * gsq(phi1) - p["a2"] * gsq(phi2)
         + 0.5 * p["S_phi"] * _ip(phi1, phi1, L)
         + 0.5 * p["S_phi"] * _ip(phi2, phi2, L)
         + 0.5 * p["omega0"] * (gsq(M[0]) + gsq(M[1]))
         + 0.5 * p["S_m"] * (_ip(M[0], M[0], L) + _ip(M[1], M[1], L))
         + _ip(U, U, L) + 0.5 * q ** 2 - p["B"] * float(np.prod(L)) - 0.5
         + 0.5 * p["a1"] * gsq(phi1 - p1p) + 0.5 * p["a2"] * gsq(phi2 - p2p))
    return float(E)

import numpy as np

def pfc_ieq_zec_run(n_steps=10, dt=2.0e-3, grid=(64, 64), cell=(64.0, 64.0), params=None, quantity="energy"):
    try:
        if np.ndim(n_steps) != 0:
            raise ValueError("n_steps must be a non-negative integer")
        if isinstance(n_steps, (int, np.integer)):
            n_steps = int(n_steps)
        else:
            value = float(n_steps)
            if not np.isfinite(value) or not value.is_integer():
                raise ValueError("n_steps must be a non-negative integer")
            n_steps = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("n_steps must be a non-negative integer") from exc
    if n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    try:
        if np.ndim(dt) != 0:
            raise ValueError("dt must be a finite positive scalar")
        dt = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt must be a finite positive scalar") from exc
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    if str(quantity) not in ("energy", "energy0", "emod", "emod0", "q",
                             "mass", "drift"):
        raise ValueError("quantity must be one of 'energy', 'energy0', 'emod', "
                         "'emod0', 'q', 'mass', 'drift'")
    try:
        dimensions = np.asarray(grid, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("grid must contain two finite integer dimensions") from exc
    if (dimensions.shape != (2,) or not np.all(np.isfinite(dimensions))
            or np.any(dimensions < 4) or np.any(dimensions != np.floor(dimensions))):
        raise ValueError("grid must contain two integer dimensions of at least four")
    grid = (int(dimensions[0]), int(dimensions[1]))
    L = _check_cell(cell, grid)

    S, Hf = _pfc_initial_state(grid, L, dt, params)
    phi1, phi2 = S[0][0].copy(), S[0][1].copy()
    q = 1.0
    area = float(np.prod(L))
    if str(quantity) == "energy0":
        return pfc_free_energy(phi1, phi2, S[0][2:4], L, params, Hf) / area
    if str(quantity) == "emod0":
        return pfc_modified_energy(phi1, phi2, S[0][2:4], S[0][8], q, L,
                                           params) / area

    prev1, prev2 = phi1, phi2
    for _ in range(int(n_steps)):
        out = pfc_time_step(S, q, L, dt, params, Hf)
        q = float(out[9].flat[0])
        prev1, prev2 = S[0][0], S[0][1]
        S = np.stack([out[:9], S[0], S[1]])

    f1, f2 = S[0][0], S[0][1]
    Mf, Uf = S[0][2:4], S[0][8]
    if str(quantity) == "emod":
        return pfc_modified_energy(f1, f2, Mf, Uf, q, L, params,
                                           prev1, prev2) / area
    if str(quantity) == "q":
        return float(q)
    if str(quantity) == "mass":
        return float(abs(np.mean(f1) - np.mean(phi1))
                     + abs(np.mean(f2) - np.mean(phi2)))
    if str(quantity) == "drift":
        Ue = pfc_ieq_coefficients(f1, f2, Mf, L, params, Hf)[0]
        return float(np.max(np.abs(Uf - Ue)))
    return pfc_free_energy(f1, f2, Mf, L, params, Hf) / area
SCICODE_GOLD_EOF
