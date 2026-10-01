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
from math import comb
from typing import Callable
from scipy.special import ellipk, ellipe
from numpy.polynomial.legendre import leggauss


def _check_scalar(value, name, positive=False, nonneg=False):
    """Return value as float, raising ValueError unless it is a finite real number with the requested sign."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(name + " must be finite")
    if positive and value <= 0.0:
        raise ValueError(name + " must be positive")
    if nonneg and value < 0.0:
        raise ValueError(name + " must be non-negative")
    return value


def _check_k_array(k, name):
    """Return k as a one-dimensional float array of finite non-negative values, else raise ValueError."""
    k = np.asarray(k, dtype=float)
    if k.ndim != 1 or k.size == 0 or not np.all(np.isfinite(k)) or np.any(k < 0.0):
        raise ValueError(name + " must be a one-dimensional array of finite non-negative numbers")
    return k


def _grid(_GRID={}):
    """Composite Gauss-Legendre radial grid: [0, 2^-10] then doubling panels up to 2^20, 16 nodes per panel (cached)."""
    if "k" not in _GRID:
        x, wref = leggauss(16)
        edges = [0.0] + [2.0 ** p for p in range(-10, 21)]
        ks, ws = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            h = (b - a) / 2
            ks.append(a + h * (x + 1))
            ws.append(wref * h)
        _GRID.update(k=np.concatenate(ks), w=np.concatenate(ws), edges=np.array(edges), x=x, wref=wref,
                     C=np.linalg.inv(np.vander(x, 16, increasing=True)))
    return _GRID


def _log_moments(ui, n):
    """mu_p = int_{-1}^{1} u^p ln|u - ui| du for p < n, in closed form."""
    def _F(x, q):
        return 0.0 if x == 0 else x ** (q + 1) * (np.log(abs(x)) - 1.0 / (q + 1)) / (q + 1)
    hi, lo = 1.0 - ui, -1.0 - ui
    base = [_F(hi, q) - _F(lo, q) for q in range(n)]
    return np.array([sum(comb(p, q) * ui ** (p - q) * base[q] for q in range(p + 1)) for p in range(n)])


def _monolayer_rows(kp, k):
    """Angular average of 4 pi/|k - k'| times k'/(2 pi) for rows kp and nodes k: kappa = 4k'K(m)/(pi s), s = k + k',
    m = 4kk'/s^2, together with the analytic coefficient c = 8k'P4(1 - m)/(pi^2 s) of its ln|k - k'| singularity
    (P4 is the fourth-order Taylor polynomial of K at zero argument). Coincident points get their limiting values."""
    K1, K2 = np.meshgrid(kp, k, indexing="ij")
    s = K1 + K2
    m = 4 * K1 * K2 / s ** 2
    m1 = 1 - m
    with np.errstate(divide="ignore", invalid="ignore"):
        kap = 4 * K2 * ellipk(m) / (np.pi * s)
    series = (1.0, 1 / 4, 9 / 64, 25 / 256, 1225 / 16384)
    poly = (np.pi / 2) * sum(series[j] * m1 ** j for j in range(5))
    c = 8 * K2 * poly / (np.pi ** 2 * s)
    same = np.abs(K1 - K2) <= 1e-14 * np.maximum(K1, 1e-300)
    kap = np.where(same, (2 / np.pi) * np.log(8 * np.maximum(K1, 1e-300)), kap)
    c = np.where(same, 2 / np.pi, c)
    return np.where(np.isfinite(kap), kap, 0.0), np.where(np.isfinite(c), c, 0.0), same


def _bilayer_rows(kp, k, d, kap0, same, npsi=48):
    """Interlayer correction kappa_d - kappa_0 = (k'/pi) int dtheta (exp(-d q) - 1)/q with q = |k - k'|: the powers of q
    up to q^4 are subtracted and integrated in closed form (angular moments in K and E), the remainder is integrated by
    Gauss-Legendre in psi with q^2 = (k - k')^2 + 4kk' sin^2(psi). Also returns the coefficient of the ln|k - k'| term."""
    K1, K2 = np.meshgrid(kp, k, indexing="ij")
    s = K1 + K2
    a = K1 ** 2 + K2 ** 2
    b = 2 * K1 * K2
    m = 4 * K1 * K2 / s ** 2
    qm = np.abs(K1 - K2)
    nsub = np.where(d * s <= 12.0, 5, np.where(d * s <= 40.0, 3, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        E = ellipe(m)
        K = ellipk(m)
    Kf = np.where(np.isfinite(K), K, 0.0)
    I1 = 4 * s * E
    I2 = 2 * np.pi * a
    I3 = (4 * s / 3) * (4 * a * E - (K1 - K2) ** 2 * Kf)
    I4 = 2 * np.pi * a ** 2 + np.pi * b ** 2
    analytic = (-d * 2 * np.pi + np.where(nsub >= 3, d ** 2 / 2 * I1 - d ** 3 / 6 * I2, 0.0)
                + np.where(nsub >= 5, d ** 4 / 24 * I3 - d ** 5 / 120 * I4, 0.0))
    xp, wp = leggauss(npsi)
    psi = (np.pi / 4) * (xp + 1)
    wpsi = wp * (np.pi / 4)
    acc = np.zeros_like(K1)
    for p_, wq in zip(psi, wpsi):
        q = np.sqrt(qm ** 2 + 2 * b * np.sin(p_) ** 2)
        dq = d * q
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r1 = (np.exp(-dq) - 1 + dq) / q
            r3 = r1 + (-dq ** 2 / 2 + dq ** 3 / 6) / q
            r5 = r3 + (-dq ** 4 / 24 + dq ** 5 / 120) / q
        rem = np.where(nsub == 5, r5, np.where(nsub == 3, r3, r1))
        acc += wq * np.where(q > 0, rem, 0.0)
    reg = (K2 / np.pi) * (analytic + 4 * acc)
    reg = np.where((d * qm > 40.0) & ~same, -kap0, reg)
    m1 = 1 - m
    with np.errstate(divide="ignore", invalid="ignore"):
        Km1 = ellipk(m1)
        Em1 = ellipe(m1)
        Kp = (Em1 - (1 - m1) * Km1) / (2 * m1 * (1 - m1))
    Kp = np.where(m1 > 1e-12, Kp, np.pi / 8)
    gE = (Km1 * m1 - 2 * (1 - m1) * m1 * Kp) / np.pi
    creg = (K2 / np.pi) * s * (np.where(nsub >= 3, 4 * d ** 2 * gE, 0.0)
                               + np.where(nsub >= 5, (d ** 4 / 9) * (4 * a * gE - (K1 - K2) ** 2 * Km1 / np.pi), 0.0))
    return np.where(np.isfinite(reg), reg, 0.0), np.where(np.isfinite(creg), creg, 0.0)


def _nystrom_rows(kp, kern, coef, near=0.35):
    """Quadrature rows for a kernel kern(k, k') = -coef(k, k') ln|k - k'| + smooth on the grid nodes: plain Gauss-Legendre
    weights away from the singular point, exact log moments of the Lagrange basis on the panel containing k and on a
    neighbour panel whenever k lies within `near` panel widths of the shared edge."""
    g = _grid()
    k, w, edges, wref, C = g["k"], g["w"], g["edges"], g["wref"], g["C"]
    n = 16
    P = len(edges) - 1
    Mm = kern * w[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = kern + coef * np.log(np.abs(kp[:, None] - k[None, :]))
    R = np.where(np.isfinite(R), R, kern)
    for i, ki in enumerate(kp):
        p = int(min(np.searchsorted(edges, ki, side="right") - 1, P - 1))
        a, b = edges[p], edges[p + 1]
        panels = [p]
        if p > 0 and (ki - a) < near * (b - a):
            panels.append(p - 1)
        if p < P - 1 and (b - ki) < near * (b - a):
            panels.append(p + 1)
        for pj in panels:
            aj, bj = edges[pj], edges[pj + 1]
            hj = (bj - aj) / 2
            ui = (ki - (aj + hj)) / hj
            Wlog = hj * (C.T @ _log_moments(ui, n)) + hj * np.log(hj) * wref
            sl = slice(pj * n, (pj + 1) * n)
            Mm[i, sl] = w[sl] * R[i, sl] - coef[i, sl] * Wlog
    return Mm


def _rows(kp, d):
    """Rows of the discretised exchange operator V_d for evaluation points kp (each >= 2^-10) over the full grid."""
    k = _grid()["k"]
    kap0, c0, same = _monolayer_rows(kp, k)
    Mm = _nystrom_rows(kp, kap0, c0)
    if d > 0:
        reg, creg = _bilayer_rows(kp, k, d, kap0, same)
        Mm = Mm + _nystrom_rows(kp, reg, creg)
    return Mm


def _operator(d, _KERNELS={}):
    """Square discretised operator on the trusted nodes (k >= 2^-10). The innermost panel is integrated over, its
    unknowns being represented by an even-polynomial extrapolation from the trusted nodes below 0.1. Returns a dict with
    the trusted nodes k, the measure mu for int d^2k/(2 pi)^2, and the matrix M with (V_d f)_i = sum_j M_ij f_j."""
    key = round(float(d), 12)
    if key in _KERNELS:
        return _KERNELS[key]
    g = _grid()
    k, w = g["k"], g["w"]
    n = 16
    N = len(k)
    M = _rows(k[n:], d)
    J = np.arange(n)
    T = np.arange(n, N)
    kT = k[T]
    fit = T[kT < 0.1][:48]
    P = np.linalg.pinv(np.vander(k[fit] ** 2, 4, increasing=True))
    L = np.zeros((n, N - n))
    L[:, fit - n] = np.vander(k[J] ** 2, 4, increasing=True) @ P
    Mred = M[:, T] + M[:, J] @ L
    mu = w * k / (2 * np.pi)
    op = dict(k=kT, mu=mu[T] + L.T @ mu[J], M=Mred, L=L, fit=fit - n)
    _KERNELS[key] = op
    return op


def _even_fit(k, f, kmax=0.1, deg=3):
    """Least-squares coefficients of c0 + c1 k^2 + ... + c_deg k^(2 deg) through the samples with k < kmax."""
    sel = k < kmax
    X = np.vander(k[sel] ** 2, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(X, f[sel], rcond=None)
    return c


def _apply_to_grid_values(fvals, k_eval, d):
    """(V_d f)(k_eval) from values of f on the full grid: direct quadrature rows for k_eval >= 2^-10, and an even
    polynomial fitted to the operator's values on the trusted nodes below 0.1 for smaller k_eval."""
    g = _grid()
    out = np.empty_like(k_eval)
    big = k_eval >= 2.0 ** -10
    if np.any(big):
        out[big] = _rows(k_eval[big], d) @ fvals
    if np.any(~big):
        kf = g["k"][16:][g["k"][16:] < 0.1]
        c = _even_fit(kf, _rows(kf, d) @ fvals)
        out[~big] = np.polynomial.polynomial.polyval(k_eval[~big] ** 2, c)
    return out


def exchange_integral(f: Callable, k_eval: np.ndarray, d: float) -> np.ndarray:
    """Reference implementation."""
    d = _check_scalar(d, "d", nonneg=True)
    k_eval = _check_k_array(k_eval, "k_eval")
    k = _grid()["k"]
    fvals = np.asarray(f(k.copy()), dtype=float)
    if fvals.shape != k.shape or not np.all(np.isfinite(fvals)):
        raise ValueError("f must return finite values with the shape of its argument")
    return _apply_to_grid_values(fvals, k_eval, d)

import numpy as np
from scipy.linalg import eigh


def _exciton(d, _EXCITONS={}):
    """Lowest eigenpair of k^2 - V_d on the trusted nodes: (E_b, phi on trusted nodes), phi normalised and positive."""
    key = round(float(d), 12)
    if key in _EXCITONS:
        return _EXCITONS[key]
    op = _operator(d)
    k = op["k"]
    A = np.diag(k * k) - op["M"]
    # the operator is symmetric under the measure up to quadrature error: start from the symmetrised problem,
    # then polish the lowest eigenpair of the actual matrix by inverse iteration with Rayleigh-quotient shifts
    sq = np.sqrt(op["mu"])
    S = A * (sq[:, None] / sq[None, :])
    vals, vecs = eigh(0.5 * (S + S.T))
    lam = vals[0]
    phi = vecs[:, 0] / sq
    for _ in range(3):
        v = np.linalg.solve(A - lam * np.eye(len(k)), phi)
        v /= np.sqrt(np.sum(v * v))
        lam = (v @ (A @ v)) / (v @ v)
        phi = v
    phi = phi / np.sqrt(np.sum(op["mu"] * phi * phi))
    if np.sum(phi[k < 0.5]) < 0:
        phi = -phi
    _EXCITONS[key] = (-lam, phi)
    return _EXCITONS[key]


def _full_values(op, fT):
    """Values of a trusted-node vector on the full grid, the innermost panel filled by the operator's extrapolation."""
    return np.concatenate([op["L"] @ fT, fT])


def exciton_state(d: float, k_out: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    d = _check_scalar(d, "d", nonneg=True)
    k_out = _check_k_array(k_out, "k_out")
    Eb, phi = _exciton(d)
    op = _operator(d)
    # Nystrom interpolation: phi(k) = (V_d phi)(k) / (k^2 + E_b)
    vphi = _apply_to_grid_values(_full_values(op, phi), k_out, d)
    return np.concatenate([[Eb], vphi / (k_out ** 2 + Eb)])

import numpy as np


def inverse_compressibility(d: float) -> float:
    """Reference implementation: Eq. (A8) of the source, 1/C_G - 2 [<phi^2, V_0 phi^2> - <phi^3, V_d phi>]."""
    d = _check_scalar(d, "d", nonneg=True)
    op = _operator(d)
    op0 = _operator(0.0)
    Eb, phi = _exciton(d)
    intra = np.sum(op["mu"] * phi ** 2 * (op0["M"] @ phi ** 2))
    inter = np.sum(op["mu"] * phi ** 3 * (op["M"] @ phi))
    return float(8 * np.pi * d - 2 * (intra - inter))

import numpy as np


def _hf(EG, d, r, start=None, tol=1e-13, maxit=400, m_hist=6, beta=0.5, _HF={}):
    """Anderson-accelerated fixed-point solution of Eq. (20) on the trusted nodes for the pair (v^2, uv)."""
    key = (round(float(EG), 12), round(float(d), 12), round(float(r), 12))
    if key in _HF and start is None:
        return _HF[key]
    op = _operator(d)
    op0 = _operator(0.0)
    k, mu = op["k"], op["mu"]
    N = len(k)
    Eb, phi = _exciton(d)
    dmu = inverse_compressibility(d)
    hs = 0.5 * k * k * (r - 1) / (r + 1)
    if EG >= Eb:
        z = np.zeros(N)
        Eq = 0.5 * (k * k + EG)
        out = dict(k=k, v2=z, uv=z, F=z, D=z, xi=Eq, Eq=Eq, n=0.0, Ep=hs + Eq, Em=hs - Eq, EG=EG, d=d, r=r, Eb=Eb, dmu=dmu, trivial=True)
        _HF[key] = out
        return out
    if start is None:
        lam = np.sqrt(max((Eb - EG) / dmu, 1e-6))
        x = np.concatenate([np.minimum(lam ** 2 * phi ** 2, 0.5), lam * phi])
    else:
        x = np.concatenate([start["v2"], start["uv"]])

    def _step(x):
        v2, uv = x[:N], x[N:]
        n = np.sum(mu * v2)
        F = 2 * (op0["M"] @ v2)
        D = op["M"] @ uv
        xi = 0.5 * (k * k + EG + 8 * np.pi * d * n - F)
        Eq = np.sqrt(xi * xi + D * D)
        return np.concatenate([0.5 * (1 - xi / Eq), D / (2 * Eq)])

    X, Fh = [], []
    err = np.inf
    for it in range(maxit):
        gx = _step(x)
        f = gx - x
        err = np.max(np.abs(f))
        if err < tol:
            x = gx
            break
        X.append(x.copy())
        Fh.append(f.copy())
        if len(X) > m_hist:
            X.pop(0)
            Fh.pop(0)
        if len(X) >= 2:
            dF = np.array([Fh[j + 1] - Fh[j] for j in range(len(Fh) - 1)]).T
            dX = np.array([X[j + 1] - X[j] for j in range(len(X) - 1)]).T
            gamma, *_ = np.linalg.lstsq(dF, f, rcond=None)
            x = x + beta * f - (dX + beta * dF) @ gamma
        else:
            x = x + beta * f
    if not err < tol:
        raise ValueError("the self-consistent iteration did not converge")
    v2, uv = x[:N], x[N:]
    n = np.sum(mu * v2)
    F = 2 * (op0["M"] @ v2)
    D = op["M"] @ uv
    xi = 0.5 * (k * k + EG + 8 * np.pi * d * n - F)
    Eq = np.sqrt(xi * xi + D * D)
    out = dict(k=k, v2=v2, uv=uv, F=F, D=D, xi=xi, Eq=Eq, n=n, Ep=hs + Eq, Em=hs - Eq, EG=EG, d=d, r=r, Eb=Eb, dmu=dmu, trivial=False)
    if start is None:
        _HF[key] = out
    return out


def _state_at(s, k_out):
    """v^2, uv, E+, E- of a converged state at arbitrary magnitudes, from the self-energies evaluated there."""
    d, r, EG = s["d"], s["r"], s["EG"]
    hs = 0.5 * k_out * k_out * (r - 1) / (r + 1)
    if s["trivial"]:
        Eq = 0.5 * (k_out * k_out + EG)
        z = np.zeros_like(k_out)
        return np.vstack([z, z, hs + Eq, hs - Eq])
    op = _operator(d)
    op0 = _operator(0.0)
    F = 2 * _apply_to_grid_values(_full_values(op0, s["v2"]), k_out, 0.0)
    D = _apply_to_grid_values(_full_values(op, s["uv"]), k_out, d)
    xi = 0.5 * (k_out * k_out + EG + 8 * np.pi * d * s["n"] - F)
    Eq = np.sqrt(xi * xi + D * D)
    return np.vstack([0.5 * (1 - xi / Eq), D / (2 * Eq), hs + Eq, hs - Eq])


def condensate_state(EG: float, d: float, r: float, k_out: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    EG = _check_scalar(EG, "EG")
    d = _check_scalar(d, "d", nonneg=True)
    r = _check_scalar(r, "r", positive=True)
    k_out = _check_k_array(k_out, "k_out")
    s = _hf(EG, d, r)
    return _state_at(s, k_out)

import numpy as np


def exciton_density(EG: float, d: float) -> float:
    """Reference implementation."""
    EG = _check_scalar(EG, "EG")
    d = _check_scalar(d, "d", nonneg=True)
    return float(_hf(EG, d, 1.0)["n"])

import numpy as np


def _gap_for_density(n_target, d):
    """Secant iteration on E_G with warm-started self-consistent solutions."""
    Eb, phi = _exciton(d)
    dmu = inverse_compressibility(d)
    if n_target == 0.0:
        return Eb
    E0 = Eb - dmu * n_target
    s0 = _hf(E0, d, 1.0)
    f0 = s0["n"] - n_target
    E1 = Eb - dmu * n_target * (n_target / s0["n"]) if s0["n"] > 0 else E0 - 0.5 * dmu * n_target
    s1 = _hf(E1, d, 1.0, start=s0)
    f1 = s1["n"] - n_target
    for _ in range(40):
        if f1 == f0:
            break
        E2 = min(E1 - f1 * (E1 - E0) / (f1 - f0), Eb - 1e-12)
        s2 = _hf(E2, d, 1.0, start=s1)
        f2 = s2["n"] - n_target
        E0, f0, E1, f1, s1 = E1, f1, E2, f2, s2
        if abs(f2) < 1e-13 * n_target:
            break
    return float(E1)


def gap_for_density(n_target: float, d: float) -> float:
    """Reference implementation."""
    n_target = _check_scalar(n_target, "n_target", nonneg=True)
    d = _check_scalar(d, "d", nonneg=True)
    return _gap_for_density(n_target, d)

import numpy as np
from scipy.optimize import brentq


def _band_curves(s):
    """Even-polynomial fits of E+, E-, v^2 near k = 0 and a monotonicity check on the trusted grid up to k = 60."""
    k = s["k"]
    sel = k < 60.0
    if np.any(np.diff(s["Ep"][sel]) <= 0) or np.any(np.diff(s["Em"][sel]) >= 0):
        raise ValueError("quasiparticle bands are not monotonic on k >= 0")
    return {nm: _even_fit(k, s[nm]) for nm in ("Ep", "Em", "v2")}


def _k_of_bias(s, band, e):
    """Momentum at which the band ('Ep' or 'Em') equals the energy e, by bracketing on the trusted grid."""
    k = s["k"]
    arr = s[band]
    if band == "Ep":
        j = int(np.searchsorted(arr, e))
    else:
        j = int(np.searchsorted(-arr, -e))
    lo = 0.0 if j == 0 else k[j - 1]
    hi = k[min(j, len(k) - 1)] if j < len(k) else k[-1] * 2
    row = 2 if band == "Ep" else 3
    g = lambda kk: _state_at(s, np.array([kk]))[row, 0] - e
    if lo == 0.0 and g(0.0) * g(hi) > 0:
        return 0.0
    return brentq(g, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=200)


def _band_and_weight(s, band, kk):
    """Value, derivative dE/dk (five-point stencil) and the projected weight (u^2 or v^2) of a band at kk."""
    h = 1e-3 * (1.0 + kk)
    pts = np.array([kk - 2 * h, kk - h, kk, kk + h, kk + 2 * h])
    if pts[0] < 0:
        pts = np.abs(pts)
    st = _state_at(s, pts)
    row = 2 if band == "Ep" else 3
    e = st[row]
    de = (e[0] - 8 * e[1] + 8 * e[3] - e[4]) / (12 * h)
    w = st[0, 2] if band == "Em" else 1 - st[0, 2]
    return e[2], de, w


def averaged_conductance(EG: float, d: float, r: float, biases: np.ndarray) -> np.ndarray:
    """Reference implementation: (dI/dV)/G_0 = 2 k w_k / |dE/dk| at k(V) (Eqs. 38 and 49 of the source)."""
    EG = _check_scalar(EG, "EG")
    d = _check_scalar(d, "d", nonneg=True)
    r = _check_scalar(r, "r", positive=True)
    biases = np.asarray(biases, dtype=float)
    if biases.ndim != 1 or biases.size == 0 or not np.all(np.isfinite(biases)):
        raise ValueError("biases must be a one-dimensional array of finite numbers")
    s = _hf(EG, d, r)
    fits = _band_curves(s)
    E0p, E0m = fits["Ep"][0], fits["Em"][0]
    out = np.zeros((2, len(biases)))
    for i, e in enumerate(biases):
        if E0m + 1e-9 < e < E0p - 1e-9:
            continue
        band = "Ep" if e >= E0p - 1e-9 else "Em"
        edge = E0p if band == "Ep" else E0m
        if abs(e - edge) <= 1e-9:
            a2 = fits[band][1]
            w0 = 1 - fits["v2"][0] if band == "Ep" else fits["v2"][0]
            out[0, i] = 1.0 / abs(a2)
            out[1, i] = w0 / abs(a2)
            continue
        kk = _k_of_bias(s, band, e)
        _, de, w = _band_and_weight(s, band, kk)
        out[0, i] = 2 * kk / abs(de)
        out[1, i] = 2 * kk * w / abs(de)
    return out

import numpy as np
from numpy.polynomial.legendre import leggauss


def satellite_current(EG: float, d: float, r: float, V: float) -> float:
    """Reference implementation: I(V) = -2 int_0^{k(V)} k v_k^2 dk (Eqs. 49 and 50 of the source)."""
    EG = _check_scalar(EG, "EG")
    d = _check_scalar(d, "d", nonneg=True)
    r = _check_scalar(r, "r", positive=True)
    if isinstance(V, bool) or not isinstance(V, (int, float, np.integer, np.floating)) or np.isnan(V) or V == np.inf:
        raise ValueError("V must be a real number or -inf")
    V = float(V)
    s = _hf(EG, d, r)
    if s["trivial"]:
        return 0.0
    fits = _band_curves(s)
    if V >= fits["Em"][0]:
        return 0.0
    if V == -np.inf:
        return float(-4 * np.pi * s["n"])
    kv = _k_of_bias(s, "Em", V)
    x, w = leggauss(64)
    kk = 0.5 * kv * (x + 1)
    v2 = _state_at(s, kk)[0]
    return float(-2 * np.sum(0.5 * kv * w * kk * v2))

import numpy as np
from scipy.interpolate import CubicSpline


def satellite_onset_conductance(I_sat: float, d: float, r: float) -> float:
    """Reference implementation chaining the oracles of steps 1 to 8."""
    I_sat = _check_scalar(I_sat, "I_sat")
    if I_sat >= 0.0:
        raise ValueError("I_sat must be negative")
    d = _check_scalar(d, "d", nonneg=True)
    r = _check_scalar(r, "r", positive=True)
    n_ex = -I_sat / (4 * np.pi)
    # step 2 and step 1: bound state and its eigen-equation residual
    k_dense = np.concatenate([[0.0], np.logspace(-3, 2.5, 160)])
    ex = exciton_state(d, k_dense)
    Eb, phi_d = ex[0], ex[1:]
    spline = CubicSpline(k_dense, phi_d, bc_type=((1, 0.0), (1, 0.0)))
    phi_call = lambda q: np.where(q <= k_dense[-1], spline(np.minimum(q, k_dense[-1])), phi_d[-1] * (k_dense[-1] / np.maximum(q, 1e-300)) ** 3)
    k_chk = np.array([0.5, 2.0])
    resid = exchange_integral(phi_call, k_chk, d) / ((k_chk ** 2 + Eb) * spline(k_chk)) - 1
    if np.max(np.abs(resid)) > 1e-4:
        raise ValueError("bound state does not satisfy its eigen-equation")
    # step 3 and step 6: first-order gap estimate and the self-consistent inversion
    dmu = inverse_compressibility(d)
    EG_first = Eb - dmu * n_ex
    EG = gap_for_density(n_ex, d)
    if not abs(EG - EG_first) <= 0.05 * dmu * n_ex + 1e-12:
        raise ValueError("self-consistent gap is inconsistent with the first-order estimate")
    # step 5: density check
    if abs(exciton_density(EG, d) - n_ex) > 1e-8 * n_ex:
        raise ValueError("density does not reproduce the target")
    # step 8: the full satellite current must reproduce the measurement
    if abs(satellite_current(EG, d, r, -np.inf) - I_sat) > 1e-8 * abs(I_sat):
        raise ValueError("satellite sum rule is violated")
    # step 4 and step 7: onset conductance as the limit from inside the lower band
    st = condensate_state(EG, d, r, np.array([0.0]))
    E0m = st[3, 0]
    onset = averaged_conductance(EG, d, r, np.array([E0m]))[1, 0]
    return float(onset)
SCICODE_GOLD_EOF
