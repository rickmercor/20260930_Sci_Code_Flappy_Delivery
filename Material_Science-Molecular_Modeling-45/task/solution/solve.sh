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
from numpy.polynomial.legendre import leggauss, legvander


def nanoreactor_parameters(core_radius, cavity_radius, site_fraction, diffusivity, reactivity, permeability, hindrance):
    R = float(core_radius); R0 = float(cavity_radius); phi = float(site_fraction)
    D = float(diffusivity); kappa = float(reactivity); perm = float(permeability); p = float(hindrance)
    if not (R > 0.0) or not (R0 > R):
        raise ValueError("need 0 < core radius < cavity radius")
    if not (0.0 < phi <= 1.0):
        raise ValueError("site fraction must lie in (0, 1]")
    if not (D > 0.0) or not (kappa > 0.0) or not (perm > 0.0):
        raise ValueError("diffusivity, reactivity and permeability must be positive")
    if not (p >= 0.0) or not np.isfinite(p):
        raise ValueError("hindrance exponent must be finite and non-negative")
    eps = R / R0
    h = 1.0 - eps
    theta0 = float(np.arccos(1.0 - 2.0 * phi))
    da = kappa * R / D
    bi = perm * R / D
    ks = 4.0 * np.pi * R * D
    eps0 = eps ** (1.0 + p) if not np.isfinite(bi) else eps ** (1.0 + p) - (1.0 + p) * eps * eps / bi
    j_shell = (1.0 + p) / (1.0 - eps0)
    j_iso = j_shell if not np.isfinite(da) else (1.0 + p) * da / ((1.0 + p) + da * (1.0 - eps0))
    return np.array([eps, h, theta0, da, bi, p, ks, j_shell, j_iso], dtype=float)

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def dual_series_coefficients(order, eps, biot, hindrance):
    n = int(order); e = float(eps); bi = float(biot); p = float(hindrance)
    if n < 0 or n != order:
        raise ValueError("order must be a non-negative integer")
    if not (0.0 <= e < 1.0):
        raise ValueError("eps must lie in [0, 1)")
    if not (bi > 0.0):
        raise ValueError("biot must be positive (inf for a perfectly permeable shell)")
    if not (p >= 0.0) or not np.isfinite(p):
        raise ValueError("hindrance exponent must be finite and non-negative")
    l = np.arange(n + 1, dtype=float)
    root = np.sqrt((1.0 + p) ** 2 + 4.0 * l * (l + 1.0))
    sp = 0.5 * (-(1.0 + p) + root)
    sm = 0.5 * (-(1.0 + p) - root)
    if e == 0.0:
        el = np.zeros(n + 1)
    elif np.isfinite(bi):
        el = e ** (sp - sm) * (bi + sm * e ** (1.0 - p)) / (bi + sp * e ** (1.0 - p))
    else:
        el = e ** (sp - sm)
    denom = -sm + el * sp
    w = 1.0 - (l + 0.5) / denom
    q = 1.0 - (l + 0.5) * (1.0 - el) / denom
    return np.vstack([sp, sm, el, w, q])

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def minkov_matrix(order, theta0):
    n = int(order); t = float(theta0)
    if n < 0 or n != order:
        raise ValueError("order must be a non-negative integer")
    if not (0.0 < t <= np.pi):
        raise ValueError("theta0 must lie in (0, pi]")
    l = np.arange(n + 1, dtype=float)[:, None]
    m = np.arange(n + 1, dtype=float)[None, :]
    s = l + m + 1.0
    d = l - m
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(d != 0.0, np.sin(d * t) / np.where(d == 0.0, 1.0, d), t)
    return (np.sin(s * t) / s + off) / np.pi

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def perfect_sink_solution(q, Q):
    q = np.asarray(q, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float)
    n = q.size - 1
    if n < 0 or Q.ndim != 2 or Q.shape != (n + 1, n + 1):
        raise ValueError("q must have length order + 1 and Q must be (order + 1, order + 1)")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(Q)):
        raise ValueError("q and Q must be finite")
    A = np.eye(n + 1) - Q * q[None, :]
    return np.linalg.solve(A, Q[:, 0].copy())

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def rate_correction_summary(order, theta0, eps, biot, hindrance):
    n = int(order); t = float(theta0); e = float(eps); bi = float(biot); p = float(hindrance)
    if not (0.0 < t <= np.pi):
        raise ValueError("theta0 must lie in (0, pi]")
    if not (0.0 <= e < 1.0):
        raise ValueError("eps must lie in [0, 1)")
    if not (bi > 0.0):
        raise ValueError("biot must be positive")
    if not (p >= 0.0) or not np.isfinite(p):
        raise ValueError("hindrance exponent must be finite and non-negative")
    q = dual_series_coefficients(n, e, bi, p)[4]
    Q = minkov_matrix(n, t)
    X = perfect_sink_solution(q, Q)
    J = 0.5 * X[0]
    mnorm = float(np.max(np.sum(np.abs(Q * q[None, :]), axis=1)))
    j0 = 0.5 * Q[0, 0]
    j1 = 0.5 * Q[0, 0] / (1.0 - q[0] * Q[0, 0])
    delta1 = 100.0 * (J - j1) / J
    f_esf = 0.5 * perfect_sink_solution(dual_series_coefficients(n, 0.0, bi, p)[4], Q)[0]
    eps0 = e ** (1.0 + p) if not np.isfinite(bi) else e ** (1.0 + p) - (1.0 + p) * e * e / bi
    j_shell = (1.0 + p) / (1.0 - eps0)
    coupling = J / (f_esf * j_shell)
    return np.array([J, j0, j1, delta1, mnorm, f_esf, coupling], dtype=float)

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def _legendre_rows(x, order):
    """P_l(x) for l = 0..order at the points x, shape (len(x), order + 1)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return legvander(x, int(order))

def _cap_integrals(cos_theta, order):
    """b_l = integral of P_l(x) dx from cos_theta to 1, l = 0..order (closed form)."""
    n = int(order)
    c = float(cos_theta)
    P = _legendre_rows(np.array([c]), n + 1)[0]
    b = np.empty(n + 1)
    b[0] = 1.0 - c
    if n >= 1:
        l = np.arange(1, n + 1, dtype=float)
        b[1:] = (P[0:n] - P[2:n + 2]) / (2.0 * l + 1.0)
    return b

def local_fields(X, coefficients, eps, points, theta_c):
    X = np.asarray(X, dtype=float).ravel()
    co = np.asarray(coefficients, dtype=float)
    e = float(eps)
    if X.size == 0 or co.ndim != 2 or co.shape[0] != 5 or co.shape[1] != X.size:
        raise ValueError("coefficients must be a (5, order + 1) array matching X")
    if not (0.0 <= e < 1.0):
        raise ValueError("eps must lie in [0, 1)")
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (m, 2) array of (xi, theta)")
    tc = float(theta_c)
    if not (0.0 < tc <= np.pi):
        raise ValueError("theta_c must lie in (0, pi]")
    xi = pts[:, 0]; th = pts[:, 1]
    outer = np.inf if e == 0.0 else 1.0 / e
    if np.any(xi < 1.0) or np.any(xi > outer) or np.any(th < 0.0) or np.any(th > np.pi):
        raise ValueError("probe points must lie in the shell 1 <= xi <= 1/eps, 0 <= theta <= pi")
    n = X.size - 1
    l = np.arange(n + 1, dtype=float)
    sp = co[0]; sm = co[1]; el = co[2]; w = co[3]
    A = (1.0 - w) * X                                                   # exterior moments
    P = _legendre_rows(np.cos(th), n)
    radial = xi[:, None] ** sm[None, :] - el[None, :] * xi[:, None] ** sp[None, :]
    u = (radial * A[None, :] * P).sum(axis=1)
    b = _cap_integrals(np.cos(tc), n)
    flux_fraction = float(np.sum((l + 0.5) * X * b) / X[0])
    if e == 0.0:
        wall_fraction = np.nan
    else:
        xw = 1.0 / e
        dwall = -A * (sm * xw ** (sm - 1.0) - el * sp * xw ** (sp - 1.0))          # entry flux density coefficients
        bh = _cap_integrals(0.0, n)
        wall_fraction = float(np.sum(dwall * bh) / (2.0 * dwall[0]))
    return np.concatenate([u, [flux_fraction, wall_fraction]])

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def _legendre_rows(x, order):
    """P_l(x) for l = 0..order at the points x, shape (len(x), order + 1)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return legvander(x, int(order))

def partially_reactive_solution(q, theta0, damkohler):
    q = np.asarray(q, dtype=float).ravel()
    t = float(theta0); da = float(damkohler)
    n = q.size - 1
    if n < 0 or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite array of length order + 1")
    if not (0.0 < t <= np.pi):
        raise ValueError("theta0 must lie in (0, pi]")
    if not (np.isfinite(da) and da > 0.0):
        raise ValueError("damkohler must be a finite positive number")
    c = float(np.cos(t))
    xg, wg = leggauss(n + 2)
    x = 0.5 * (1.0 - c) * xg + 0.5 * (1.0 + c); wq = 0.5 * (1.0 - c) * wg
    V = _legendre_rows(x, n)
    K = (V * wq[:, None]).T @ V
    b = (V * wq[:, None]).sum(axis=0)
    A = np.eye(n + 1) + da * K * (1.0 - q)[None, :]
    return np.linalg.solve(A, da * b)

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def radiation_islae_regularity(q, Q, damkohler):
    q = np.asarray(q, dtype=float).ravel(); Q = np.asarray(Q, dtype=float)
    da = float(damkohler)
    n = q.size - 1
    if n < 0 or Q.shape != (n + 1, n + 1):
        raise ValueError("q must have length order + 1 and Q must be (order + 1, order + 1)")
    if not (np.isfinite(da) and da > 0.0):
        raise ValueError("damkohler must be a finite positive number")
    l = np.arange(n + 1, dtype=float)
    qda = q - (l + 0.5) / da
    M = Q * qda[None, :]
    mnorm = float(np.max(np.sum(np.abs(M), axis=1)))
    X = np.linalg.solve(np.eye(n + 1) - M, Q[:, 0].copy())
    return np.array([mnorm, 0.5 * X[0]], dtype=float)

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def series_resistance_deviation(j_sink, theta0, damkohler, j_exact):
    js = float(j_sink); t = float(theta0); da = float(damkohler); je = float(j_exact)
    if not (js > 0.0) or not (je > 0.0):
        raise ValueError("correction factors must be positive")
    if not (0.0 < t <= np.pi):
        raise ValueError("theta0 must lie in (0, pi]")
    if not (np.isfinite(da) and da > 0.0):
        raise ValueError("damkohler must be a finite positive number")
    j_react = 0.5 * da * (1.0 - np.cos(t))
    j_ck = 1.0 / (1.0 / js + 1.0 / j_react)
    return np.array([j_react, j_ck, 100.0 * (j_ck - je) / je], dtype=float)

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def nanoreactor_audit(design_table, order):
    rows = [tuple(float(v) for v in r) for r in design_table]
    if not rows or any(len(r) != 7 for r in rows):
        raise ValueError("design_table must hold rows (R, R0, site_fraction, D, kappa, permeability, hindrance)")
    n = int(order)
    if n < 1 or n != order:
        raise ValueError("order must be a positive integer")
    NC = 29
    out = np.zeros((1 + len(rows), NC))
    for i, (R, R0, phi, D, kappa, perm, p) in enumerate(rows, 1):
        par = nanoreactor_parameters(R, R0, phi, D, kappa, perm, p)
        eps, h, t0, da, bi, pp, ks, j_shell, j_iso = par
        co = dual_series_coefficients(n, eps, bi, pp)
        Q = minkov_matrix(n, t0)
        X = perfect_sink_solution(co[4], Q)
        summ = rate_correction_summary(n, t0, eps, bi, pp)
        J, j0, j1, d1, mnorm, f_esf, coup = summ
        xi_mid = 0.5 * (1.0 + 1.0 / eps)
        loc = local_fields(X, co, eps, [[xi_mid, 0.0], [xi_mid, np.pi]], 0.5 * t0)
        u_front, u_back, f_half, g_front = loc[0], loc[1], loc[2], loc[3]
        Xd = partially_reactive_solution(co[4], t0, da)
        j_da = 0.5 * Xd[0]
        reg = radiation_islae_regularity(co[4], Q, da)
        sr = series_resistance_deviation(J, t0, da, j_da)
        loc_da = local_fields(Xd, co, eps, [[xi_mid, 0.0]], 0.5 * t0)
        out[i] = [eps, h, t0, da, bi, pp, ks, j_shell, j_iso, J, j0, j1, d1, mnorm, f_esf, coup,
                  f_half, u_front, u_back, g_front, j_da, reg[0], reg[1], sr[0], sr[1], sr[2],
                  ks * j_da, loc_da[1], j_da / J]
    out[0, 0] = out[1, 28]
    out[0, 1] = len(rows)
    out[0, 2] = n
    out[0, 3] = float(np.sum(out[1:, 25]))
    return out
SCICODE_GOLD_EOF
