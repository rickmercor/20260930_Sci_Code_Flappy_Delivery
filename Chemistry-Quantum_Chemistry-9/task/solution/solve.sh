#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math

import numpy as np


def _gl01(n):
    """Gauss-Legendre nodes and weights mapped from [-1, 1] onto (0, 1)."""
    t, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (t + 1.0), 0.5 * w


def radial_quadrature(n_nodes: int, scale: float) -> "np.ndarray":
    """Gauss-Legendre radial quadrature on the half line."""
    if int(n_nodes) < 1:
        raise ValueError("n_nodes must be at least one")
    if float(scale) <= 0.0:
        raise ValueError("scale must be positive")
    u, w = _gl01(n_nodes)
    r = float(scale) * u / (1.0 - u)
    weights = w * float(scale) / (1.0 - u) ** 2
    return np.vstack((r, weights))

import math

import numpy as np


def _sto_norm(principal, exponent):
    """Norm of the radial Slater primitive r**(principal-1) * exp(-exponent r)."""
    if principal == 1:
        return math.sqrt(exponent ** 3 / math.pi)
    return math.sqrt(exponent ** 5 / (3.0 * math.pi))


def _pair_terms(a, b):
    """Collect the product of two orbital tables as {(power, decay): coefficient}."""
    out = {}
    for pa, za, ca in a:
        if ca == 0.0:
            continue
        for pb, zb, cb in b:
            if cb == 0.0:
                continue
            key = (int(round(pa)) + int(round(pb)) - 2, float(za) + float(zb))
            w = ca * cb * _sto_norm(int(round(pa)), float(za)) * _sto_norm(int(round(pb)), float(zb))
            out[key] = out.get(key, 0.0) + w
    return out


def _overlap(a, b):
    """Overlap of two orbital tables, integrated over all space."""
    s = 0.0
    for (power, decay), w in _pair_terms(a, b).items():
        s += w * 4.0 * math.pi * math.factorial(power + 2) / decay ** (power + 3)
    return s


def orthonormal_orbitals(inner_exponents: "np.ndarray", inner_coefficients: "np.ndarray", outer_exponent: float) -> "np.ndarray":
    """Normalised inner orbital and the outer orbital orthogonalised against it."""
    ze = np.asarray(inner_exponents, dtype=float).ravel()
    zc = np.asarray(inner_coefficients, dtype=float).ravel()
    if ze.size != zc.size or ze.size == 0:
        raise ValueError("inner_exponents and inner_coefficients must have the same nonzero length")
    if np.any(ze <= 0.0) or float(outer_exponent) <= 0.0:
        raise ValueError("every Slater exponent must be positive")
    nin = ze.size
    inner = [[1.0, ze[i], zc[i]] for i in range(nin)]
    s = math.sqrt(_overlap(inner, inner))
    inner = [[1.0, ze[i], zc[i] / s] for i in range(nin)]
    outer_raw = [[2.0, float(outer_exponent), 1.0]]
    mix = _overlap(outer_raw, inner)
    outer = outer_raw + [[1.0, ze[i], -mix * inner[i][2]] for i in range(nin)]
    t = math.sqrt(_overlap(outer, outer))
    outer = [[row[0], row[1], row[2] / t] for row in outer]
    width = nin + 1
    table = np.zeros((2, width, 3))
    for i, row in enumerate(inner):
        table[0, i] = row
    for i, row in enumerate(outer):
        table[1, i] = row
    return table

import math

import numpy as np


def _sto_norm(principal, exponent):
    """Norm of the radial Slater primitive r**(principal-1) * exp(-exponent r)."""
    if principal == 1:
        return math.sqrt(exponent ** 3 / math.pi)
    return math.sqrt(exponent ** 5 / (3.0 * math.pi))


def _sto_eval(principal, exponent, r, order):
    """Value (order 0), d/dr (order 1) or d2/dr2 (order 2) of a normalised primitive."""
    c = _sto_norm(principal, exponent)
    e = np.exp(-exponent * r)
    if principal == 1:
        if order == 0:
            return c * e
        if order == 1:
            return -exponent * c * e
        return exponent * exponent * c * e
    if order == 0:
        return c * r * e
    if order == 1:
        return c * (1.0 - exponent * r) * e
    return c * (exponent * exponent * r - 2.0 * exponent) * e


def _expand(orbital, r, order):
    """Evaluate one padded (nterm, 3) orbital table on r."""
    out = np.zeros_like(r, dtype=float)
    for principal, exponent, coef in orbital:
        if coef == 0.0:
            continue
        out = out + coef * _sto_eval(int(round(principal)), float(exponent), r, order)
    return out


def spin_channel_fields(orbitals: "np.ndarray", occupations: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    """Density, gradient, Laplacian, kinetic-energy density and hole curvature."""
    orb = np.asarray(orbitals, dtype=float)
    occ = np.asarray(occupations, dtype=float).ravel()
    r = np.asarray(radii, dtype=float).ravel()
    if orb.ndim != 3 or orb.shape[2] != 3:
        raise ValueError("orbitals must have shape (n_orbitals, n_terms, 3)")
    if orb.shape[0] != occ.size:
        raise ValueError("orbitals and occupations must describe the same number of orbitals")
    if np.any(occ < 0.0) or np.any(occ > 1.0):
        raise ValueError("each occupation must lie between zero and one")
    if np.any(r <= 0.0):
        raise ValueError("every radius must be positive")
    rho = np.zeros_like(r)
    grad = np.zeros_like(r)
    lap = np.zeros_like(r)
    tau = np.zeros_like(r)
    for k in range(orb.shape[0]):
        f0 = _expand(orb[k], r, 0)
        f1 = _expand(orb[k], r, 1)
        f2 = _expand(orb[k], r, 2)
        rho = rho + occ[k] * f0 * f0
        grad = grad + 2.0 * occ[k] * f0 * f1
        lap = lap + 2.0 * occ[k] * (f0 * (f2 + 2.0 * f1 / r) + f1 * f1)
        tau = tau + occ[k] * f1 * f1
    safe = np.where(rho > 0.0, rho, 1.0)
    d = tau - 0.25 * grad * grad / safe
    curv = (lap - 2.0 * d) / 6.0
    return np.vstack((rho, grad, lap, tau, curv))

import math

import numpy as np


def b86b_exchange_density(density: "np.ndarray", gradient: "np.ndarray") -> "np.ndarray":
    """B86b exchange energy density of one spin channel."""
    rho = np.asarray(density, dtype=float).ravel()
    g = np.asarray(gradient, dtype=float).ravel()
    if rho.size != g.size:
        raise ValueError("density and gradient must have the same length")
    if np.any(rho <= 0.0):
        raise ValueError("every density value must be positive")
    cx = 1.5 * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    r43 = rho ** (4.0 / 3.0)
    x2 = g * g / rho ** (8.0 / 3.0)
    return -cx * r43 - 0.00375 * r43 * x2 / (1.0 + 0.007 * x2) ** 0.8

import math

import numpy as np


def _sto_norm(principal, exponent):
    """Norm of the radial Slater primitive r**(principal-1) * exp(-exponent r)."""
    if principal == 1:
        return math.sqrt(exponent ** 3 / math.pi)
    return math.sqrt(exponent ** 5 / (3.0 * math.pi))


def _sto_eval(principal, exponent, r, order):
    """Value (order 0), d/dr (order 1) or d2/dr2 (order 2) of a normalised primitive."""
    c = _sto_norm(principal, exponent)
    e = np.exp(-exponent * r)
    if principal == 1:
        if order == 0:
            return c * e
        if order == 1:
            return -exponent * c * e
        return exponent * exponent * c * e
    if order == 0:
        return c * r * e
    if order == 1:
        return c * (1.0 - exponent * r) * e
    return c * (exponent * exponent * r - 2.0 * exponent) * e


def _expand(orbital, r, order):
    """Evaluate one padded (nterm, 3) orbital table on r."""
    out = np.zeros_like(r, dtype=float)
    for principal, exponent, coef in orbital:
        if coef == 0.0:
            continue
        out = out + coef * _sto_eval(int(round(principal)), float(exponent), r, order)
    return out


def _pair_terms(a, b):
    """Collect the product of two orbital tables as {(power, decay): coefficient}."""
    out = {}
    for pa, za, ca in a:
        if ca == 0.0:
            continue
        for pb, zb, cb in b:
            if cb == 0.0:
                continue
            key = (int(round(pa)) + int(round(pb)) - 2, float(za) + float(zb))
            w = ca * cb * _sto_norm(int(round(pa)), float(za)) * _sto_norm(int(round(pb)), float(zb))
            out[key] = out.get(key, 0.0) + w
    return out


def _gamma_lo(n, x):
    """Lower incomplete gamma P(n+1, x) for integer n, by its finite exponential sum."""
    s = np.zeros_like(x)
    term = np.ones_like(x)
    for m in range(n + 1):
        s = s + term
        term = term * x / (m + 1)
    return 1.0 - np.exp(-x) * s


def _gamma_hi(n, x):
    s = np.zeros_like(x)
    term = np.ones_like(x)
    for m in range(n + 1):
        s = s + term
        term = term * x / (m + 1)
    return np.exp(-x) * s


def _radial_coulomb(power, decay, r):
    """Potential at radius r of the spherical density s**power * exp(-decay s)."""
    x = decay * r
    lo = math.factorial(power + 2) / decay ** (power + 3) * _gamma_lo(power + 2, x)
    hi = math.factorial(power + 1) / decay ** (power + 2) * _gamma_hi(power + 1, x)
    return 4.0 * math.pi * (lo / r + hi)


def exact_exchange_density(orbitals: "np.ndarray", occupations: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    """Exact (Hartree-Fock) exchange energy density of one spin channel."""
    orb = np.asarray(orbitals, dtype=float)
    occ = np.asarray(occupations, dtype=float).ravel()
    r = np.asarray(radii, dtype=float).ravel()
    if orb.ndim != 3 or orb.shape[2] != 3:
        raise ValueError("orbitals must have shape (n_orbitals, n_terms, 3)")
    if orb.shape[0] != occ.size:
        raise ValueError("orbitals and occupations must describe the same number of orbitals")
    if np.any(occ < 0.0) or np.any(occ > 1.0):
        raise ValueError("each occupation must lie between zero and one")
    if np.any(r <= 0.0):
        raise ValueError("every radius must be positive")
    total = np.zeros_like(r)
    for i in range(orb.shape[0]):
        fi = _expand(orb[i], r, 0)
        for j in range(orb.shape[0]):
            fj = _expand(orb[j], r, 0)
            pot = np.zeros_like(r)
            for (power, decay), w in _pair_terms(orb[i], orb[j]).items():
                pot = pot + w * _radial_coulomb(power, decay, r)
            total = total + occ[i] * occ[j] * fi * fj * pot
    return -0.5 * total

import math

import numpy as np


def _brx(x, rhs):
    """Residual and derivative of the Becke-Roussel defining equation."""
    e = np.exp(-2.0 * x / 3.0)
    return x * e / (x - 2.0) - rhs, 2.0 / 3.0 * (2.0 * x - x * x - 3.0) / (x - 2.0) ** 2 * e


def _bisect(fun, lo, hi, iters=200):
    """Vectorised bisection on a monotone residual, bracketed by lo and hi."""
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    flo = fun(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = fun(mid)
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo)
        flo = np.where(same, fm, flo)
        hi = np.where(same, hi, mid)
    return 0.5 * (lo + hi)


def _br_potential(density, curvature, hole_normalisation):
    """Becke-Roussel model exchange potential at the reference point."""
    rho = np.asarray(density, dtype=float).ravel()
    q = np.asarray(curvature, dtype=float).ravel()
    n = np.asarray(hole_normalisation, dtype=float).ravel()
    if rho.size != q.size:
        raise ValueError("density and curvature must have the same length")
    if n.size == 1:
        n = np.full(rho.size, float(n[0]))
    if n.size != rho.size:
        raise ValueError("hole_normalisation must be a scalar or match the density length")
    if np.any(rho <= 0.0):
        raise ValueError("every density value must be positive")
    if np.any(n <= 0.0):
        raise ValueError("every hole normalisation must be positive")
    flat = q == 0.0
    safe = np.where(flat, 1.0, q)
    rhs = (2.0 / 3.0) * (math.pi * rho / n) ** (2.0 / 3.0) * rho / safe
    lo = np.where(rhs < 0.0, 1e-12, 2.0 + 1e-13)
    hi = np.where(rhs < 0.0, 2.0 - 1e-13, 400.0)
    x = _bisect(lambda v: _brx(v, rhs)[0], lo, hi)
    for _ in range(80):
        f, df = _brx(x, rhs)
        x = x - f / df
    # A vanishing curvature sends the right-hand side to infinity from either side, and
    # the left-hand side diverges only at x = 2, so x = 2 IS the limit -- not an error.
    x = np.where(flat, 2.0, x)
    e = np.exp(-x)
    alpha = (8.0 * math.pi * rho / e / n) ** (1.0 / 3.0)
    b = x / alpha
    return -n * (1.0 - e - 0.5 * x * e) / b


def effective_hole_normalisation(density: "np.ndarray", curvature: "np.ndarray", exact_exchange_energy_density: "np.ndarray") -> "np.ndarray":
    """Effective exchange-hole normalisation from the inverse Becke-Roussel procedure."""
    rho = np.asarray(density, dtype=float).ravel()
    q = np.asarray(curvature, dtype=float).ravel()
    eps = np.asarray(exact_exchange_energy_density, dtype=float).ravel()
    if not (rho.size == q.size == eps.size):
        raise ValueError("density, curvature and exact_exchange_energy_density must have the same length")
    if np.any(rho <= 0.0):
        raise ValueError("every density value must be positive")
    if np.any(eps >= 0.0):
        raise ValueError("every exact exchange energy density must be negative")
    target = 2.0 * eps / rho
    root = _bisect(lambda n: _br_potential(rho, q, n) - target,
                   np.full(rho.size, 1e-8), np.ones(rho.size))
    return np.minimum(root, 1.0)

import math

import numpy as np


def xc_hole_normalisation(same_spin_norm: "np.ndarray", other_spin_norm: "np.ndarray") -> "np.ndarray":
    """Effective exchange-correlation hole normalisation of the same-spin channel."""
    a = np.asarray(same_spin_norm, dtype=float).ravel()
    b = np.asarray(other_spin_norm, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("both normalisation arrays must have the same length")
    if np.any(a < 0.0) or np.any(a > 1.0) or np.any(b < 0.0) or np.any(b > 1.0):
        raise ValueError("every hole normalisation must lie between zero and one")
    big = np.finfo(float).max
    first = np.where(b > 0.0, (1.0 - a) / np.where(b > 0.0, b, 1.0), big)
    second = np.where(a > 0.0, (1.0 - b) / np.where(a > 0.0, a, 1.0), big)
    f = np.minimum(np.minimum(first, second), 1.0)
    return a + f * b

import math

import numpy as np


def _erf_array(values):
    """Error function evaluated elementwise, using only the standard library."""
    arr = np.asarray(values, dtype=float)
    flat = np.array([math.erf(float(v)) for v in arr.ravel()], dtype=float)
    return flat.reshape(arr.shape)


def local_mixing_function(density: "np.ndarray", b86b_energy_density: "np.ndarray", xc_hole_norm: "np.ndarray", c_param: float, b_param: float) -> "np.ndarray":
    """Position-dependent exact-exchange mixing fraction of the LHnz local hybrid."""
    rho = np.asarray(density, dtype=float).ravel()
    eb = np.asarray(b86b_energy_density, dtype=float).ravel()
    n = np.asarray(xc_hole_norm, dtype=float).ravel()
    if not (rho.size == eb.size == n.size):
        raise ValueError("density, b86b_energy_density and xc_hole_norm must have the same length")
    if np.any(rho <= 0.0):
        raise ValueError("every density value must be positive")
    if np.any(eb == 0.0):
        raise ValueError("no B86b exchange energy density may vanish")
    if float(c_param) <= 0.0:
        raise ValueError("c_param must be positive")
    if float(b_param) < 0.0:
        raise ValueError("b_param must not be negative")
    z = np.abs(rho / eb)
    s = float(b_param) * np.sin(math.pi * n) ** 2 + 1.0
    return _erf_array(float(c_param) * s * z)

import math

import numpy as np


def channel_admixture(mixing: "np.ndarray", exact_density: "np.ndarray", semilocal_density: "np.ndarray", radii: "np.ndarray", weights: "np.ndarray") -> float:
    """One spin channel's contribution to the exact-exchange admixture energy."""
    g = np.asarray(mixing, dtype=float).ravel()
    ex = np.asarray(exact_density, dtype=float).ravel()
    eb = np.asarray(semilocal_density, dtype=float).ravel()
    r = np.asarray(radii, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if not (g.size == ex.size == eb.size == r.size == w.size):
        raise ValueError("mixing, both energy densities, radii and weights must have the same length")
    if np.any(r <= 0.0):
        raise ValueError("every radius must be positive")
    if np.any(g < 0.0) or np.any(g > 1.0):
        raise ValueError("every mixing fraction must lie between zero and one")
    return 4.0 * math.pi * float(np.sum(w * r ** 2 * g * (ex - eb)))

import math

import numpy as np


def _gl01(n):
    """Gauss-Legendre nodes and weights mapped from [-1, 1] onto (0, 1)."""
    t, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (t + 1.0), 0.5 * w


def _sto_norm(principal, exponent):
    """Norm of the radial Slater primitive r**(principal-1) * exp(-exponent r)."""
    if principal == 1:
        return math.sqrt(exponent ** 3 / math.pi)
    return math.sqrt(exponent ** 5 / (3.0 * math.pi))


def _sto_eval(principal, exponent, r, order):
    """Value (order 0), d/dr (order 1) or d2/dr2 (order 2) of a normalised primitive."""
    c = _sto_norm(principal, exponent)
    e = np.exp(-exponent * r)
    if principal == 1:
        if order == 0:
            return c * e
        if order == 1:
            return -exponent * c * e
        return exponent * exponent * c * e
    if order == 0:
        return c * r * e
    if order == 1:
        return c * (1.0 - exponent * r) * e
    return c * (exponent * exponent * r - 2.0 * exponent) * e


def _expand(orbital, r, order):
    """Evaluate one padded (nterm, 3) orbital table on r."""
    out = np.zeros_like(r, dtype=float)
    for principal, exponent, coef in orbital:
        if coef == 0.0:
            continue
        out = out + coef * _sto_eval(int(round(principal)), float(exponent), r, order)
    return out


def _pair_terms(a, b):
    """Collect the product of two orbital tables as {(power, decay): coefficient}."""
    out = {}
    for pa, za, ca in a:
        if ca == 0.0:
            continue
        for pb, zb, cb in b:
            if cb == 0.0:
                continue
            key = (int(round(pa)) + int(round(pb)) - 2, float(za) + float(zb))
            w = ca * cb * _sto_norm(int(round(pa)), float(za)) * _sto_norm(int(round(pb)), float(zb))
            out[key] = out.get(key, 0.0) + w
    return out


def _overlap(a, b):
    """Overlap of two orbital tables, integrated over all space."""
    s = 0.0
    for (power, decay), w in _pair_terms(a, b).items():
        s += w * 4.0 * math.pi * math.factorial(power + 2) / decay ** (power + 3)
    return s


def _gamma_lo(n, x):
    """Lower incomplete gamma P(n+1, x) for integer n, by its finite exponential sum."""
    s = np.zeros_like(x)
    term = np.ones_like(x)
    for m in range(n + 1):
        s = s + term
        term = term * x / (m + 1)
    return 1.0 - np.exp(-x) * s


def _gamma_hi(n, x):
    s = np.zeros_like(x)
    term = np.ones_like(x)
    for m in range(n + 1):
        s = s + term
        term = term * x / (m + 1)
    return np.exp(-x) * s


def _radial_coulomb(power, decay, r):
    """Potential at radius r of the spherical density s**power * exp(-decay s)."""
    x = decay * r
    lo = math.factorial(power + 2) / decay ** (power + 3) * _gamma_lo(power + 2, x)
    hi = math.factorial(power + 1) / decay ** (power + 2) * _gamma_hi(power + 1, x)
    return 4.0 * math.pi * (lo / r + hi)


def _brx(x, rhs):
    """Residual and derivative of the Becke-Roussel defining equation."""
    e = np.exp(-2.0 * x / 3.0)
    return x * e / (x - 2.0) - rhs, 2.0 / 3.0 * (2.0 * x - x * x - 3.0) / (x - 2.0) ** 2 * e


def _bisect(fun, lo, hi, iters=200):
    """Vectorised bisection on a monotone residual, bracketed by lo and hi."""
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    flo = fun(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = fun(mid)
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo)
        flo = np.where(same, fm, flo)
        hi = np.where(same, hi, mid)
    return 0.5 * (lo + hi)


def _br_potential(density, curvature, hole_normalisation):
    """Becke-Roussel model exchange potential at the reference point."""
    rho = np.asarray(density, dtype=float).ravel()
    q = np.asarray(curvature, dtype=float).ravel()
    n = np.asarray(hole_normalisation, dtype=float).ravel()
    if rho.size != q.size:
        raise ValueError("density and curvature must have the same length")
    if n.size == 1:
        n = np.full(rho.size, float(n[0]))
    if n.size != rho.size:
        raise ValueError("hole_normalisation must be a scalar or match the density length")
    if np.any(rho <= 0.0):
        raise ValueError("every density value must be positive")
    if np.any(n <= 0.0):
        raise ValueError("every hole normalisation must be positive")
    flat = q == 0.0
    safe = np.where(flat, 1.0, q)
    rhs = (2.0 / 3.0) * (math.pi * rho / n) ** (2.0 / 3.0) * rho / safe
    lo = np.where(rhs < 0.0, 1e-12, 2.0 + 1e-13)
    hi = np.where(rhs < 0.0, 2.0 - 1e-13, 400.0)
    x = _bisect(lambda v: _brx(v, rhs)[0], lo, hi)
    for _ in range(80):
        f, df = _brx(x, rhs)
        x = x - f / df
    # A vanishing curvature sends the right-hand side to infinity from either side, and
    # the left-hand side diverges only at x = 2, so x = 2 IS the limit -- not an error.
    x = np.where(flat, 2.0, x)
    e = np.exp(-x)
    alpha = (8.0 * math.pi * rho / e / n) ** (1.0 / 3.0)
    b = x / alpha
    return -n * (1.0 - e - 0.5 * x * e) / b


def _erf_array(values):
    """Error function evaluated elementwise, using only the standard library."""
    arr = np.asarray(values, dtype=float)
    flat = np.array([math.erf(float(v)) for v in arr.ravel()], dtype=float)
    return flat.reshape(arr.shape)


def lhnz_exchange_correction(inner_exponents: "np.ndarray", inner_coefficients: "np.ndarray", outer_exponent: float, outer_occupation: float, c_param: float, b_param: float, n_nodes: int, scale: float) -> float:
    """Local-hybrid exchange correction of the LHnz functional for the atom."""
    if not 0.0 <= float(outer_occupation) <= 1.0:
        raise ValueError("outer_occupation must lie between zero and one")
    grid = radial_quadrature(n_nodes, scale)
    r, w = grid[0], grid[1]
    orb = orthonormal_orbitals(inner_exponents, inner_coefficients, outer_exponent)
    occ = (np.array([1.0, float(outer_occupation)]), np.array([1.0]))
    sets = (orb, orb[:1])
    keep = []
    fields = []
    epsx = []
    epsb = []
    for k in range(2):
        f = spin_channel_fields(sets[k], occ[k], r)
        m = f[0] > 1e-14
        keep.append(m)
        fields.append(f[:, m])
        epsx.append(exact_exchange_density(sets[k], occ[k], r[m]))
        epsb.append(b86b_exchange_density(f[0][m], f[1][m]))
    norms = [effective_hole_normalisation(fields[k][0], fields[k][4], epsx[k])
             for k in range(2)]
    full = []
    for k in range(2):
        v = np.ones_like(r)
        v[keep[k]] = norms[k]
        full.append(v)
    total = 0.0
    for k in range(2):
        m = keep[k]
        nxc = xc_hole_normalisation(full[k][m], full[1 - k][m])
        g = local_mixing_function(fields[k][0], epsb[k], nxc, c_param, b_param)
        total += channel_admixture(g, epsx[k], epsb[k], r[m], w[m])
    return total
SCICODE_GOLD_EOF
