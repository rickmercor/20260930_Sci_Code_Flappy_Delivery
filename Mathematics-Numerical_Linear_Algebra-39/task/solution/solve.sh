#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np, math
from math import comb


def _kappa():
    """the source's deactivation threshold of the acceleration (Algorithm 5.1)"""
    return 0.01

def _dim():
    """the order of the synthetic matrices"""
    return 16

def _check_index(l):
    if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or not (0 <= int(l) <= 7):
        raise ValueError("l must be an integer in 0..7")
    return int(l)

def _table(l):
    """the source's tabulated closed-form members for l = 4..7 (Table 2), monomial order b0..b8"""
    t = {4: [0, 0, 0, 0, 0, 56, -140, 120, -35], 5: [0, 0, 0, 0, 0, 0, 28, -48, 21], 6: [0, 0, 0, 0, 0, 0, 0, 8, -7], 7: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    return np.array(t[l], dtype=np.float64)

def _horner(b, x):
    y = 0.0
    for k in range(8, -1, -1):
        y = y * x + b[k]
    return y

def _stop_constant(l):
    return [np.inf, 28.0, 56.0, 82.0, 82.0, 56.0, 28.0, np.inf][l]

def _stop_order(l):
    return [1, 2, 3, 4, 4, 3, 2, 1][l]

def _esym(st):
    """elementary symmetric polynomials e_0..e_7 of the seven stationary points"""
    e = np.zeros(8); e[0] = 1.0
    for v in st:
        e[1:] = e[1:] + v * e[:-1]
    return e

def _pcoef(st):
    """monomial coefficients of P(x) = int_0^x prod_k (t - s_k) dt, the parametrisation (4.1) with unit scale and zero offset"""
    e = _esym(st); c = np.zeros(9)
    for k in range(8):
        c[8 - k] = ((-1.0) ** k) * e[k] / (8 - k)
    return c

def _pdiff(st, x1, x0):
    """int_{x0}^{x1} prod_k (t - s_k) dt by the four-point Gauss rule, exact for the degree-seven integrand and free of cancellation"""
    g = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
    w = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    m = 0.5 * (x1 + x0); h = 0.5 * (x1 - x0); t = m + h * g
    f = np.ones(4)
    for s in st:
        f = f * (t - s)
    return h * float(np.dot(w, f))

def _check_bounds(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        raise ValueError("bounds must be real numbers")
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b > 1.0 or not (a < b):
        raise ValueError("bounds must satisfy 0 <= lam_lumo < lam_homo <= 1")
    return a, b

def _alternation_sets(st, L, a, b):
    """lower and upper alternation points of the left interval and upper and lower points of the right interval"""
    R = 7 - L
    lo = ([0.0] if L % 2 == 0 else []) + [st[k - 1] for k in range(1, L + 1) if k % 2 == 1]
    up = [a] + [st[k - 1] for k in range(1, L + 1) if k % 2 == 0] + ([0.0] if L % 2 == 1 else [])
    ru = [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 1] + ([1.0] if R % 2 == 0 else [])
    rl = [b] + [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 0] + ([1.0] if R % 2 == 1 else [])
    return lo, up, ru, rl

def _free_index(L, a, b):
    fl = [] if a == 0.0 else list(range(L)); fr = [] if b == 1.0 else list(range(L, 7))
    return fl + fr

def _full_st(free, L, a, b):
    st = np.zeros(7)
    if b == 1.0: st[L:] = 1.0
    st[_free_index(L, a, b)] = free
    return st

def _residual_free(free, L, a, b):
    """scale-free form of the equioscillatory conditions: equal values of P on each alternation set, normalised by the amplitude of the interval"""
    R = 7 - L; st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b); res = []
    if a > 0.0 and L > 0:
        amp = _pdiff(st, a, st[0])
        for x in lo[1:]: res.append(_pdiff(st, x, lo[0]) / amp)
        for x in up[1:]: res.append(_pdiff(st, x, up[0]) / amp)
    if b < 1.0 and R > 0:
        amp = _pdiff(st, st[L], b)
        for x in ru[1:]: res.append(_pdiff(st, x, ru[0]) / amp)
        for x in rl[1:]: res.append(_pdiff(st, x, rl[0]) / amp)
    return np.array(res)

def _strict_ordered(st, L, a, b):
    """the ordering (4.7) with strict inequalities on the free stationary points"""
    R = 7 - L
    if L > 0 and a > 0.0:
        if not (st[0] < a and st[L - 1] > 0.0): return False
        if L > 1 and not np.all(np.diff(st[:L]) < 0.0): return False
    if R > 0 and b < 1.0:
        if not (st[L] > b and st[6] < 1.0): return False
        if R > 1 and not np.all(np.diff(st[L:]) > 0.0): return False
    return True

def _assemble(free, L, a, b):
    """the nine parameters: stationary points, scale s8 and offset s9 fixed by p = 0 at the lower-left level and p = 1 at the upper-right level"""
    st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b)
    xl = 0.0 if a == 0.0 else lo[0]
    xr = 1.0 if b == 1.0 else ru[0]
    s8 = 1.0 / _pdiff(st, xr, xl); s9 = -s8 * _pdiff(st, xl, 0.0)
    return np.concatenate([st, [s8, s9]])

def _jacobian(free, L, a, b):
    n = free.size; f0 = _residual_free(free, L, a, b); J = np.zeros((f0.size, n)); h = 1e-7
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (_residual_free(free + d, L, a, b) - _residual_free(free - d, L, a, b)) / (2.0 * h)
    return J

def _newton_free(L, a, b, z0):
    """damped Newton iteration on the scale-free system that keeps the ordering (4.7)"""
    z = z0.copy(); F = _residual_free(z, L, a, b); nf = float(np.max(np.abs(F)))
    for it in range(60):
        if nf < 1e-12: break
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -F)
        except np.linalg.LinAlgError:
            return z, False
        lam = 1.0; accepted = False
        while lam > 1e-4:
            zn = z + lam * dz
            if _strict_ordered(_full_st(zn, L, a, b), L, a, b):
                Fn = _residual_free(zn, L, a, b); nfn = float(np.max(np.abs(Fn)))
                if nfn < nf or lam < 0.1:
                    z, F, nf = zn, Fn, nfn; accepted = True; break
            lam *= 0.5
        if not accepted: return z, False
    for it in range(3):
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -_residual_free(z, L, a, b))
        except np.linalg.LinAlgError:
            break
        zn = z + dz
        if _strict_ordered(_full_st(zn, L, a, b), L, a, b) and np.max(np.abs(_residual_free(zn, L, a, b))) <= np.max(np.abs(_residual_free(z, L, a, b))): z = zn
    nf = float(np.max(np.abs(_residual_free(z, L, a, b))))
    return z, (nf < 1e-11 and _strict_ordered(_full_st(z, L, a, b), L, a, b))

def _guess(L, a, b, kind, power):
    """Chebyshev extrema (kind 0) or Chebyshev zeros (kind 1) on each interval, the relative positions raised to a power, as the initial stationary points"""
    R = 7 - L; st = np.zeros(7)
    if L > 0 and a > 0.0:
        k = np.arange(1, L + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (L + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * L + 2)))
        st[:L] = (a * u ** power)[::-1]
    if R > 0 and b < 1.0:
        k = np.arange(1, R + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (R + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * R + 2)))
        st[L:] = b + (1.0 - b) * (1.0 - (1.0 - u) ** power)
    return st[_free_index(L, a, b)]

def _solve_equioscillatory(L, a, b):
    """the unique solution of the equioscillatory conditions with the ordering (4.7), from deterministic Chebyshev starts of increasing compression"""
    if len(_free_index(L, a, b)) == 0:
        return _assemble(np.zeros(0), L, a, b)
    for power in (1.0, 1.5, 0.75, 2.0, 0.5, 3.0):
        for kind in (0, 1):
            z, ok = _newton_free(L, a, b, _guess(L, a, b, kind, power))
            if ok: return _assemble(z, L, a, b)
    raise ValueError("the equioscillatory conditions could not be solved for L=%d on (%g, %g)" % (L, a, b))

def _coefficients(s):
    c = _pcoef(s[:7]) * s[7]; c[0] = c[0] + s[8]
    return c

def _transform(energies):
    """the source's normalisation (2.2): an eigenvalue e of H maps to (lmax - e)/(lmax - lmin); Table 1 order lmin, out_homo, in_homo, in_lumo, out_lumo, lmax"""
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size != 6 or not np.all(np.isfinite(e)): raise ValueError("energies must hold six finite values")
    lmin, oh, ih, il, ol, lmax = e
    if not (lmin < oh <= ih < il <= ol < lmax): raise ValueError("energies must be ordered lmin < out_homo <= in_homo < in_lumo <= out_lumo < lmax")
    T = (lmax - np.array([il, ih, ol, oh])) / (lmax - lmin)
    return T   # in_lumo, in_homo, out_lumo, out_homo on [0, 1]

def _build(bounds, seed):
    """synthetic 16 by 16 matrix consistent with the bounds: the extreme eigenvalues of each interval at the midpoints of the bound intervals, the rest equidistant, on the sign-fixed QR frame of the mod-47 integer constructor"""
    n = _dim(); inl, inh, outl, outh = bounds
    lam_lumo = 0.5 * (outl + inl); lam_homo = 0.5 * (inh + outh)
    mu = 0.5 * (inl + inh); nocc = int(round(n * (1.0 - mu)))
    lam = np.concatenate([np.linspace(0.0, lam_lumo, n - nocc), np.linspace(lam_homo, 1.0, nocc)])
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (((i + 1) * (j + 2) * (i + j + 3) + seed * (i + 4) * (j + 5)) % 47) - 23
    Q, _r = np.linalg.qr(M)
    for c in range(n):
        col = Q[:, c]
        for v in col:
            if abs(v) > 1e-12:
                if v < 0:
                    Q[:, c] = -col
                break
    X0 = Q @ np.diag(lam) @ Q.T
    return (X0 + X0.T) / 2.0, nocc

def sp8_closed_form(l: int) -> "np.ndarray":
    k = _check_index(l)
    if k >= 4:
        return _table(k).copy()
    q = _table(7 - k)
    out = np.zeros(9, dtype=np.float64)
    out[0] = 1.0
    for m in range(9):
        if q[m] == 0.0:
            continue
        for j in range(m + 1):
            out[j] -= q[m] * comb(m, j) * ((-1.0) ** j)
    return out

import numpy as np, math
from math import comb


def _kappa():
    """the source's deactivation threshold of the acceleration (Algorithm 5.1)"""
    return 0.01

def _dim():
    """the order of the synthetic matrices"""
    return 16

def _check_index(l):
    if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or not (0 <= int(l) <= 7):
        raise ValueError("l must be an integer in 0..7")
    return int(l)

def _table(l):
    """the source's tabulated closed-form members for l = 4..7 (Table 2), monomial order b0..b8"""
    t = {4: [0, 0, 0, 0, 0, 56, -140, 120, -35], 5: [0, 0, 0, 0, 0, 0, 28, -48, 21], 6: [0, 0, 0, 0, 0, 0, 0, 8, -7], 7: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    return np.array(t[l], dtype=np.float64)

def _horner(b, x):
    y = 0.0
    for k in range(8, -1, -1):
        y = y * x + b[k]
    return y

def _stop_constant(l):
    return [np.inf, 28.0, 56.0, 82.0, 82.0, 56.0, 28.0, np.inf][l]

def _stop_order(l):
    return [1, 2, 3, 4, 4, 3, 2, 1][l]

def _esym(st):
    """elementary symmetric polynomials e_0..e_7 of the seven stationary points"""
    e = np.zeros(8); e[0] = 1.0
    for v in st:
        e[1:] = e[1:] + v * e[:-1]
    return e

def _pcoef(st):
    """monomial coefficients of P(x) = int_0^x prod_k (t - s_k) dt, the parametrisation (4.1) with unit scale and zero offset"""
    e = _esym(st); c = np.zeros(9)
    for k in range(8):
        c[8 - k] = ((-1.0) ** k) * e[k] / (8 - k)
    return c

def _pdiff(st, x1, x0):
    """int_{x0}^{x1} prod_k (t - s_k) dt by the four-point Gauss rule, exact for the degree-seven integrand and free of cancellation"""
    g = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
    w = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    m = 0.5 * (x1 + x0); h = 0.5 * (x1 - x0); t = m + h * g
    f = np.ones(4)
    for s in st:
        f = f * (t - s)
    return h * float(np.dot(w, f))

def _check_bounds(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        raise ValueError("bounds must be real numbers")
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b > 1.0 or not (a < b):
        raise ValueError("bounds must satisfy 0 <= lam_lumo < lam_homo <= 1")
    return a, b

def _alternation_sets(st, L, a, b):
    """lower and upper alternation points of the left interval and upper and lower points of the right interval"""
    R = 7 - L
    lo = ([0.0] if L % 2 == 0 else []) + [st[k - 1] for k in range(1, L + 1) if k % 2 == 1]
    up = [a] + [st[k - 1] for k in range(1, L + 1) if k % 2 == 0] + ([0.0] if L % 2 == 1 else [])
    ru = [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 1] + ([1.0] if R % 2 == 0 else [])
    rl = [b] + [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 0] + ([1.0] if R % 2 == 1 else [])
    return lo, up, ru, rl

def _free_index(L, a, b):
    fl = [] if a == 0.0 else list(range(L)); fr = [] if b == 1.0 else list(range(L, 7))
    return fl + fr

def _full_st(free, L, a, b):
    st = np.zeros(7)
    if b == 1.0: st[L:] = 1.0
    st[_free_index(L, a, b)] = free
    return st

def _residual_free(free, L, a, b):
    """scale-free form of the equioscillatory conditions: equal values of P on each alternation set, normalised by the amplitude of the interval"""
    R = 7 - L; st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b); res = []
    if a > 0.0 and L > 0:
        amp = _pdiff(st, a, st[0])
        for x in lo[1:]: res.append(_pdiff(st, x, lo[0]) / amp)
        for x in up[1:]: res.append(_pdiff(st, x, up[0]) / amp)
    if b < 1.0 and R > 0:
        amp = _pdiff(st, st[L], b)
        for x in ru[1:]: res.append(_pdiff(st, x, ru[0]) / amp)
        for x in rl[1:]: res.append(_pdiff(st, x, rl[0]) / amp)
    return np.array(res)

def _strict_ordered(st, L, a, b):
    """the ordering (4.7) with strict inequalities on the free stationary points"""
    R = 7 - L
    if L > 0 and a > 0.0:
        if not (st[0] < a and st[L - 1] > 0.0): return False
        if L > 1 and not np.all(np.diff(st[:L]) < 0.0): return False
    if R > 0 and b < 1.0:
        if not (st[L] > b and st[6] < 1.0): return False
        if R > 1 and not np.all(np.diff(st[L:]) > 0.0): return False
    return True

def _assemble(free, L, a, b):
    """the nine parameters: stationary points, scale s8 and offset s9 fixed by p = 0 at the lower-left level and p = 1 at the upper-right level"""
    st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b)
    xl = 0.0 if a == 0.0 else lo[0]
    xr = 1.0 if b == 1.0 else ru[0]
    s8 = 1.0 / _pdiff(st, xr, xl); s9 = -s8 * _pdiff(st, xl, 0.0)
    return np.concatenate([st, [s8, s9]])

def _jacobian(free, L, a, b):
    n = free.size; f0 = _residual_free(free, L, a, b); J = np.zeros((f0.size, n)); h = 1e-7
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (_residual_free(free + d, L, a, b) - _residual_free(free - d, L, a, b)) / (2.0 * h)
    return J

def _newton_free(L, a, b, z0):
    """damped Newton iteration on the scale-free system that keeps the ordering (4.7)"""
    z = z0.copy(); F = _residual_free(z, L, a, b); nf = float(np.max(np.abs(F)))
    for it in range(60):
        if nf < 1e-12: break
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -F)
        except np.linalg.LinAlgError:
            return z, False
        lam = 1.0; accepted = False
        while lam > 1e-4:
            zn = z + lam * dz
            if _strict_ordered(_full_st(zn, L, a, b), L, a, b):
                Fn = _residual_free(zn, L, a, b); nfn = float(np.max(np.abs(Fn)))
                if nfn < nf or lam < 0.1:
                    z, F, nf = zn, Fn, nfn; accepted = True; break
            lam *= 0.5
        if not accepted: return z, False
    for it in range(3):
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -_residual_free(z, L, a, b))
        except np.linalg.LinAlgError:
            break
        zn = z + dz
        if _strict_ordered(_full_st(zn, L, a, b), L, a, b) and np.max(np.abs(_residual_free(zn, L, a, b))) <= np.max(np.abs(_residual_free(z, L, a, b))): z = zn
    nf = float(np.max(np.abs(_residual_free(z, L, a, b))))
    return z, (nf < 1e-11 and _strict_ordered(_full_st(z, L, a, b), L, a, b))

def _guess(L, a, b, kind, power):
    """Chebyshev extrema (kind 0) or Chebyshev zeros (kind 1) on each interval, the relative positions raised to a power, as the initial stationary points"""
    R = 7 - L; st = np.zeros(7)
    if L > 0 and a > 0.0:
        k = np.arange(1, L + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (L + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * L + 2)))
        st[:L] = (a * u ** power)[::-1]
    if R > 0 and b < 1.0:
        k = np.arange(1, R + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (R + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * R + 2)))
        st[L:] = b + (1.0 - b) * (1.0 - (1.0 - u) ** power)
    return st[_free_index(L, a, b)]

def _solve_equioscillatory(L, a, b):
    """the unique solution of the equioscillatory conditions with the ordering (4.7), from deterministic Chebyshev starts of increasing compression"""
    if len(_free_index(L, a, b)) == 0:
        return _assemble(np.zeros(0), L, a, b)
    for power in (1.0, 1.5, 0.75, 2.0, 0.5, 3.0):
        for kind in (0, 1):
            z, ok = _newton_free(L, a, b, _guess(L, a, b, kind, power))
            if ok: return _assemble(z, L, a, b)
    raise ValueError("the equioscillatory conditions could not be solved for L=%d on (%g, %g)" % (L, a, b))

def _coefficients(s):
    c = _pcoef(s[:7]) * s[7]; c[0] = c[0] + s[8]
    return c

def _transform(energies):
    """the source's normalisation (2.2): an eigenvalue e of H maps to (lmax - e)/(lmax - lmin); Table 1 order lmin, out_homo, in_homo, in_lumo, out_lumo, lmax"""
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size != 6 or not np.all(np.isfinite(e)): raise ValueError("energies must hold six finite values")
    lmin, oh, ih, il, ol, lmax = e
    if not (lmin < oh <= ih < il <= ol < lmax): raise ValueError("energies must be ordered lmin < out_homo <= in_homo < in_lumo <= out_lumo < lmax")
    T = (lmax - np.array([il, ih, ol, oh])) / (lmax - lmin)
    return T   # in_lumo, in_homo, out_lumo, out_homo on [0, 1]

def _build(bounds, seed):
    """synthetic 16 by 16 matrix consistent with the bounds: the extreme eigenvalues of each interval at the midpoints of the bound intervals, the rest equidistant, on the sign-fixed QR frame of the mod-47 integer constructor"""
    n = _dim(); inl, inh, outl, outh = bounds
    lam_lumo = 0.5 * (outl + inl); lam_homo = 0.5 * (inh + outh)
    mu = 0.5 * (inl + inh); nocc = int(round(n * (1.0 - mu)))
    lam = np.concatenate([np.linspace(0.0, lam_lumo, n - nocc), np.linspace(lam_homo, 1.0, nocc)])
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (((i + 1) * (j + 2) * (i + j + 3) + seed * (i + 4) * (j + 5)) % 47) - 23
    Q, _r = np.linalg.qr(M)
    for c in range(n):
        col = Q[:, c]
        for v in col:
            if abs(v) > 1e-12:
                if v < 0:
                    Q[:, c] = -col
                break
    X0 = Q @ np.diag(lam) @ Q.T
    return (X0 + X0.T) / 2.0, nocc

def sp8_from_stationary_points(s: "np.ndarray") -> "np.ndarray":
    ss = np.asarray(s, dtype=np.float64).ravel()
    if ss.size != 9 or not np.all(np.isfinite(ss)):
        raise ValueError("s must hold nine finite values")
    return _coefficients(ss)

import numpy as np, math
from math import comb


def _kappa():
    """the source's deactivation threshold of the acceleration (Algorithm 5.1)"""
    return 0.01

def _dim():
    """the order of the synthetic matrices"""
    return 16

def _check_index(l):
    if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or not (0 <= int(l) <= 7):
        raise ValueError("l must be an integer in 0..7")
    return int(l)

def _table(l):
    """the source's tabulated closed-form members for l = 4..7 (Table 2), monomial order b0..b8"""
    t = {4: [0, 0, 0, 0, 0, 56, -140, 120, -35], 5: [0, 0, 0, 0, 0, 0, 28, -48, 21], 6: [0, 0, 0, 0, 0, 0, 0, 8, -7], 7: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    return np.array(t[l], dtype=np.float64)

def _horner(b, x):
    y = 0.0
    for k in range(8, -1, -1):
        y = y * x + b[k]
    return y

def _stop_constant(l):
    return [np.inf, 28.0, 56.0, 82.0, 82.0, 56.0, 28.0, np.inf][l]

def _stop_order(l):
    return [1, 2, 3, 4, 4, 3, 2, 1][l]

def _esym(st):
    """elementary symmetric polynomials e_0..e_7 of the seven stationary points"""
    e = np.zeros(8); e[0] = 1.0
    for v in st:
        e[1:] = e[1:] + v * e[:-1]
    return e

def _pcoef(st):
    """monomial coefficients of P(x) = int_0^x prod_k (t - s_k) dt, the parametrisation (4.1) with unit scale and zero offset"""
    e = _esym(st); c = np.zeros(9)
    for k in range(8):
        c[8 - k] = ((-1.0) ** k) * e[k] / (8 - k)
    return c

def _pdiff(st, x1, x0):
    """int_{x0}^{x1} prod_k (t - s_k) dt by the four-point Gauss rule, exact for the degree-seven integrand and free of cancellation"""
    g = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
    w = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    m = 0.5 * (x1 + x0); h = 0.5 * (x1 - x0); t = m + h * g
    f = np.ones(4)
    for s in st:
        f = f * (t - s)
    return h * float(np.dot(w, f))

def _check_bounds(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        raise ValueError("bounds must be real numbers")
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b > 1.0 or not (a < b):
        raise ValueError("bounds must satisfy 0 <= lam_lumo < lam_homo <= 1")
    return a, b

def _alternation_sets(st, L, a, b):
    """lower and upper alternation points of the left interval and upper and lower points of the right interval"""
    R = 7 - L
    lo = ([0.0] if L % 2 == 0 else []) + [st[k - 1] for k in range(1, L + 1) if k % 2 == 1]
    up = [a] + [st[k - 1] for k in range(1, L + 1) if k % 2 == 0] + ([0.0] if L % 2 == 1 else [])
    ru = [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 1] + ([1.0] if R % 2 == 0 else [])
    rl = [b] + [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 0] + ([1.0] if R % 2 == 1 else [])
    return lo, up, ru, rl

def _free_index(L, a, b):
    fl = [] if a == 0.0 else list(range(L)); fr = [] if b == 1.0 else list(range(L, 7))
    return fl + fr

def _full_st(free, L, a, b):
    st = np.zeros(7)
    if b == 1.0: st[L:] = 1.0
    st[_free_index(L, a, b)] = free
    return st

def _residual_free(free, L, a, b):
    """scale-free form of the equioscillatory conditions: equal values of P on each alternation set, normalised by the amplitude of the interval"""
    R = 7 - L; st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b); res = []
    if a > 0.0 and L > 0:
        amp = _pdiff(st, a, st[0])
        for x in lo[1:]: res.append(_pdiff(st, x, lo[0]) / amp)
        for x in up[1:]: res.append(_pdiff(st, x, up[0]) / amp)
    if b < 1.0 and R > 0:
        amp = _pdiff(st, st[L], b)
        for x in ru[1:]: res.append(_pdiff(st, x, ru[0]) / amp)
        for x in rl[1:]: res.append(_pdiff(st, x, rl[0]) / amp)
    return np.array(res)

def _strict_ordered(st, L, a, b):
    """the ordering (4.7) with strict inequalities on the free stationary points"""
    R = 7 - L
    if L > 0 and a > 0.0:
        if not (st[0] < a and st[L - 1] > 0.0): return False
        if L > 1 and not np.all(np.diff(st[:L]) < 0.0): return False
    if R > 0 and b < 1.0:
        if not (st[L] > b and st[6] < 1.0): return False
        if R > 1 and not np.all(np.diff(st[L:]) > 0.0): return False
    return True

def _assemble(free, L, a, b):
    """the nine parameters: stationary points, scale s8 and offset s9 fixed by p = 0 at the lower-left level and p = 1 at the upper-right level"""
    st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b)
    xl = 0.0 if a == 0.0 else lo[0]
    xr = 1.0 if b == 1.0 else ru[0]
    s8 = 1.0 / _pdiff(st, xr, xl); s9 = -s8 * _pdiff(st, xl, 0.0)
    return np.concatenate([st, [s8, s9]])

def _jacobian(free, L, a, b):
    n = free.size; f0 = _residual_free(free, L, a, b); J = np.zeros((f0.size, n)); h = 1e-7
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (_residual_free(free + d, L, a, b) - _residual_free(free - d, L, a, b)) / (2.0 * h)
    return J

def _newton_free(L, a, b, z0):
    """damped Newton iteration on the scale-free system that keeps the ordering (4.7)"""
    z = z0.copy(); F = _residual_free(z, L, a, b); nf = float(np.max(np.abs(F)))
    for it in range(60):
        if nf < 1e-12: break
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -F)
        except np.linalg.LinAlgError:
            return z, False
        lam = 1.0; accepted = False
        while lam > 1e-4:
            zn = z + lam * dz
            if _strict_ordered(_full_st(zn, L, a, b), L, a, b):
                Fn = _residual_free(zn, L, a, b); nfn = float(np.max(np.abs(Fn)))
                if nfn < nf or lam < 0.1:
                    z, F, nf = zn, Fn, nfn; accepted = True; break
            lam *= 0.5
        if not accepted: return z, False
    for it in range(3):
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -_residual_free(z, L, a, b))
        except np.linalg.LinAlgError:
            break
        zn = z + dz
        if _strict_ordered(_full_st(zn, L, a, b), L, a, b) and np.max(np.abs(_residual_free(zn, L, a, b))) <= np.max(np.abs(_residual_free(z, L, a, b))): z = zn
    nf = float(np.max(np.abs(_residual_free(z, L, a, b))))
    return z, (nf < 1e-11 and _strict_ordered(_full_st(z, L, a, b), L, a, b))

def _guess(L, a, b, kind, power):
    """Chebyshev extrema (kind 0) or Chebyshev zeros (kind 1) on each interval, the relative positions raised to a power, as the initial stationary points"""
    R = 7 - L; st = np.zeros(7)
    if L > 0 and a > 0.0:
        k = np.arange(1, L + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (L + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * L + 2)))
        st[:L] = (a * u ** power)[::-1]
    if R > 0 and b < 1.0:
        k = np.arange(1, R + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (R + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * R + 2)))
        st[L:] = b + (1.0 - b) * (1.0 - (1.0 - u) ** power)
    return st[_free_index(L, a, b)]

def _solve_equioscillatory(L, a, b):
    """the unique solution of the equioscillatory conditions with the ordering (4.7), from deterministic Chebyshev starts of increasing compression"""
    if len(_free_index(L, a, b)) == 0:
        return _assemble(np.zeros(0), L, a, b)
    for power in (1.0, 1.5, 0.75, 2.0, 0.5, 3.0):
        for kind in (0, 1):
            z, ok = _newton_free(L, a, b, _guess(L, a, b, kind, power))
            if ok: return _assemble(z, L, a, b)
    raise ValueError("the equioscillatory conditions could not be solved for L=%d on (%g, %g)" % (L, a, b))

def _coefficients(s):
    c = _pcoef(s[:7]) * s[7]; c[0] = c[0] + s[8]
    return c

def _transform(energies):
    """the source's normalisation (2.2): an eigenvalue e of H maps to (lmax - e)/(lmax - lmin); Table 1 order lmin, out_homo, in_homo, in_lumo, out_lumo, lmax"""
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size != 6 or not np.all(np.isfinite(e)): raise ValueError("energies must hold six finite values")
    lmin, oh, ih, il, ol, lmax = e
    if not (lmin < oh <= ih < il <= ol < lmax): raise ValueError("energies must be ordered lmin < out_homo <= in_homo < in_lumo <= out_lumo < lmax")
    T = (lmax - np.array([il, ih, ol, oh])) / (lmax - lmin)
    return T   # in_lumo, in_homo, out_lumo, out_homo on [0, 1]

def _build(bounds, seed):
    """synthetic 16 by 16 matrix consistent with the bounds: the extreme eigenvalues of each interval at the midpoints of the bound intervals, the rest equidistant, on the sign-fixed QR frame of the mod-47 integer constructor"""
    n = _dim(); inl, inh, outl, outh = bounds
    lam_lumo = 0.5 * (outl + inl); lam_homo = 0.5 * (inh + outh)
    mu = 0.5 * (inl + inh); nocc = int(round(n * (1.0 - mu)))
    lam = np.concatenate([np.linspace(0.0, lam_lumo, n - nocc), np.linspace(lam_homo, 1.0, nocc)])
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (((i + 1) * (j + 2) * (i + j + 3) + seed * (i + 4) * (j + 5)) % 47) - 23
    Q, _r = np.linalg.qr(M)
    for c in range(n):
        col = Q[:, c]
        for v in col:
            if abs(v) > 1e-12:
                if v < 0:
                    Q[:, c] = -col
                break
    X0 = Q @ np.diag(lam) @ Q.T
    return (X0 + X0.T) / 2.0, nocc

def equioscillation_residual(s: "np.ndarray", L: int, lam_lumo: float, lam_homo: float) -> "np.ndarray":
    ss = np.asarray(s, dtype=np.float64).ravel()
    if ss.size != 9 or not np.all(np.isfinite(ss)):
        raise ValueError("s must hold nine finite values")
    k = _check_index(L); R = 7 - k
    a, b = _check_bounds(lam_lumo, lam_homo)
    c = _coefficients(ss); st = ss[:7]
    pa = _horner(c, a); pb = _horner(c, b)
    out = []
    if a == 0.0:
        for j in range(1, k + 1): out.append(st[j - 1])
    else:
        for j in range(1, k + 1): out.append(_horner(c, st[j - 1]) - (0.0 if j % 2 == 1 else pa))
    if b == 1.0:
        for j in range(1, R + 1): out.append(st[k + j - 1] - 1.0)
    else:
        for j in range(1, R + 1): out.append(_horner(c, st[k + j - 1]) - (1.0 if j % 2 == 1 else pb))
    out.append(_horner(c, 0.0) - (0.0 if (a == 0.0 or k % 2 == 0) else pa))
    out.append(_horner(c, 1.0) - (1.0 if (b == 1.0 or R % 2 == 0) else pb))
    return np.array(out, dtype=np.float64)

import numpy as np, math
from math import comb


def _kappa():
    """the source's deactivation threshold of the acceleration (Algorithm 5.1)"""
    return 0.01

def _dim():
    """the order of the synthetic matrices"""
    return 16

def _check_index(l):
    if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or not (0 <= int(l) <= 7):
        raise ValueError("l must be an integer in 0..7")
    return int(l)

def _table(l):
    """the source's tabulated closed-form members for l = 4..7 (Table 2), monomial order b0..b8"""
    t = {4: [0, 0, 0, 0, 0, 56, -140, 120, -35], 5: [0, 0, 0, 0, 0, 0, 28, -48, 21], 6: [0, 0, 0, 0, 0, 0, 0, 8, -7], 7: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    return np.array(t[l], dtype=np.float64)

def _horner(b, x):
    y = 0.0
    for k in range(8, -1, -1):
        y = y * x + b[k]
    return y

def _stop_constant(l):
    return [np.inf, 28.0, 56.0, 82.0, 82.0, 56.0, 28.0, np.inf][l]

def _stop_order(l):
    return [1, 2, 3, 4, 4, 3, 2, 1][l]

def _esym(st):
    """elementary symmetric polynomials e_0..e_7 of the seven stationary points"""
    e = np.zeros(8); e[0] = 1.0
    for v in st:
        e[1:] = e[1:] + v * e[:-1]
    return e

def _pcoef(st):
    """monomial coefficients of P(x) = int_0^x prod_k (t - s_k) dt, the parametrisation (4.1) with unit scale and zero offset"""
    e = _esym(st); c = np.zeros(9)
    for k in range(8):
        c[8 - k] = ((-1.0) ** k) * e[k] / (8 - k)
    return c

def _pdiff(st, x1, x0):
    """int_{x0}^{x1} prod_k (t - s_k) dt by the four-point Gauss rule, exact for the degree-seven integrand and free of cancellation"""
    g = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
    w = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    m = 0.5 * (x1 + x0); h = 0.5 * (x1 - x0); t = m + h * g
    f = np.ones(4)
    for s in st:
        f = f * (t - s)
    return h * float(np.dot(w, f))

def _check_bounds(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        raise ValueError("bounds must be real numbers")
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b > 1.0 or not (a < b):
        raise ValueError("bounds must satisfy 0 <= lam_lumo < lam_homo <= 1")
    return a, b

def _alternation_sets(st, L, a, b):
    """lower and upper alternation points of the left interval and upper and lower points of the right interval"""
    R = 7 - L
    lo = ([0.0] if L % 2 == 0 else []) + [st[k - 1] for k in range(1, L + 1) if k % 2 == 1]
    up = [a] + [st[k - 1] for k in range(1, L + 1) if k % 2 == 0] + ([0.0] if L % 2 == 1 else [])
    ru = [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 1] + ([1.0] if R % 2 == 0 else [])
    rl = [b] + [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 0] + ([1.0] if R % 2 == 1 else [])
    return lo, up, ru, rl

def _free_index(L, a, b):
    fl = [] if a == 0.0 else list(range(L)); fr = [] if b == 1.0 else list(range(L, 7))
    return fl + fr

def _full_st(free, L, a, b):
    st = np.zeros(7)
    if b == 1.0: st[L:] = 1.0
    st[_free_index(L, a, b)] = free
    return st

def _residual_free(free, L, a, b):
    """scale-free form of the equioscillatory conditions: equal values of P on each alternation set, normalised by the amplitude of the interval"""
    R = 7 - L; st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b); res = []
    if a > 0.0 and L > 0:
        amp = _pdiff(st, a, st[0])
        for x in lo[1:]: res.append(_pdiff(st, x, lo[0]) / amp)
        for x in up[1:]: res.append(_pdiff(st, x, up[0]) / amp)
    if b < 1.0 and R > 0:
        amp = _pdiff(st, st[L], b)
        for x in ru[1:]: res.append(_pdiff(st, x, ru[0]) / amp)
        for x in rl[1:]: res.append(_pdiff(st, x, rl[0]) / amp)
    return np.array(res)

def _strict_ordered(st, L, a, b):
    """the ordering (4.7) with strict inequalities on the free stationary points"""
    R = 7 - L
    if L > 0 and a > 0.0:
        if not (st[0] < a and st[L - 1] > 0.0): return False
        if L > 1 and not np.all(np.diff(st[:L]) < 0.0): return False
    if R > 0 and b < 1.0:
        if not (st[L] > b and st[6] < 1.0): return False
        if R > 1 and not np.all(np.diff(st[L:]) > 0.0): return False
    return True

def _assemble(free, L, a, b):
    """the nine parameters: stationary points, scale s8 and offset s9 fixed by p = 0 at the lower-left level and p = 1 at the upper-right level"""
    st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b)
    xl = 0.0 if a == 0.0 else lo[0]
    xr = 1.0 if b == 1.0 else ru[0]
    s8 = 1.0 / _pdiff(st, xr, xl); s9 = -s8 * _pdiff(st, xl, 0.0)
    return np.concatenate([st, [s8, s9]])

def _jacobian(free, L, a, b):
    n = free.size; f0 = _residual_free(free, L, a, b); J = np.zeros((f0.size, n)); h = 1e-7
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (_residual_free(free + d, L, a, b) - _residual_free(free - d, L, a, b)) / (2.0 * h)
    return J

def _newton_free(L, a, b, z0):
    """damped Newton iteration on the scale-free system that keeps the ordering (4.7)"""
    z = z0.copy(); F = _residual_free(z, L, a, b); nf = float(np.max(np.abs(F)))
    for it in range(60):
        if nf < 1e-12: break
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -F)
        except np.linalg.LinAlgError:
            return z, False
        lam = 1.0; accepted = False
        while lam > 1e-4:
            zn = z + lam * dz
            if _strict_ordered(_full_st(zn, L, a, b), L, a, b):
                Fn = _residual_free(zn, L, a, b); nfn = float(np.max(np.abs(Fn)))
                if nfn < nf or lam < 0.1:
                    z, F, nf = zn, Fn, nfn; accepted = True; break
            lam *= 0.5
        if not accepted: return z, False
    for it in range(3):
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -_residual_free(z, L, a, b))
        except np.linalg.LinAlgError:
            break
        zn = z + dz
        if _strict_ordered(_full_st(zn, L, a, b), L, a, b) and np.max(np.abs(_residual_free(zn, L, a, b))) <= np.max(np.abs(_residual_free(z, L, a, b))): z = zn
    nf = float(np.max(np.abs(_residual_free(z, L, a, b))))
    return z, (nf < 1e-11 and _strict_ordered(_full_st(z, L, a, b), L, a, b))

def _guess(L, a, b, kind, power):
    """Chebyshev extrema (kind 0) or Chebyshev zeros (kind 1) on each interval, the relative positions raised to a power, as the initial stationary points"""
    R = 7 - L; st = np.zeros(7)
    if L > 0 and a > 0.0:
        k = np.arange(1, L + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (L + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * L + 2)))
        st[:L] = (a * u ** power)[::-1]
    if R > 0 and b < 1.0:
        k = np.arange(1, R + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (R + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * R + 2)))
        st[L:] = b + (1.0 - b) * (1.0 - (1.0 - u) ** power)
    return st[_free_index(L, a, b)]

def _solve_equioscillatory(L, a, b):
    """the unique solution of the equioscillatory conditions with the ordering (4.7), from deterministic Chebyshev starts of increasing compression"""
    if len(_free_index(L, a, b)) == 0:
        return _assemble(np.zeros(0), L, a, b)
    for power in (1.0, 1.5, 0.75, 2.0, 0.5, 3.0):
        for kind in (0, 1):
            z, ok = _newton_free(L, a, b, _guess(L, a, b, kind, power))
            if ok: return _assemble(z, L, a, b)
    raise ValueError("the equioscillatory conditions could not be solved for L=%d on (%g, %g)" % (L, a, b))

def _coefficients(s):
    c = _pcoef(s[:7]) * s[7]; c[0] = c[0] + s[8]
    return c

def _transform(energies):
    """the source's normalisation (2.2): an eigenvalue e of H maps to (lmax - e)/(lmax - lmin); Table 1 order lmin, out_homo, in_homo, in_lumo, out_lumo, lmax"""
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size != 6 or not np.all(np.isfinite(e)): raise ValueError("energies must hold six finite values")
    lmin, oh, ih, il, ol, lmax = e
    if not (lmin < oh <= ih < il <= ol < lmax): raise ValueError("energies must be ordered lmin < out_homo <= in_homo < in_lumo <= out_lumo < lmax")
    T = (lmax - np.array([il, ih, ol, oh])) / (lmax - lmin)
    return T   # in_lumo, in_homo, out_lumo, out_homo on [0, 1]

def _build(bounds, seed):
    """synthetic 16 by 16 matrix consistent with the bounds: the extreme eigenvalues of each interval at the midpoints of the bound intervals, the rest equidistant, on the sign-fixed QR frame of the mod-47 integer constructor"""
    n = _dim(); inl, inh, outl, outh = bounds
    lam_lumo = 0.5 * (outl + inl); lam_homo = 0.5 * (inh + outh)
    mu = 0.5 * (inl + inh); nocc = int(round(n * (1.0 - mu)))
    lam = np.concatenate([np.linspace(0.0, lam_lumo, n - nocc), np.linspace(lam_homo, 1.0, nocc)])
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (((i + 1) * (j + 2) * (i + j + 3) + seed * (i + 4) * (j + 5)) % 47) - 23
    Q, _r = np.linalg.qr(M)
    for c in range(n):
        col = Q[:, c]
        for v in col:
            if abs(v) > 1e-12:
                if v < 0:
                    Q[:, c] = -col
                break
    X0 = Q @ np.diag(lam) @ Q.T
    return (X0 + X0.T) / 2.0, nocc

def sp8_equioscillatory(L: int, lam_lumo: float, lam_homo: float) -> "np.ndarray":
    k = _check_index(L)
    a, b = _check_bounds(lam_lumo, lam_homo)
    return _solve_equioscillatory(k, a, b).astype(np.float64)

import numpy as np, math
from math import comb


def _kappa():
    """the source's deactivation threshold of the acceleration (Algorithm 5.1)"""
    return 0.01

def _dim():
    """the order of the synthetic matrices"""
    return 16

def _check_index(l):
    if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or not (0 <= int(l) <= 7):
        raise ValueError("l must be an integer in 0..7")
    return int(l)

def _table(l):
    """the source's tabulated closed-form members for l = 4..7 (Table 2), monomial order b0..b8"""
    t = {4: [0, 0, 0, 0, 0, 56, -140, 120, -35], 5: [0, 0, 0, 0, 0, 0, 28, -48, 21], 6: [0, 0, 0, 0, 0, 0, 0, 8, -7], 7: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    return np.array(t[l], dtype=np.float64)

def _horner(b, x):
    y = 0.0
    for k in range(8, -1, -1):
        y = y * x + b[k]
    return y

def _stop_constant(l):
    return [np.inf, 28.0, 56.0, 82.0, 82.0, 56.0, 28.0, np.inf][l]

def _stop_order(l):
    return [1, 2, 3, 4, 4, 3, 2, 1][l]

def _esym(st):
    """elementary symmetric polynomials e_0..e_7 of the seven stationary points"""
    e = np.zeros(8); e[0] = 1.0
    for v in st:
        e[1:] = e[1:] + v * e[:-1]
    return e

def _pcoef(st):
    """monomial coefficients of P(x) = int_0^x prod_k (t - s_k) dt, the parametrisation (4.1) with unit scale and zero offset"""
    e = _esym(st); c = np.zeros(9)
    for k in range(8):
        c[8 - k] = ((-1.0) ** k) * e[k] / (8 - k)
    return c

def _pdiff(st, x1, x0):
    """int_{x0}^{x1} prod_k (t - s_k) dt by the four-point Gauss rule, exact for the degree-seven integrand and free of cancellation"""
    g = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
    w = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    m = 0.5 * (x1 + x0); h = 0.5 * (x1 - x0); t = m + h * g
    f = np.ones(4)
    for s in st:
        f = f * (t - s)
    return h * float(np.dot(w, f))

def _check_bounds(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        raise ValueError("bounds must be real numbers")
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b > 1.0 or not (a < b):
        raise ValueError("bounds must satisfy 0 <= lam_lumo < lam_homo <= 1")
    return a, b

def _alternation_sets(st, L, a, b):
    """lower and upper alternation points of the left interval and upper and lower points of the right interval"""
    R = 7 - L
    lo = ([0.0] if L % 2 == 0 else []) + [st[k - 1] for k in range(1, L + 1) if k % 2 == 1]
    up = [a] + [st[k - 1] for k in range(1, L + 1) if k % 2 == 0] + ([0.0] if L % 2 == 1 else [])
    ru = [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 1] + ([1.0] if R % 2 == 0 else [])
    rl = [b] + [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 0] + ([1.0] if R % 2 == 1 else [])
    return lo, up, ru, rl

def _free_index(L, a, b):
    fl = [] if a == 0.0 else list(range(L)); fr = [] if b == 1.0 else list(range(L, 7))
    return fl + fr

def _full_st(free, L, a, b):
    st = np.zeros(7)
    if b == 1.0: st[L:] = 1.0
    st[_free_index(L, a, b)] = free
    return st

def _residual_free(free, L, a, b):
    """scale-free form of the equioscillatory conditions: equal values of P on each alternation set, normalised by the amplitude of the interval"""
    R = 7 - L; st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b); res = []
    if a > 0.0 and L > 0:
        amp = _pdiff(st, a, st[0])
        for x in lo[1:]: res.append(_pdiff(st, x, lo[0]) / amp)
        for x in up[1:]: res.append(_pdiff(st, x, up[0]) / amp)
    if b < 1.0 and R > 0:
        amp = _pdiff(st, st[L], b)
        for x in ru[1:]: res.append(_pdiff(st, x, ru[0]) / amp)
        for x in rl[1:]: res.append(_pdiff(st, x, rl[0]) / amp)
    return np.array(res)

def _strict_ordered(st, L, a, b):
    """the ordering (4.7) with strict inequalities on the free stationary points"""
    R = 7 - L
    if L > 0 and a > 0.0:
        if not (st[0] < a and st[L - 1] > 0.0): return False
        if L > 1 and not np.all(np.diff(st[:L]) < 0.0): return False
    if R > 0 and b < 1.0:
        if not (st[L] > b and st[6] < 1.0): return False
        if R > 1 and not np.all(np.diff(st[L:]) > 0.0): return False
    return True

def _assemble(free, L, a, b):
    """the nine parameters: stationary points, scale s8 and offset s9 fixed by p = 0 at the lower-left level and p = 1 at the upper-right level"""
    st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b)
    xl = 0.0 if a == 0.0 else lo[0]
    xr = 1.0 if b == 1.0 else ru[0]
    s8 = 1.0 / _pdiff(st, xr, xl); s9 = -s8 * _pdiff(st, xl, 0.0)
    return np.concatenate([st, [s8, s9]])

def _jacobian(free, L, a, b):
    n = free.size; f0 = _residual_free(free, L, a, b); J = np.zeros((f0.size, n)); h = 1e-7
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (_residual_free(free + d, L, a, b) - _residual_free(free - d, L, a, b)) / (2.0 * h)
    return J

def _newton_free(L, a, b, z0):
    """damped Newton iteration on the scale-free system that keeps the ordering (4.7)"""
    z = z0.copy(); F = _residual_free(z, L, a, b); nf = float(np.max(np.abs(F)))
    for it in range(60):
        if nf < 1e-12: break
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -F)
        except np.linalg.LinAlgError:
            return z, False
        lam = 1.0; accepted = False
        while lam > 1e-4:
            zn = z + lam * dz
            if _strict_ordered(_full_st(zn, L, a, b), L, a, b):
                Fn = _residual_free(zn, L, a, b); nfn = float(np.max(np.abs(Fn)))
                if nfn < nf or lam < 0.1:
                    z, F, nf = zn, Fn, nfn; accepted = True; break
            lam *= 0.5
        if not accepted: return z, False
    for it in range(3):
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -_residual_free(z, L, a, b))
        except np.linalg.LinAlgError:
            break
        zn = z + dz
        if _strict_ordered(_full_st(zn, L, a, b), L, a, b) and np.max(np.abs(_residual_free(zn, L, a, b))) <= np.max(np.abs(_residual_free(z, L, a, b))): z = zn
    nf = float(np.max(np.abs(_residual_free(z, L, a, b))))
    return z, (nf < 1e-11 and _strict_ordered(_full_st(z, L, a, b), L, a, b))

def _guess(L, a, b, kind, power):
    """Chebyshev extrema (kind 0) or Chebyshev zeros (kind 1) on each interval, the relative positions raised to a power, as the initial stationary points"""
    R = 7 - L; st = np.zeros(7)
    if L > 0 and a > 0.0:
        k = np.arange(1, L + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (L + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * L + 2)))
        st[:L] = (a * u ** power)[::-1]
    if R > 0 and b < 1.0:
        k = np.arange(1, R + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (R + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * R + 2)))
        st[L:] = b + (1.0 - b) * (1.0 - (1.0 - u) ** power)
    return st[_free_index(L, a, b)]

def _solve_equioscillatory(L, a, b):
    """the unique solution of the equioscillatory conditions with the ordering (4.7), from deterministic Chebyshev starts of increasing compression"""
    if len(_free_index(L, a, b)) == 0:
        return _assemble(np.zeros(0), L, a, b)
    for power in (1.0, 1.5, 0.75, 2.0, 0.5, 3.0):
        for kind in (0, 1):
            z, ok = _newton_free(L, a, b, _guess(L, a, b, kind, power))
            if ok: return _assemble(z, L, a, b)
    raise ValueError("the equioscillatory conditions could not be solved for L=%d on (%g, %g)" % (L, a, b))

def _coefficients(s):
    c = _pcoef(s[:7]) * s[7]; c[0] = c[0] + s[8]
    return c

def _transform(energies):
    """the source's normalisation (2.2): an eigenvalue e of H maps to (lmax - e)/(lmax - lmin); Table 1 order lmin, out_homo, in_homo, in_lumo, out_lumo, lmax"""
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size != 6 or not np.all(np.isfinite(e)): raise ValueError("energies must hold six finite values")
    lmin, oh, ih, il, ol, lmax = e
    if not (lmin < oh <= ih < il <= ol < lmax): raise ValueError("energies must be ordered lmin < out_homo <= in_homo < in_lumo <= out_lumo < lmax")
    T = (lmax - np.array([il, ih, ol, oh])) / (lmax - lmin)
    return T   # in_lumo, in_homo, out_lumo, out_homo on [0, 1]

def _build(bounds, seed):
    """synthetic 16 by 16 matrix consistent with the bounds: the extreme eigenvalues of each interval at the midpoints of the bound intervals, the rest equidistant, on the sign-fixed QR frame of the mod-47 integer constructor"""
    n = _dim(); inl, inh, outl, outh = bounds
    lam_lumo = 0.5 * (outl + inl); lam_homo = 0.5 * (inh + outh)
    mu = 0.5 * (inl + inh); nocc = int(round(n * (1.0 - mu)))
    lam = np.concatenate([np.linspace(0.0, lam_lumo, n - nocc), np.linspace(lam_homo, 1.0, nocc)])
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (((i + 1) * (j + 2) * (i + j + 3) + seed * (i + 4) * (j + 5)) % 47) - 23
    Q, _r = np.linalg.qr(M)
    for c in range(n):
        col = Q[:, c]
        for v in col:
            if abs(v) > 1e-12:
                if v < 0:
                    Q[:, c] = -col
                break
    X0 = Q @ np.diag(lam) @ Q.T
    return (X0 + X0.T) / 2.0, nocc

def sastre_coefficients(b: "np.ndarray") -> "np.ndarray":
    bb = np.asarray(b, dtype=np.float64)
    if bb.shape != (9,) or bb[8] == 0.0:
        raise ValueError("need 9 monomial coefficients with b8 nonzero")
    f4 = bb[8]
    c1 = bb[7] / (2.0 * f4)
    t2 = bb[6] / f4 - c1 ** 2
    t1 = bb[5] / f4 - c1 * t2
    d0 = 0.25 * (1.0 - t2 ** 2 + 4.0 * bb[4] / f4 - 4.0 * c1 * t1)
    e2 = 0.5 * (t2 + 1.0)
    d2 = 0.5 * (t2 - 1.0)
    e1 = c1 * d0 + t1 * e2 - bb[3] / f4
    d1 = t1 - e1
    f2 = bb[2] - f4 * (d0 * e2 + d1 * e1)
    f1 = bb[1] - f4 * d0 * e1
    f0 = bb[0]
    return np.array([c1, d0, d1, d2, e1, e2, f0, f1, f2, f4], dtype=np.float64)

import numpy as np, math
from math import comb


def _kappa():
    """the source's deactivation threshold of the acceleration (Algorithm 5.1)"""
    return 0.01

def _dim():
    """the order of the synthetic matrices"""
    return 16

def _check_index(l):
    if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or not (0 <= int(l) <= 7):
        raise ValueError("l must be an integer in 0..7")
    return int(l)

def _table(l):
    """the source's tabulated closed-form members for l = 4..7 (Table 2), monomial order b0..b8"""
    t = {4: [0, 0, 0, 0, 0, 56, -140, 120, -35], 5: [0, 0, 0, 0, 0, 0, 28, -48, 21], 6: [0, 0, 0, 0, 0, 0, 0, 8, -7], 7: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    return np.array(t[l], dtype=np.float64)

def _horner(b, x):
    y = 0.0
    for k in range(8, -1, -1):
        y = y * x + b[k]
    return y

def _stop_constant(l):
    return [np.inf, 28.0, 56.0, 82.0, 82.0, 56.0, 28.0, np.inf][l]

def _stop_order(l):
    return [1, 2, 3, 4, 4, 3, 2, 1][l]

def _esym(st):
    """elementary symmetric polynomials e_0..e_7 of the seven stationary points"""
    e = np.zeros(8); e[0] = 1.0
    for v in st:
        e[1:] = e[1:] + v * e[:-1]
    return e

def _pcoef(st):
    """monomial coefficients of P(x) = int_0^x prod_k (t - s_k) dt, the parametrisation (4.1) with unit scale and zero offset"""
    e = _esym(st); c = np.zeros(9)
    for k in range(8):
        c[8 - k] = ((-1.0) ** k) * e[k] / (8 - k)
    return c

def _pdiff(st, x1, x0):
    """int_{x0}^{x1} prod_k (t - s_k) dt by the four-point Gauss rule, exact for the degree-seven integrand and free of cancellation"""
    g = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
    w = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    m = 0.5 * (x1 + x0); h = 0.5 * (x1 - x0); t = m + h * g
    f = np.ones(4)
    for s in st:
        f = f * (t - s)
    return h * float(np.dot(w, f))

def _check_bounds(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        raise ValueError("bounds must be real numbers")
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b > 1.0 or not (a < b):
        raise ValueError("bounds must satisfy 0 <= lam_lumo < lam_homo <= 1")
    return a, b

def _alternation_sets(st, L, a, b):
    """lower and upper alternation points of the left interval and upper and lower points of the right interval"""
    R = 7 - L
    lo = ([0.0] if L % 2 == 0 else []) + [st[k - 1] for k in range(1, L + 1) if k % 2 == 1]
    up = [a] + [st[k - 1] for k in range(1, L + 1) if k % 2 == 0] + ([0.0] if L % 2 == 1 else [])
    ru = [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 1] + ([1.0] if R % 2 == 0 else [])
    rl = [b] + [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 0] + ([1.0] if R % 2 == 1 else [])
    return lo, up, ru, rl

def _free_index(L, a, b):
    fl = [] if a == 0.0 else list(range(L)); fr = [] if b == 1.0 else list(range(L, 7))
    return fl + fr

def _full_st(free, L, a, b):
    st = np.zeros(7)
    if b == 1.0: st[L:] = 1.0
    st[_free_index(L, a, b)] = free
    return st

def _residual_free(free, L, a, b):
    """scale-free form of the equioscillatory conditions: equal values of P on each alternation set, normalised by the amplitude of the interval"""
    R = 7 - L; st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b); res = []
    if a > 0.0 and L > 0:
        amp = _pdiff(st, a, st[0])
        for x in lo[1:]: res.append(_pdiff(st, x, lo[0]) / amp)
        for x in up[1:]: res.append(_pdiff(st, x, up[0]) / amp)
    if b < 1.0 and R > 0:
        amp = _pdiff(st, st[L], b)
        for x in ru[1:]: res.append(_pdiff(st, x, ru[0]) / amp)
        for x in rl[1:]: res.append(_pdiff(st, x, rl[0]) / amp)
    return np.array(res)

def _strict_ordered(st, L, a, b):
    """the ordering (4.7) with strict inequalities on the free stationary points"""
    R = 7 - L
    if L > 0 and a > 0.0:
        if not (st[0] < a and st[L - 1] > 0.0): return False
        if L > 1 and not np.all(np.diff(st[:L]) < 0.0): return False
    if R > 0 and b < 1.0:
        if not (st[L] > b and st[6] < 1.0): return False
        if R > 1 and not np.all(np.diff(st[L:]) > 0.0): return False
    return True

def _assemble(free, L, a, b):
    """the nine parameters: stationary points, scale s8 and offset s9 fixed by p = 0 at the lower-left level and p = 1 at the upper-right level"""
    st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b)
    xl = 0.0 if a == 0.0 else lo[0]
    xr = 1.0 if b == 1.0 else ru[0]
    s8 = 1.0 / _pdiff(st, xr, xl); s9 = -s8 * _pdiff(st, xl, 0.0)
    return np.concatenate([st, [s8, s9]])

def _jacobian(free, L, a, b):
    n = free.size; f0 = _residual_free(free, L, a, b); J = np.zeros((f0.size, n)); h = 1e-7
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (_residual_free(free + d, L, a, b) - _residual_free(free - d, L, a, b)) / (2.0 * h)
    return J

def _newton_free(L, a, b, z0):
    """damped Newton iteration on the scale-free system that keeps the ordering (4.7)"""
    z = z0.copy(); F = _residual_free(z, L, a, b); nf = float(np.max(np.abs(F)))
    for it in range(60):
        if nf < 1e-12: break
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -F)
        except np.linalg.LinAlgError:
            return z, False
        lam = 1.0; accepted = False
        while lam > 1e-4:
            zn = z + lam * dz
            if _strict_ordered(_full_st(zn, L, a, b), L, a, b):
                Fn = _residual_free(zn, L, a, b); nfn = float(np.max(np.abs(Fn)))
                if nfn < nf or lam < 0.1:
                    z, F, nf = zn, Fn, nfn; accepted = True; break
            lam *= 0.5
        if not accepted: return z, False
    for it in range(3):
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -_residual_free(z, L, a, b))
        except np.linalg.LinAlgError:
            break
        zn = z + dz
        if _strict_ordered(_full_st(zn, L, a, b), L, a, b) and np.max(np.abs(_residual_free(zn, L, a, b))) <= np.max(np.abs(_residual_free(z, L, a, b))): z = zn
    nf = float(np.max(np.abs(_residual_free(z, L, a, b))))
    return z, (nf < 1e-11 and _strict_ordered(_full_st(z, L, a, b), L, a, b))

def _guess(L, a, b, kind, power):
    """Chebyshev extrema (kind 0) or Chebyshev zeros (kind 1) on each interval, the relative positions raised to a power, as the initial stationary points"""
    R = 7 - L; st = np.zeros(7)
    if L > 0 and a > 0.0:
        k = np.arange(1, L + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (L + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * L + 2)))
        st[:L] = (a * u ** power)[::-1]
    if R > 0 and b < 1.0:
        k = np.arange(1, R + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (R + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * R + 2)))
        st[L:] = b + (1.0 - b) * (1.0 - (1.0 - u) ** power)
    return st[_free_index(L, a, b)]

def _solve_equioscillatory(L, a, b):
    """the unique solution of the equioscillatory conditions with the ordering (4.7), from deterministic Chebyshev starts of increasing compression"""
    if len(_free_index(L, a, b)) == 0:
        return _assemble(np.zeros(0), L, a, b)
    for power in (1.0, 1.5, 0.75, 2.0, 0.5, 3.0):
        for kind in (0, 1):
            z, ok = _newton_free(L, a, b, _guess(L, a, b, kind, power))
            if ok: return _assemble(z, L, a, b)
    raise ValueError("the equioscillatory conditions could not be solved for L=%d on (%g, %g)" % (L, a, b))

def _coefficients(s):
    c = _pcoef(s[:7]) * s[7]; c[0] = c[0] + s[8]
    return c

def _transform(energies):
    """the source's normalisation (2.2): an eigenvalue e of H maps to (lmax - e)/(lmax - lmin); Table 1 order lmin, out_homo, in_homo, in_lumo, out_lumo, lmax"""
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size != 6 or not np.all(np.isfinite(e)): raise ValueError("energies must hold six finite values")
    lmin, oh, ih, il, ol, lmax = e
    if not (lmin < oh <= ih < il <= ol < lmax): raise ValueError("energies must be ordered lmin < out_homo <= in_homo < in_lumo <= out_lumo < lmax")
    T = (lmax - np.array([il, ih, ol, oh])) / (lmax - lmin)
    return T   # in_lumo, in_homo, out_lumo, out_homo on [0, 1]

def _build(bounds, seed):
    """synthetic 16 by 16 matrix consistent with the bounds: the extreme eigenvalues of each interval at the midpoints of the bound intervals, the rest equidistant, on the sign-fixed QR frame of the mod-47 integer constructor"""
    n = _dim(); inl, inh, outl, outh = bounds
    lam_lumo = 0.5 * (outl + inl); lam_homo = 0.5 * (inh + outh)
    mu = 0.5 * (inl + inh); nocc = int(round(n * (1.0 - mu)))
    lam = np.concatenate([np.linspace(0.0, lam_lumo, n - nocc), np.linspace(lam_homo, 1.0, nocc)])
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (((i + 1) * (j + 2) * (i + j + 3) + seed * (i + 4) * (j + 5)) % 47) - 23
    Q, _r = np.linalg.qr(M)
    for c in range(n):
        col = Q[:, c]
        for v in col:
            if abs(v) > 1e-12:
                if v < 0:
                    Q[:, c] = -col
                break
    X0 = Q @ np.diag(lam) @ Q.T
    return (X0 + X0.T) / 2.0, nocc

def workspace_scalars(coeffs: "np.ndarray") -> "np.ndarray":
    co = np.asarray(coeffs, dtype=np.float64)
    if co.shape != (10,):
        raise ValueError("coeffs must have shape (10,)")
    if not np.all(np.isfinite(co)):
        raise ValueError("coeffs must be finite")
    c1, d0, d1, d2, e1, e2, f0, f1, f2, f4 = co
    r1 = d1 - 0.5 * c1 * (d2 - 0.25 * c1 ** 2)
    r2 = d2 - 0.25 * c1 ** 2
    r3 = e1 - d1 - 0.5 * c1
    r4 = f1 - f2 * (e1 - d1)
    return np.array([r1, r2, r3, r4], dtype=np.float64)

import numpy as np, math
from math import comb


def _kappa():
    """the source's deactivation threshold of the acceleration (Algorithm 5.1)"""
    return 0.01

def _dim():
    """the order of the synthetic matrices"""
    return 16

def _check_index(l):
    if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or not (0 <= int(l) <= 7):
        raise ValueError("l must be an integer in 0..7")
    return int(l)

def _table(l):
    """the source's tabulated closed-form members for l = 4..7 (Table 2), monomial order b0..b8"""
    t = {4: [0, 0, 0, 0, 0, 56, -140, 120, -35], 5: [0, 0, 0, 0, 0, 0, 28, -48, 21], 6: [0, 0, 0, 0, 0, 0, 0, 8, -7], 7: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    return np.array(t[l], dtype=np.float64)

def _horner(b, x):
    y = 0.0
    for k in range(8, -1, -1):
        y = y * x + b[k]
    return y

def _stop_constant(l):
    return [np.inf, 28.0, 56.0, 82.0, 82.0, 56.0, 28.0, np.inf][l]

def _stop_order(l):
    return [1, 2, 3, 4, 4, 3, 2, 1][l]

def _esym(st):
    """elementary symmetric polynomials e_0..e_7 of the seven stationary points"""
    e = np.zeros(8); e[0] = 1.0
    for v in st:
        e[1:] = e[1:] + v * e[:-1]
    return e

def _pcoef(st):
    """monomial coefficients of P(x) = int_0^x prod_k (t - s_k) dt, the parametrisation (4.1) with unit scale and zero offset"""
    e = _esym(st); c = np.zeros(9)
    for k in range(8):
        c[8 - k] = ((-1.0) ** k) * e[k] / (8 - k)
    return c

def _pdiff(st, x1, x0):
    """int_{x0}^{x1} prod_k (t - s_k) dt by the four-point Gauss rule, exact for the degree-seven integrand and free of cancellation"""
    g = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
    w = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    m = 0.5 * (x1 + x0); h = 0.5 * (x1 - x0); t = m + h * g
    f = np.ones(4)
    for s in st:
        f = f * (t - s)
    return h * float(np.dot(w, f))

def _check_bounds(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        raise ValueError("bounds must be real numbers")
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b > 1.0 or not (a < b):
        raise ValueError("bounds must satisfy 0 <= lam_lumo < lam_homo <= 1")
    return a, b

def _alternation_sets(st, L, a, b):
    """lower and upper alternation points of the left interval and upper and lower points of the right interval"""
    R = 7 - L
    lo = ([0.0] if L % 2 == 0 else []) + [st[k - 1] for k in range(1, L + 1) if k % 2 == 1]
    up = [a] + [st[k - 1] for k in range(1, L + 1) if k % 2 == 0] + ([0.0] if L % 2 == 1 else [])
    ru = [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 1] + ([1.0] if R % 2 == 0 else [])
    rl = [b] + [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 0] + ([1.0] if R % 2 == 1 else [])
    return lo, up, ru, rl

def _free_index(L, a, b):
    fl = [] if a == 0.0 else list(range(L)); fr = [] if b == 1.0 else list(range(L, 7))
    return fl + fr

def _full_st(free, L, a, b):
    st = np.zeros(7)
    if b == 1.0: st[L:] = 1.0
    st[_free_index(L, a, b)] = free
    return st

def _residual_free(free, L, a, b):
    """scale-free form of the equioscillatory conditions: equal values of P on each alternation set, normalised by the amplitude of the interval"""
    R = 7 - L; st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b); res = []
    if a > 0.0 and L > 0:
        amp = _pdiff(st, a, st[0])
        for x in lo[1:]: res.append(_pdiff(st, x, lo[0]) / amp)
        for x in up[1:]: res.append(_pdiff(st, x, up[0]) / amp)
    if b < 1.0 and R > 0:
        amp = _pdiff(st, st[L], b)
        for x in ru[1:]: res.append(_pdiff(st, x, ru[0]) / amp)
        for x in rl[1:]: res.append(_pdiff(st, x, rl[0]) / amp)
    return np.array(res)

def _strict_ordered(st, L, a, b):
    """the ordering (4.7) with strict inequalities on the free stationary points"""
    R = 7 - L
    if L > 0 and a > 0.0:
        if not (st[0] < a and st[L - 1] > 0.0): return False
        if L > 1 and not np.all(np.diff(st[:L]) < 0.0): return False
    if R > 0 and b < 1.0:
        if not (st[L] > b and st[6] < 1.0): return False
        if R > 1 and not np.all(np.diff(st[L:]) > 0.0): return False
    return True

def _assemble(free, L, a, b):
    """the nine parameters: stationary points, scale s8 and offset s9 fixed by p = 0 at the lower-left level and p = 1 at the upper-right level"""
    st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b)
    xl = 0.0 if a == 0.0 else lo[0]
    xr = 1.0 if b == 1.0 else ru[0]
    s8 = 1.0 / _pdiff(st, xr, xl); s9 = -s8 * _pdiff(st, xl, 0.0)
    return np.concatenate([st, [s8, s9]])

def _jacobian(free, L, a, b):
    n = free.size; f0 = _residual_free(free, L, a, b); J = np.zeros((f0.size, n)); h = 1e-7
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (_residual_free(free + d, L, a, b) - _residual_free(free - d, L, a, b)) / (2.0 * h)
    return J

def _newton_free(L, a, b, z0):
    """damped Newton iteration on the scale-free system that keeps the ordering (4.7)"""
    z = z0.copy(); F = _residual_free(z, L, a, b); nf = float(np.max(np.abs(F)))
    for it in range(60):
        if nf < 1e-12: break
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -F)
        except np.linalg.LinAlgError:
            return z, False
        lam = 1.0; accepted = False
        while lam > 1e-4:
            zn = z + lam * dz
            if _strict_ordered(_full_st(zn, L, a, b), L, a, b):
                Fn = _residual_free(zn, L, a, b); nfn = float(np.max(np.abs(Fn)))
                if nfn < nf or lam < 0.1:
                    z, F, nf = zn, Fn, nfn; accepted = True; break
            lam *= 0.5
        if not accepted: return z, False
    for it in range(3):
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -_residual_free(z, L, a, b))
        except np.linalg.LinAlgError:
            break
        zn = z + dz
        if _strict_ordered(_full_st(zn, L, a, b), L, a, b) and np.max(np.abs(_residual_free(zn, L, a, b))) <= np.max(np.abs(_residual_free(z, L, a, b))): z = zn
    nf = float(np.max(np.abs(_residual_free(z, L, a, b))))
    return z, (nf < 1e-11 and _strict_ordered(_full_st(z, L, a, b), L, a, b))

def _guess(L, a, b, kind, power):
    """Chebyshev extrema (kind 0) or Chebyshev zeros (kind 1) on each interval, the relative positions raised to a power, as the initial stationary points"""
    R = 7 - L; st = np.zeros(7)
    if L > 0 and a > 0.0:
        k = np.arange(1, L + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (L + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * L + 2)))
        st[:L] = (a * u ** power)[::-1]
    if R > 0 and b < 1.0:
        k = np.arange(1, R + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (R + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * R + 2)))
        st[L:] = b + (1.0 - b) * (1.0 - (1.0 - u) ** power)
    return st[_free_index(L, a, b)]

def _solve_equioscillatory(L, a, b):
    """the unique solution of the equioscillatory conditions with the ordering (4.7), from deterministic Chebyshev starts of increasing compression"""
    if len(_free_index(L, a, b)) == 0:
        return _assemble(np.zeros(0), L, a, b)
    for power in (1.0, 1.5, 0.75, 2.0, 0.5, 3.0):
        for kind in (0, 1):
            z, ok = _newton_free(L, a, b, _guess(L, a, b, kind, power))
            if ok: return _assemble(z, L, a, b)
    raise ValueError("the equioscillatory conditions could not be solved for L=%d on (%g, %g)" % (L, a, b))

def _coefficients(s):
    c = _pcoef(s[:7]) * s[7]; c[0] = c[0] + s[8]
    return c

def _transform(energies):
    """the source's normalisation (2.2): an eigenvalue e of H maps to (lmax - e)/(lmax - lmin); Table 1 order lmin, out_homo, in_homo, in_lumo, out_lumo, lmax"""
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size != 6 or not np.all(np.isfinite(e)): raise ValueError("energies must hold six finite values")
    lmin, oh, ih, il, ol, lmax = e
    if not (lmin < oh <= ih < il <= ol < lmax): raise ValueError("energies must be ordered lmin < out_homo <= in_homo < in_lumo <= out_lumo < lmax")
    T = (lmax - np.array([il, ih, ol, oh])) / (lmax - lmin)
    return T   # in_lumo, in_homo, out_lumo, out_homo on [0, 1]

def _build(bounds, seed):
    """synthetic 16 by 16 matrix consistent with the bounds: the extreme eigenvalues of each interval at the midpoints of the bound intervals, the rest equidistant, on the sign-fixed QR frame of the mod-47 integer constructor"""
    n = _dim(); inl, inh, outl, outh = bounds
    lam_lumo = 0.5 * (outl + inl); lam_homo = 0.5 * (inh + outh)
    mu = 0.5 * (inl + inh); nocc = int(round(n * (1.0 - mu)))
    lam = np.concatenate([np.linspace(0.0, lam_lumo, n - nocc), np.linspace(lam_homo, 1.0, nocc)])
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (((i + 1) * (j + 2) * (i + j + 3) + seed * (i + 4) * (j + 5)) % 47) - 23
    Q, _r = np.linalg.qr(M)
    for c in range(n):
        col = Q[:, c]
        for v in col:
            if abs(v) > 1e-12:
                if v < 0:
                    Q[:, c] = -col
                break
    X0 = Q @ np.diag(lam) @ Q.T
    return (X0 + X0.T) / 2.0, nocc

def evaluate_p8(X: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    XX = np.asarray(X, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if XX.ndim != 2 or XX.shape[0] != XX.shape[1]:
        raise ValueError("X must be a square matrix")
    if bb.shape != (9,):
        raise ValueError("b must have shape (9,)")
    if bb[8] == 0.0:
        raise ValueError("b[8] must be nonzero for the degree-eight scheme")
    co = sastre_coefficients(bb)
    c1, d0, d1, d2, e1, e2, f0, f1, f2, f4 = co
    r1, r2, r3, r4 = workspace_scalars(co)
    I = np.eye(XX.shape[0])
    M1 = XX.copy()
    M2 = M1 @ M1
    M2 = M2 + 0.5 * c1 * M1
    M3 = M2 @ M2
    M3 = M3 + r1 * M1
    M3 = M3 + r2 * M2
    M2 = M2 + r3 * M1
    M1 = r4 * M1 + f2 * M2 + f0 * I
    M2 = M2 + M3
    M3 = M3 + d0 * I
    M1 = f4 * (M2 @ M3) + M1
    return (M1 + M1.T) / 2.0

import numpy as np, math
from math import comb


def _kappa():
    """the source's deactivation threshold of the acceleration (Algorithm 5.1)"""
    return 0.01

def _dim():
    """the order of the synthetic matrices"""
    return 16

def _check_index(l):
    if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or not (0 <= int(l) <= 7):
        raise ValueError("l must be an integer in 0..7")
    return int(l)

def _table(l):
    """the source's tabulated closed-form members for l = 4..7 (Table 2), monomial order b0..b8"""
    t = {4: [0, 0, 0, 0, 0, 56, -140, 120, -35], 5: [0, 0, 0, 0, 0, 0, 28, -48, 21], 6: [0, 0, 0, 0, 0, 0, 0, 8, -7], 7: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    return np.array(t[l], dtype=np.float64)

def _horner(b, x):
    y = 0.0
    for k in range(8, -1, -1):
        y = y * x + b[k]
    return y

def _stop_constant(l):
    return [np.inf, 28.0, 56.0, 82.0, 82.0, 56.0, 28.0, np.inf][l]

def _stop_order(l):
    return [1, 2, 3, 4, 4, 3, 2, 1][l]

def _esym(st):
    """elementary symmetric polynomials e_0..e_7 of the seven stationary points"""
    e = np.zeros(8); e[0] = 1.0
    for v in st:
        e[1:] = e[1:] + v * e[:-1]
    return e

def _pcoef(st):
    """monomial coefficients of P(x) = int_0^x prod_k (t - s_k) dt, the parametrisation (4.1) with unit scale and zero offset"""
    e = _esym(st); c = np.zeros(9)
    for k in range(8):
        c[8 - k] = ((-1.0) ** k) * e[k] / (8 - k)
    return c

def _pdiff(st, x1, x0):
    """int_{x0}^{x1} prod_k (t - s_k) dt by the four-point Gauss rule, exact for the degree-seven integrand and free of cancellation"""
    g = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
    w = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    m = 0.5 * (x1 + x0); h = 0.5 * (x1 - x0); t = m + h * g
    f = np.ones(4)
    for s in st:
        f = f * (t - s)
    return h * float(np.dot(w, f))

def _check_bounds(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        raise ValueError("bounds must be real numbers")
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b > 1.0 or not (a < b):
        raise ValueError("bounds must satisfy 0 <= lam_lumo < lam_homo <= 1")
    return a, b

def _alternation_sets(st, L, a, b):
    """lower and upper alternation points of the left interval and upper and lower points of the right interval"""
    R = 7 - L
    lo = ([0.0] if L % 2 == 0 else []) + [st[k - 1] for k in range(1, L + 1) if k % 2 == 1]
    up = [a] + [st[k - 1] for k in range(1, L + 1) if k % 2 == 0] + ([0.0] if L % 2 == 1 else [])
    ru = [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 1] + ([1.0] if R % 2 == 0 else [])
    rl = [b] + [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 0] + ([1.0] if R % 2 == 1 else [])
    return lo, up, ru, rl

def _free_index(L, a, b):
    fl = [] if a == 0.0 else list(range(L)); fr = [] if b == 1.0 else list(range(L, 7))
    return fl + fr

def _full_st(free, L, a, b):
    st = np.zeros(7)
    if b == 1.0: st[L:] = 1.0
    st[_free_index(L, a, b)] = free
    return st

def _residual_free(free, L, a, b):
    """scale-free form of the equioscillatory conditions: equal values of P on each alternation set, normalised by the amplitude of the interval"""
    R = 7 - L; st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b); res = []
    if a > 0.0 and L > 0:
        amp = _pdiff(st, a, st[0])
        for x in lo[1:]: res.append(_pdiff(st, x, lo[0]) / amp)
        for x in up[1:]: res.append(_pdiff(st, x, up[0]) / amp)
    if b < 1.0 and R > 0:
        amp = _pdiff(st, st[L], b)
        for x in ru[1:]: res.append(_pdiff(st, x, ru[0]) / amp)
        for x in rl[1:]: res.append(_pdiff(st, x, rl[0]) / amp)
    return np.array(res)

def _strict_ordered(st, L, a, b):
    """the ordering (4.7) with strict inequalities on the free stationary points"""
    R = 7 - L
    if L > 0 and a > 0.0:
        if not (st[0] < a and st[L - 1] > 0.0): return False
        if L > 1 and not np.all(np.diff(st[:L]) < 0.0): return False
    if R > 0 and b < 1.0:
        if not (st[L] > b and st[6] < 1.0): return False
        if R > 1 and not np.all(np.diff(st[L:]) > 0.0): return False
    return True

def _assemble(free, L, a, b):
    """the nine parameters: stationary points, scale s8 and offset s9 fixed by p = 0 at the lower-left level and p = 1 at the upper-right level"""
    st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b)
    xl = 0.0 if a == 0.0 else lo[0]
    xr = 1.0 if b == 1.0 else ru[0]
    s8 = 1.0 / _pdiff(st, xr, xl); s9 = -s8 * _pdiff(st, xl, 0.0)
    return np.concatenate([st, [s8, s9]])

def _jacobian(free, L, a, b):
    n = free.size; f0 = _residual_free(free, L, a, b); J = np.zeros((f0.size, n)); h = 1e-7
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (_residual_free(free + d, L, a, b) - _residual_free(free - d, L, a, b)) / (2.0 * h)
    return J

def _newton_free(L, a, b, z0):
    """damped Newton iteration on the scale-free system that keeps the ordering (4.7)"""
    z = z0.copy(); F = _residual_free(z, L, a, b); nf = float(np.max(np.abs(F)))
    for it in range(60):
        if nf < 1e-12: break
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -F)
        except np.linalg.LinAlgError:
            return z, False
        lam = 1.0; accepted = False
        while lam > 1e-4:
            zn = z + lam * dz
            if _strict_ordered(_full_st(zn, L, a, b), L, a, b):
                Fn = _residual_free(zn, L, a, b); nfn = float(np.max(np.abs(Fn)))
                if nfn < nf or lam < 0.1:
                    z, F, nf = zn, Fn, nfn; accepted = True; break
            lam *= 0.5
        if not accepted: return z, False
    for it in range(3):
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -_residual_free(z, L, a, b))
        except np.linalg.LinAlgError:
            break
        zn = z + dz
        if _strict_ordered(_full_st(zn, L, a, b), L, a, b) and np.max(np.abs(_residual_free(zn, L, a, b))) <= np.max(np.abs(_residual_free(z, L, a, b))): z = zn
    nf = float(np.max(np.abs(_residual_free(z, L, a, b))))
    return z, (nf < 1e-11 and _strict_ordered(_full_st(z, L, a, b), L, a, b))

def _guess(L, a, b, kind, power):
    """Chebyshev extrema (kind 0) or Chebyshev zeros (kind 1) on each interval, the relative positions raised to a power, as the initial stationary points"""
    R = 7 - L; st = np.zeros(7)
    if L > 0 and a > 0.0:
        k = np.arange(1, L + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (L + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * L + 2)))
        st[:L] = (a * u ** power)[::-1]
    if R > 0 and b < 1.0:
        k = np.arange(1, R + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (R + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * R + 2)))
        st[L:] = b + (1.0 - b) * (1.0 - (1.0 - u) ** power)
    return st[_free_index(L, a, b)]

def _solve_equioscillatory(L, a, b):
    """the unique solution of the equioscillatory conditions with the ordering (4.7), from deterministic Chebyshev starts of increasing compression"""
    if len(_free_index(L, a, b)) == 0:
        return _assemble(np.zeros(0), L, a, b)
    for power in (1.0, 1.5, 0.75, 2.0, 0.5, 3.0):
        for kind in (0, 1):
            z, ok = _newton_free(L, a, b, _guess(L, a, b, kind, power))
            if ok: return _assemble(z, L, a, b)
    raise ValueError("the equioscillatory conditions could not be solved for L=%d on (%g, %g)" % (L, a, b))

def _coefficients(s):
    c = _pcoef(s[:7]) * s[7]; c[0] = c[0] + s[8]
    return c

def _transform(energies):
    """the source's normalisation (2.2): an eigenvalue e of H maps to (lmax - e)/(lmax - lmin); Table 1 order lmin, out_homo, in_homo, in_lumo, out_lumo, lmax"""
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size != 6 or not np.all(np.isfinite(e)): raise ValueError("energies must hold six finite values")
    lmin, oh, ih, il, ol, lmax = e
    if not (lmin < oh <= ih < il <= ol < lmax): raise ValueError("energies must be ordered lmin < out_homo <= in_homo < in_lumo <= out_lumo < lmax")
    T = (lmax - np.array([il, ih, ol, oh])) / (lmax - lmin)
    return T   # in_lumo, in_homo, out_lumo, out_homo on [0, 1]

def _build(bounds, seed):
    """synthetic 16 by 16 matrix consistent with the bounds: the extreme eigenvalues of each interval at the midpoints of the bound intervals, the rest equidistant, on the sign-fixed QR frame of the mod-47 integer constructor"""
    n = _dim(); inl, inh, outl, outh = bounds
    lam_lumo = 0.5 * (outl + inl); lam_homo = 0.5 * (inh + outh)
    mu = 0.5 * (inl + inh); nocc = int(round(n * (1.0 - mu)))
    lam = np.concatenate([np.linspace(0.0, lam_lumo, n - nocc), np.linspace(lam_homo, 1.0, nocc)])
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (((i + 1) * (j + 2) * (i + j + 3) + seed * (i + 4) * (j + 5)) % 47) - 23
    Q, _r = np.linalg.qr(M)
    for c in range(n):
        col = Q[:, c]
        for v in col:
            if abs(v) > 1e-12:
                if v < 0:
                    Q[:, c] = -col
                break
    X0 = Q @ np.diag(lam) @ Q.T
    return (X0 + X0.T) / 2.0, nocc

def select_member(bounds: "np.ndarray", i: int, kappa: float) -> "np.ndarray":
    if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 1:
        raise ValueError("i must be a 1-based integer iteration index >= 1")
    kap = float(kappa)
    if not (0.0 < kap < 0.5):
        raise ValueError("kappa must lie strictly between 0 and 1/2")
    bd = np.asarray(bounds, dtype=np.float64).ravel()
    if bd.size != 4 or not np.all(np.isfinite(bd)):
        raise ValueError("bounds must hold four finite values")
    inl, inh, outl, outh = bd
    if not (0.0 <= outl <= inl < inh <= outh <= 1.0):
        raise ValueError("bounds must be ordered 0 <= out_lumo <= in_lumo < in_homo <= out_homo <= 1")
    a = outl if outl > kap else 0.0
    b = outh if outh < 1.0 - kap else 1.0
    if inl < kap and 1.0 - kap < inh:
        return np.array([3.0 if i % 2 == 1 else 4.0, a, b], dtype=np.float64)
    gaps = []
    for L in range(8):
        c = sp8_from_stationary_points(sp8_equioscillatory(L, a, b))
        gaps.append(_horner(c, inh) - _horner(c, inl))
    return np.array([float(int(np.argmax(gaps))), a, b], dtype=np.float64)

import numpy as np, math
from math import comb


def _kappa():
    """the source's deactivation threshold of the acceleration (Algorithm 5.1)"""
    return 0.01

def _dim():
    """the order of the synthetic matrices"""
    return 16

def _check_index(l):
    if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or not (0 <= int(l) <= 7):
        raise ValueError("l must be an integer in 0..7")
    return int(l)

def _table(l):
    """the source's tabulated closed-form members for l = 4..7 (Table 2), monomial order b0..b8"""
    t = {4: [0, 0, 0, 0, 0, 56, -140, 120, -35], 5: [0, 0, 0, 0, 0, 0, 28, -48, 21], 6: [0, 0, 0, 0, 0, 0, 0, 8, -7], 7: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    return np.array(t[l], dtype=np.float64)

def _horner(b, x):
    y = 0.0
    for k in range(8, -1, -1):
        y = y * x + b[k]
    return y

def _stop_constant(l):
    return [np.inf, 28.0, 56.0, 82.0, 82.0, 56.0, 28.0, np.inf][l]

def _stop_order(l):
    return [1, 2, 3, 4, 4, 3, 2, 1][l]

def _esym(st):
    """elementary symmetric polynomials e_0..e_7 of the seven stationary points"""
    e = np.zeros(8); e[0] = 1.0
    for v in st:
        e[1:] = e[1:] + v * e[:-1]
    return e

def _pcoef(st):
    """monomial coefficients of P(x) = int_0^x prod_k (t - s_k) dt, the parametrisation (4.1) with unit scale and zero offset"""
    e = _esym(st); c = np.zeros(9)
    for k in range(8):
        c[8 - k] = ((-1.0) ** k) * e[k] / (8 - k)
    return c

def _pdiff(st, x1, x0):
    """int_{x0}^{x1} prod_k (t - s_k) dt by the four-point Gauss rule, exact for the degree-seven integrand and free of cancellation"""
    g = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
    w = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    m = 0.5 * (x1 + x0); h = 0.5 * (x1 - x0); t = m + h * g
    f = np.ones(4)
    for s in st:
        f = f * (t - s)
    return h * float(np.dot(w, f))

def _check_bounds(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        raise ValueError("bounds must be real numbers")
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b > 1.0 or not (a < b):
        raise ValueError("bounds must satisfy 0 <= lam_lumo < lam_homo <= 1")
    return a, b

def _alternation_sets(st, L, a, b):
    """lower and upper alternation points of the left interval and upper and lower points of the right interval"""
    R = 7 - L
    lo = ([0.0] if L % 2 == 0 else []) + [st[k - 1] for k in range(1, L + 1) if k % 2 == 1]
    up = [a] + [st[k - 1] for k in range(1, L + 1) if k % 2 == 0] + ([0.0] if L % 2 == 1 else [])
    ru = [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 1] + ([1.0] if R % 2 == 0 else [])
    rl = [b] + [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 0] + ([1.0] if R % 2 == 1 else [])
    return lo, up, ru, rl

def _free_index(L, a, b):
    fl = [] if a == 0.0 else list(range(L)); fr = [] if b == 1.0 else list(range(L, 7))
    return fl + fr

def _full_st(free, L, a, b):
    st = np.zeros(7)
    if b == 1.0: st[L:] = 1.0
    st[_free_index(L, a, b)] = free
    return st

def _residual_free(free, L, a, b):
    """scale-free form of the equioscillatory conditions: equal values of P on each alternation set, normalised by the amplitude of the interval"""
    R = 7 - L; st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b); res = []
    if a > 0.0 and L > 0:
        amp = _pdiff(st, a, st[0])
        for x in lo[1:]: res.append(_pdiff(st, x, lo[0]) / amp)
        for x in up[1:]: res.append(_pdiff(st, x, up[0]) / amp)
    if b < 1.0 and R > 0:
        amp = _pdiff(st, st[L], b)
        for x in ru[1:]: res.append(_pdiff(st, x, ru[0]) / amp)
        for x in rl[1:]: res.append(_pdiff(st, x, rl[0]) / amp)
    return np.array(res)

def _strict_ordered(st, L, a, b):
    """the ordering (4.7) with strict inequalities on the free stationary points"""
    R = 7 - L
    if L > 0 and a > 0.0:
        if not (st[0] < a and st[L - 1] > 0.0): return False
        if L > 1 and not np.all(np.diff(st[:L]) < 0.0): return False
    if R > 0 and b < 1.0:
        if not (st[L] > b and st[6] < 1.0): return False
        if R > 1 and not np.all(np.diff(st[L:]) > 0.0): return False
    return True

def _assemble(free, L, a, b):
    """the nine parameters: stationary points, scale s8 and offset s9 fixed by p = 0 at the lower-left level and p = 1 at the upper-right level"""
    st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b)
    xl = 0.0 if a == 0.0 else lo[0]
    xr = 1.0 if b == 1.0 else ru[0]
    s8 = 1.0 / _pdiff(st, xr, xl); s9 = -s8 * _pdiff(st, xl, 0.0)
    return np.concatenate([st, [s8, s9]])

def _jacobian(free, L, a, b):
    n = free.size; f0 = _residual_free(free, L, a, b); J = np.zeros((f0.size, n)); h = 1e-7
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (_residual_free(free + d, L, a, b) - _residual_free(free - d, L, a, b)) / (2.0 * h)
    return J

def _newton_free(L, a, b, z0):
    """damped Newton iteration on the scale-free system that keeps the ordering (4.7)"""
    z = z0.copy(); F = _residual_free(z, L, a, b); nf = float(np.max(np.abs(F)))
    for it in range(60):
        if nf < 1e-12: break
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -F)
        except np.linalg.LinAlgError:
            return z, False
        lam = 1.0; accepted = False
        while lam > 1e-4:
            zn = z + lam * dz
            if _strict_ordered(_full_st(zn, L, a, b), L, a, b):
                Fn = _residual_free(zn, L, a, b); nfn = float(np.max(np.abs(Fn)))
                if nfn < nf or lam < 0.1:
                    z, F, nf = zn, Fn, nfn; accepted = True; break
            lam *= 0.5
        if not accepted: return z, False
    for it in range(3):
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -_residual_free(z, L, a, b))
        except np.linalg.LinAlgError:
            break
        zn = z + dz
        if _strict_ordered(_full_st(zn, L, a, b), L, a, b) and np.max(np.abs(_residual_free(zn, L, a, b))) <= np.max(np.abs(_residual_free(z, L, a, b))): z = zn
    nf = float(np.max(np.abs(_residual_free(z, L, a, b))))
    return z, (nf < 1e-11 and _strict_ordered(_full_st(z, L, a, b), L, a, b))

def _guess(L, a, b, kind, power):
    """Chebyshev extrema (kind 0) or Chebyshev zeros (kind 1) on each interval, the relative positions raised to a power, as the initial stationary points"""
    R = 7 - L; st = np.zeros(7)
    if L > 0 and a > 0.0:
        k = np.arange(1, L + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (L + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * L + 2)))
        st[:L] = (a * u ** power)[::-1]
    if R > 0 and b < 1.0:
        k = np.arange(1, R + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (R + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * R + 2)))
        st[L:] = b + (1.0 - b) * (1.0 - (1.0 - u) ** power)
    return st[_free_index(L, a, b)]

def _solve_equioscillatory(L, a, b):
    """the unique solution of the equioscillatory conditions with the ordering (4.7), from deterministic Chebyshev starts of increasing compression"""
    if len(_free_index(L, a, b)) == 0:
        return _assemble(np.zeros(0), L, a, b)
    for power in (1.0, 1.5, 0.75, 2.0, 0.5, 3.0):
        for kind in (0, 1):
            z, ok = _newton_free(L, a, b, _guess(L, a, b, kind, power))
            if ok: return _assemble(z, L, a, b)
    raise ValueError("the equioscillatory conditions could not be solved for L=%d on (%g, %g)" % (L, a, b))

def _coefficients(s):
    c = _pcoef(s[:7]) * s[7]; c[0] = c[0] + s[8]
    return c

def _transform(energies):
    """the source's normalisation (2.2): an eigenvalue e of H maps to (lmax - e)/(lmax - lmin); Table 1 order lmin, out_homo, in_homo, in_lumo, out_lumo, lmax"""
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size != 6 or not np.all(np.isfinite(e)): raise ValueError("energies must hold six finite values")
    lmin, oh, ih, il, ol, lmax = e
    if not (lmin < oh <= ih < il <= ol < lmax): raise ValueError("energies must be ordered lmin < out_homo <= in_homo < in_lumo <= out_lumo < lmax")
    T = (lmax - np.array([il, ih, ol, oh])) / (lmax - lmin)
    return T   # in_lumo, in_homo, out_lumo, out_homo on [0, 1]

def _build(bounds, seed):
    """synthetic 16 by 16 matrix consistent with the bounds: the extreme eigenvalues of each interval at the midpoints of the bound intervals, the rest equidistant, on the sign-fixed QR frame of the mod-47 integer constructor"""
    n = _dim(); inl, inh, outl, outh = bounds
    lam_lumo = 0.5 * (outl + inl); lam_homo = 0.5 * (inh + outh)
    mu = 0.5 * (inl + inh); nocc = int(round(n * (1.0 - mu)))
    lam = np.concatenate([np.linspace(0.0, lam_lumo, n - nocc), np.linspace(lam_homo, 1.0, nocc)])
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (((i + 1) * (j + 2) * (i + j + 3) + seed * (i + 4) * (j + 5)) % 47) - 23
    Q, _r = np.linalg.qr(M)
    for c in range(n):
        col = Q[:, c]
        for v in col:
            if abs(v) > 1e-12:
                if v < 0:
                    Q[:, c] = -col
                break
    X0 = Q @ np.diag(lam) @ Q.T
    return (X0 + X0.T) / 2.0, nocc

def propagate_bounds(bounds: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    bd = np.asarray(bounds, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64)
    if bd.size != 4 or not np.all(np.isfinite(bd)):
        raise ValueError("bounds must hold four finite values")
    if bb.shape != (9,):
        raise ValueError("b must have shape (9,)")
    return np.array([_horner(bb, x) for x in bd], dtype=np.float64)

import numpy as np, math
from math import comb


def _kappa():
    """the source's deactivation threshold of the acceleration (Algorithm 5.1)"""
    return 0.01

def _dim():
    """the order of the synthetic matrices"""
    return 16

def _check_index(l):
    if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or not (0 <= int(l) <= 7):
        raise ValueError("l must be an integer in 0..7")
    return int(l)

def _table(l):
    """the source's tabulated closed-form members for l = 4..7 (Table 2), monomial order b0..b8"""
    t = {4: [0, 0, 0, 0, 0, 56, -140, 120, -35], 5: [0, 0, 0, 0, 0, 0, 28, -48, 21], 6: [0, 0, 0, 0, 0, 0, 0, 8, -7], 7: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    return np.array(t[l], dtype=np.float64)

def _horner(b, x):
    y = 0.0
    for k in range(8, -1, -1):
        y = y * x + b[k]
    return y

def _stop_constant(l):
    return [np.inf, 28.0, 56.0, 82.0, 82.0, 56.0, 28.0, np.inf][l]

def _stop_order(l):
    return [1, 2, 3, 4, 4, 3, 2, 1][l]

def _esym(st):
    """elementary symmetric polynomials e_0..e_7 of the seven stationary points"""
    e = np.zeros(8); e[0] = 1.0
    for v in st:
        e[1:] = e[1:] + v * e[:-1]
    return e

def _pcoef(st):
    """monomial coefficients of P(x) = int_0^x prod_k (t - s_k) dt, the parametrisation (4.1) with unit scale and zero offset"""
    e = _esym(st); c = np.zeros(9)
    for k in range(8):
        c[8 - k] = ((-1.0) ** k) * e[k] / (8 - k)
    return c

def _pdiff(st, x1, x0):
    """int_{x0}^{x1} prod_k (t - s_k) dt by the four-point Gauss rule, exact for the degree-seven integrand and free of cancellation"""
    g = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
    w = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    m = 0.5 * (x1 + x0); h = 0.5 * (x1 - x0); t = m + h * g
    f = np.ones(4)
    for s in st:
        f = f * (t - s)
    return h * float(np.dot(w, f))

def _check_bounds(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        raise ValueError("bounds must be real numbers")
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b > 1.0 or not (a < b):
        raise ValueError("bounds must satisfy 0 <= lam_lumo < lam_homo <= 1")
    return a, b

def _alternation_sets(st, L, a, b):
    """lower and upper alternation points of the left interval and upper and lower points of the right interval"""
    R = 7 - L
    lo = ([0.0] if L % 2 == 0 else []) + [st[k - 1] for k in range(1, L + 1) if k % 2 == 1]
    up = [a] + [st[k - 1] for k in range(1, L + 1) if k % 2 == 0] + ([0.0] if L % 2 == 1 else [])
    ru = [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 1] + ([1.0] if R % 2 == 0 else [])
    rl = [b] + [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 0] + ([1.0] if R % 2 == 1 else [])
    return lo, up, ru, rl

def _free_index(L, a, b):
    fl = [] if a == 0.0 else list(range(L)); fr = [] if b == 1.0 else list(range(L, 7))
    return fl + fr

def _full_st(free, L, a, b):
    st = np.zeros(7)
    if b == 1.0: st[L:] = 1.0
    st[_free_index(L, a, b)] = free
    return st

def _residual_free(free, L, a, b):
    """scale-free form of the equioscillatory conditions: equal values of P on each alternation set, normalised by the amplitude of the interval"""
    R = 7 - L; st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b); res = []
    if a > 0.0 and L > 0:
        amp = _pdiff(st, a, st[0])
        for x in lo[1:]: res.append(_pdiff(st, x, lo[0]) / amp)
        for x in up[1:]: res.append(_pdiff(st, x, up[0]) / amp)
    if b < 1.0 and R > 0:
        amp = _pdiff(st, st[L], b)
        for x in ru[1:]: res.append(_pdiff(st, x, ru[0]) / amp)
        for x in rl[1:]: res.append(_pdiff(st, x, rl[0]) / amp)
    return np.array(res)

def _strict_ordered(st, L, a, b):
    """the ordering (4.7) with strict inequalities on the free stationary points"""
    R = 7 - L
    if L > 0 and a > 0.0:
        if not (st[0] < a and st[L - 1] > 0.0): return False
        if L > 1 and not np.all(np.diff(st[:L]) < 0.0): return False
    if R > 0 and b < 1.0:
        if not (st[L] > b and st[6] < 1.0): return False
        if R > 1 and not np.all(np.diff(st[L:]) > 0.0): return False
    return True

def _assemble(free, L, a, b):
    """the nine parameters: stationary points, scale s8 and offset s9 fixed by p = 0 at the lower-left level and p = 1 at the upper-right level"""
    st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b)
    xl = 0.0 if a == 0.0 else lo[0]
    xr = 1.0 if b == 1.0 else ru[0]
    s8 = 1.0 / _pdiff(st, xr, xl); s9 = -s8 * _pdiff(st, xl, 0.0)
    return np.concatenate([st, [s8, s9]])

def _jacobian(free, L, a, b):
    n = free.size; f0 = _residual_free(free, L, a, b); J = np.zeros((f0.size, n)); h = 1e-7
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (_residual_free(free + d, L, a, b) - _residual_free(free - d, L, a, b)) / (2.0 * h)
    return J

def _newton_free(L, a, b, z0):
    """damped Newton iteration on the scale-free system that keeps the ordering (4.7)"""
    z = z0.copy(); F = _residual_free(z, L, a, b); nf = float(np.max(np.abs(F)))
    for it in range(60):
        if nf < 1e-12: break
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -F)
        except np.linalg.LinAlgError:
            return z, False
        lam = 1.0; accepted = False
        while lam > 1e-4:
            zn = z + lam * dz
            if _strict_ordered(_full_st(zn, L, a, b), L, a, b):
                Fn = _residual_free(zn, L, a, b); nfn = float(np.max(np.abs(Fn)))
                if nfn < nf or lam < 0.1:
                    z, F, nf = zn, Fn, nfn; accepted = True; break
            lam *= 0.5
        if not accepted: return z, False
    for it in range(3):
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -_residual_free(z, L, a, b))
        except np.linalg.LinAlgError:
            break
        zn = z + dz
        if _strict_ordered(_full_st(zn, L, a, b), L, a, b) and np.max(np.abs(_residual_free(zn, L, a, b))) <= np.max(np.abs(_residual_free(z, L, a, b))): z = zn
    nf = float(np.max(np.abs(_residual_free(z, L, a, b))))
    return z, (nf < 1e-11 and _strict_ordered(_full_st(z, L, a, b), L, a, b))

def _guess(L, a, b, kind, power):
    """Chebyshev extrema (kind 0) or Chebyshev zeros (kind 1) on each interval, the relative positions raised to a power, as the initial stationary points"""
    R = 7 - L; st = np.zeros(7)
    if L > 0 and a > 0.0:
        k = np.arange(1, L + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (L + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * L + 2)))
        st[:L] = (a * u ** power)[::-1]
    if R > 0 and b < 1.0:
        k = np.arange(1, R + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (R + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * R + 2)))
        st[L:] = b + (1.0 - b) * (1.0 - (1.0 - u) ** power)
    return st[_free_index(L, a, b)]

def _solve_equioscillatory(L, a, b):
    """the unique solution of the equioscillatory conditions with the ordering (4.7), from deterministic Chebyshev starts of increasing compression"""
    if len(_free_index(L, a, b)) == 0:
        return _assemble(np.zeros(0), L, a, b)
    for power in (1.0, 1.5, 0.75, 2.0, 0.5, 3.0):
        for kind in (0, 1):
            z, ok = _newton_free(L, a, b, _guess(L, a, b, kind, power))
            if ok: return _assemble(z, L, a, b)
    raise ValueError("the equioscillatory conditions could not be solved for L=%d on (%g, %g)" % (L, a, b))

def _coefficients(s):
    c = _pcoef(s[:7]) * s[7]; c[0] = c[0] + s[8]
    return c

def _transform(energies):
    """the source's normalisation (2.2): an eigenvalue e of H maps to (lmax - e)/(lmax - lmin); Table 1 order lmin, out_homo, in_homo, in_lumo, out_lumo, lmax"""
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size != 6 or not np.all(np.isfinite(e)): raise ValueError("energies must hold six finite values")
    lmin, oh, ih, il, ol, lmax = e
    if not (lmin < oh <= ih < il <= ol < lmax): raise ValueError("energies must be ordered lmin < out_homo <= in_homo < in_lumo <= out_lumo < lmax")
    T = (lmax - np.array([il, ih, ol, oh])) / (lmax - lmin)
    return T   # in_lumo, in_homo, out_lumo, out_homo on [0, 1]

def _build(bounds, seed):
    """synthetic 16 by 16 matrix consistent with the bounds: the extreme eigenvalues of each interval at the midpoints of the bound intervals, the rest equidistant, on the sign-fixed QR frame of the mod-47 integer constructor"""
    n = _dim(); inl, inh, outl, outh = bounds
    lam_lumo = 0.5 * (outl + inl); lam_homo = 0.5 * (inh + outh)
    mu = 0.5 * (inl + inh); nocc = int(round(n * (1.0 - mu)))
    lam = np.concatenate([np.linspace(0.0, lam_lumo, n - nocc), np.linspace(lam_homo, 1.0, nocc)])
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (((i + 1) * (j + 2) * (i + j + 3) + seed * (i + 4) * (j + 5)) % 47) - 23
    Q, _r = np.linalg.qr(M)
    for c in range(n):
        col = Q[:, c]
        for v in col:
            if abs(v) > 1e-12:
                if v < 0:
                    Q[:, c] = -col
                break
    X0 = Q @ np.diag(lam) @ Q.T
    return (X0 + X0.T) / 2.0, nocc

def stop_check(tr_prev: float, tr_curr: float, l: int) -> float:
    k = _check_index(l)
    if tr_curr <= 0.0:
        return 2.0
    C = _stop_constant(k)
    if not np.isfinite(C):
        return 0.0
    if tr_curr > C * (tr_prev ** _stop_order(k)):
        return 1.0
    return 0.0

import numpy as np, math
from math import comb


def _kappa():
    """the source's deactivation threshold of the acceleration (Algorithm 5.1)"""
    return 0.01

def _dim():
    """the order of the synthetic matrices"""
    return 16

def _check_index(l):
    if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or not (0 <= int(l) <= 7):
        raise ValueError("l must be an integer in 0..7")
    return int(l)

def _table(l):
    """the source's tabulated closed-form members for l = 4..7 (Table 2), monomial order b0..b8"""
    t = {4: [0, 0, 0, 0, 0, 56, -140, 120, -35], 5: [0, 0, 0, 0, 0, 0, 28, -48, 21], 6: [0, 0, 0, 0, 0, 0, 0, 8, -7], 7: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    return np.array(t[l], dtype=np.float64)

def _horner(b, x):
    y = 0.0
    for k in range(8, -1, -1):
        y = y * x + b[k]
    return y

def _stop_constant(l):
    return [np.inf, 28.0, 56.0, 82.0, 82.0, 56.0, 28.0, np.inf][l]

def _stop_order(l):
    return [1, 2, 3, 4, 4, 3, 2, 1][l]

def _esym(st):
    """elementary symmetric polynomials e_0..e_7 of the seven stationary points"""
    e = np.zeros(8); e[0] = 1.0
    for v in st:
        e[1:] = e[1:] + v * e[:-1]
    return e

def _pcoef(st):
    """monomial coefficients of P(x) = int_0^x prod_k (t - s_k) dt, the parametrisation (4.1) with unit scale and zero offset"""
    e = _esym(st); c = np.zeros(9)
    for k in range(8):
        c[8 - k] = ((-1.0) ** k) * e[k] / (8 - k)
    return c

def _pdiff(st, x1, x0):
    """int_{x0}^{x1} prod_k (t - s_k) dt by the four-point Gauss rule, exact for the degree-seven integrand and free of cancellation"""
    g = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
    w = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    m = 0.5 * (x1 + x0); h = 0.5 * (x1 - x0); t = m + h * g
    f = np.ones(4)
    for s in st:
        f = f * (t - s)
    return h * float(np.dot(w, f))

def _check_bounds(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        raise ValueError("bounds must be real numbers")
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b > 1.0 or not (a < b):
        raise ValueError("bounds must satisfy 0 <= lam_lumo < lam_homo <= 1")
    return a, b

def _alternation_sets(st, L, a, b):
    """lower and upper alternation points of the left interval and upper and lower points of the right interval"""
    R = 7 - L
    lo = ([0.0] if L % 2 == 0 else []) + [st[k - 1] for k in range(1, L + 1) if k % 2 == 1]
    up = [a] + [st[k - 1] for k in range(1, L + 1) if k % 2 == 0] + ([0.0] if L % 2 == 1 else [])
    ru = [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 1] + ([1.0] if R % 2 == 0 else [])
    rl = [b] + [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 0] + ([1.0] if R % 2 == 1 else [])
    return lo, up, ru, rl

def _free_index(L, a, b):
    fl = [] if a == 0.0 else list(range(L)); fr = [] if b == 1.0 else list(range(L, 7))
    return fl + fr

def _full_st(free, L, a, b):
    st = np.zeros(7)
    if b == 1.0: st[L:] = 1.0
    st[_free_index(L, a, b)] = free
    return st

def _residual_free(free, L, a, b):
    """scale-free form of the equioscillatory conditions: equal values of P on each alternation set, normalised by the amplitude of the interval"""
    R = 7 - L; st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b); res = []
    if a > 0.0 and L > 0:
        amp = _pdiff(st, a, st[0])
        for x in lo[1:]: res.append(_pdiff(st, x, lo[0]) / amp)
        for x in up[1:]: res.append(_pdiff(st, x, up[0]) / amp)
    if b < 1.0 and R > 0:
        amp = _pdiff(st, st[L], b)
        for x in ru[1:]: res.append(_pdiff(st, x, ru[0]) / amp)
        for x in rl[1:]: res.append(_pdiff(st, x, rl[0]) / amp)
    return np.array(res)

def _strict_ordered(st, L, a, b):
    """the ordering (4.7) with strict inequalities on the free stationary points"""
    R = 7 - L
    if L > 0 and a > 0.0:
        if not (st[0] < a and st[L - 1] > 0.0): return False
        if L > 1 and not np.all(np.diff(st[:L]) < 0.0): return False
    if R > 0 and b < 1.0:
        if not (st[L] > b and st[6] < 1.0): return False
        if R > 1 and not np.all(np.diff(st[L:]) > 0.0): return False
    return True

def _assemble(free, L, a, b):
    """the nine parameters: stationary points, scale s8 and offset s9 fixed by p = 0 at the lower-left level and p = 1 at the upper-right level"""
    st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b)
    xl = 0.0 if a == 0.0 else lo[0]
    xr = 1.0 if b == 1.0 else ru[0]
    s8 = 1.0 / _pdiff(st, xr, xl); s9 = -s8 * _pdiff(st, xl, 0.0)
    return np.concatenate([st, [s8, s9]])

def _jacobian(free, L, a, b):
    n = free.size; f0 = _residual_free(free, L, a, b); J = np.zeros((f0.size, n)); h = 1e-7
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (_residual_free(free + d, L, a, b) - _residual_free(free - d, L, a, b)) / (2.0 * h)
    return J

def _newton_free(L, a, b, z0):
    """damped Newton iteration on the scale-free system that keeps the ordering (4.7)"""
    z = z0.copy(); F = _residual_free(z, L, a, b); nf = float(np.max(np.abs(F)))
    for it in range(60):
        if nf < 1e-12: break
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -F)
        except np.linalg.LinAlgError:
            return z, False
        lam = 1.0; accepted = False
        while lam > 1e-4:
            zn = z + lam * dz
            if _strict_ordered(_full_st(zn, L, a, b), L, a, b):
                Fn = _residual_free(zn, L, a, b); nfn = float(np.max(np.abs(Fn)))
                if nfn < nf or lam < 0.1:
                    z, F, nf = zn, Fn, nfn; accepted = True; break
            lam *= 0.5
        if not accepted: return z, False
    for it in range(3):
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -_residual_free(z, L, a, b))
        except np.linalg.LinAlgError:
            break
        zn = z + dz
        if _strict_ordered(_full_st(zn, L, a, b), L, a, b) and np.max(np.abs(_residual_free(zn, L, a, b))) <= np.max(np.abs(_residual_free(z, L, a, b))): z = zn
    nf = float(np.max(np.abs(_residual_free(z, L, a, b))))
    return z, (nf < 1e-11 and _strict_ordered(_full_st(z, L, a, b), L, a, b))

def _guess(L, a, b, kind, power):
    """Chebyshev extrema (kind 0) or Chebyshev zeros (kind 1) on each interval, the relative positions raised to a power, as the initial stationary points"""
    R = 7 - L; st = np.zeros(7)
    if L > 0 and a > 0.0:
        k = np.arange(1, L + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (L + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * L + 2)))
        st[:L] = (a * u ** power)[::-1]
    if R > 0 and b < 1.0:
        k = np.arange(1, R + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (R + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * R + 2)))
        st[L:] = b + (1.0 - b) * (1.0 - (1.0 - u) ** power)
    return st[_free_index(L, a, b)]

def _solve_equioscillatory(L, a, b):
    """the unique solution of the equioscillatory conditions with the ordering (4.7), from deterministic Chebyshev starts of increasing compression"""
    if len(_free_index(L, a, b)) == 0:
        return _assemble(np.zeros(0), L, a, b)
    for power in (1.0, 1.5, 0.75, 2.0, 0.5, 3.0):
        for kind in (0, 1):
            z, ok = _newton_free(L, a, b, _guess(L, a, b, kind, power))
            if ok: return _assemble(z, L, a, b)
    raise ValueError("the equioscillatory conditions could not be solved for L=%d on (%g, %g)" % (L, a, b))

def _coefficients(s):
    c = _pcoef(s[:7]) * s[7]; c[0] = c[0] + s[8]
    return c

def _transform(energies):
    """the source's normalisation (2.2): an eigenvalue e of H maps to (lmax - e)/(lmax - lmin); Table 1 order lmin, out_homo, in_homo, in_lumo, out_lumo, lmax"""
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size != 6 or not np.all(np.isfinite(e)): raise ValueError("energies must hold six finite values")
    lmin, oh, ih, il, ol, lmax = e
    if not (lmin < oh <= ih < il <= ol < lmax): raise ValueError("energies must be ordered lmin < out_homo <= in_homo < in_lumo <= out_lumo < lmax")
    T = (lmax - np.array([il, ih, ol, oh])) / (lmax - lmin)
    return T   # in_lumo, in_homo, out_lumo, out_homo on [0, 1]

def _build(bounds, seed):
    """synthetic 16 by 16 matrix consistent with the bounds: the extreme eigenvalues of each interval at the midpoints of the bound intervals, the rest equidistant, on the sign-fixed QR frame of the mod-47 integer constructor"""
    n = _dim(); inl, inh, outl, outh = bounds
    lam_lumo = 0.5 * (outl + inl); lam_homo = 0.5 * (inh + outh)
    mu = 0.5 * (inl + inh); nocc = int(round(n * (1.0 - mu)))
    lam = np.concatenate([np.linspace(0.0, lam_lumo, n - nocc), np.linspace(lam_homo, 1.0, nocc)])
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (((i + 1) * (j + 2) * (i + j + 3) + seed * (i + 4) * (j + 5)) % 47) - 23
    Q, _r = np.linalg.qr(M)
    for c in range(n):
        col = Q[:, c]
        for v in col:
            if abs(v) > 1e-12:
                if v < 0:
                    Q[:, c] = -col
                break
    X0 = Q @ np.diag(lam) @ Q.T
    return (X0 + X0.T) / 2.0, nocc

def sp8_accelerated_audit(energies: "np.ndarray", seed: int, niter: int) -> "np.ndarray":
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(niter, bool) or not isinstance(niter, (int, np.integer)) or niter < 2:
        raise ValueError("niter must be an integer >= 2")
    bounds = _transform(energies)
    X, nocc = _build(bounds, int(seed))
    kap = _kappa()
    tr_prev = float(np.trace(X - X @ X))
    rows = []
    for i in range(1, int(niter) + 1):
        sel = select_member(bounds, i, kap)
        L = int(sel[0]); a = float(sel[1]); b = float(sel[2])
        if a == 0.0 and b == 1.0:
            c = sp8_closed_form(L)
        else:
            s_member = sp8_equioscillatory(L, a, b)
            if float(np.max(np.abs(equioscillation_residual(s_member, L, a, b)))) > 1e-8:
                raise ValueError("the applied member does not satisfy its defining conditions")
            c = sp8_from_stationary_points(s_member)
        X = evaluate_p8(X, c)
        bounds = np.clip(propagate_bounds(bounds, c), 0.0, 1.0)
        bounds[2] = min(bounds[2], bounds[0]); bounds[3] = max(bounds[3], bounds[1])
        tr = float(np.trace(X - X @ X))
        rows.append([float(L), bounds[0], bounds[1], bounds[2], bounds[3], tr])
        code = stop_check(tr_prev, tr, L)
        if i <= 2 and (code == 2.0 or (a == 0.0 and b == 1.0 and code == 1.0)):
            raise ValueError("a termination test fired in the conditioning phase")
        tr_prev = tr
    return np.asarray(rows, dtype=np.float64)
SCICODE_GOLD_EOF
