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
from fractions import Fraction

def _check_pr(p, r):
    p = int(p)
    r = int(r)
    if p < 1 or r < 0 or r > p - 1:
        raise ValueError('degree p must be >= 1 and regularity r must satisfy 0 <= r <= p - 1')
    return (p, r)

def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else Fraction(0)) + (b[i] if i < len(b) else Fraction(0)) for i in range(n)]

def _pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out

def _peval(a, x):
    s = Fraction(0)
    for c in reversed(a):
        s = s * x + c
    return s

def _knots(p, r, nmesh):
    return [Fraction(0)] * (p + 1) + sum(([Fraction(e)] * (p - r) for e in range(1, nmesh)), []) + [Fraction(nmesh)] * (p + 1)

def _bsplines(p, r, nmesh):
    ks = _knots(p, r, nmesh)
    cur = []
    for j in range(len(ks) - 1):
        d = {}
        if ks[j] < ks[j + 1]:
            for e in range(int(ks[j]), int(ks[j + 1])):
                d[e] = [Fraction(1)]
        cur.append(d)
    for k in range(1, p + 1):
        nxt = []
        for j in range(len(ks) - k - 1):
            d = {}
            den1 = ks[j + k] - ks[j]
            den2 = ks[j + k + 1] - ks[j + 1]
            if den1 != 0:
                for e, poly in cur[j].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([-ks[j] / den1, Fraction(1) / den1], poly))
            if den2 != 0:
                for e, poly in cur[j + 1].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([ks[j + k + 1] / den2, Fraction(-1) / den2], poly))
            nxt.append({e: v for e, v in d.items() if any((x != 0 for x in v))})
        cur = nxt
    return cur

def spline_basis_values(p, r, nmesh, tvals):
    p, r = _check_pr(p, r)
    nmesh = int(nmesh)
    if nmesh < 1:
        raise ValueError('nmesh must be a positive integer')
    tv = np.atleast_1d(np.asarray(tvals, dtype=float))
    if tv.ndim != 1 or np.any(tv < 0.0) or np.any(tv > nmesh) or (not np.all(np.isfinite(tv))):
        raise ValueError('evaluation points must lie in [0, nmesh]')
    phi = _bsplines(p, r, nmesh)
    out = np.zeros((len(phi), len(tv)))
    for i, f in enumerate(phi):
        for jj, x in enumerate(tv):
            e = min(int(np.floor(x)), nmesh - 1)
            if e in f:
                out[i, jj] = float(_peval(f[e], Fraction(float(x))))
    return out

import numpy as np
from fractions import Fraction

def _check_pr(p, r):
    p = int(p)
    r = int(r)
    if p < 1 or r < 0 or r > p - 1:
        raise ValueError('degree p must be >= 1 and regularity r must satisfy 0 <= r <= p - 1')
    return (p, r)

def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else Fraction(0)) + (b[i] if i < len(b) else Fraction(0)) for i in range(n)]

def _pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out

def _pderiv(a):
    return [a[i] * i for i in range(1, len(a))] or [Fraction(0)]

def _pint(a, lo, hi):
    s = Fraction(0)
    for i, c in enumerate(a):
        s += c * (Fraction(hi) ** (i + 1) - Fraction(lo) ** (i + 1)) / (i + 1)
    return s

def _knots(p, r, nmesh):
    return [Fraction(0)] * (p + 1) + sum(([Fraction(e)] * (p - r) for e in range(1, nmesh)), []) + [Fraction(nmesh)] * (p + 1)

def _bsplines(p, r, nmesh):
    ks = _knots(p, r, nmesh)
    cur = []
    for j in range(len(ks) - 1):
        d = {}
        if ks[j] < ks[j + 1]:
            for e in range(int(ks[j]), int(ks[j + 1])):
                d[e] = [Fraction(1)]
        cur.append(d)
    for k in range(1, p + 1):
        nxt = []
        for j in range(len(ks) - k - 1):
            d = {}
            den1 = ks[j + k] - ks[j]
            den2 = ks[j + k + 1] - ks[j + 1]
            if den1 != 0:
                for e, poly in cur[j].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([-ks[j] / den1, Fraction(1) / den1], poly))
            if den2 != 0:
                for e, poly in cur[j + 1].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([ks[j + k + 1] / den2, Fraction(-1) / den2], poly))
            nxt.append({e: v for e, v in d.items() if any((x != 0 for x in v))})
        cur = nxt
    return cur

def _galerkin_exact(p, r, nmesh):
    phi = _bsplines(p, r, nmesh)
    n = nmesh * (p - r) + r

    def _inner(f, g, d1, d2):
        s = Fraction(0)
        for e in set(f) & set(g):
            a = f[e]
            b = g[e]
            for _ in range(d1):
                a = _pderiv(a)
            for _ in range(d2):
                b = _pderiv(b)
            s += _pint(_pmul(a, b), e, e + 1)
        return s
    M = [[_inner(phi[j], phi[l - 1], 0, 0) for j in range(1, n + 1)] for l in range(1, n + 1)]
    B = [[_inner(phi[j], phi[l - 1], 1, 1) for j in range(1, n + 1)] for l in range(1, n + 1)]
    return (M, B, n)

def petrov_galerkin_matrices(p, r, nmesh):
    p, r = _check_pr(p, r)
    nmesh = int(nmesh)
    if nmesh < 2:
        raise ValueError('nmesh must be at least 2')
    M, B, n = _galerkin_exact(p, r, nmesh)
    out = np.zeros((2, n, n))
    for i in range(n):
        for j in range(n):
            out[0, i, j] = float(M[i][j])
            out[1, i, j] = float(B[i][j])
    return out

import numpy as np
from fractions import Fraction

def _check_pr(p, r):
    p = int(p)
    r = int(r)
    if p < 1 or r < 0 or r > p - 1:
        raise ValueError('degree p must be >= 1 and regularity r must satisfy 0 <= r <= p - 1')
    return (p, r)

def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else Fraction(0)) + (b[i] if i < len(b) else Fraction(0)) for i in range(n)]

def _pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out

def _pderiv(a):
    return [a[i] * i for i in range(1, len(a))] or [Fraction(0)]

def _pint(a, lo, hi):
    s = Fraction(0)
    for i, c in enumerate(a):
        s += c * (Fraction(hi) ** (i + 1) - Fraction(lo) ** (i + 1)) / (i + 1)
    return s

def _knots(p, r, nmesh):
    return [Fraction(0)] * (p + 1) + sum(([Fraction(e)] * (p - r) for e in range(1, nmesh)), []) + [Fraction(nmesh)] * (p + 1)

def _bsplines(p, r, nmesh):
    ks = _knots(p, r, nmesh)
    cur = []
    for j in range(len(ks) - 1):
        d = {}
        if ks[j] < ks[j + 1]:
            for e in range(int(ks[j]), int(ks[j + 1])):
                d[e] = [Fraction(1)]
        cur.append(d)
    for k in range(1, p + 1):
        nxt = []
        for j in range(len(ks) - k - 1):
            d = {}
            den1 = ks[j + k] - ks[j]
            den2 = ks[j + k + 1] - ks[j + 1]
            if den1 != 0:
                for e, poly in cur[j].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([-ks[j] / den1, Fraction(1) / den1], poly))
            if den2 != 0:
                for e, poly in cur[j + 1].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([ks[j + k + 1] / den2, Fraction(-1) / den2], poly))
            nxt.append({e: v for e, v in d.items() if any((x != 0 for x in v))})
        cur = nxt
    return cur

def _galerkin_exact(p, r, nmesh):
    phi = _bsplines(p, r, nmesh)
    n = nmesh * (p - r) + r

    def _inner(f, g, d1, d2):
        s = Fraction(0)
        for e in set(f) & set(g):
            a = f[e]
            b = g[e]
            for _ in range(d1):
                a = _pderiv(a)
            for _ in range(d2):
                b = _pderiv(b)
            s += _pint(_pmul(a, b), e, e + 1)
        return s
    M = [[_inner(phi[j], phi[l - 1], 0, 0) for j in range(1, n + 1)] for l in range(1, n + 1)]
    B = [[_inner(phi[j], phi[l - 1], 1, 1) for j in range(1, n + 1)] for l in range(1, n + 1)]
    return (M, B, n)

def _interior_blocks_exact(p, r):
    N = p - r
    nmesh = 2 * p + 4
    M, B, n = _galerkin_exact(p, r, nmesh)
    mid = nmesh // 2
    out = {}
    for d in range(-p, p + 1):
        bj = mid + d
        if 0 <= bj < n // N:
            Mb = [[M[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            Bb = [[B[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            if any((x != 0 for row in Mb for x in row)) or any((x != 0 for row in Bb for x in row)):
                out[d] = (Mb, Bb)
    return out

def interior_symbol_blocks(p, r):
    if int(p) < 1 or int(r) < 0 or int(r) > int(p) - 1:
        raise ValueError('invalid degree or regularity')
    p, r = _check_pr(p, r)
    N = p - r
    blocks = _interior_blocks_exact(p, r)
    out = np.zeros((2 * p + 1, 2, N, N))
    for d, (Mb, Bb) in blocks.items():
        for a in range(N):
            for b in range(N):
                out[d + p, 0, a, b] = float(Mb[a][b])
                out[d + p, 1, a, b] = float(Bb[a][b])
    return out

import numpy as np
from fractions import Fraction

def _check_pr(p, r):
    p = int(p)
    r = int(r)
    if p < 1 or r < 0 or r > p - 1:
        raise ValueError('degree p must be >= 1 and regularity r must satisfy 0 <= r <= p - 1')
    return (p, r)

def _check_rho(rho):
    if isinstance(rho, Fraction):
        rq = rho
    else:
        rq = Fraction(rho).limit_denominator(10 ** 12) if isinstance(rho, float) else Fraction(rho)
    if rq < 0:
        raise ValueError('rho must be non-negative')
    return rq

def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else Fraction(0)) + (b[i] if i < len(b) else Fraction(0)) for i in range(n)]

def _pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out

def _pderiv(a):
    return [a[i] * i for i in range(1, len(a))] or [Fraction(0)]

def _pint(a, lo, hi):
    s = Fraction(0)
    for i, c in enumerate(a):
        s += c * (Fraction(hi) ** (i + 1) - Fraction(lo) ** (i + 1)) / (i + 1)
    return s

def _ptrim(a):
    a = list(a)
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a

def _knots(p, r, nmesh):
    return [Fraction(0)] * (p + 1) + sum(([Fraction(e)] * (p - r) for e in range(1, nmesh)), []) + [Fraction(nmesh)] * (p + 1)

def _bsplines(p, r, nmesh):
    ks = _knots(p, r, nmesh)
    cur = []
    for j in range(len(ks) - 1):
        d = {}
        if ks[j] < ks[j + 1]:
            for e in range(int(ks[j]), int(ks[j + 1])):
                d[e] = [Fraction(1)]
        cur.append(d)
    for k in range(1, p + 1):
        nxt = []
        for j in range(len(ks) - k - 1):
            d = {}
            den1 = ks[j + k] - ks[j]
            den2 = ks[j + k + 1] - ks[j + 1]
            if den1 != 0:
                for e, poly in cur[j].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([-ks[j] / den1, Fraction(1) / den1], poly))
            if den2 != 0:
                for e, poly in cur[j + 1].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([ks[j + k + 1] / den2, Fraction(-1) / den2], poly))
            nxt.append({e: v for e, v in d.items() if any((x != 0 for x in v))})
        cur = nxt
    return cur

def _galerkin_exact(p, r, nmesh):
    phi = _bsplines(p, r, nmesh)
    n = nmesh * (p - r) + r

    def _inner(f, g, d1, d2):
        s = Fraction(0)
        for e in set(f) & set(g):
            a = f[e]
            b = g[e]
            for _ in range(d1):
                a = _pderiv(a)
            for _ in range(d2):
                b = _pderiv(b)
            s += _pint(_pmul(a, b), e, e + 1)
        return s
    M = [[_inner(phi[j], phi[l - 1], 0, 0) for j in range(1, n + 1)] for l in range(1, n + 1)]
    B = [[_inner(phi[j], phi[l - 1], 1, 1) for j in range(1, n + 1)] for l in range(1, n + 1)]
    return (M, B, n)

def _interior_blocks_exact(p, r):
    N = p - r
    nmesh = 2 * p + 4
    M, B, n = _galerkin_exact(p, r, nmesh)
    mid = nmesh // 2
    out = {}
    for d in range(-p, p + 1):
        bj = mid + d
        if 0 <= bj < n // N:
            Mb = [[M[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            Bb = [[B[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            if any((x != 0 for row in Mb for x in row)) or any((x != 0 for row in Bb for x in row)):
                out[d] = (Mb, Bb)
    return out

def _symbol_polys(p, r, rho):
    blocks = _interior_blocks_exact(p, r)
    N = p - r
    k = max([d for d in blocks if d > 0] + [0])
    deg = k + max((-d for d in blocks))
    S = [[[Fraction(0)] * (deg + 1) for _ in range(N)] for _ in range(N)]
    for d, (Mb, Bb) in blocks.items():
        power = k - d
        for a in range(N):
            for b in range(N):
                S[a][b][power] += -Bb[a][b] + rho * Mb[a][b]
    return (S, k)

def _pdet(S):
    n = len(S)
    if n == 1:
        return S[0][0]
    tot = [Fraction(0)]
    for j in range(n):
        minor = [[S[i][c] for c in range(n) if c != j] for i in range(1, n)]
        term = _pmul(S[0][j], _pdet(minor))
        if j % 2:
            term = [-x for x in term]
        tot = _padd(tot, term)
    return tot

def _det_symbol_exact(p, r, rho):
    S, k = _symbol_polys(p, r, rho)
    return _ptrim(_pdet(S))

def symbol_determinant(p, r, rho):
    if float(rho) < 0.0:
        raise ValueError('rho must be non-negative')
    p, r = _check_pr(p, r)
    rq = _check_rho(rho)
    c = _det_symbol_exact(p, r, rq)
    return np.array([float(x) for x in c[::-1]])

import numpy as np
from fractions import Fraction

def _check_pr(p, r):
    p = int(p)
    r = int(r)
    if p < 1 or r < 0 or r > p - 1:
        raise ValueError('degree p must be >= 1 and regularity r must satisfy 0 <= r <= p - 1')
    return (p, r)

def _check_rho(rho):
    if isinstance(rho, Fraction):
        rq = rho
    else:
        rq = Fraction(rho).limit_denominator(10 ** 12) if isinstance(rho, float) else Fraction(rho)
    if rq < 0:
        raise ValueError('rho must be non-negative')
    return rq

def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else Fraction(0)) + (b[i] if i < len(b) else Fraction(0)) for i in range(n)]

def _pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out

def _pderiv(a):
    return [a[i] * i for i in range(1, len(a))] or [Fraction(0)]

def _pint(a, lo, hi):
    s = Fraction(0)
    for i, c in enumerate(a):
        s += c * (Fraction(hi) ** (i + 1) - Fraction(lo) ** (i + 1)) / (i + 1)
    return s

def _ptrim(a):
    a = list(a)
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a

def _peval(a, x):
    s = Fraction(0)
    for c in reversed(a):
        s = s * x + c
    return s

def _knots(p, r, nmesh):
    return [Fraction(0)] * (p + 1) + sum(([Fraction(e)] * (p - r) for e in range(1, nmesh)), []) + [Fraction(nmesh)] * (p + 1)

def _bsplines(p, r, nmesh):
    ks = _knots(p, r, nmesh)
    cur = []
    for j in range(len(ks) - 1):
        d = {}
        if ks[j] < ks[j + 1]:
            for e in range(int(ks[j]), int(ks[j + 1])):
                d[e] = [Fraction(1)]
        cur.append(d)
    for k in range(1, p + 1):
        nxt = []
        for j in range(len(ks) - k - 1):
            d = {}
            den1 = ks[j + k] - ks[j]
            den2 = ks[j + k + 1] - ks[j + 1]
            if den1 != 0:
                for e, poly in cur[j].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([-ks[j] / den1, Fraction(1) / den1], poly))
            if den2 != 0:
                for e, poly in cur[j + 1].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([ks[j + k + 1] / den2, Fraction(-1) / den2], poly))
            nxt.append({e: v for e, v in d.items() if any((x != 0 for x in v))})
        cur = nxt
    return cur

def _galerkin_exact(p, r, nmesh):
    phi = _bsplines(p, r, nmesh)
    n = nmesh * (p - r) + r

    def _inner(f, g, d1, d2):
        s = Fraction(0)
        for e in set(f) & set(g):
            a = f[e]
            b = g[e]
            for _ in range(d1):
                a = _pderiv(a)
            for _ in range(d2):
                b = _pderiv(b)
            s += _pint(_pmul(a, b), e, e + 1)
        return s
    M = [[_inner(phi[j], phi[l - 1], 0, 0) for j in range(1, n + 1)] for l in range(1, n + 1)]
    B = [[_inner(phi[j], phi[l - 1], 1, 1) for j in range(1, n + 1)] for l in range(1, n + 1)]
    return (M, B, n)

def _interior_blocks_exact(p, r):
    N = p - r
    nmesh = 2 * p + 4
    M, B, n = _galerkin_exact(p, r, nmesh)
    mid = nmesh // 2
    out = {}
    for d in range(-p, p + 1):
        bj = mid + d
        if 0 <= bj < n // N:
            Mb = [[M[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            Bb = [[B[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            if any((x != 0 for row in Mb for x in row)) or any((x != 0 for row in Bb for x in row)):
                out[d] = (Mb, Bb)
    return out

def _symbol_polys(p, r, rho):
    blocks = _interior_blocks_exact(p, r)
    N = p - r
    k = max([d for d in blocks if d > 0] + [0])
    deg = k + max((-d for d in blocks))
    S = [[[Fraction(0)] * (deg + 1) for _ in range(N)] for _ in range(N)]
    for d, (Mb, Bb) in blocks.items():
        power = k - d
        for a in range(N):
            for b in range(N):
                S[a][b][power] += -Bb[a][b] + rho * Mb[a][b]
    return (S, k)

def _pdet(S):
    n = len(S)
    if n == 1:
        return S[0][0]
    tot = [Fraction(0)]
    for j in range(n):
        minor = [[S[i][c] for c in range(n) if c != j] for i in range(1, n)]
        term = _pmul(S[0][j], _pdet(minor))
        if j % 2:
            term = [-x for x in term]
        tot = _padd(tot, term)
    return tot

def _det_symbol_exact(p, r, rho):
    S, k = _symbol_polys(p, r, rho)
    return _ptrim(_pdet(S))

def _is_palindromic(c):
    return all((c[i] == c[-1 - i] for i in range(len(c))))

def _dickson_reduce(c):
    m = (len(c) - 1) // 2
    C = [[Fraction(2)], [Fraction(0), Fraction(1)]]
    for j in range(2, m + 1):
        C.append(_padd(_pmul([Fraction(0), Fraction(1)], C[j - 1]), [-x for x in C[j - 2]]))
    q = [c[m]]
    for j in range(1, m + 1):
        q = _padd(q, [c[m + j] * x for x in C[j]])
    return _ptrim(q)

def _real_roots_poly(coeffs_low):
    c = _ptrim(coeffs_low)
    if len(c) <= 1:
        return []
    cf = np.array([float(x) for x in c[::-1]])
    roots = np.roots(cf)
    out = []
    dc = _pderiv(c)
    for z in roots:
        if abs(z.imag) > 1e-07 * (1.0 + abs(z.real)):
            continue
        x = Fraction(float(z.real)).limit_denominator(10 ** 15)
        for _ in range(60):
            fx = _peval(c, x)
            dfx = _peval(dc, x)
            if dfx == 0:
                break
            step = fx / dfx
            x = x - step
            if abs(step) < Fraction(1, 10 ** 30) * (1 + abs(x)):
                break
            x = x.limit_denominator(10 ** 40)
        if abs(_peval(c, x)) <= Fraction(1, 10 ** 18) * (1 + sum((abs(t) for t in c))):
            out.append(x)
    res = []
    for x in sorted(out):
        if not res or abs(x - res[-1]) > Fraction(1, 10 ** 12):
            res.append(x)
    return res

def _strip_origin(c):
    j0 = 0
    while len(c) > 1 and c[0] == 0:
        c = c[1:]
        j0 += 1
    return (j0, c)

def _classify_exact(p, r, rho):
    j0, c = _strip_origin(_det_symbol_exact(p, r, rho))
    deg = len(c) - 1
    if _is_palindromic(c) and deg % 2 == 0:
        q = _dickson_reduce(c)
        ys = _real_roots_poly(q)
        m = deg // 2
        on = 0
        inside = []
        outside = []
        qf = np.array([float(x) for x in q[::-1]])
        yroots = np.roots(qf) if len(qf) > 1 else np.array([])
        for yv in yroots:
            if abs(yv.imag) > 1e-09 * (1 + abs(yv.real)):
                t1 = (yv + np.sqrt(yv * yv - 4)) / 2
                t2 = (yv - np.sqrt(yv * yv - 4)) / 2
                for tv in (t1, t2):
                    (inside if abs(tv) < 1 else outside).append(abs(tv))
            else:
                yr = yv.real
                ysign = Fraction(2) if yr > 0 else Fraction(-2)
                at_boundary = abs(abs(yr) - 2.0) < 1e-06 and sum((cq * ysign ** i for i, cq in enumerate(q))) == 0
                if at_boundary or abs(yr) < 2.0:
                    on += 2
                else:
                    a = abs(yr) / 2.0
                    tout = a + np.sqrt(a * a - 1.0)
                    tin = 1.0 / tout
                    inside.append(tin)
                    outside.append(tout)
        s = len(inside) + j0
        l = len(outside)
        z = on
        return (s, z, l, max(inside) if inside else 0.0, min(outside) if outside else 0.0)
    cf = np.array([float(x) for x in c[::-1]])
    roots = np.roots(cf)
    inside = [abs(t) for t in roots if abs(t) < 1 - 1e-09]
    outside = [abs(t) for t in roots if abs(t) > 1 + 1e-09]
    on = len(roots) - len(inside) - len(outside)
    return (len(inside) + j0, on, len(outside), max(inside) if inside else 0.0, min(outside) if outside else 0.0)

def zero_location_type(p, r, rho):
    if float(rho) < 0.0:
        raise ValueError('rho must be non-negative')
    p, r = _check_pr(p, r)
    rq = _check_rho(rho)
    s, z, l, mi, mo = _classify_exact(p, r, rq)
    return np.array([float(s), float(z), float(l), float(mi), float(mo)])

import numpy as np
from fractions import Fraction
def _check_pr(p, r):
    p = int(p)
    r = int(r)
    if p < 1 or r < 0 or r > p - 1:
        raise ValueError('degree p must be >= 1 and regularity r must satisfy 0 <= r <= p - 1')
    return (p, r)

def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else Fraction(0)) + (b[i] if i < len(b) else Fraction(0)) for i in range(n)]

def _pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out

def _pderiv(a):
    return [a[i] * i for i in range(1, len(a))] or [Fraction(0)]

def _pint(a, lo, hi):
    s = Fraction(0)
    for i, c in enumerate(a):
        s += c * (Fraction(hi) ** (i + 1) - Fraction(lo) ** (i + 1)) / (i + 1)
    return s

def _ptrim(a):
    a = list(a)
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a

def _peval(a, x):
    s = Fraction(0)
    for c in reversed(a):
        s = s * x + c
    return s

def _knots(p, r, nmesh):
    return [Fraction(0)] * (p + 1) + sum(([Fraction(e)] * (p - r) for e in range(1, nmesh)), []) + [Fraction(nmesh)] * (p + 1)

def _bsplines(p, r, nmesh):
    ks = _knots(p, r, nmesh)
    cur = []
    for j in range(len(ks) - 1):
        d = {}
        if ks[j] < ks[j + 1]:
            for e in range(int(ks[j]), int(ks[j + 1])):
                d[e] = [Fraction(1)]
        cur.append(d)
    for k in range(1, p + 1):
        nxt = []
        for j in range(len(ks) - k - 1):
            d = {}
            den1 = ks[j + k] - ks[j]
            den2 = ks[j + k + 1] - ks[j + 1]
            if den1 != 0:
                for e, poly in cur[j].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([-ks[j] / den1, Fraction(1) / den1], poly))
            if den2 != 0:
                for e, poly in cur[j + 1].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([ks[j + k + 1] / den2, Fraction(-1) / den2], poly))
            nxt.append({e: v for e, v in d.items() if any((x != 0 for x in v))})
        cur = nxt
    return cur

def _galerkin_exact(p, r, nmesh):
    phi = _bsplines(p, r, nmesh)
    n = nmesh * (p - r) + r

    def _inner(f, g, d1, d2):
        s = Fraction(0)
        for e in set(f) & set(g):
            a = f[e]
            b = g[e]
            for _ in range(d1):
                a = _pderiv(a)
            for _ in range(d2):
                b = _pderiv(b)
            s += _pint(_pmul(a, b), e, e + 1)
        return s
    M = [[_inner(phi[j], phi[l - 1], 0, 0) for j in range(1, n + 1)] for l in range(1, n + 1)]
    B = [[_inner(phi[j], phi[l - 1], 1, 1) for j in range(1, n + 1)] for l in range(1, n + 1)]
    return (M, B, n)

def _interior_blocks_exact(p, r):
    N = p - r
    nmesh = 2 * p + 4
    M, B, n = _galerkin_exact(p, r, nmesh)
    mid = nmesh // 2
    out = {}
    for d in range(-p, p + 1):
        bj = mid + d
        if 0 <= bj < n // N:
            Mb = [[M[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            Bb = [[B[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            if any((x != 0 for row in Mb for x in row)) or any((x != 0 for row in Bb for x in row)):
                out[d] = (Mb, Bb)
    return out

def _symbol_polys(p, r, rho):
    blocks = _interior_blocks_exact(p, r)
    N = p - r
    k = max([d for d in blocks if d > 0] + [0])
    deg = k + max((-d for d in blocks))
    S = [[[Fraction(0)] * (deg + 1) for _ in range(N)] for _ in range(N)]
    for d, (Mb, Bb) in blocks.items():
        power = k - d
        for a in range(N):
            for b in range(N):
                S[a][b][power] += -Bb[a][b] + rho * Mb[a][b]
    return (S, k)

def _pdet(S):
    n = len(S)
    if n == 1:
        return S[0][0]
    tot = [Fraction(0)]
    for j in range(n):
        minor = [[S[i][c] for c in range(n) if c != j] for i in range(1, n)]
        term = _pmul(S[0][j], _pdet(minor))
        if j % 2:
            term = [-x for x in term]
        tot = _padd(tot, term)
    return tot

def _det_symbol_exact(p, r, rho):
    S, k = _symbol_polys(p, r, rho)
    return _ptrim(_pdet(S))

def _is_palindromic(c):
    return all((c[i] == c[-1 - i] for i in range(len(c))))

def _dickson_reduce(c):
    m = (len(c) - 1) // 2
    C = [[Fraction(2)], [Fraction(0), Fraction(1)]]
    for j in range(2, m + 1):
        C.append(_padd(_pmul([Fraction(0), Fraction(1)], C[j - 1]), [-x for x in C[j - 2]]))
    q = [c[m]]
    for j in range(1, m + 1):
        q = _padd(q, [c[m + j] * x for x in C[j]])
    return _ptrim(q)

def _real_roots_poly(coeffs_low):
    c = _ptrim(coeffs_low)
    if len(c) <= 1:
        return []
    cf = np.array([float(x) for x in c[::-1]])
    roots = np.roots(cf)
    out = []
    dc = _pderiv(c)
    for z in roots:
        if abs(z.imag) > 1e-07 * (1.0 + abs(z.real)):
            continue
        x = Fraction(float(z.real)).limit_denominator(10 ** 15)
        for _ in range(60):
            fx = _peval(c, x)
            dfx = _peval(dc, x)
            if dfx == 0:
                break
            step = fx / dfx
            x = x - step
            if abs(step) < Fraction(1, 10 ** 30) * (1 + abs(x)):
                break
            x = x.limit_denominator(10 ** 40)
        if abs(_peval(c, x)) <= Fraction(1, 10 ** 18) * (1 + sum((abs(t) for t in c))):
            out.append(x)
    res = []
    for x in sorted(out):
        if not res or abs(x - res[-1]) > Fraction(1, 10 ** 12):
            res.append(x)
    return res

def _strip_origin(c):
    j0 = 0
    while len(c) > 1 and c[0] == 0:
        c = c[1:]
        j0 += 1
    return (j0, c)

def _classify_exact(p, r, rho):
    j0, c = _strip_origin(_det_symbol_exact(p, r, rho))
    deg = len(c) - 1
    if _is_palindromic(c) and deg % 2 == 0:
        q = _dickson_reduce(c)
        ys = _real_roots_poly(q)
        m = deg // 2
        on = 0
        inside = []
        outside = []
        qf = np.array([float(x) for x in q[::-1]])
        yroots = np.roots(qf) if len(qf) > 1 else np.array([])
        for yv in yroots:
            if abs(yv.imag) > 1e-09 * (1 + abs(yv.real)):
                t1 = (yv + np.sqrt(yv * yv - 4)) / 2
                t2 = (yv - np.sqrt(yv * yv - 4)) / 2
                for tv in (t1, t2):
                    (inside if abs(tv) < 1 else outside).append(abs(tv))
            else:
                yr = yv.real
                ysign = Fraction(2) if yr > 0 else Fraction(-2)
                at_boundary = abs(abs(yr) - 2.0) < 1e-06 and sum((cq * ysign ** i for i, cq in enumerate(q))) == 0
                if at_boundary or abs(yr) < 2.0:
                    on += 2
                else:
                    a = abs(yr) / 2.0
                    tout = a + np.sqrt(a * a - 1.0)
                    tin = 1.0 / tout
                    inside.append(tin)
                    outside.append(tout)
        s = len(inside) + j0
        l = len(outside)
        z = on
        return (s, z, l, max(inside) if inside else 0.0, min(outside) if outside else 0.0)
    cf = np.array([float(x) for x in c[::-1]])
    roots = np.roots(cf)
    inside = [abs(t) for t in roots if abs(t) < 1 - 1e-09]
    outside = [abs(t) for t in roots if abs(t) > 1 + 1e-09]
    on = len(roots) - len(inside) - len(outside)
    return (len(inside) + j0, on, len(outside), max(inside) if inside else 0.0, min(outside) if outside else 0.0)

def _interp_poly_in_rho(fn, degree):
    xs = [Fraction(i) for i in range(degree + 1)]
    ys = [fn(x) for x in xs]
    coef = list(ys)
    for j in range(1, degree + 1):
        for i in range(degree, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (xs[i] - xs[i - j])
    poly = [Fraction(0)]
    for i in range(degree, -1, -1):
        poly = _padd(_pmul(poly, [-xs[i], Fraction(1)]), [coef[i]])
    return _ptrim(poly)

def _thresholds_exact(p, r):
    N = p - r
    j0, c0 = _strip_origin(_det_symbol_exact(p, r, Fraction(0)))
    deg = len(c0) - 1
    if not (_is_palindromic(c0) and deg % 2 == 0):
        raise RuntimeError('non-palindromic symbol determinant')
    m = deg // 2

    def _q_at(rho):
        return _dickson_reduce(_strip_origin(_det_symbol_exact(p, r, rho))[1])
    cands = []
    for yv in (Fraction(2), Fraction(-2)):
        poly = _interp_poly_in_rho(lambda rr: _peval(_q_at(rr), yv), N)
        cands += _real_roots_poly(poly)
    if m >= 2:

        def _disc(rr):
            q = _q_at(rr)
            dq = _pderiv(q)
            n1 = len(q) - 1
            n2 = len(dq) - 1
            size = n1 + n2
            Smat = [[Fraction(0)] * size for _ in range(size)]
            for i in range(n2):
                for j, cc in enumerate(q[::-1]):
                    Smat[i][i + j] = cc
            for i in range(n1):
                for j, cc in enumerate(dq[::-1]):
                    Smat[n2 + i][i + j] = cc
            return _fdet(Smat)
        poly = _interp_poly_in_rho(_disc, N * (2 * m - 2) + 2)
        cands += _real_roots_poly(poly)
    out = []
    for x in sorted(set(cands)):
        if x <= 0:
            continue
        eps = Fraction(1, 10 ** 7) * (1 + x)
        lo = _classify_exact(p, r, x - eps)[:3]
        hi = _classify_exact(p, r, x + eps)[:3]
        if lo != hi:
            out.append(x)
    return out

def _fdet(A):
    n = len(A)
    A = [row[:] for row in A]
    det = Fraction(1)
    for c in range(n):
        piv = next((i for i in range(c, n) if A[i][c] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != c:
            A[c], A[piv] = (A[piv], A[c])
            det = -det
        det *= A[c][c]
        for i in range(c + 1, n):
            if A[i][c] != 0:
                f = A[i][c] / A[c][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[c])]
    return det
    
def growth_thresholds(p, r):
    if int(p) < 1 or int(r) < 0 or int(r) > int(p) - 1:
        raise ValueError('invalid degree or regularity')
    p, r = _check_pr(p, r)
    th = _thresholds_exact(p, r)
    return np.array([float(x) for x in th])

import numpy as np
from fractions import Fraction

def _check_pr(p, r):
    p = int(p)
    r = int(r)
    if p < 1 or r < 0 or r > p - 1:
        raise ValueError('degree p must be >= 1 and regularity r must satisfy 0 <= r <= p - 1')
    return (p, r)

def _check_rho(rho):
    if isinstance(rho, Fraction):
        rq = rho
    else:
        rq = Fraction(rho).limit_denominator(10 ** 12) if isinstance(rho, float) else Fraction(rho)
    if rq < 0:
        raise ValueError('rho must be non-negative')
    return rq

def _check_nb(nblocks):
    nb = int(nblocks)
    if nb < 1:
        raise ValueError('nblocks must be a positive integer')
    return nb

def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else Fraction(0)) + (b[i] if i < len(b) else Fraction(0)) for i in range(n)]

def _pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out

def _pderiv(a):
    return [a[i] * i for i in range(1, len(a))] or [Fraction(0)]

def _pint(a, lo, hi):
    s = Fraction(0)
    for i, c in enumerate(a):
        s += c * (Fraction(hi) ** (i + 1) - Fraction(lo) ** (i + 1)) / (i + 1)
    return s

def _knots(p, r, nmesh):
    return [Fraction(0)] * (p + 1) + sum(([Fraction(e)] * (p - r) for e in range(1, nmesh)), []) + [Fraction(nmesh)] * (p + 1)

def _bsplines(p, r, nmesh):
    ks = _knots(p, r, nmesh)
    cur = []
    for j in range(len(ks) - 1):
        d = {}
        if ks[j] < ks[j + 1]:
            for e in range(int(ks[j]), int(ks[j + 1])):
                d[e] = [Fraction(1)]
        cur.append(d)
    for k in range(1, p + 1):
        nxt = []
        for j in range(len(ks) - k - 1):
            d = {}
            den1 = ks[j + k] - ks[j]
            den2 = ks[j + k + 1] - ks[j + 1]
            if den1 != 0:
                for e, poly in cur[j].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([-ks[j] / den1, Fraction(1) / den1], poly))
            if den2 != 0:
                for e, poly in cur[j + 1].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([ks[j + k + 1] / den2, Fraction(-1) / den2], poly))
            nxt.append({e: v for e, v in d.items() if any((x != 0 for x in v))})
        cur = nxt
    return cur

def _galerkin_exact(p, r, nmesh):
    phi = _bsplines(p, r, nmesh)
    n = nmesh * (p - r) + r

    def _inner(f, g, d1, d2):
        s = Fraction(0)
        for e in set(f) & set(g):
            a = f[e]
            b = g[e]
            for _ in range(d1):
                a = _pderiv(a)
            for _ in range(d2):
                b = _pderiv(b)
            s += _pint(_pmul(a, b), e, e + 1)
        return s
    M = [[_inner(phi[j], phi[l - 1], 0, 0) for j in range(1, n + 1)] for l in range(1, n + 1)]
    B = [[_inner(phi[j], phi[l - 1], 1, 1) for j in range(1, n + 1)] for l in range(1, n + 1)]
    return (M, B, n)

def _interior_blocks_exact(p, r):
    N = p - r
    nmesh = 2 * p + 4
    M, B, n = _galerkin_exact(p, r, nmesh)
    mid = nmesh // 2
    out = {}
    for d in range(-p, p + 1):
        bj = mid + d
        if 0 <= bj < n // N:
            Mb = [[M[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            Bb = [[B[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            if any((x != 0 for row in Mb for x in row)) or any((x != 0 for row in Bb for x in row)):
                out[d] = (Mb, Bb)
    return out

def _section_exact(p, r, rho, nb):
    blocks = _interior_blocks_exact(p, r)
    N = p - r
    T = [[Fraction(0)] * (N * nb) for _ in range(N * nb)]
    for bi in range(nb):
        for d, (Mb, Bb) in blocks.items():
            bj = bi + d
            if 0 <= bj < nb:
                for a in range(N):
                    for b in range(N):
                        T[N * bi + a][N * bj + b] = -Bb[a][b] + rho * Mb[a][b]
    return T

def block_toeplitz_section(p, r, rho, nblocks):
    if float(rho) < 0.0 or int(nblocks) < 1:
        raise ValueError('rho must be non-negative and nblocks positive')
    p, r = _check_pr(p, r)
    rq = _check_rho(rho)
    nb = _check_nb(nblocks)
    T = _section_exact(p, r, rq, nb)
    return np.array([[float(x) for x in row] for row in T])

import numpy as np
from fractions import Fraction

def _check_pr(p, r):
    p = int(p)
    r = int(r)
    if p < 1 or r < 0 or r > p - 1:
        raise ValueError('degree p must be >= 1 and regularity r must satisfy 0 <= r <= p - 1')
    return (p, r)

def _check_rho(rho):
    if isinstance(rho, Fraction):
        rq = rho
    else:
        rq = Fraction(rho).limit_denominator(10 ** 12) if isinstance(rho, float) else Fraction(rho)
    if rq < 0:
        raise ValueError('rho must be non-negative')
    return rq

def _check_nb(nblocks):
    nb = int(nblocks)
    if nb < 1:
        raise ValueError('nblocks must be a positive integer')
    return nb

def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else Fraction(0)) + (b[i] if i < len(b) else Fraction(0)) for i in range(n)]

def _pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out

def _pderiv(a):
    return [a[i] * i for i in range(1, len(a))] or [Fraction(0)]

def _pint(a, lo, hi):
    s = Fraction(0)
    for i, c in enumerate(a):
        s += c * (Fraction(hi) ** (i + 1) - Fraction(lo) ** (i + 1)) / (i + 1)
    return s

def _knots(p, r, nmesh):
    return [Fraction(0)] * (p + 1) + sum(([Fraction(e)] * (p - r) for e in range(1, nmesh)), []) + [Fraction(nmesh)] * (p + 1)

def _bsplines(p, r, nmesh):
    ks = _knots(p, r, nmesh)
    cur = []
    for j in range(len(ks) - 1):
        d = {}
        if ks[j] < ks[j + 1]:
            for e in range(int(ks[j]), int(ks[j + 1])):
                d[e] = [Fraction(1)]
        cur.append(d)
    for k in range(1, p + 1):
        nxt = []
        for j in range(len(ks) - k - 1):
            d = {}
            den1 = ks[j + k] - ks[j]
            den2 = ks[j + k + 1] - ks[j + 1]
            if den1 != 0:
                for e, poly in cur[j].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([-ks[j] / den1, Fraction(1) / den1], poly))
            if den2 != 0:
                for e, poly in cur[j + 1].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([ks[j + k + 1] / den2, Fraction(-1) / den2], poly))
            nxt.append({e: v for e, v in d.items() if any((x != 0 for x in v))})
        cur = nxt
    return cur

def _galerkin_exact(p, r, nmesh):
    phi = _bsplines(p, r, nmesh)
    n = nmesh * (p - r) + r

    def _inner(f, g, d1, d2):
        s = Fraction(0)
        for e in set(f) & set(g):
            a = f[e]
            b = g[e]
            for _ in range(d1):
                a = _pderiv(a)
            for _ in range(d2):
                b = _pderiv(b)
            s += _pint(_pmul(a, b), e, e + 1)
        return s
    M = [[_inner(phi[j], phi[l - 1], 0, 0) for j in range(1, n + 1)] for l in range(1, n + 1)]
    B = [[_inner(phi[j], phi[l - 1], 1, 1) for j in range(1, n + 1)] for l in range(1, n + 1)]
    return (M, B, n)

def _interior_blocks_exact(p, r):
    N = p - r
    nmesh = 2 * p + 4
    M, B, n = _galerkin_exact(p, r, nmesh)
    mid = nmesh // 2
    out = {}
    for d in range(-p, p + 1):
        bj = mid + d
        if 0 <= bj < n // N:
            Mb = [[M[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            Bb = [[B[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            if any((x != 0 for row in Mb for x in row)) or any((x != 0 for row in Bb for x in row)):
                out[d] = (Mb, Bb)
    return out

def _section_exact(p, r, rho, nb):
    blocks = _interior_blocks_exact(p, r)
    N = p - r
    T = [[Fraction(0)] * (N * nb) for _ in range(N * nb)]
    for bi in range(nb):
        for d, (Mb, Bb) in blocks.items():
            bj = bi + d
            if 0 <= bj < nb:
                for a in range(N):
                    for b in range(N):
                        T[N * bi + a][N * bj + b] = -Bb[a][b] + rho * Mb[a][b]
    return T

def _inverse_exact(T):
    n = len(T)
    A = [row[:] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(T)]
    for c in range(n):
        piv = next((i for i in range(c, n) if A[i][c] != 0), None)
        if piv is None:
            raise RuntimeError('singular block Toeplitz section')
        A[c], A[piv] = (A[piv], A[c])
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        rowc = A[c]
        for i in range(n):
            if i != c and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], rowc)]
    return [row[n:] for row in A]

def _kappa_exact(p, r, rho, nb):
    T = _section_exact(p, r, rho, nb)
    Ti = _inverse_exact(T)
    Tf = np.array([[float(x) for x in row] for row in T])
    Tif = np.array([[float(x) for x in row] for row in Ti])
    smax = float(np.linalg.norm(Tf, 2))
    sinv = float(np.linalg.norm(Tif, 2))
    return (smax * sinv, smax, 1.0 / sinv)

def exact_condition_number(p, r, rho, nblocks):
    if float(rho) < 0.0 or int(nblocks) < 1:
        raise ValueError('rho must be non-negative and nblocks positive')
    p, r = _check_pr(p, r)
    rq = _check_rho(rho)
    nb = _check_nb(nblocks)
    k, smax, smin = _kappa_exact(p, r, rq, nb)
    return np.array([np.log10(k), smax, np.log10(smin)])

import numpy as np
from fractions import Fraction

def _check_pr(p, r):
    p = int(p)
    r = int(r)
    if p < 1 or r < 0 or r > p - 1:
        raise ValueError('degree p must be >= 1 and regularity r must satisfy 0 <= r <= p - 1')
    return (p, r)

def _check_rho(rho):
    if isinstance(rho, Fraction):
        rq = rho
    else:
        rq = Fraction(rho).limit_denominator(10 ** 12) if isinstance(rho, float) else Fraction(rho)
    if rq < 0:
        raise ValueError('rho must be non-negative')
    return rq

def _padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else Fraction(0)) + (b[i] if i < len(b) else Fraction(0)) for i in range(n)]

def _pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out

def _pderiv(a):
    return [a[i] * i for i in range(1, len(a))] or [Fraction(0)]

def _pint(a, lo, hi):
    s = Fraction(0)
    for i, c in enumerate(a):
        s += c * (Fraction(hi) ** (i + 1) - Fraction(lo) ** (i + 1)) / (i + 1)
    return s

def _knots(p, r, nmesh):
    return [Fraction(0)] * (p + 1) + sum(([Fraction(e)] * (p - r) for e in range(1, nmesh)), []) + [Fraction(nmesh)] * (p + 1)

def _bsplines(p, r, nmesh):
    ks = _knots(p, r, nmesh)
    cur = []
    for j in range(len(ks) - 1):
        d = {}
        if ks[j] < ks[j + 1]:
            for e in range(int(ks[j]), int(ks[j + 1])):
                d[e] = [Fraction(1)]
        cur.append(d)
    for k in range(1, p + 1):
        nxt = []
        for j in range(len(ks) - k - 1):
            d = {}
            den1 = ks[j + k] - ks[j]
            den2 = ks[j + k + 1] - ks[j + 1]
            if den1 != 0:
                for e, poly in cur[j].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([-ks[j] / den1, Fraction(1) / den1], poly))
            if den2 != 0:
                for e, poly in cur[j + 1].items():
                    d[e] = _padd(d.get(e, [Fraction(0)]), _pmul([ks[j + k + 1] / den2, Fraction(-1) / den2], poly))
            nxt.append({e: v for e, v in d.items() if any((x != 0 for x in v))})
        cur = nxt
    return cur

def _galerkin_exact(p, r, nmesh):
    phi = _bsplines(p, r, nmesh)
    n = nmesh * (p - r) + r

    def _inner(f, g, d1, d2):
        s = Fraction(0)
        for e in set(f) & set(g):
            a = f[e]
            b = g[e]
            for _ in range(d1):
                a = _pderiv(a)
            for _ in range(d2):
                b = _pderiv(b)
            s += _pint(_pmul(a, b), e, e + 1)
        return s
    M = [[_inner(phi[j], phi[l - 1], 0, 0) for j in range(1, n + 1)] for l in range(1, n + 1)]
    B = [[_inner(phi[j], phi[l - 1], 1, 1) for j in range(1, n + 1)] for l in range(1, n + 1)]
    return (M, B, n)

def _interior_blocks_exact(p, r):
    N = p - r
    nmesh = 2 * p + 4
    M, B, n = _galerkin_exact(p, r, nmesh)
    mid = nmesh // 2
    out = {}
    for d in range(-p, p + 1):
        bj = mid + d
        if 0 <= bj < n // N:
            Mb = [[M[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            Bb = [[B[N * mid + a][N * bj + b] for b in range(N)] for a in range(N)]
            if any((x != 0 for row in Mb for x in row)) or any((x != 0 for row in Bb for x in row)):
                out[d] = (Mb, Bb)
    return out

def _section_exact(p, r, rho, nb):
    blocks = _interior_blocks_exact(p, r)
    N = p - r
    T = [[Fraction(0)] * (N * nb) for _ in range(N * nb)]
    for bi in range(nb):
        for d, (Mb, Bb) in blocks.items():
            bj = bi + d
            if 0 <= bj < nb:
                for a in range(N):
                    for b in range(N):
                        T[N * bi + a][N * bj + b] = -Bb[a][b] + rho * Mb[a][b]
    return T

def _inverse_exact(T):
    n = len(T)
    A = [row[:] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(T)]
    for c in range(n):
        piv = next((i for i in range(c, n) if A[i][c] != 0), None)
        if piv is None:
            raise RuntimeError('singular block Toeplitz section')
        A[c], A[piv] = (A[piv], A[c])
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        rowc = A[c]
        for i in range(n):
            if i != c and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], rowc)]
    return [row[n:] for row in A]

def _kappa_exact(p, r, rho, nb):
    T = _section_exact(p, r, rho, nb)
    Ti = _inverse_exact(T)
    Tf = np.array([[float(x) for x in row] for row in T])
    Tif = np.array([[float(x) for x in row] for row in Ti])
    smax = float(np.linalg.norm(Tf, 2))
    sinv = float(np.linalg.norm(Tif, 2))
    return (smax * sinv, smax, 1.0 / sinv)

def growth_factors(p, r, rho, nb_list):
    p, r = _check_pr(p, r)
    rq = _check_rho(rho)
    nbs = [int(x) for x in np.atleast_1d(nb_list)]
    if any((x < 1 for x in nbs)):
        raise ValueError('block counts must be positive integers')
    out = []
    for nb in nbs:
        k1 = _kappa_exact(p, r, rq, nb)[0]
        k2 = _kappa_exact(p, r, rq, nb + 1)[0]
        out.append(k2 / k1)
    return np.array(out)

import numpy as np
from fractions import Fraction

def _check_pr(p, r):
    p = int(p)
    r = int(r)
    if p < 1 or r < 0 or r > p - 1:
        raise ValueError('degree p must be >= 1 and regularity r must satisfy 0 <= r <= p - 1')
    return (p, r)

def _check_rho(rho):
    if isinstance(rho, Fraction):
        rq = rho
    else:
        rq = Fraction(rho).limit_denominator(10 ** 12) if isinstance(rho, float) else Fraction(rho)
    if rq < 0:
        raise ValueError('rho must be non-negative')
    return rq

def _check_nb(nblocks):
    nb = int(nblocks)
    if nb < 1:
        raise ValueError('nblocks must be a positive integer')
    return nb

def iga_conditioning_audit(p, r, rho, nb_final):
    p, r = _check_pr(p, r)
    rq = _check_rho(rho)
    nb = _check_nb(nb_final)
    if nb < 3:
        raise ValueError('nb_final must be at least 3')
    nmesh = 2 * p + 4
    vals = spline_basis_values(p, r, nmesh, np.array([0.3, 1.7, nmesh - 0.25]))
    if np.max(np.abs(vals.sum(axis=0) - 1.0)) > 1e-12:
        raise RuntimeError('basis does not form a partition of unity')
    G = petrov_galerkin_matrices(p, r, nmesh)
    blocks = interior_symbol_blocks(p, r)
    N = p - r
    mid = nmesh // 2
    for d in range(-p, p + 1):
        bj = mid + d
        if 0 <= bj < (nmesh * N + r) // N:
            if np.max(np.abs(G[0, N * mid:N * mid + N, N * bj:N * bj + N] - blocks[d + p, 0])) > 1e-12 or np.max(np.abs(G[1, N * mid:N * mid + N, N * bj:N * bj + N] - blocks[d + p, 1])) > 1e-12:
                raise RuntimeError('interior blocks do not match the assembled matrices')
    th = growth_thresholds(p, r)
    typ = zero_location_type(p, r, rq)
    kap = exact_condition_number(p, r, rq, nb)
    gf = growth_factors(p, r, rq, [nb - 1])
    sec = block_toeplitz_section(p, r, rq, 2)
    detc = symbol_determinant(p, r, rq)
    t20 = growth_thresholds(2, 0)
    t30 = growth_thresholds(3, 0)
    t31 = growth_thresholds(3, 1)
    if len(t20) != 3 or abs(t20[-1] - 60.0) > 1e-09:
        raise RuntimeError('the (2, 0) thresholds of the source are not reproduced')
    row0 = np.zeros(4)
    row0[:min(3, len(th))] = th[:3]
    row0[3] = float(len(th))
    row1 = np.array([typ[0], typ[1], typ[2], 1.0 / typ[3] if typ[3] > 0 else 0.0])
    row2 = np.array([kap[0], kap[1], kap[2], gf[0]])
    row3 = np.array([t20[-1], t30[-1], t31[0], float(sec[0, 0])])
    audit = np.vstack([row0, row1, row2, row3])
    nz = np.nonzero(detc)[0]
    core = detc[nz[0]:nz[-1] + 1]
    if not np.all(np.isfinite(audit)) or abs(core[0] - core[-1]) > 1e-09 * abs(core[0]):
        raise RuntimeError('audit not finite or symbol determinant not self-reciprocal')
    return audit
SCICODE_GOLD_EOF
