"""
Return, in increasing order, every positive value of rho at which the zero location type of the previous step changes, to at least ten significant figures. Such a change happens only where a zero of the determinant crosses the unit circle, that is where a real root of the reduced polynomial in y crosses 2 or -2, or where a pair of real roots of the reduced polynomial turns complex; the candidate values are the positive roots of the exact polynomials in rho obtained by evaluating the reduced polynomial at y = 2 and at y = -2 and by its discriminant, and a candidate is kept only if the type really differs on the two sides of it. Reject invalid p or r.

The thresholds separate parameter ranges of at most polynomial growth of the condition numbers from ranges of exponential growth, and they are algebraic numbers determined exactly by the spline space, so they can be checked against the source, which lists them for its three cases. The window between two close thresholds can be very narrow, so the candidates must be located exactly rather than by scanning rho.

Returns
-------
A one-dimensional float64 array of threshold values in increasing order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def growth_thresholds(p, r):
    """Return, in increasing order, every positive value of rho at which the zero location type
    of the previous step changes, to at least ten significant figures.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.

    Returns:
        numpy.ndarray of the positive values of rho, in increasing order, at which the
        zero location type changes.

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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
    
def _oracle_growth_thresholds(p, r):
    if int(p) < 1 or int(r) < 0 or int(r) > int(p) - 1:
        raise ValueError('invalid degree or regularity')
    p, r = _check_pr(p, r)
    th = _thresholds_exact(p, r)
    return np.array([float(x) for x in th])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\np, r = 2, 0\n',
         'call': 'growth_thresholds(p, r)',
         'gold_call': '_oracle_growth_thresholds(p, r)'},
        {'setup': 'import numpy as np\np, r = 3, 0\n',
         'call': 'growth_thresholds(p, r)',
         'gold_call': '_oracle_growth_thresholds(p, r)'},
        {'setup': 'import numpy as np\np, r = 3, 1\n',
         'call': 'growth_thresholds(p, r)',
         'gold_call': '_oracle_growth_thresholds(p, r)'},
        {'setup': 'import numpy as np\np, r = 4, 2\n',
         'call': 'growth_thresholds(p, r)',
         'gold_call': '_oracle_growth_thresholds(p, r)'},
    ]
