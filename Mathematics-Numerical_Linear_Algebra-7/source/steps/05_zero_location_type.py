"""
Classify the zeros of det S_rho(t) from the previous step with respect to the unit circle: return the number of zeros strictly inside (zeros at the origin included), on, and strictly outside the circle, followed by the largest modulus among the inside zeros and the smallest modulus among the outside zeros, each 0 if that group is empty. Zeros on the circle must be identified exactly, not by a floating tolerance on the modulus: after removing the factor t^j from a zero of multiplicity j at the origin the determinant is self-reciprocal, so the substitution y = t + 1/t reduces it to a polynomial in y of half the degree whose real roots in [-2, 2] correspond to pairs of zeros on the circle and whose other roots correspond to pairs inside and outside. Reject invalid p, r or a negative rho.

For these symbols the zeros come in pairs t and 1/t. A pair on the unit circle means the operator is not Fredholm and the condition numbers grow at most polynomially; a pair off the circle contributes one zero inside and one outside and, when it is not compensated, exponential growth whose rate is set by the zero closest to the circle. The classification therefore has to be exact at the level of counting, and the two moduli fix the rate.

Returns
-------
A (5,) float64 array: inside count, on-circle count, outside count, largest inside modulus, smallest outside modulus.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def zero_location_type(p, r, rho):
    """Classify the zeros of det S_rho(t) from the previous step with respect to the unit
    circle: return the number of zeros strictly inside (zeros at the origin included), on,
    and strictly outside the circle, followed by the largest modulus among the inside zeros
    and the smallest modulus among the outside zeros, each 0 if that group is empty.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.
        rho (float or Fraction): non-negative parameter of the wave scheme.

    Returns:
        numpy.ndarray of shape (5,): numbers of zeros strictly inside, on and strictly
        outside the unit circle, the largest modulus among the inside zeros and the
        smallest modulus among the outside zeros (0 when a group is empty).

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
        ValueError: if rho is negative.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================

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

def _oracle_zero_location_type(p, r, rho):
    if float(rho) < 0.0:
        raise ValueError('rho must be non-negative')
    p, r = _check_pr(p, r)
    rq = _check_rho(rho)
    s, z, l, mi, mo = _classify_exact(p, r, rq)
    return np.array([float(s), float(z), float(l), float(mi), float(mo)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\np, r, rho = 4, 2, 60.0\n',
         'call': 'zero_location_type(p, r, rho)',
         'gold_call': '_oracle_zero_location_type(p, r, rho)'},
        {'setup': 'import numpy as np\np, r, rho = 4, 2, 9.875\n',
         'call': 'zero_location_type(p, r, rho)',
         'gold_call': '_oracle_zero_location_type(p, r, rho)'},
        {'setup': 'import numpy as np\np, r, rho = 2, 0, 11.0\n',
         'call': 'zero_location_type(p, r, rho)',
         'gold_call': '_oracle_zero_location_type(p, r, rho)'},
        {'setup': 'import numpy as np\np, r, rho = 3, 1, 20.0\n',
         'call': 'zero_location_type(p, r, rho)',
         'gold_call': '_oracle_zero_location_type(p, r, rho)'},
    ]
