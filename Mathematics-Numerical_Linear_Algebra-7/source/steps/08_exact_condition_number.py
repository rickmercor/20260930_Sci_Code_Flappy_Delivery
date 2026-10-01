"""
Return the base-10 logarithm of the spectral condition number of the block Toeplitz section of the previous step, together with its largest singular value and the base-10 logarithm of its smallest singular value. The condition numbers of interest exceed the reciprocal of the double precision unit roundoff by many orders of magnitude, so a floating-point singular value decomposition of the section cannot deliver them: the reciprocal of the smallest singular value has to be obtained as the spectral norm of the exactly computed inverse, whose rational entries are only converted to floating point at the end, or by an equivalent exact or extended-precision route, and the result must be accurate to a relative error of 1e-10. Reject invalid p, r, a negative rho, nblocks < 1, and a singular section.

In the exponential regime the smallest singular value of the section decays like a geometric sequence in the number of blocks and soon falls below the rounding level of double precision arithmetic, where any floating-point factorisation returns noise of the order of the unit roundoff times the norm; the source's own condition number plots saturate there. Because the entries are rational, the inverse can be computed exactly, and the two spectral norms are then well conditioned quantities.

Returns
-------
A (3,) float64 array: log10 of the condition number, the largest singular value, log10 of the smallest singular value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exact_condition_number(p, r, rho, nblocks):
    """Return the base-10 logarithm of the spectral condition number of the block Toeplitz
    section of the previous step, together with its largest singular value and the base-10
    logarithm of its smallest singular value.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.
        rho (float or Fraction): non-negative parameter of the wave scheme.
        nblocks (int): number of blocks of the section, at least 1.

    Returns:
        numpy.ndarray of shape (3,): log10 of the spectral condition number, the largest
        singular value, and log10 of the smallest singular value of the section.

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
        ValueError: if rho is negative or nblocks < 1.
        RuntimeError: if the section is singular.
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

def _oracle_exact_condition_number(p, r, rho, nblocks):
    if float(rho) < 0.0 or int(nblocks) < 1:
        raise ValueError('rho must be non-negative and nblocks positive')
    p, r = _check_pr(p, r)
    rq = _check_rho(rho)
    nb = _check_nb(nblocks)
    k, smax, smin = _kappa_exact(p, r, rq, nb)
    return np.array([np.log10(k), smax, np.log10(smin)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\np, r, rho, nblocks = 4, 2, 60.0, 20\n',
         'call': 'exact_condition_number(p, r, rho, nblocks)',
         'gold_call': '_oracle_exact_condition_number(p, r, rho, nblocks)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\np, r, rho, nblocks = 2, 0, 70.0, 14\n',
         'call': 'exact_condition_number(p, r, rho, nblocks)',
         'gold_call': '_oracle_exact_condition_number(p, r, rho, nblocks)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\np, r, rho, nblocks = 3, 1, 50.0, 14\n',
         'call': 'exact_condition_number(p, r, rho, nblocks)',
         'gold_call': '_oracle_exact_condition_number(p, r, rho, nblocks)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\np, r, rho, nblocks = 4, 2, 9.875, 25\n',
         'call': 'exact_condition_number(p, r, rho, nblocks)',
         'gold_call': '_oracle_exact_condition_number(p, r, rho, nblocks)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\np, r, rho, nblocks = 4, 2, 60.0, 40\n',
         'call': 'exact_condition_number(p, r, rho, nblocks)',
         'gold_call': '_oracle_exact_condition_number(p, r, rho, nblocks)',
         'tol': 1e-08},
    ]
