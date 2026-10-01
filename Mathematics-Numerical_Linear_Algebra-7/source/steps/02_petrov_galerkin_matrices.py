"""
Return the two n x n Petrov-Galerkin matrices of the space-time scheme on the mesh of the previous step, with n = nmesh (p - r) + r. The trial space is spanned by the basis functions with indices 1 to n, which vanish at t = 0, and the test space by the functions with indices 0 to n - 1, which vanish at t = nmesh; the mass matrix has entries M[l, j] = integral over [0, nmesh] of phi_j times phi_{l-1}, and the stiffness matrix has entries B[l, j] = integral of the first derivatives of the same two functions, for l, j = 1, ..., n, both computed exactly. With unit elements these are already the mesh-independent scaled matrices of the source, whose entries are rational numbers. Reject invalid p, r or nmesh < 2.

The scheme is a Petrov-Galerkin method: the trial functions carry the initial condition u(0) = 0 and the test functions the end condition v(T) = 0, which shifts the two index ranges by one and makes both matrices non-symmetric. This shift, not the spline degree, is what turns the bilinear form into a block Toeplitz operator with a matrix-polynomial symbol, so the assignment of the test index to the row must be exactly as stated.

Returns
-------
A (2, n, n) float64 array holding the mass matrix and the stiffness matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def petrov_galerkin_matrices(p, r, nmesh):
    """Return the two n x n Petrov-Galerkin matrices of the space-time scheme on the mesh of
    the previous step, with n = nmesh (p - r) + r.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.
        nmesh (int): number of uniform unit elements, at least 2.

    Returns:
        numpy.ndarray of shape (2, n, n) with n = nmesh (p - r) + r: the mass matrix in
        [0] and the stiffness matrix in [1].

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
        ValueError: if nmesh < 2.
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

def _oracle_petrov_galerkin_matrices(p, r, nmesh):
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\np, r, nmesh = 2, 0, 4\n',
         'call': 'petrov_galerkin_matrices(p, r, nmesh)',
         'gold_call': '_oracle_petrov_galerkin_matrices(p, r, nmesh)'},
        {'setup': 'import numpy as np\np, r, nmesh = 3, 0, 3\n',
         'call': 'petrov_galerkin_matrices(p, r, nmesh)',
         'gold_call': '_oracle_petrov_galerkin_matrices(p, r, nmesh)'},
        {'setup': 'import numpy as np\np, r, nmesh = 3, 1, 4\n',
         'call': 'petrov_galerkin_matrices(p, r, nmesh)',
         'gold_call': '_oracle_petrov_galerkin_matrices(p, r, nmesh)'},
        {'setup': 'import numpy as np\np, r, nmesh = 4, 2, 5\n',
         'call': 'petrov_galerkin_matrices(p, r, nmesh)',
         'gold_call': '_oracle_petrov_galerkin_matrices(p, r, nmesh)'},
    ]
