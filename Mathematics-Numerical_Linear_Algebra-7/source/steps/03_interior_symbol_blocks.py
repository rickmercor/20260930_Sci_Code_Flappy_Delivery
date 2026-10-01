"""
Return the N x N blocks, N = p - r, that define the block Toeplitz structure of the two matrices of the previous step away from the boundaries: partition rows and columns into consecutive blocks of size N starting at the first row and column (rows and columns 0 to N - 1 form block 0), and for a block row deep inside the matrix return the mass block and the stiffness block found at block offset d from the diagonal, where d is the block column index minus the block row index, for d from -p to p, in an array indexed by d + p, with zero blocks where the offset is empty. The blocks must be independent of the block row chosen and of nmesh once the mesh is large enough for the interior to exist; the last r rows and columns of the matrices are boundary perturbations and are not part of the pattern. Reject invalid p or r.

Inside the matrix, every block row is the same as the previous one shifted by one block, because the mesh is uniform and the splines are translates of each other; only a few blocks near the corners deviate, and for r > 0 the matrix carries r extra rows and columns. The block Toeplitz operator whose finite sections we study is the infinite matrix built from these interior blocks alone.

Returns
-------
A (2p + 1, 2, N, N) float64 array of interior blocks, mass first, indexed by offset plus p.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interior_symbol_blocks(p, r):
    """Return the N x N blocks, N = p - r, that define the block Toeplitz structure of the two
    matrices of the previous step away from the boundaries: partition rows and columns into
    consecutive blocks of size N starting at the first row and column (rows and columns 0 to
    N - 1 form block 0), and for a block row deep inside the matrix return the mass block
    and the stiffness block found at block offset d from the diagonal, where d is the block
    column index minus the block row index, for d from -p to p, in an array indexed by d +
    p, with zero blocks where the offset is empty.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.

    Returns:
        numpy.ndarray of shape (2p + 1, 2, N, N), N = p - r: entry [d + p, 0] is the
        mass block and [d + p, 1] the stiffness block at block offset d = block column
        index minus block row index (blocks of size N counted from row and column 0),
        zero when absent.

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

def _oracle_interior_symbol_blocks(p, r):
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\np, r = 2, 0\n',
         'call': 'interior_symbol_blocks(p, r)',
         'gold_call': '_oracle_interior_symbol_blocks(p, r)'},
        {'setup': 'import numpy as np\np, r = 3, 0\n',
         'call': 'interior_symbol_blocks(p, r)',
         'gold_call': '_oracle_interior_symbol_blocks(p, r)'},
        {'setup': 'import numpy as np\np, r = 3, 1\n',
         'call': 'interior_symbol_blocks(p, r)',
         'gold_call': '_oracle_interior_symbol_blocks(p, r)'},
        {'setup': 'import numpy as np\np, r = 4, 2\n',
         'call': 'interior_symbol_blocks(p, r)',
         'gold_call': '_oracle_interior_symbol_blocks(p, r)'},
    ]
