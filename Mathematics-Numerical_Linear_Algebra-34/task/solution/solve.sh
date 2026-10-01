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
from math import gcd


def identify_scaling_structure(exponents: np.ndarray) -> np.ndarray:
    """Reference implementation: exact rational nullspace of [A | -1]."""
    E = np.asarray(exponents, dtype=np.int64)
    if E.ndim != 2 or E.shape[0] == 0:
        raise ValueError("exponents must be a non-empty two-dimensional array")

    n_mono, n_var = E.shape
    rows = [[Fraction(int(E[i, j])) for j in range(n_var)] + [Fraction(-1)]
            for i in range(n_mono)]
    n_col = n_var + 1

    piv, r = [], 0
    for col in range(n_col):
        p = next((i for i in range(r, n_mono) if rows[i][col] != 0), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        pv = rows[r][col]
        rows[r] = [v / pv for v in rows[r]]
        for i in range(n_mono):
            if i != r and rows[i][col] != 0:
                f = rows[i][col]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        piv.append(col)
        r += 1
        if r == n_mono:
            break

    free = [col for col in range(n_col) if col not in piv]
    if len(free) != 1:
        raise ValueError("scaling structure is not uniquely determined")

    f = free[0]
    sol = [Fraction(0)] * n_col
    sol[f] = Fraction(1)
    for i, col in enumerate(piv):
        sol[col] = -rows[i][f]

    den = 1
    for v in sol:
        den = den * v.denominator // gcd(den, v.denominator)
    ints = [int(v * den) for v in sol]

    g = 0
    for v in ints:
        g = gcd(g, abs(v))
    ints = [v // g for v in ints]

    if ints[-1] < 0:
        ints = [-v for v in ints]
    if any(v <= 0 for v in ints):
        raise ValueError("no strictly positive integer solution exists")

    return np.array(ints, dtype=np.float64)

import numpy as np


def compute_certified_maximum(coeffs: np.ndarray, exponents: np.ndarray) -> float:
    """Reference implementation: bound each monomial by one on the box."""
    c = np.asarray(coeffs, dtype=np.float64)
    E = np.asarray(exponents, dtype=np.int64)

    if c.ndim != 1 or E.ndim != 2:
        raise ValueError("coeffs must be one-dimensional and exponents two-dimensional")
    if c.size != E.shape[0] or c.size == 0:
        raise ValueError("coeffs and exponents must be non-empty and agree in length")

    return float(np.abs(c).sum())

import numpy as np
from fractions import Fraction


def compute_box_moments(coeffs: np.ndarray, exponents: np.ndarray, max_order: int) -> np.ndarray:
    """Reference implementation: exact multinomial expansion, then term-by-term integration."""
    c = np.asarray(coeffs, dtype=np.float64)
    E = np.asarray(exponents, dtype=np.int64)

    if c.ndim != 1 or E.ndim != 2:
        raise ValueError("coeffs must be one-dimensional and exponents two-dimensional")
    if c.size != E.shape[0] or c.size == 0:
        raise ValueError("coeffs and exponents must be non-empty and agree in length")
    if int(max_order) < 0:
        raise ValueError("max_order must be non-negative")

    C = [Fraction(float(v)) for v in c]
    n_var = E.shape[1]

    g_poly = {}
    for coeff, row in zip(C, E):
        key = tuple(int(v) for v in row)
        g_poly[key] = g_poly.get(key, Fraction(0)) + coeff

    def _h_integrate(poly):
        """Integrate a monomial dict against the uniform probability measure on the box."""
        total = Fraction(0)
        for a, coeff in poly.items():
            if any(e % 2 for e in a):
                continue
            t = coeff
            for e in a:
                t /= (e + 1)
            total += t
        return total

    def _h_multiply(p, q):
        """Multiply two polynomials represented as exponent-tuple dicts."""
        out = {}
        for a, ca in p.items():
            for b, cb in q.items():
                k = tuple(x + y for x, y in zip(a, b))
                out[k] = out.get(k, Fraction(0)) + ca * cb
        return out

    cur = {tuple([0] * n_var): Fraction(1)}
    ys = [_h_integrate(cur)]
    for _ in range(int(max_order)):
        cur = _h_multiply(cur, g_poly)
        ys.append(_h_integrate(cur))

    return np.array([float(v) for v in ys], dtype=np.float64)

import numpy as np
from fractions import Fraction


def compute_reference_moments(weights: np.ndarray, weighted_degree: int, max_order: int) -> np.ndarray:
    """Reference implementation: exact rational evaluation of the weighted moment constants."""
    w = np.asarray(weights, dtype=np.int64)
    if w.size == 0 or np.any(w <= 0):
        raise ValueError("weights must be strictly positive integers")

    m = int(weighted_degree)
    if m < 1:
        raise ValueError("weighted degree must be a positive integer")

    if int(max_order) < 0:
        raise ValueError("max_order must be non-negative")

    weight_sum = Fraction(int(w.sum()))

    out = []
    for k in range(int(max_order) + 1):
        out.append(float(weight_sum / (weight_sum + Fraction(k * m))))

    return np.array(out, dtype=np.float64)

import numpy as np
from fractions import Fraction


def compute_upper_bound(box_moments: np.ndarray, reference_moments: np.ndarray, dimension: int) -> float:
    """Reference implementation: exact rational bisection on the definiteness of the affine matrix family."""
    y = np.asarray(box_moments, dtype=np.float64)
    z = np.asarray(reference_moments, dtype=np.float64)

    if y.ndim != 1 or z.ndim != 1:
        raise ValueError("moment sequences must be one-dimensional")
    if y.size != z.size or y.size == 0:
        raise ValueError("moment sequences must be non-empty and of equal length")

    Y = [Fraction(float(v)) for v in y]
    Z = [Fraction(float(v)) for v in z]

    s = (len(Y) - 1) // 2 + 1
    A = [[Y[i + j] for j in range(s)] for i in range(s)]
    B = [[Z[i + j] for j in range(s)] for i in range(s)]

    def _h_pos_def(M):
        """Exact positive-definiteness test by symmetric Gaussian elimination."""
        n = len(M)
        m = [row[:] for row in M]
        for i in range(n):
            p = m[i][i]
            if p <= 0:
                return False
            inv = Fraction(1) / p
            for r in range(i + 1, n):
                f = m[r][i] * inv
                if f == 0:
                    continue
                for c in range(i, n):
                    m[r][c] -= f * m[i][c]
        return True

    if not _h_pos_def(B):
        raise ValueError("subtracted part is not positive definite")

    lo, hi = Fraction(0), Fraction(2) ** 40
    for _ in range(120):
        mid = (lo + hi) / 2
        M = [[A[i][j] - mid * B[i][j] for j in range(s)] for i in range(s)]
        if _h_pos_def(M):
            lo = mid
        else:
            hi = mid

    return float(2 ** int(dimension)) * float(lo)

import numpy as np
from fractions import Fraction


def compute_lower_bound(box_moments: np.ndarray, reference_moments: np.ndarray, gbar: float, dimension: int) -> float:
    """Reference implementation: exact rational bisection on the weighted affine matrix family."""
    y = np.asarray(box_moments, dtype=np.float64)
    z = np.asarray(reference_moments, dtype=np.float64)

    if y.ndim != 1 or z.ndim != 1:
        raise ValueError("moment sequences must be one-dimensional")
    if y.size != z.size or y.size == 0:
        raise ValueError("moment sequences must be non-empty and of equal length")

    gb = Fraction(float(gbar))
    if gb <= 1:
        raise ValueError("gbar must exceed 1")

    q = (y.size - 1) // 2
    if q == 0:
        raise ValueError("budget is too small to form the weighted construction")

    Y = [Fraction(float(v)) for v in y]
    Z = [Fraction(float(v)) for v in z]
    h = [-gb, gb + 1, Fraction(-1)]

    A = [[sum(h[k] * Y[i + j + k] for k in range(3)) for j in range(q)] for i in range(q)]
    B = [[-sum(h[k] * Z[i + j + k] for k in range(3)) for j in range(q)] for i in range(q)]

    def _h_pos_def(M):
        """Exact positive-definiteness test by symmetric Gaussian elimination."""
        n = len(M)
        m = [row[:] for row in M]
        for i in range(n):
            p = m[i][i]
            if p <= 0:
                return False
            inv = Fraction(1) / p
            for r in range(i + 1, n):
                f = m[r][i] * inv
                if f == 0:
                    continue
                for c in range(i, n):
                    m[r][c] -= f * m[i][c]
        return True

    if not _h_pos_def(B):
        raise ValueError("subtracted part is not positive definite")

    lo, hi = Fraction(-(2 ** 40)), Fraction(2 ** 40)
    for _ in range(140):
        mid = (lo + hi) / 2
        M = [[A[i][j] - mid * B[i][j] for j in range(q)] for i in range(q)]
        if _h_pos_def(M):
            lo = mid
        else:
            hi = mid

    return -float(2 ** int(dimension)) * float(lo)

import numpy as np


def compute_certified_volume_bound(coeffs: np.ndarray, exponents: np.ndarray, max_order: int) -> float:
    """Reference implementation: compose the reference implementations of the preceding steps."""
    E = np.asarray(exponents, dtype=np.int64)
    if E.ndim != 2:
        raise ValueError("exponents must be two-dimensional")

    dimension = E.shape[1]

    structure = identify_scaling_structure(E)
    weights = np.rint(structure[:dimension]).astype(np.int64)
    weighted_degree = int(round(float(structure[dimension])))

    gbar = compute_certified_maximum(coeffs, E)

    box_moments = compute_box_moments(coeffs, E, max_order)
    reference_moments = compute_reference_moments(weights, weighted_degree, max_order)

    upper = compute_upper_bound(box_moments, reference_moments, dimension)
    lower = compute_lower_bound(box_moments, reference_moments, gbar, dimension)

    if lower > upper:
        raise ValueError("computed bracket is inconsistent")

    return float(upper)
SCICODE_GOLD_EOF
