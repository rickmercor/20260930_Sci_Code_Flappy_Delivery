"""
Return every quadratic (degree p) B-spline basis function and its first parametric derivative at a single parameter value from an open knot vector, evaluated by the recursion the paper uses to build its global patch (the paper's Eqs. 34-36). Row 0 holds the n basis values N_{i,p}(xi) and row 1 holds dN_{i,p}/dxi, in ascending basis index. Recover the exact recursion and the derivative rule from the paper.

The global field in the s-version isogeometric method is spanned by non-uniform rational-free B-splines; the basis and its derivative are the objects that make the global approximation C^{p-1} continuous across knot spans, which is the property the coupling relies on.

Returns
-------
return (2, n) float64: row 0 the basis values, row 1 the parametric derivatives
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bspline_basis(knots, p, xi):
    """knots: (m+1,) open knot vector; p: degree (>=1); xi: parameter in the knot range.
    Returns a float64 array (2, n) where n = len(knots)-p-1: row 0 the basis values
    N_{i,p}(xi) and row 1 the derivatives dN_{i,p}/dxi (paper Eqs. 34-36). Raises
    ValueError if len(knots) < p+2 or p < 1."""
    n = len(np.asarray(knots)) - p - 1
    return np.zeros((2, max(n, 0)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 1: quadratic B-spline basis and derivative (Cox-de Boor, Eqs. 34-36)."""

import numpy as np


def _oracle_bspline_basis(knots, p, xi):
    knots = np.asarray(knots, dtype=np.float64)
    if p < 1 or len(knots) < p + 2:
        raise ValueError("need p >= 1 and len(knots) >= p+2")
    m = len(knots) - 1
    n = m - p - 1
    N = np.zeros((n + 1, p + 1), dtype=np.float64)
    for i in range(n + 1):
        hi = knots[i + 1]
        if (knots[i] <= xi < hi) or (xi == knots[-1] and knots[i] <= xi <= hi and hi == knots[-1]):
            N[i, 0] = 1.0
    for q in range(1, p + 1):
        for i in range(n + 1):
            a = 0.0
            d1 = knots[i + q] - knots[i]
            if d1 > 0:
                a = (xi - knots[i]) / d1 * N[i, q - 1]
            b = 0.0
            if i + 1 <= n:
                d2 = knots[i + q + 1] - knots[i + 1]
                if d2 > 0:
                    b = (knots[i + q + 1] - xi) / d2 * N[i + 1, q - 1]
            N[i, q] = a + b
    vals = N[:, p].copy()
    der = np.zeros(n + 1, dtype=np.float64)
    for i in range(n + 1):
        t1 = 0.0
        d1 = knots[i + p] - knots[i]
        if d1 > 0:
            t1 = p / d1 * N[i, p - 1]
        t2 = 0.0
        if i + 1 <= n:
            d2 = knots[i + p + 1] - knots[i + 1]
            if d2 > 0:
                t2 = p / d2 * N[i + 1, p - 1]
        der[i] = t1 - t2
    return np.vstack([vals, der])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nknots=_n.array([-1.5,-1.5,-1.5,0.,1.5,1.5,1.5]);p=2;xi=0.3', "call": 'bspline_basis(knots, p, xi)', "gold_call": '_oracle_bspline_basis(knots, p, xi)', "tol": 1e-10},
        {"setup": 'import numpy as _n\nknots=_n.array([-1.5,-1.5,-1.5,0.,1.5,1.5,1.5]);p=2;xi=-0.7', "call": 'bspline_basis(knots, p, xi)', "gold_call": '_oracle_bspline_basis(knots, p, xi)', "tol": 1e-10},
        {"setup": 'import numpy as _n\nknots=_n.array([0.,0.,0.,1.,2.,3.,3.,3.]);p=2;xi=1.4', "call": 'bspline_basis(knots, p, xi)', "gold_call": '_oracle_bspline_basis(knots, p, xi)', "tol": 1e-10},
    ]
