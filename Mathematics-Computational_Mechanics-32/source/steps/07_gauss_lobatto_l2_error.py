"""
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Return the L2 error between the vectorised reference function fun and the degree-k DG function with coefficient vector coef, evaluated exactly as the source does in Section 5: on every cell a (k+3)-point Gauss-Lobatto rule (end points included) applied to the squared pointwise difference, summed over cells, square root taken. Raise ValueError if N or k is invalid, coef does not have N*(k+1) entries, or fun does not return an array of the input's shape.

Convergence studies of DG methods report errors through a fixed cell quadrature rather than the exact integral; the rule must be reproduced, weights and nodes, for the reported numbers to be comparable.

Returns
-------
float, the Gauss-Lobatto L2 error.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def gauss_lobatto_l2_error(coef, fun, N, k):
    """float, the Gauss-Lobatto L2 error."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _legendre_values(k, xi):
    """P_0..P_k at the points xi (shape (len(xi), k+1)), three-term recurrence."""
    import numpy as np
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    out = np.zeros((xi.size, k + 1))
    out[:, 0] = 1.0
    if k >= 1:
        out[:, 1] = xi
    for m in range(1, k):
        out[:, m + 1] = ((2 * m + 1) * xi * out[:, m] - m * out[:, m - 1]) / (m + 1)
    return out


def _check_space(N, k):
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if int(k) != k or k < 0:
        raise ValueError("k must be a non-negative integer")


def _oracle_gauss_lobatto_l2_error(coef, fun, N, k):
    """e_u of Section 5: (k+3)-point Gauss-Lobatto rule per cell.

    e_u = ( sum_j sum_i (h/2) w_i (fun(x_i^j) - u_h(x_i^j))^2 )^(1/2), with the Lobatto
    nodes +-1 and the roots of P'_{k+2}, weights 2 / ((k+3)(k+2) P_{k+2}(x_i)^2).
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k)
    coef = np.asarray(coef, dtype=np.float64).ravel()
    n = k + 1
    if coef.size != N * n:
        raise ValueError("coef must have N*(k+1) entries")
    npts = k + 3
    Pn = np.polynomial.legendre.Legendre.basis(npts - 1)
    inner = np.real(Pn.deriv().roots())
    xi = np.sort(np.concatenate(([-1.0], inner, [1.0])))
    Pval = _legendre_values(npts - 1, xi)
    w = 2.0 / (npts * (npts - 1) * Pval[:, npts - 1] ** 2)
    P = _legendre_values(k, xi)
    h = 1.0 / N
    err2 = 0.0
    for j in range(N):
        x = (j + 0.5) * h + 0.5 * h * xi
        f = np.asarray(fun(x), dtype=np.float64)
        if f.shape != x.shape:
            raise ValueError("fun must map an array of points to an array of the same shape")
        uh = P @ coef[j * n:(j + 1) * n]
        err2 += 0.5 * h * float(np.sum(w * (f - uh) ** 2))
    return float(np.sqrt(err2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "coef = np.array([0.5, 0.2, -0.1, 0.3, 0.7, -0.4, 0.0, 0.1])\nfun = lambda x: np.sin(2.0 * np.pi * x)\nN, k = 4, 1",
            "call": "gauss_lobatto_l2_error(coef, fun, N, k)",
            "gold_call": "_oracle_gauss_lobatto_l2_error(coef, fun, N, k)",
        },
        {
            "setup": "coef = np.array([1.0, -1.0])\nfun = lambda x: np.cos(2.0 * np.pi * x)\nN, k = 2, 0",
            "call": "gauss_lobatto_l2_error(coef, fun, N, k)",
            "gold_call": "_oracle_gauss_lobatto_l2_error(coef, fun, N, k)",
        },
        {
            "setup": "coef = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 0.05, 0.0, 0.0, 0.0, 0.5, 0.0])\nfun = lambda x: x * (1.0 - x)\nN, k = 4, 2",
            "call": "gauss_lobatto_l2_error(coef, fun, N, k)",
            "gold_call": "_oracle_gauss_lobatto_l2_error(coef, fun, N, k)",
        },
    ]
