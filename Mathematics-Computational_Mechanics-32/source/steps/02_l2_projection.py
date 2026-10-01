"""
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Return the coefficient vector of the L2 projection of the vectorised function fun onto the piecewise-polynomial space of degree k, Eq (3.1) of the source, i.e. on every cell the polynomial whose inner products with P_0..P_k match those of fun; integrate accurately enough that smooth functions are projected to round-off. Raise ValueError if N is not a positive integer, k is not a non-negative integer, or fun does not return an array of the same shape as its input.

Discontinuous Galerkin methods represent the solution cell by cell in a polynomial basis and start from the L2 projection of the initial data, which is the best approximation in the energy norm of the wave equation.

Returns
-------
ndarray of float64 with shape (N*(k+1),): the cell-major modal coefficients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def l2_projection(fun, N, k):
    """ndarray of float64 with shape (N*(k+1),): the cell-major modal coefficients."""
    return np.zeros(N * (k + 1), dtype=np.float64)

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


def _oracle_l2_projection(fun, N, k):
    """Modal Legendre coefficients of the L2 projection P_h fun onto V_h^k, Eq (3.1).

    Cells I_j = ((j-1)h, jh), h = 1/N, local coordinate xi = 2(x - x_j)/h in [-1,1], basis
    P_0..P_k (Legendre, NOT normalised).  Coefficient m of cell j is
    (2m+1)/2 * int_{-1}^{1} fun P_m dxi, evaluated with a (k+8)-point Gauss-Legendre rule.
    Returns a flat array ordered cell-major: entry j*(k+1)+m.
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k)
    h = 1.0 / N
    xg, wg = np.polynomial.legendre.leggauss(k + 8)
    P = _legendre_values(k, xg)
    scale = (2.0 * np.arange(k + 1) + 1.0) / 2.0
    coef = np.zeros(N * (k + 1))
    for j in range(N):
        x = (j + 0.5) * h + 0.5 * h * xg
        f = np.asarray(fun(x), dtype=np.float64)
        if f.shape != x.shape:
            raise ValueError("fun must map an array of points to an array of the same shape")
        coef[j * (k + 1):(j + 1) * (k + 1)] = scale * (P.T @ (wg * f))
    return coef

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "fun = lambda x: np.sin(2.0 * np.pi * x)\nN, k = 8, 1",
            "call": "l2_projection(fun, N, k)",
            "gold_call": "_oracle_l2_projection(fun, N, k)",
        },
        {
            "setup": "fun = lambda x: np.cos(4.0 * np.pi * x)\nN, k = 4, 2",
            "call": "l2_projection(fun, N, k)",
            "gold_call": "_oracle_l2_projection(fun, N, k)",
        },
        {
            "setup": "fun = lambda x: np.exp(np.sin(2.0 * np.pi * x))\nN, k = 16, 0",
            "call": "l2_projection(fun, N, k)",
            "gold_call": "_oracle_l2_projection(fun, N, k)",
        },
    ]
