"""
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Return the dense matrix that maps the coefficient vector of a function u_h in the degree-k space to the coefficient vector of the L2 projection onto the same space of the shifted function x -> u_h(x + s), s > 0, with u_h continued periodically, i.e. the building block of the operators of Eq (2.5) of the source. The shifted function is itself piecewise polynomial with its own breakpoints, and the projection integrals must be exact, not approximated by a fixed quadrature on the cell. Raise ValueError if N is not a positive integer, k is not a non-negative integer, or s is not positive.

The nonlocal DG scheme couples a cell to the cells its horizon reaches through difference quotients of shifted copies of the solution. On a uniform mesh the shift by s decomposes into whole-cell steps plus a remainder, and every cell sees exactly two neighbouring cells of the shifted function.

Returns
-------
ndarray of float64 with shape (N*(k+1), N*(k+1)).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def shift_projection_matrix(N, k, s):
    """ndarray of float64 with shape (N*(k+1), N*(k+1))."""
    return np.zeros((N * (k + 1), N * (k + 1)), dtype=np.float64)

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


def _oracle_shift_projection_matrix(N, k, s):
    """Matrix of S(s): coefficients of P_h[u_h(. + s)] on the periodic mesh, s > 0.

    This is the exact building block of H_j in (2.5): for x in I_j, x + s lies in cell
    j+m on [x_{j-1/2}, x_{j+1/2} - r] and in cell j+m+1 on the rest, where s = m h + r,
    0 <= r < h.  Each piece is a polynomial, so the two integrals are evaluated EXACTLY with
    a (k+2)-point Gauss-Legendre rule on the sub-interval (never a single rule across the
    crossing point).  With rho = 2r/h, block A multiplies the coefficients of cell j+m and
    block B those of cell j+m+1 (indices mod N):
        A = Mhat^-1 int_{-1}^{1-rho} phi(xi) phi(xi+rho)^T dxi,
        B = Mhat^-1 int_{1-rho}^{1}  phi(xi) phi(xi+rho-2)^T dxi.
    S(h) is the pure cell permutation; S(s) preserves constants.
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k); s = float(s)
    if s <= 0.0:
        raise ValueError("s must be positive")
    h = 1.0 / N
    m = int(np.floor(s / h + 1e-12))
    r = s - m * h
    if r < 0.0:
        r = 0.0
    rho = 2.0 * r / h
    xg, wg = np.polynomial.legendre.leggauss(k + 2)
    Minv = np.diag((2.0 * np.arange(k + 1) + 1.0) / 2.0)

    def block(lo, hi, shift):
        if hi - lo <= 0.0:
            return np.zeros((k + 1, k + 1))
        xi = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        PA = _legendre_values(k, xi)
        PB = _legendre_values(k, xi + shift)
        return Minv @ (PA.T @ (w[:, None] * PB))

    A = block(-1.0, 1.0 - rho, rho)
    B = block(1.0 - rho, 1.0, rho - 2.0)
    n = k + 1
    S = np.zeros((N * n, N * n))
    for j in range(N):
        ja = (j + m) % N
        jb = (j + m + 1) % N
        S[j * n:(j + 1) * n, ja * n:(ja + 1) * n] += A
        S[j * n:(j + 1) * n, jb * n:(jb + 1) * n] += B
    return S

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "N, k, s = 8, 1, 0.3",
            "call": "shift_projection_matrix(N, k, s)",
            "gold_call": "_oracle_shift_projection_matrix(N, k, s)",
        },
        {
            "setup": "N, k, s = 8, 1, 0.125",
            "call": "shift_projection_matrix(N, k, s)",
            "gold_call": "_oracle_shift_projection_matrix(N, k, s)",
        },
        {
            "setup": "N, k, s = 4, 2, 0.0625",
            "call": "shift_projection_matrix(N, k, s)",
            "gold_call": "_oracle_shift_projection_matrix(N, k, s)",
        },
        {
            "setup": "N, k, s = 8, 0, 1.3",
            "call": "shift_projection_matrix(N, k, s)",
            "gold_call": "_oracle_shift_projection_matrix(N, k, s)",
        },
    ]
