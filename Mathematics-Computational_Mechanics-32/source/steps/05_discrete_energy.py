"""
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Given two consecutive coefficient vectors u_new = u^{n+1} and u_old = u^n of the fully discrete scheme with time step ht, and the operator matrix B of the previous step (d^2 u/dt^2 = B u), return the fully discrete energy of Theorem 4.1 of the source: the squared L2 norm of the difference quotient (u^{n+1} - u^n)/ht plus the nonlocal energy of the two states, each expressed through B and the mass matrix. Raise ValueError if ht is not positive, if N or k is invalid, or if the vectors and B do not match N*(k+1).

The Crank-Nicolson time integration of the source conserves a discrete energy exactly, step after step, for any mesh and any time step. Evaluating it is the sharpest test of a time-stepping implementation: a wrong average or a wrong start-up shows up as a drift.

Returns
-------
float, the fully discrete energy of the pair (u^{n+1}, u^n).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def discrete_energy(u_new, u_old, ht, N, k, B):
    """float, the fully discrete energy of the pair (u^{n+1}, u^n)."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _mass_diag(N, k):
    """Diagonal of the mass matrix of the Legendre modal basis on N uniform cells of (0,1)."""
    import numpy as np
    h = 1.0 / N
    return np.tile(h / (2.0 * np.arange(k + 1) + 1.0), N)


def _check_space(N, k):
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if int(k) != k or k < 0:
        raise ValueError("k must be a non-negative integer")


def _oracle_discrete_energy(u_new, u_old, ht, N, k, B):
    """Fully discrete energy of Theorem 4.1 for the pair (u^{n+1}, u^n).

    E = ||(u^{n+1} - u^n)/h_t||^2_{L2} + int_0^d s^2 gamma (||q^{n+1}||^2 + ||q^n||^2) ds,
    with q^n = H(s) u^n, and int s^2 gamma ||H(s) u||^2 ds = -(B_h u, u)_M / 2, so
    E = ||(u^{n+1}-u^n)/h_t||_M^2 - [(B u^{n+1}, u^{n+1})_M + (B u^n, u^n)_M] / 2.
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k); ht = float(ht)
    if ht <= 0.0:
        raise ValueError("ht must be positive")
    u_new = np.asarray(u_new, dtype=np.float64).ravel()
    u_old = np.asarray(u_old, dtype=np.float64).ravel()
    B = np.asarray(B, dtype=np.float64)
    dim = N * (k + 1)
    if u_new.size != dim or u_old.size != dim or B.shape != (dim, dim):
        raise ValueError("u_new, u_old and B must match N*(k+1)")
    Md = _mass_diag(N, k)
    v = (u_new - u_old) / ht
    kin = float(np.sum(Md * v * v))
    pot = -0.5 * (float(u_new @ (Md * (B @ u_new))) + float(u_old @ (Md * (B @ u_old))))
    return kin + pot

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "N, k, ht = 4, 0, 0.1\nB = 16.0 * np.array([[-2.0, 1.0, 0.0, 1.0], [1.0, -2.0, 1.0, 0.0], [0.0, 1.0, -2.0, 1.0], [1.0, 0.0, 1.0, -2.0]])\nu_new = np.array([1.0, 0.5, -0.25, 0.125])\nu_old = np.array([0.9, 0.6, -0.2, 0.1])",
            "call": "discrete_energy(u_new, u_old, ht, N, k, B)",
            "gold_call": "_oracle_discrete_energy(u_new, u_old, ht, N, k, B)",
        },
        {
            "setup": "N, k, ht = 2, 1, 0.05\nB = np.array([[-8.0, 0.0, 8.0, 4.0], [0.0, -24.0, -12.0, -12.0], [8.0, 4.0, -8.0, 0.0], [-12.0, -12.0, 0.0, -24.0]])\nu_new = np.array([0.3, -0.1, 0.2, 0.4])\nu_old = np.array([0.25, -0.05, 0.15, 0.45])",
            "call": "discrete_energy(u_new, u_old, ht, N, k, B)",
            "gold_call": "_oracle_discrete_energy(u_new, u_old, ht, N, k, B)",
        },
        {
            "setup": "N, k, ht = 4, 0, 0.5\nB = np.zeros((4, 4))\nu_new = np.ones(4)\nu_old = np.zeros(4)",
            "call": "discrete_energy(u_new, u_old, ht, N, k, B)",
            "gold_call": "_oracle_discrete_energy(u_new, u_old, ht, N, k, B)",
        },
    ]
