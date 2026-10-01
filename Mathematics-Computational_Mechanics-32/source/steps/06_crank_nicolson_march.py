"""
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Advance the semi-discrete system d^2 u/dt^2 = B u from the coefficient vector u0 at t = 0 with zero initial velocity over nsteps uniform steps of size ht using the two-level Crank-Nicolson scheme of Eq (4.1) of the source (second-order central difference in time, the operator applied to the source's time average of the auxiliary states). Start the two-step recursion as the task statement prescribes for zero initial velocity: the fictitious value at t = -ht equals the value at t = +ht, and the scheme at n = 0 then determines u^1. Return a (2, N*(k+1)) array whose first row is u at t = nsteps*ht and whose second row is u at t = (nsteps-1)*ht. Raise ValueError if ht is not positive, nsteps is not a positive integer, N or k is invalid, or the sizes do not match.

The scheme is implicit and unconditionally stable, so the time step is limited only by accuracy; each step is one linear solve with a fixed matrix.

Returns
-------
ndarray of float64 with shape (2, N*(k+1)): rows u^{nsteps} and u^{nsteps-1}.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def crank_nicolson_march(u0, ht, nsteps, N, k, B):
    """ndarray of float64 with shape (2, N*(k+1)): rows u^{nsteps} and u^{nsteps-1}."""
    return np.zeros((2, N * (k + 1)), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_space(N, k):
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if int(k) != k or k < 0:
        raise ValueError("k must be a non-negative integer")


def _oracle_crank_nicolson_march(u0, ht, nsteps, N, k, B):
    """Crank-Nicolson scheme (4.1) for d^2u/dt^2 = B u with u_t(0) = 0, nsteps steps.

    (u^{n+1} - 2u^n + u^{n-1})/h_t^2 = B (u^{n+1} + u^{n-1})/2   (mean-value operator on q_h)
    <=> (I - h_t^2 B/2) u^{n+1} = 2 u^n - (I - h_t^2 B/2) u^{n-1}.
    Start-up (declared in the task, the paper is silent): the ghost value u^{-1} := u^{1}
    encodes u_t(x,0) = 0, so the n = 0 equation gives (I - h_t^2 B/2) u^1 = u^0.
    Returns a (2, dim) array: row 0 is u^{nsteps}, row 1 is u^{nsteps-1}.
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k); ht = float(ht)
    if ht <= 0.0:
        raise ValueError("ht must be positive")
    if int(nsteps) != nsteps or nsteps < 1:
        raise ValueError("nsteps must be a positive integer")
    nsteps = int(nsteps)
    u0 = np.asarray(u0, dtype=np.float64).ravel()
    B = np.asarray(B, dtype=np.float64)
    dim = N * (k + 1)
    if u0.size != dim or B.shape != (dim, dim):
        raise ValueError("u0 and B must match N*(k+1)")
    C = np.eye(dim) - 0.5 * ht * ht * B
    Cinv = np.linalg.inv(C)
    u_prev = u0.copy()
    u_cur = Cinv @ u0
    for _ in range(1, nsteps):
        u_next = Cinv @ (2.0 * u_cur - C @ u_prev)
        u_prev, u_cur = u_cur, u_next
    return np.vstack([u_cur, u_prev])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "N, k, ht, nsteps = 4, 0, 0.1, 5\nB = 16.0 * np.array([[-2.0, 1.0, 0.0, 1.0], [1.0, -2.0, 1.0, 0.0], [0.0, 1.0, -2.0, 1.0], [1.0, 0.0, 1.0, -2.0]])\nu0 = np.array([1.0, 0.0, -1.0, 0.0])",
            "call": "crank_nicolson_march(u0, ht, nsteps, N, k, B)",
            "gold_call": "_oracle_crank_nicolson_march(u0, ht, nsteps, N, k, B)",
        },
        {
            "setup": "N, k, ht, nsteps = 2, 1, 0.05, 40\nB = np.array([[-8.0, 0.0, 8.0, 4.0], [0.0, -24.0, -12.0, -12.0], [8.0, 4.0, -8.0, 0.0], [-12.0, -12.0, 0.0, -24.0]])\nu0 = np.array([0.5, 0.25, -0.5, 0.25])",
            "call": "crank_nicolson_march(u0, ht, nsteps, N, k, B)",
            "gold_call": "_oracle_crank_nicolson_march(u0, ht, nsteps, N, k, B)",
        },
        {
            "setup": "N, k, ht, nsteps = 4, 0, 0.25, 1\nB = np.diag([-1.0, -4.0, -9.0, -16.0])\nu0 = np.array([1.0, 1.0, 1.0, 1.0])",
            "call": "crank_nicolson_march(u0, ht, nsteps, N, k, B)",
            "gold_call": "_oracle_crank_nicolson_march(u0, ht, nsteps, N, k, B)",
        },
    ]
