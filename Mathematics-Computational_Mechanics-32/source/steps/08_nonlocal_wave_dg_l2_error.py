"""
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Orchestrator. Solve the nonlocal wave equation of Eq (2.1) of the source on the periodic unit interval with the kernel family of Eq (5.1) (exponent alpha, horizon delta), initial data u(x,0) = sin(2 pi x) + cos(4 pi x)/2 and zero initial velocity, with the DG scheme of Eq (2.6) of degree k on N cells and the Crank-Nicolson scheme of Eq (4.1) with step ht up to time T. Project the initial data (step 2), assemble the operator (step 4, which uses step 3), march in time (step 6), verify with step 5 that the discrete energy after the first step and after the last step agree to a relative 1e-9 (raise ValueError otherwise), build the exact solution of the nonlocal problem from the eigenvalues of step 1 for the two modes present, and return the Gauss-Lobatto L2 error of step 7 at time T. Call the earlier step functions rather than reimplementing them. Raise ValueError if T is not a positive integer multiple of ht or any argument is invalid.

Because the periodic nonlocal operator is diagonal in Fourier space, a two-mode initial condition has a closed-form exact solution once the two eigenvalues are known, which turns the DG-CN solver into a self-contained convergence and dispersion test without any manufactured source term.

Returns
-------
float, the Gauss-Lobatto L2 error e_u(T) of the DG-CN solution.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def nonlocal_wave_dg_l2_error(N, k, alpha, delta, ht, T):
    """float, the Gauss-Lobatto L2 error e_u(T) of the DG-CN solution."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _gauss_jacobi(n, a, b):
    """Golub-Welsch nodes/weights for the weight (1-x)^a (1+x)^b on [-1, 1], numpy only."""
    import math
    import numpy as np
    n = int(n)
    if n < 1 or a <= -1.0 or b <= -1.0:
        raise ValueError("need n >= 1 and a, b > -1")
    k = np.arange(n, dtype=np.float64)
    ab = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        diag = (b * b - a * a) / ((2.0 * k + ab) * (2.0 * k + ab + 2.0))
    diag[0] = (b - a) / (ab + 2.0)
    kk = np.arange(1, n, dtype=np.float64)
    num = 4.0 * kk * (kk + a) * (kk + b) * (kk + ab)
    den = (2.0 * kk + ab) ** 2 * (2.0 * kk + ab + 1.0) * (2.0 * kk + ab - 1.0)
    off2 = num / den
    if n > 1 and abs(ab) < 1e-300:            # k = 1 with a + b = 0 is the 0/0 case
        off2[0] = 4.0 * (1.0 + a) * (1.0 + b) / 12.0
    J = np.diag(diag)
    if n > 1:
        off = np.sqrt(off2)
        J += np.diag(off, 1) + np.diag(off, -1)
    x, V = np.linalg.eigh(J)
    mu0 = 2.0 ** (ab + 1.0) * math.gamma(a + 1.0) * math.gamma(b + 1.0) / math.gamma(ab + 2.0)
    w = mu0 * V[0, :] ** 2
    return x, w


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


def _oracle_nonlocal_wave_dg_l2_error(N, k, alpha, delta, ht, T):
    """Orchestrator: e_u(T) of the DG-CN solution for u0 = sin(2 pi x) + cos(4 pi x)/2.

    Exact solution: L_d acts on e^{i xi x} as lambda_d(xi) (step 1), so with u_t(x,0) = 0
    u(x,t) = cos(w1 t) sin(2 pi x) + cos(w2 t) cos(4 pi x)/2, w_m = sqrt(lambda_d(2 pi m)).
    Chain: L2 projection of u0 (step 2) -> B_h (step 4, fed the shift matrices of step 3) -> CN march (step 6)
    -> energy check with step 5 (conservation to 1e-9 relative, else ValueError)
    -> e_u with the Gauss-Lobatto rule (step 7).
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k); alpha = float(alpha); delta = float(delta)
    ht = float(ht); T = float(T)
    if ht <= 0.0 or T <= 0.0:
        raise ValueError("ht and T must be positive")
    nsteps = int(round(T / ht))
    if nsteps < 1 or abs(nsteps * ht - T) > 1e-9 * max(1.0, T):
        raise ValueError("T must be a positive integer multiple of ht")
    w1 = np.sqrt(_oracle_nonlocal_symbol(2.0 * np.pi, alpha, delta))
    w2 = np.sqrt(_oracle_nonlocal_symbol(4.0 * np.pi, alpha, delta))

    def u0(x):
        return np.sin(2.0 * np.pi * x) + 0.5 * np.cos(4.0 * np.pi * x)

    def uT(x):
        return (np.cos(w1 * T) * np.sin(2.0 * np.pi * x)
                + 0.5 * np.cos(w2 * T) * np.cos(4.0 * np.pi * x))

    c0 = _oracle_l2_projection(u0, N, k)
    B = _oracle_nonlocal_dg_operator(N, k, alpha, delta, _oracle_shift_projection_matrix)
    pair = _oracle_crank_nicolson_march(c0, ht, nsteps, N, k, B)
    first = _oracle_crank_nicolson_march(c0, ht, 1, N, k, B)
    E1 = _oracle_discrete_energy(first[0], first[1], ht, N, k, B)
    EN = _oracle_discrete_energy(pair[0], pair[1], ht, N, k, B)
    if abs(EN - E1) > 1e-9 * max(1.0, abs(E1)):
        raise ValueError("discrete energy is not conserved")
    return _oracle_gauss_lobatto_l2_error(pair[0], uT, N, k)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "N, k, alpha, delta, ht, T = 8, 1, 1.5, 0.25, 0.01, 2.0",
            "call": "nonlocal_wave_dg_l2_error(N, k, alpha, delta, ht, T)",
            "gold_call": "_oracle_nonlocal_wave_dg_l2_error(N, k, alpha, delta, ht, T)",
        },
        {
            "setup": "N, k, alpha, delta, ht, T = 4, 2, 0.5, 0.375, 0.02, 1.0",
            "call": "nonlocal_wave_dg_l2_error(N, k, alpha, delta, ht, T)",
            "gold_call": "_oracle_nonlocal_wave_dg_l2_error(N, k, alpha, delta, ht, T)",
        },
        {
            "setup": "N, k, alpha, delta, ht, T = 8, 0, 2.5, 0.3, 0.05, 0.5",
            "call": "nonlocal_wave_dg_l2_error(N, k, alpha, delta, ht, T)",
            "gold_call": "_oracle_nonlocal_wave_dg_l2_error(N, k, alpha, delta, ht, T)",
        },
    ]
