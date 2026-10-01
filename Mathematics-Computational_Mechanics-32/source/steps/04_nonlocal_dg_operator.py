"""
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Return the dense matrix B of the semi-discrete DG scheme of the source, Eq (2.6), for the reformulated system (2.4) with the operators of Eq (2.5) and the kernel family of Eq (5.1) with exponent alpha and horizon delta, written as d^2 u/dt^2 = B u for the coefficient vector u of the degree-k solution (the mass matrix already inverted). The auxiliary variable of (2.4) must be represented in the same degree-k space for every interaction distance s; the shifted-projection matrices it needs are supplied by the callable shift_matrix(N, k, s) (the previous step), which must be used for every distance. The integral over s in (0, delta] must be converged to at least eight significant digits, accounting for the singular kernel and the breakpoints of the integrand at multiples of h. Raise ValueError if N is not a positive integer, k is not a non-negative integer, delta is not positive, alpha is not in (0, 3), or shift_matrix is not callable.

Nonlocal operators have no derivatives to integrate by parts, so the DG method of the source reformulates the equation with an auxiliary field indexed by the interaction distance, discretises that field in the same broken polynomial space, and recovers a local DG method with a specific flux choice as the horizon shrinks. The resulting operator is symmetric and negative semidefinite in the mass inner product, which is what makes the scheme energy conserving.

Returns
-------
ndarray of float64 with shape (N*(k+1), N*(k+1)): the matrix B with d^2 u/dt^2 = B u.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def nonlocal_dg_operator(N, k, alpha, delta, shift_matrix):
    """ndarray of float64 with shape (N*(k+1), N*(k+1)): the matrix B with d^2 u/dt^2 = B u."""
    return np.zeros((N * (k + 1), N * (k + 1)), dtype=np.float64)

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


def _oracle_nonlocal_dg_operator(N, k, alpha, delta, shift_matrix):
    """Dense matrix B_h of the semi-discrete scheme (2.6): d^2 u_h / dt^2 = B_h u_h.

    shift_matrix(N, k, s) is the previous step (the orchestrator passes its oracle).

    From (2.4)-(2.6) with q_h(.;s) = H(s) u_h, H(s) = (S(s) - I)/s (L2 projection of the
    forward quotient), and K_j acting on q_h through the backward quotient, Lemma 2.3 gives
    K(q_h, v; s) = -H(v, q_h; s) = -(H(s) v, q_h)_M, hence
        B_h = -2 int_0^delta s^2 gamma_d(s) M^-1 H(s)^T M H(s) ds,
    M-symmetric and negative semidefinite with (B_h u, u)_M = -2 int s^2 gamma ||H(s)u||_M^2.
    The s-integrand is piecewise smooth with breakpoints at multiples of h and behaves like
    s^(2-alpha) at 0, so it is integrated with a 24-point Gauss-Jacobi rule (weight s^(2-a))
    on [0, min(h,delta)] and 24-point Gauss-Legendre rules on every following interval of
    length h (last one truncated at delta).  Result is converged to round-off.
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k); alpha = float(alpha); delta = float(delta)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if not (0.0 < alpha < 3.0):
        raise ValueError("alpha must lie in (0, 3)")
    if not callable(shift_matrix):
        raise ValueError("shift_matrix must be callable")
    h = 1.0 / N
    n = k + 1
    dim = N * n
    c = (3.0 - alpha) / (2.0 * delta ** (3.0 - alpha))
    beta = 2.0 - alpha
    # quadrature in s: nodes and weights that already include s^2 gamma_d(s) ds
    b0 = min(h, delta)
    xj, wj = _gauss_jacobi(24, 0.0, beta)
    nodes = [0.5 * b0 * (xj + 1.0)]
    weights = [c * wj * (0.5 * b0) ** (1.0 + beta)]
    edges = [b0]
    while edges[-1] < delta - 1e-14:
        edges.append(min(edges[-1] + h, delta))
    xg, wg = np.polynomial.legendre.leggauss(24)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sq = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        nodes.append(sq)
        weights.append(0.5 * (hi - lo) * wg * c * sq ** beta)
    sq = np.concatenate(nodes)
    wq = np.concatenate(weights)
    Md = _mass_diag(N, k)
    I = np.eye(dim)
    B = np.zeros((dim, dim))
    for s, w in zip(sq, wq):
        H = (np.asarray(shift_matrix(N, k, s), dtype=np.float64) - I) / s
        B -= 2.0 * w * (H.T @ (Md[:, None] * H))
    return B / Md[:, None]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "def shift_matrix(N, k, s):\n    h = 1.0 / N; m = int(np.floor(s / h + 1e-12)); r = s - m * h; S = np.zeros((N, N))\n    for j in range(N):\n        S[j, (j + m) % N] += 1.0 - r / h; S[j, (j + m + 1) % N] += r / h\n    return S\nN, k, alpha, delta = 8, 0, 1.5, 0.25",
            "call": "nonlocal_dg_operator(N, k, alpha, delta, shift_matrix)",
            "gold_call": "_oracle_nonlocal_dg_operator(N, k, alpha, delta, shift_matrix)",
        },
        {
            "setup": "def shift_matrix(N, k, s):\n    h = 1.0 / N; m = int(np.floor(s / h + 1e-12)); r = s - m * h; S = np.zeros((N, N))\n    for j in range(N):\n        S[j, (j + m) % N] += 1.0 - r / h; S[j, (j + m + 1) % N] += r / h\n    return S\nN, k, alpha, delta = 4, 0, 0.5, 0.375",
            "call": "nonlocal_dg_operator(N, k, alpha, delta, shift_matrix)",
            "gold_call": "_oracle_nonlocal_dg_operator(N, k, alpha, delta, shift_matrix)",
        },
        {
            "setup": "def shift_matrix(N, k, s):\n    h = 1.0 / N; m = int(np.floor(s / h + 1e-12)); r = s - m * h; S = np.zeros((N, N))\n    for j in range(N):\n        S[j, (j + m) % N] += 1.0 - r / h; S[j, (j + m + 1) % N] += r / h\n    return S\nN, k, alpha, delta = 8, 0, 2.5, 0.3",
            "call": "nonlocal_dg_operator(N, k, alpha, delta, shift_matrix)",
            "gold_call": "_oracle_nonlocal_dg_operator(N, k, alpha, delta, shift_matrix)",
        },
        {
            "setup": "def shift_matrix(N, k, s):\n    h = 1.0 / N; m = int(np.floor(s / h + 1e-12)); r = s - m * h; S = np.zeros((N, N))\n    for j in range(N):\n        S[j, (j + m) % N] += 1.0 - r / h; S[j, (j + m + 1) % N] += r / h\n    return S\nN, k, alpha, delta = 16, 0, 0.25, 0.0625",
            "call": "nonlocal_dg_operator(N, k, alpha, delta, shift_matrix)",
            "gold_call": "_oracle_nonlocal_dg_operator(N, k, alpha, delta, shift_matrix)",
        },
    ]
