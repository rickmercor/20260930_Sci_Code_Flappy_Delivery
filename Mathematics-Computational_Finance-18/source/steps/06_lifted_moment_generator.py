"""
Build the generator G of the closed linear system for the first and second moments of the lifted state from the drift and jump blocks of the previous step, the volatility of variance xi, the exponential-sum weights omega and v0. Coordinates of the moment vector y: the second moments S_ij = E[z_i z_j] for i <= j in row-major order (n(n+1)/2 entries), then the first moments m_i = E[z_i] (n entries), then the constant 1; return G with dy/dt = G y, shape (n(n+1)/2 + n + 1) square, where the price-jump and variance-jump intensities are the coordinates z_N and z_{N+1}. Raise ValueError if the block shapes are inconsistent with n = len(omega) + 2, if any block is not finite, or if xi or v0 is not finite and nonnegative.

For an affine jump-diffusion the moments of every order close on themselves: the drift is affine, the diffusion covariance is affine in the state, and the jump compensators are the intensities, which are coordinates of the state. The first and second moments therefore obey one linear system whose solution is exact through a matrix exponential.

Returns
-------
ndarray of float64, square of size n(n+1)/2 + n + 1 with n = len(omega) + 2, the moment generator G.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lifted_moment_generator(A: "np.ndarray", c: "np.ndarray", xi: float, omega: "np.ndarray", v0: float, j_s: "np.ndarray", j_v: "np.ndarray", Q_s: "np.ndarray", Q_v: "np.ndarray") -> "np.ndarray":
    """Build the generator G of the closed linear system for the first and second moments of
    the lifted state.

    Parameters
    ----------
    A : np.ndarray
        Drift matrix of the lifted state, shape (n, n).
    c : np.ndarray
        Constant drift vector of the lifted state, shape (n,).
    xi : float
        Volatility of variance, finite and nonnegative.
    omega : np.ndarray
        Exponential-sum weights, shape (N,) with n = N + 2.
    v0 : float
        Initial variance, finite and nonnegative.
    j_s : np.ndarray
        Mean jump vector of a price jump, shape (n,).
    j_v : np.ndarray
        Mean jump vector of a variance jump, shape (n,).
    Q_s : np.ndarray
        Second-moment jump products of a price jump, shape (n, n).
    Q_v : np.ndarray
        Second-moment jump products of a variance jump, shape (n, n).

    Returns
    -------
    result : np.ndarray
        ndarray of float64, square of size n(n+1)/2 + n + 1 with n = len(omega) + 2, the
        moment generator G.

    Raises
    ------
    ValueError
        If the block shapes are inconsistent with n = len(omega) + 2.
        If any block is not finite.
        If xi or v0 is not finite and nonnegative.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def _oracle_lifted_moment_generator(A: "np.ndarray", c: "np.ndarray", xi: float, omega: "np.ndarray", v0: float, j_s: "np.ndarray", j_v: "np.ndarray", Q_s: "np.ndarray", Q_v: "np.ndarray") -> "np.ndarray":
    """Generator of the closed linear system for the first and second moments of z.

    Coordinates of the moment vector y: the second moments S_ij = E[z_i z_j] for i <= j in
    row-major order (n(n+1)/2 entries), then the first moments m_i = E[z_i] (n entries),
    then the constant 1. Returns G with dy/dt = G y.
    For each pair (i, j): dS_ij/dt = E[z_i a_j + z_j a_i] + E[sigma_i sigma_j]
    + sum over jump types k of E[lam_k (z_i dz_j + z_j dz_i + dz_i dz_j)], where a = A z + c,
    sigma_i sigma_j = xi^2 (v0 + omega . U) for factor pairs and 0 otherwise (every factor is
    driven by the same Brownian increment), lam_S = z_N, lam_V = z_{N+1}, and the jump
    expectations use j_k and Q_k. First moments: dm/dt = A m + c + m_N j_s + m_{N+1} j_v.
    """
    A = np.asarray(A, dtype=np.float64); c = np.asarray(c, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    j_s = np.asarray(j_s, dtype=np.float64); j_v = np.asarray(j_v, dtype=np.float64)
    Q_s = np.asarray(Q_s, dtype=np.float64); Q_v = np.asarray(Q_v, dtype=np.float64)
    n = A.shape[0] if A.ndim == 2 else -1
    N = omega.size
    if A.shape != (n, n) or n != N + 2 or c.shape != (n,) or j_s.shape != (n,) or j_v.shape != (n,) or Q_s.shape != (n, n) or Q_v.shape != (n, n):
        raise ValueError("block shapes must be consistent with n = len(omega) + 2")
    if not all(np.all(np.isfinite(v)) for v in (A, c, omega, j_s, j_v, Q_s, Q_v)):
        raise ValueError("blocks must be finite")
    if not np.isfinite(xi) or xi < 0.0 or not np.isfinite(v0) or v0 < 0.0:
        raise ValueError("xi and v0 must be finite and nonnegative")
    idx = {}
    k = 0
    for i in range(n):
        for j in range(i, n):
            idx[(i, j)] = k
            k += 1
    nS = k
    dim = nS + n + 1

    def I(i, j):
        return idx[(i, j)] if i <= j else idx[(j, i)]

    Gm = np.zeros((dim, dim))
    lam_S, lam_V = N, N + 1
    for i in range(n):
        Gm[nS + i, nS:nS + n] += A[i]
        Gm[nS + i, dim - 1] += c[i]
        Gm[nS + i, nS + lam_S] += j_s[i]
        Gm[nS + i, nS + lam_V] += j_v[i]
    for i in range(n):
        for j in range(i, n):
            r = I(i, j)
            for kk in np.nonzero(A[j])[0]:
                Gm[r, I(i, kk)] += A[j, kk]
            for kk in np.nonzero(A[i])[0]:
                Gm[r, I(j, kk)] += A[i, kk]
            Gm[r, nS + i] += c[j]
            Gm[r, nS + j] += c[i]
            if i < N and j < N:
                Gm[r, dim - 1] += xi * xi * v0
                Gm[r, nS:nS + N] += xi * xi * omega
            for lam_idx, jk, Qk in ((lam_S, j_s, Q_s), (lam_V, j_v, Q_v)):
                if jk[j] != 0.0:
                    Gm[r, I(i, lam_idx)] += jk[j]
                if jk[i] != 0.0:
                    Gm[r, I(j, lam_idx)] += jk[i]
                if Qk[i, j] != 0.0:
                    Gm[r, nS + lam_idx] += Qk[i, j]
    return Gm

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "np = __import__('numpy')\nx = np.array([0.5, 2.0, 8.0])\nomega = np.array([0.3, 0.2, 0.1])\nx_ref = x.copy()\nomega_ref = omega.copy()\nB = lifted_moment_blocks(x, omega, 4.924, 0.288, 0.1081, 0.0865, np.array([0.523, 0.687]), np.array([[0.30, 0.15], [0.20, 0.25]]), np.array([3.0, 2.5]))\nB_ref = _oracle_lifted_moment_blocks(x_ref, omega_ref, 4.924, 0.288, 0.1081, 0.0865, np.array([0.523, 0.687]), np.array([[0.30, 0.15], [0.20, 0.25]]), np.array([3.0, 2.5]))",
            "call": "lifted_moment_generator(B[0], B[1], 0.275, omega, 0.1081, B[2], B[3], B[4], B[5]).ravel().tolist()",
            "gold_call": "_oracle_lifted_moment_generator(B_ref[0], B_ref[1], 0.275, omega_ref, 0.1081, B_ref[2], B_ref[3], B_ref[4], B_ref[5]).ravel().tolist()",
        },
        {
            "setup": "np = __import__('numpy')\nx = np.array([1.0])\nomega = np.array([2.0])\nx_ref = x.copy()\nomega_ref = omega.copy()\nB = lifted_moment_blocks(x, omega, 1.2641, 0.1391, 0.0722, 0.193, np.array([0.6832, 3.6342]), np.array([[0.30, 0.15], [0.20, 0.25]]), np.array([3.0, 2.5]))\nB_ref = _oracle_lifted_moment_blocks(x_ref, omega_ref, 1.2641, 0.1391, 0.0722, 0.193, np.array([0.6832, 3.6342]), np.array([[0.30, 0.15], [0.20, 0.25]]), np.array([3.0, 2.5]))",
            "call": "lifted_moment_generator(B[0], B[1], 0.9116, omega, 0.0722, B[2], B[3], B[4], B[5]).ravel().tolist()",
            "gold_call": "_oracle_lifted_moment_generator(B_ref[0], B_ref[1], 0.9116, omega_ref, 0.0722, B_ref[2], B_ref[3], B_ref[4], B_ref[5]).ravel().tolist()",
        },
        {
            "setup": "np = __import__('numpy')\nx = np.array([1e-3, 0.1, 1.0, 10.0])\nomega = np.array([0.01, 0.05, 0.2, 0.5])\nx_ref = x.copy()\nomega_ref = omega.copy()\nB = lifted_moment_blocks(x, omega, 0.0, 0.05, 0.02, 0.5, np.array([1.0, 0.2]), np.array([[0.0, 0.9], [0.4, 0.0]]), np.array([1.5, 0.8]))\nB_ref = _oracle_lifted_moment_blocks(x_ref, omega_ref, 0.0, 0.05, 0.02, 0.5, np.array([1.0, 0.2]), np.array([[0.0, 0.9], [0.4, 0.0]]), np.array([1.5, 0.8]))",
            "call": "lifted_moment_generator(B[0], B[1], 0.0, omega, 0.02, B[2], B[3], B[4], B[5]).ravel().tolist()",
            "gold_call": "_oracle_lifted_moment_generator(B_ref[0], B_ref[1], 0.0, omega_ref, 0.02, B_ref[2], B_ref[3], B_ref[4], B_ref[5]).ravel().tolist()",
        },
        {
            "setup": "np = __import__('numpy')\ndef run_model():\n    try:\n        lifted_moment_generator(np.eye(4), np.zeros(4), 0.2, np.array([0.5, 0.5]), 0.1, np.zeros(4), np.zeros(4), np.zeros((4, 4)), np.zeros((3, 3)))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_lifted_moment_generator(np.eye(4), np.zeros(4), 0.2, np.array([0.5, 0.5]), 0.1, np.zeros(4), np.zeros(4), np.zeros((4, 4)), np.zeros((3, 3)))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
