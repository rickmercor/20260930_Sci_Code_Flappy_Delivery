"""
Build the affine drift and the jump blocks of the lifted state z = (U_1, ..., U_N, lam_S, lam_V), types ordered (price jumps S, variance jumps V), for the source's lifted model with variance V = v0 + sum_k omega_k U_k and the positive-part truncation inactive. Return (A, c, j_s, j_v, Q_s, Q_v): the matrix A and vector c of the continuous drift A z + c; for each jump type k the mean jump vector j_k, whose entry i is the expected change of z_i at a type-k jump; and the matrix Q_k, whose entry (i, j) is the expected product of the changes of z_i and z_j at a type-k jump. Raise ValueError if x and omega are not non-empty 1-D arrays of equal length, if any rate is not finite and strictly positive, if kappa, theta, v0 or mu_v is not finite and nonnegative, if lam_inf, eta or beta has the wrong shape or a non-finite entry, if a baseline or excitation is negative, or if a decay rate is not positive.

Under the lift the variance is an affine function of the factors and the intensities are affine in the counts, so the continuous drift is affine in the state, every jump shifts the state by a vector whose mean and second moment are fixed by the source's model. These blocks are what the closed moment system is assembled from.

Returns
-------
tuple (A, c, j_s, j_v, Q_s, Q_v) of float64 arrays with shapes (n, n), (n,), (n,), (n,), (n, n), (n, n), n = len(x) + 2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lifted_moment_blocks(x: "np.ndarray", omega: "np.ndarray", kappa: float, theta: float, v0: float, mu_v: float, lam_inf: "np.ndarray", eta: "np.ndarray", beta: "np.ndarray") -> tuple:
    """Build the affine drift and the jump blocks of the lifted state (U_1, ..., U_N, lam_S,
    lam_V).

    Parameters
    ----------
    x : np.ndarray
        Exponential-sum nodes (factor decay rates), a non-empty 1-D array of strictly
        positive values.
    omega : np.ndarray
        Exponential-sum weights, a 1-D array of the same length as the nodes.
    kappa : float
        Mean-reversion speed of the variance, finite and nonnegative.
    theta : float
        Base level of the variance, finite and nonnegative.
    v0 : float
        Initial variance, finite and nonnegative.
    mu_v : float
        Mean variance-jump size, finite and nonnegative.
    lam_inf : np.ndarray
        Baseline jump intensities, shape (2,), ordered (price jumps, variance jumps).
    eta : np.ndarray
        Immediate excitations, shape (2, 2); column k holds the increments of the two
        intensities caused by a jump of type k, types ordered (price jumps, variance jumps).
    beta : np.ndarray
        Decay rates of the two intensities, shape (2,).

    Returns
    -------
    result : tuple
        tuple (A, c, j_s, j_v, Q_s, Q_v) of float64 arrays with shapes (n, n), (n,), (n,),
        (n,), (n, n), (n, n), n = len(x) + 2.

    Raises
    ------
    ValueError
        If x and omega are not non-empty 1-D arrays of equal length.
        If any rate is not finite and strictly positive.
        If kappa, theta, v0 or mu_v is not finite and nonnegative.
        If lam_inf, eta or beta has the wrong shape or a non-finite entry.
        If a baseline or an excitation is negative, or a decay rate is not positive.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def _oracle_lifted_moment_blocks(x: "np.ndarray", omega: "np.ndarray", kappa: float, theta: float, v0: float, mu_v: float, lam_inf: "np.ndarray", eta: "np.ndarray", beta: "np.ndarray") -> tuple:
    """Affine drift and jump blocks of the lifted state z = (U_1..U_N, lam_S, lam_V).

    Continuous drift a(z) = A z + c: each factor decays at its own rate, is pulled by
    kappa (theta - V) with V = v0 + sum_k omega_k U_k, and each intensity reverts at its
    decay rate to its baseline. Jumps: a price jump raises the intensities by the first
    column of eta; a variance jump adds its mark J to every factor (the exponential sum
    reproduces the kernel) and raises the intensities by the second column of eta.
    With J exponential of mean mu_v, E[J] = mu_v and E[J^2] = 2 mu_v^2.
    Returns (A, c, j_s, j_v, Q_s, Q_v): mean jump vectors j_k = E[jump of z] for type k
    and second-moment products Q_k = E[jump_i jump_j] for type k.
    """
    x = np.asarray(x, dtype=np.float64); omega = np.asarray(omega, dtype=np.float64)
    lam_inf = np.asarray(lam_inf, dtype=np.float64); eta = np.asarray(eta, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if x.ndim != 1 or x.size < 1 or omega.shape != x.shape:
        raise ValueError("x and omega must be non-empty 1-D arrays of equal length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(omega))) or np.any(x <= 0.0):
        raise ValueError("rates must be finite and strictly positive")
    for name, val in (("kappa", kappa), ("theta", theta), ("v0", v0), ("mu_v", mu_v)):
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(name + " must be finite and nonnegative")
    if lam_inf.shape != (2,) or eta.shape != (2, 2) or beta.shape != (2,):
        raise ValueError("lam_inf and beta must have shape (2,), eta shape (2, 2)")
    if not (np.all(np.isfinite(lam_inf)) and np.all(np.isfinite(eta)) and np.all(np.isfinite(beta))):
        raise ValueError("Hawkes inputs must be finite")
    if np.any(lam_inf < 0.0) or np.any(eta < 0.0) or np.any(beta <= 0.0):
        raise ValueError("baselines and excitations must be nonnegative, decay rates positive")
    N = x.size; n = N + 2
    A = np.zeros((n, n)); c = np.zeros(n)
    A[:N, :N] = -np.diag(x) - kappa * np.tile(omega, (N, 1))
    c[:N] = kappa * (theta - v0)
    A[N, N] = -beta[0]; A[N + 1, N + 1] = -beta[1]
    c[N] = beta[0] * lam_inf[0]; c[N + 1] = beta[1] * lam_inf[1]
    j_s = np.zeros(n); j_s[N] = eta[0, 0]; j_s[N + 1] = eta[1, 0]
    j_v = np.zeros(n); j_v[:N] = mu_v; j_v[N] = eta[0, 1]; j_v[N + 1] = eta[1, 1]
    Q_s = np.outer(j_s, j_s)
    Q_v = np.outer(j_v, j_v); Q_v[:N, :N] = 2.0 * mu_v * mu_v
    return A, c, j_s, j_v, Q_s, Q_v

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "np = __import__('numpy')\nx = np.array([0.5, 2.0, 8.0])\nomega = np.array([0.3, 0.2, 0.1])\nlam_inf = np.array([0.523, 0.687])\neta = np.array([[0.30, 0.15], [0.20, 0.25]])\nbeta = np.array([3.0, 2.5])\nx_ref = x.copy()\nomega_ref = omega.copy()\nlam_inf_ref = lam_inf.copy()\neta_ref = eta.copy()\nbeta_ref = beta.copy()",
            "call": "np.concatenate([np.asarray(v, dtype=float).ravel() for v in lifted_moment_blocks(x, omega, 4.924, 0.288, 0.1081, 0.0865, lam_inf, eta, beta)]).tolist()",
            "gold_call": "np.concatenate([np.asarray(v, dtype=float).ravel() for v in _oracle_lifted_moment_blocks(x_ref, omega_ref, 4.924, 0.288, 0.1081, 0.0865, lam_inf_ref, eta_ref, beta_ref)]).tolist()",
        },
        {
            "setup": "np = __import__('numpy')\nx = np.array([1.0])\nomega = np.array([2.0])\nlam_inf = np.array([0.6832, 3.6342])\neta = np.array([[0.30, 0.15], [0.20, 0.25]])\nbeta = np.array([3.0, 2.5])\nx_ref = x.copy()\nomega_ref = omega.copy()\nlam_inf_ref = lam_inf.copy()\neta_ref = eta.copy()\nbeta_ref = beta.copy()",
            "call": "np.concatenate([np.asarray(v, dtype=float).ravel() for v in lifted_moment_blocks(x, omega, 1.2641, 0.1391, 0.0722, 0.193, lam_inf, eta, beta)]).tolist()",
            "gold_call": "np.concatenate([np.asarray(v, dtype=float).ravel() for v in _oracle_lifted_moment_blocks(x_ref, omega_ref, 1.2641, 0.1391, 0.0722, 0.193, lam_inf_ref, eta_ref, beta_ref)]).tolist()",
        },
        {
            "setup": "np = __import__('numpy')\nx = np.array([1e-3, 0.1, 1.0, 10.0])\nomega = np.array([0.01, 0.05, 0.2, 0.5])\nlam_inf = np.array([1.0, 0.2])\neta = np.array([[0.0, 0.9], [0.4, 0.0]])\nbeta = np.array([1.5, 0.8])\nx_ref = x.copy()\nomega_ref = omega.copy()\nlam_inf_ref = lam_inf.copy()\neta_ref = eta.copy()\nbeta_ref = beta.copy()",
            "call": "np.concatenate([np.asarray(v, dtype=float).ravel() for v in lifted_moment_blocks(x, omega, 0.0, 0.05, 0.02, 0.5, lam_inf, eta, beta)]).tolist()",
            "gold_call": "np.concatenate([np.asarray(v, dtype=float).ravel() for v in _oracle_lifted_moment_blocks(x_ref, omega_ref, 0.0, 0.05, 0.02, 0.5, lam_inf_ref, eta_ref, beta_ref)]).tolist()",
        },
        {
            "setup": "np = __import__('numpy')\ndef run_model():\n    try:\n        lifted_moment_blocks(np.array([0.5, -2.0]), np.array([0.3, 0.2]), 1.0, 0.1, 0.05, 0.1, np.array([0.5, 0.5]), np.eye(2) * 0.1, np.array([1.0, 1.0]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_lifted_moment_blocks(np.array([0.5, -2.0]), np.array([0.3, 0.2]), 1.0, 0.1, 0.05, 0.1, np.array([0.5, 0.5]), np.eye(2) * 0.1, np.array([1.0, 1.0]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
