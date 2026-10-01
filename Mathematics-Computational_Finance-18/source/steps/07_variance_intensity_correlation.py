"""
Orchestrator. Build the nodes and density-relative weights and form the exponential-sum weights (steps 3 and 2). Compute the lift-fidelity error over dt on the n_grid grid (step 4) and raise ValueError if it is not finite or exceeds 100 basis points in magnitude; evaluate the kernel at zero lag (step 1) and raise ValueError if the weights do not sum to it within one percent. Build the drift and jump blocks (step 5) and the moment generator (step 6). Start every factor at zero and both intensities at their baselines, so the initial second moments are the outer product of the initial means and the constant coordinate is one. Propagate the moment vector to the horizon with the matrix exponential of G times the horizon, recover the covariance matrix of the state as S - m m^T, and return the correlation between the variance V = v0 + omega . U and the variance-jump intensity: their covariance divided by the product of their standard deviations. Call the earlier step functions rather than reimplementing any of them. Raise ValueError if the horizon is not a positive finite number or if either variance at the horizon is not positive.

The correlation between the variance and the variance-jump intensity is the signature of the clustered-jump channel: a jump raises both at once and the intensity feeds expected further jumps back into the variance drift. Under the lift the number is exact, so it can be compared with a simulation of the same model.

Returns
-------
float, the correlation between the variance and the variance-jump intensity at the horizon.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def variance_intensity_correlation(alpha: float, delta_star: float, m: int, n_quad: int, dt: float, n_grid: int, kappa: float, theta: float, v0: float, xi: float, mu_v: float, lam_inf: "np.ndarray", eta: "np.ndarray", beta: "np.ndarray", horizon: float) -> float:
    """Orchestrator: the correlation between the variance and the variance-jump intensity at
    the horizon under the lifted model.

    Parameters
    ----------
    alpha : float
        Roughness exponent of the fractional kernel, 0 < alpha < 1/2.
    delta_star : float
        Resolution scale of the regularization, positive.
    m : int
        Nonnegative integer that sets the block breakpoint.
    n_quad : int
        Positive number of quadrature points per block.
    dt : float
        Length of the time step, positive.
    n_grid : int
        Odd number, at least three, of uniform grid points spanning zero to dt inclusive for
        the composite Simpson rule.
    kappa : float
        Mean-reversion speed of the variance, finite and nonnegative.
    theta : float
        Base level of the variance, finite and nonnegative.
    v0 : float
        Initial variance, finite and nonnegative.
    xi : float
        Volatility of variance, finite and nonnegative.
    mu_v : float
        Mean variance-jump size, finite and nonnegative.
    lam_inf : np.ndarray
        Baseline jump intensities, shape (2,), ordered (price jumps, variance jumps).
    eta : np.ndarray
        Immediate excitations, shape (2, 2); column k holds the increments of the two
        intensities caused by a jump of type k, types ordered (price jumps, variance jumps).
    beta : np.ndarray
        Decay rates of the two intensities, shape (2,).
    horizon : float
        Horizon T at which the correlation is evaluated, positive and finite.

    Returns
    -------
    result : float
        float, the correlation between the variance and the variance-jump intensity at the
        horizon.

    Raises
    ------
    ValueError
        If the horizon is not a positive finite number.
        If the lift-fidelity error is not finite or exceeds 100 basis points in magnitude.
        If the exponential-sum weights do not sum to the kernel at zero lag within one
        percent.
        If either variance at the horizon is not positive.
        If an input is invalid for an earlier step.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def _oracle_variance_intensity_correlation(alpha: float, delta_star: float, m: int, n_quad: int, dt: float, n_grid: int, kappa: float, theta: float, v0: float, xi: float, mu_v: float, lam_inf: "np.ndarray", eta: "np.ndarray", beta: "np.ndarray", horizon: float) -> float:
    """ORCHESTRATOR. Correlation between the variance and the variance-jump intensity at
    the horizon under the lifted Volterra-Hawkes dynamics, from the exact moments.

    Calls every earlier step: 3 and 2 for the exponential sum, 4 for the lift-fidelity
    admissibility check, 5 for the drift and jump blocks, 6 for the moment generator, and 1
    for the kernel's zero-lag value, which the exponential-sum weights must approximate to
    within one percent (a second admissibility check). Initial state: all factors at zero and
    both intensities at their baselines, so the initial second moments are the outer product
    of the initial means.
    """
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be a positive finite time")
    x, q = _oracle_block_quadrature_nodes(alpha, m, n_quad)
    omega = q * _oracle_bernstein_density(alpha, delta_star, x)
    fidelity = _oracle_lift_fidelity_error_bp(alpha, delta_star, m, n_quad, dt, n_grid)
    if not np.isfinite(fidelity) or abs(fidelity) > 100.0:
        raise ValueError("the exponential-sum lift is not admissible at this configuration")
    k0 = float(_oracle_regularized_kernel(alpha, delta_star, np.array([0.0]))[0])
    if abs(float(omega.sum()) / k0 - 1.0) > 0.01:
        raise ValueError("the exponential-sum weights do not reproduce the zero-lag kernel")
    A, c, j_s, j_v, Q_s, Q_v = _oracle_lifted_moment_blocks(x, omega, kappa, theta, v0, mu_v, lam_inf, eta, beta)
    Gm = _oracle_lifted_moment_generator(A, c, xi, omega, v0, j_s, j_v, Q_s, Q_v)
    N = x.size
    n = N + 2
    nS = n * (n + 1) // 2
    m0 = np.concatenate([np.zeros(N), np.asarray(lam_inf, dtype=np.float64)])
    y0 = np.zeros(nS + n + 1)
    k = 0
    for i in range(n):
        for j in range(i, n):
            y0[k] = m0[i] * m0[j]
            k += 1
    y0[nS:nS + n] = m0
    y0[-1] = 1.0
    yT = expm(Gm * float(horizon)) @ y0
    mT = yT[nS:nS + n]
    ST = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i, n):
            ST[i, j] = ST[j, i] = yT[k]
            k += 1
    cov = ST - np.outer(mT, mT)
    wv = np.concatenate([omega, [0.0, 0.0]])
    var_v = float(wv @ cov @ wv)
    var_l = float(cov[N + 1, N + 1])
    cov_vl = float(wv @ cov[:, N + 1])
    if not (var_v > 0.0 and var_l > 0.0):
        raise ValueError("degenerate variance at the horizon")
    return cov_vl / np.sqrt(var_v * var_l)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "np = __import__('numpy')\nlam_inf = np.array([0.523, 0.687])\neta = np.array([[0.30, 0.15], [0.20, 0.25]])\nbeta = np.array([3.0, 2.5])\nlam_inf_ref = lam_inf.copy()\neta_ref = eta.copy()\nbeta_ref = beta.copy()",
            "call": "variance_intensity_correlation(0.35, 0.05, 3, 24, 0.25, 200001, 4.924, 0.288, 0.1081, 0.275, 0.0865, lam_inf, eta, beta, 0.25)",
            "gold_call": "_oracle_variance_intensity_correlation(0.35, 0.05, 3, 24, 0.25, 200001, 4.924, 0.288, 0.1081, 0.275, 0.0865, lam_inf_ref, eta_ref, beta_ref, 0.25)",
        },
        {
            "setup": "np = __import__('numpy')\nlam_inf = np.array([0.523, 0.687])\neta = np.array([[0.30, 0.15], [0.20, 0.25]])\nbeta = np.array([3.0, 2.5])\nlam_inf_ref = lam_inf.copy()\neta_ref = eta.copy()\nbeta_ref = beta.copy()",
            "call": "variance_intensity_correlation(0.35, 0.05, 3, 16, 0.25, 200001, 4.924, 0.288, 0.1081, 0.275, 0.0865, lam_inf, eta, beta, 0.1)",
            "gold_call": "_oracle_variance_intensity_correlation(0.35, 0.05, 3, 16, 0.25, 200001, 4.924, 0.288, 0.1081, 0.275, 0.0865, lam_inf_ref, eta_ref, beta_ref, 0.1)",
        },
        {
            "setup": "np = __import__('numpy')\nlam_inf = np.array([1.0, 0.2])\neta = np.array([[0.0, 0.9], [0.4, 0.0]])\nbeta = np.array([1.5, 0.8])\nlam_inf_ref = lam_inf.copy()\neta_ref = eta.copy()\nbeta_ref = beta.copy()",
            "call": "variance_intensity_correlation(0.10, 0.20, 5, 12, 1.0, 200001, 0.8, 0.05, 0.02, 0.1, 0.05, lam_inf, eta, beta, 0.5)",
            "gold_call": "_oracle_variance_intensity_correlation(0.10, 0.20, 5, 12, 1.0, 200001, 0.8, 0.05, 0.02, 0.1, 0.05, lam_inf_ref, eta_ref, beta_ref, 0.5)",
        },
        {
            "setup": "np = __import__('numpy')\nlam_inf = np.array([0.523, 0.687])\neta = np.array([[0.30, 0.15], [0.20, 0.25]])\nbeta = np.array([3.0, 2.5])\nlam_inf_ref = lam_inf.copy()\neta_ref = eta.copy()\nbeta_ref = beta.copy()",
            "call": "variance_intensity_correlation(0.35, 0.05, 3, 24, 0.25, 200001, 4.924, 0.288, 0.1081, 0.0, 0.0865, lam_inf, eta, beta, 0.25)",
            "gold_call": "_oracle_variance_intensity_correlation(0.35, 0.05, 3, 24, 0.25, 200001, 4.924, 0.288, 0.1081, 0.0, 0.0865, lam_inf_ref, eta_ref, beta_ref, 0.25)",
        },
        {
            "setup": "np = __import__('numpy')\ndef run_model():\n    try:\n        variance_intensity_correlation(0.35, 0.05, 3, 24, 0.25, 200001, 4.924, 0.288, 0.1081, 0.275, 0.0865, np.array([0.523, 0.687]), np.array([[0.30, 0.15], [0.20, 0.25]]), np.array([3.0, 2.5]), 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_variance_intensity_correlation(0.35, 0.05, 3, 24, 0.25, 200001, 4.924, 0.288, 0.1081, 0.275, 0.0865, np.array([0.523, 0.687]), np.array([[0.30, 0.15], [0.20, 0.25]]), np.array([3.0, 2.5]), 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
