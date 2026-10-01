"""
Solve for the GTFK trial-potential's self-consistent parameters (omega, delta_gamma) at a given path-average state xbar, for a Black-Karasinski intensity process with mean-reversion k, volatility sigma, long-run log-intensity theta, over horizon T.

The GTFK path-integral approximation for a Black-Karasinski intensity process replaces the true nonlinear potential with a trial quadratic ansatz centered on a given path-average state x-bar. The ansatz's own two free parameters, omega (an effective mean-reversion-like frequency) and delta_gamma (a mean-shift correction), are pinned down by a self-consistency condition: omega depends on delta_gamma through a Gaussian-averaged intensity term, and delta_gamma in turn depends on omega through the accumulated potential integrals Gamma_0 and Gamma_T. Solving this coupled system requires a fixed-point (or similarly convergent) iteration; the source paper does not specify a particular iteration scheme, initial guess, or convergence tolerance.

Returns
-------
(omega, delta_gamma) : tuple of float, the self-consistent effective frequency and mean-shift correction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def gtfk_selfconsistent(k: float, sigma: float, theta: float, T: float, xbar: float) -> tuple:
    """Solve for the GTFK trial-potential's self-consistent parameters
    (omega, delta_gamma) at a given path-average state xbar, for a
    Black-Karasinski intensity process with mean-reversion k, volatility
    sigma, long-run log-intensity theta, over horizon T.

    Parameters
    ----------
    k : float
        Mean-reversion strength (> 0).
    sigma : float
        Volatility (> 0).
    theta : float
        Long-run mean of the log-intensity process.
    T : float
        Horizon (> 0).
    xbar : float
        Path-average state at which the trial potential is centered.

    Returns
    -------
    omega : float
        The self-consistent effective frequency.
    delta_gamma : float
        The self-consistent mean-shift correction.

    Raises
    ------
    ValueError
        If k <= 0, sigma <= 0, or T <= 0.
    """
    omega = np.sqrt(k ** 2 + sigma ** 2 * np.exp(xbar))  # placeholder
    delta_gamma = 0.0  # placeholder
    return omega, delta_gamma

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_gtfk_selfconsistent(k: float, sigma: float, theta: float, T: float, xbar: float) -> tuple:
    import numpy as np
    if k <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("k, sigma, and T must all be positive")

    omega = np.sqrt(k ** 2 + sigma ** 2 * np.exp(xbar))
    deltagamma = 0.0
    for _ in range(60):
        f = omega * T / 2
        alpha = (sigma ** 2 / (2 * omega)) * (1.0 / np.tanh(f) - 1.0 / f)
        gamma = ((omega ** 2 - k ** 2) / sigma ** 2) * (deltagamma + 1) \
            + k ** 2 * xbar / sigma ** 2 - k ** 2 * theta / sigma ** 2
        Gamma0 = gamma * (np.cosh(omega * T) - 1) / omega
        GammaT = Gamma0
        gammahat = gamma * T
        deltagamma_new = (sigma ** 2 / (2 * omega)) * (
            (Gamma0 + GammaT) / (2 * np.sinh(f) ** 2) - gammahat / f)
        omega_new = np.sqrt(k ** 2 + sigma ** 2 * np.exp(alpha / 2 + xbar - deltagamma_new))
        if abs(omega_new - omega) < 1e-13 and abs(deltagamma_new - deltagamma) < 1e-13:
            omega, deltagamma = omega_new, deltagamma_new
            break
        omega, deltagamma = omega_new, deltagamma_new
    return omega, deltagamma

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal case: instance parameters, xbar = x0, T = 1 ---
        {
            "setup": """import numpy as np
def _combine(omega, dg):
    return np.array([omega, dg])
k, sigma = 0.3, 0.35
theta = -2.900422093749666
x0 = -3.101092789211817
def run_model():
    return _combine(*gtfk_selfconsistent(k, sigma, theta, 1.0, x0))
def run_gold():
    return _combine(*_oracle_gtfk_selfconsistent(k, sigma, theta, 1.0, x0))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-8,
        },
        # --- Boundary case: xbar = theta exactly (symmetric point), T = 2 ---
        {
            "setup": """import numpy as np
def _combine(omega, dg):
    return np.array([omega, dg])
k, sigma = 0.3, 0.35
theta = -2.900422093749666
def run_model():
    return _combine(*gtfk_selfconsistent(k, sigma, theta, 2.0, theta))
def run_gold():
    return _combine(*_oracle_gtfk_selfconsistent(k, sigma, theta, 2.0, theta))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-8,
        },
        # --- Edge case: non-positive T should raise ---
        {
            "setup": """import numpy as np
k, sigma, theta, x0 = 0.3, 0.35, -2.9, -3.1
def run_model():
    try:
        gtfk_selfconsistent(k, sigma, theta, 0.0, x0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_gtfk_selfconsistent(k, sigma, theta, 0.0, x0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
