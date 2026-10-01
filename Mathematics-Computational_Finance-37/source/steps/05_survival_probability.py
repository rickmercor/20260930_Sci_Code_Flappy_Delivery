"""
Compute the survival probability Q(x0, T) = E[exp(-integral_0^T exp(X_t) dt)] for a Black-Karasinski intensity process dX_t = k(theta - X_t)dt + sigma dW_t started at X_0 = x0, using the GTFK path-integral semi-analytical approximation.

Once the trial-potential parameters (omega, delta_gamma) are known for a given path-average state x-bar, the GTFK approximation gives a closed-form expression for the reduced density's contribution at that x-bar (built from the normalization factor N(x-bar) and the completed-square Gaussian integral over the process's terminal/initial coupling, with coefficients A, B, C). The full survival probability Q(x0, T) = E[exp(-integral of the intensity)] is obtained by integrating this closed-form expression over the average-point variable x-bar. The source paper leaves the outer quadrature's node count and integration range, and the inner nonlinear solve's own iteration/tolerance choices, up to the implementer, with the caveat that the result should be verified for numerical stability.

Returns
-------
Q : float, the survival probability, a real number in (0, 1].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def survival_probability(k: float, sigma: float, theta: float, x0: float, T: float) -> float:
    """Compute the survival probability Q(x0, T) = E[exp(-integral_0^T
    exp(X_t) dt)] for a Black-Karasinski intensity process dX_t =
    k(theta - X_t)dt + sigma dW_t started at X_0 = x0, using the GTFK
    path-integral semi-analytical approximation.

    Parameters
    ----------
    k : float
        Mean-reversion strength (> 0).
    sigma : float
        Volatility (> 0).
    theta : float
        Long-run mean of the log-intensity process.
    x0 : float
        Initial state.
    T : float
        Horizon (> 0).

    Returns
    -------
    Q : float
        The survival probability, a real number in (0, 1].

    Raises
    ------
    ValueError
        If k <= 0, sigma <= 0, or T <= 0.
    """
    return float(np.exp(-np.exp(x0) * T))  # placeholder (crude, not the GTFK method)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_survival_probability(k: float, sigma: float, theta: float, x0: float, T: float) -> float:
    import numpy as np
    if k <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("k, sigma, and T must all be positive")

    def solve_omega_deltagamma(xbar):
        import numpy as np
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
        return omega, deltagamma, gamma

    def integrand(xbar):
        import numpy as np
        omega, deltagamma, gamma = solve_omega_deltagamma(xbar)
        f = omega * T / 2
        alpha = (sigma ** 2 / (2 * omega)) * (1.0 / np.tanh(f) - 1.0 / f)
        Gamma0 = gamma * (np.cosh(omega * T) - 1) / omega
        GammaT = Gamma0
        gammahat = gamma * T
        sinh2f = np.sinh(2 * f)
        int1 = (np.cosh(omega * T) - 1) / omega
        int2 = 0.5 * np.sinh(omega * T) * T \
            + (np.cosh(omega * T) - np.cosh(-omega * T)) / (4 * omega)
        Gamma0T = gamma * (-1.0 / omega) * gamma * int1 + (gamma ** 2 / omega) * int2
        Gamma_val = (sigma ** 2 / omega) * (
            Gamma0T / sinh2f - (1 / (4 * f)) * ((Gamma0 + GammaT) / sinh2f - gammahat) ** 2)

        xshift = xbar - deltagamma
        const_part = (k ** 2 - omega ** 2) / (2 * sigma ** 2) * alpha \
            - omega ** 2 * deltagamma ** 2 / (2 * sigma ** 2) \
            + np.exp(alpha / 2) * np.exp(xshift)
        int_w = const_part * T \
            + (k ** 2 * (theta - xshift) ** 2 / (2 * sigma ** 2)) * T \
            + deltagamma * gamma * T

        N_xbar = np.sqrt((1.0 / (2 * np.pi * alpha)) * (1.0 / (2 * np.pi * T * sigma ** 2))) \
            * (f / np.sinh(f)) * np.exp(-int_w + Gamma_val)

        A = 1.0 / (8 * alpha) + (omega / np.tanh(f)) / (4 * sigma ** 2) + k / (2 * sigma ** 2)
        delta_ = (sigma ** 2 / (2 * omega * f)) * ((Gamma0 + GammaT) / sinh2f - gammahat)
        B = (x0 - xbar + delta_) / (2 * alpha) + Gamma0 / sinh2f + k * (x0 - theta) / sigma ** 2
        C = -(x0 - xbar + delta_) ** 2 / (2 * alpha) - (x0 - xbar) * (Gamma0 + GammaT) / sinh2f \
            + k * T / 2
        Nbar = np.sqrt(np.pi / A) * N_xbar * np.exp(C + B ** 2 / (4 * A))
        return Nbar

    n_xbar, x_range = 401, 2.5
    xbars = np.linspace(x0 - x_range, x0 + x_range, n_xbar)
    vals = np.array([integrand(xb) for xb in xbars])
    return float(np.sum((vals[:-1] + vals[1:]) * np.diff(xbars) / 2.0))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal case: the actual task instance, T = 1 ---
        {
            "setup": """import numpy as np
k, sigma = 0.3, 0.35
theta = -2.900422093749666
x0 = -3.101092789211817
def run_model():
    return survival_probability(k, sigma, theta, x0, 1.0)
def run_gold():
    return _oracle_survival_probability(k, sigma, theta, x0, 1.0)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-4,
        },
        # --- Boundary case: the actual task instance, T = 2 ---
        {
            "setup": """import numpy as np
k, sigma = 0.3, 0.35
theta = -2.900422093749666
x0 = -3.101092789211817
def run_model():
    return survival_probability(k, sigma, theta, x0, 2.0)
def run_gold():
    return _oracle_survival_probability(k, sigma, theta, x0, 2.0)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-4,
        },
        # --- Edge case: non-positive k should raise ---
        {
            "setup": """import numpy as np
theta, x0 = -2.9, -3.1
def run_model():
    try:
        survival_probability(0.0, 0.35, theta, x0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_survival_probability(0.0, 0.35, theta, x0, 1.0)
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
