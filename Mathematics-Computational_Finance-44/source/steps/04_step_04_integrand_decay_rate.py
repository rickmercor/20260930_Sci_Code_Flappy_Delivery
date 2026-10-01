"""
Compute the exponential decay rate that the damped Fourier integrand inherits from the variance dynamics at a given maturity. The rate combines a long-run variance contribution with a maturity term whose exponent is set by the roughness of the variance path.

Large-frequency asymptotics of the fractional Riccati solution determine the exponential decay of the damped Fourier integrand. The resulting rate includes a rough-specific maturity factor involving T^(1-alpha)/Gamma(2-alpha), and it sets the scale used to place the Gauss-Laguerre nodes efficiently.

Returns
-------
float Strictly positive decay rate sigma.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def integrand_decay_rate(model, horizon):
    """Return the prescribed exponential decay rate at one maturity.

    Use the source's rough-variance decay estimate for the supplied parameter
    ordering.  The estimate combines correlation, volatility of volatility,
    accumulated mean variance, and the rough initial-variance contribution at
    ``horizon``.  Evaluate the expression in double precision, including its
    finite limiting case when the roughness index equals one.

    Parameters
    ----------
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.
    horizon : float
        Strictly positive maturity.

    Returns
    -------
    float
        Strictly positive finite rate for the exponential quadrature weight.

    Raises
    ------
    ValueError
        If ``model`` violates a stated bound or ``horizon`` is not finite and
        strictly positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _unpack_model(model):
    """Validate and unpack the six model parameters."""
    np = __import__("numpy")
    arr = np.asarray(model, dtype=float).reshape(-1)
    if arr.size != 6 or not bool(np.all(np.isfinite(arr))):
        raise ValueError("model must hold six finite numbers")
    alpha, gamma, nu, rho, v0, theta = (float(v) for v in arr)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("model[0] must satisfy 0 < alpha <= 1")
    if gamma <= 0.0 or nu <= 0.0:
        raise ValueError("model[1] and model[2] must be positive")
    if not -1.0 < rho < 1.0:
        raise ValueError("model[3] must satisfy abs(rho) < 1")
    if v0 <= 0.0 or theta <= 0.0:
        raise ValueError("model[4] and model[5] must be positive")
    return alpha, gamma, nu, rho, v0, theta


def _unpack_market(market):
    """Validate and unpack the four market parameters."""
    np = __import__("numpy")
    arr = np.asarray(market, dtype=float).reshape(-1)
    if arr.size != 4 or not bool(np.all(np.isfinite(arr))):
        raise ValueError("market must hold four finite numbers")
    spot, strike, rate, horizon = (float(v) for v in arr)
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("market[0] and market[1] must be positive")
    if horizon <= 0.0:
        raise ValueError("market[3] must be positive")
    return spot, strike, rate, horizon


def _positive_int(value, name):
    """Return ``value`` as a positive Python int or raise ValueError."""
    np = __import__("numpy")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be a positive integer")
    if int(value) < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _line_argument(xi_real, xi_imag):
    """Return the flattened complex vector xi_real + 1j*xi_imag."""
    np = __import__("numpy")
    a = np.atleast_1d(np.asarray(xi_real, dtype=float)).reshape(-1)
    b = np.atleast_1d(np.asarray(xi_imag, dtype=float)).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("xi_real and xi_imag must have the same shape")
    if not (bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(b)))):
        raise ValueError("xi_real and xi_imag must be finite")
    return a + 1j * b
def _decay_rate(alpha, gamma, nu, rho, v0, theta, horizon):
    """Return sqrt(1-rho**2)/(gamma*nu)*(gamma*theta*T + v0*T**(1-alpha)/Gamma(2-alpha))."""
    math = __import__("math")
    return (math.sqrt(1.0 - rho * rho) / (gamma * nu)) * (
        gamma * theta * horizon
        + v0 * horizon ** (1.0 - alpha) / math.gamma(2.0 - alpha)
    )


def _oracle_integrand_decay_rate(model, horizon):
    """Reference implementation of the decay rate."""
    np = __import__("numpy")
    alpha, gamma, nu, rho, v0, theta = _unpack_model(model)
    horizon = float(horizon)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    return float(_decay_rate(alpha, gamma, nu, rho, v0, theta, horizon))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return short, target, long, alternative-parameter, and invalid cases."""
    euros = "model = [0.62, 0.1, 0.331, -0.681, 0.0392, 0.3156]\n"
    return [
        {
            "setup": euros + "horizon = 2.0\n",
            "call": "round(float(integrand_decay_rate(model, horizon)), 10)",
            "gold_call": "round(float(_oracle_integrand_decay_rate(model, horizon)), 10)",
        },
        {
            "setup": euros + "horizon = 0.1\n",
            "call": "round(float(integrand_decay_rate(model, horizon)), 10)",
            "gold_call": "round(float(_oracle_integrand_decay_rate(model, horizon)), 10)",
        },
        {
            "setup": euros + "horizon = 10.0\n",
            "call": "round(float(integrand_decay_rate(model, horizon)), 10)",
            "gold_call": "round(float(_oracle_integrand_decay_rate(model, horizon)), 10)",
        },
        {
            "setup": "model = [0.7151, 1.8967, 0.6144356, -0.6704, 0.06246, 0.03848]\nhorizon = 1.0\n",
            "call": "round(float(integrand_decay_rate(model, horizon)), 10)",
            "gold_call": "round(float(_oracle_integrand_decay_rate(model, horizon)), 10)",
        },
        {
            "setup": (
                "model = [0.62, 0.1, 0.331, -1.5, 0.0392, 0.3156]\n"
                "def run_model():\n"
                "    try:\n"
                "        integrand_decay_rate(model, 2.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_integrand_decay_rate(model, 2.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
