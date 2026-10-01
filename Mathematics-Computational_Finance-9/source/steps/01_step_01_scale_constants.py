"""
Scale constants of the lognormal-volatility normal-return model.

The RMS realized volatility and convergence radius define the scale and admissible strike range of the finite matrix expansion.

Returns
-------
tuple (tau, v, radius), three finite floats.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def scale_constants(sigma0: float, nu: float, rho: float, T: float) -> tuple:
    """Return the three scale constants of the model.

    The instantaneous volatility is ``sigma_t = sigma0 * exp(nu * W_t - nu**2 * t / 2)``
    for a standard Brownian motion ``W``, and the return process is driven by
    ``rho * dW_t + sqrt(1 - rho**2) * dB_t`` with ``B`` independent of ``W``.

    Compute, in this order:

    * ``tau = nu**2 * T``;
    * ``v = sigma0 * sqrt((exp(tau) - 1) / tau)``, the root-mean-square level
      ``sqrt(E[(1 / T) * int_0^T sigma_s**2 ds])``;
    * ``radius = sigma0 * sqrt(1 - rho**2) / nu``.

    Parameters
    ----------
    sigma0 : float
        Initial volatility level, strictly positive and finite.
    nu : float
        Volatility-of-volatility, strictly positive and finite.
    rho : float
        Correlation coefficient with ``-1 < rho < 1``.
    T : float
        Maturity in years, strictly positive and finite.

    Returns
    -------
    tuple
        ``(tau, v, radius)``, three finite floats.

    Raises
    ------
    ValueError
        If ``sigma0``, ``nu`` or ``T`` is not strictly positive and finite, or
        if ``rho`` is not a finite number with ``-1 < rho < 1``.
    """
    return (0.0, 0.0, 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_scale_constants(sigma0: float, nu: float, rho: float, T: float) -> tuple:
    """Reference implementation of the model scale constants."""
    np = __import__("numpy")

    for name, value in (("sigma0", sigma0), ("nu", nu), ("T", T)):
        val = float(value)
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")
    rho_val = float(rho)
    if not np.isfinite(rho_val) or not -1.0 < rho_val < 1.0:
        raise ValueError("rho must be finite with -1 < rho < 1")

    sigma0 = float(sigma0)
    nu = float(nu)
    T = float(T)
    tau = nu * nu * T
    v = sigma0 * np.sqrt(np.expm1(tau) / tau)
    radius = sigma0 * np.sqrt(1.0 - rho_val * rho_val) / nu
    return float(tau), float(v), float(radius)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
sigma0, nu, rho, T = 35.0, 0.7, -0.3, 2.0
""",
            "call": "list(scale_constants(sigma0, nu, rho, T))",
            "gold_call": "list(_oracle_scale_constants(sigma0, nu, rho, T))",
        },
        {
            "setup": """import numpy as np
sigma0, nu, rho, T = 12.5, 0.25, 0.6, 0.5
""",
            "call": "list(scale_constants(sigma0, nu, rho, T))",
            "gold_call": "list(_oracle_scale_constants(sigma0, nu, rho, T))",
        },
        {
            "setup": """import numpy as np
sigma0, nu, rho, T = 100.0, 1.4, 0.0, 5.0
""",
            "call": "list(scale_constants(sigma0, nu, rho, T))",
            "gold_call": "list(_oracle_scale_constants(sigma0, nu, rho, T))",
        },
        {
            "setup": """import numpy as np
def run_model_rho():
    try:
        scale_constants(35.0, 0.7, 1.0, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_rho():
    try:
        _oracle_scale_constants(35.0, 0.7, 1.0, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_rho()",
            "gold_call": "run_oracle_rho()",
        },
        {
            "setup": """import numpy as np
def run_model_nu():
    try:
        scale_constants(35.0, 0.0, -0.3, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_nu():
    try:
        _oracle_scale_constants(35.0, 0.0, -0.3, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_nu()",
            "gold_call": "run_oracle_nu()",
        },
    ]
