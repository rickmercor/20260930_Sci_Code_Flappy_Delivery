"""
Closed-form normal-model call value and its two underlying-price derivatives.

The conditional normal call supplies the reference price, Delta and Gamma to which the stochastic-volatility correction is added.

Returns
-------
tuple (price, delta, gamma), three float64 arrays of shape (r,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def normal_call_terms(T: float, X0: float, strikes: np.ndarray, sigma: float) -> tuple:
    """Return the constant-volatility normal call value and its first two
    derivatives with respect to the underlying level.

    With ``s = sigma * sqrt(T)`` and ``d_i = (X0 - k_i) / s``, let ``N`` be the
    standard normal cumulative distribution function and
    ``n(z) = exp(-z**2 / 2) / sqrt(2 * pi)`` its density. Return

    * ``price[i] = (X0 - k_i) * N(d_i) + s * n(d_i)``,
    * ``delta[i] = N(d_i)``,
    * ``gamma[i] = n(d_i) / s``.

    Parameters
    ----------
    T : float
        Maturity, strictly positive and finite.
    X0 : float
        Current underlying level, finite.
    strikes : numpy.ndarray
        One-dimensional, non-empty vector of finite strikes, shape ``(r,)``.
    sigma : float
        Constant absolute volatility, strictly positive and finite.

    Returns
    -------
    tuple
        ``(price, delta, gamma)``, three float64 arrays of shape ``(r,)``.

    Raises
    ------
    ValueError
        If ``T`` or ``sigma`` is not strictly positive and finite, if ``X0`` is
        not finite, or if ``strikes`` is not a one-dimensional non-empty array
        of finite values.
    """
    return (np.zeros(1), np.zeros(1), np.zeros(1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_normal_call_terms(T: float, X0: float, strikes: np.ndarray, sigma: float) -> tuple:
    """Reference implementation of the normal-model call terms."""
    np = __import__("numpy")

    for name, value in (("T", T), ("sigma", sigma)):
        val = float(value)
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if not np.isfinite(float(X0)):
        raise ValueError("X0 must be finite")
    k = np.asarray(strikes, dtype=float)
    if k.ndim != 1 or k.size == 0:
        raise ValueError("strikes must be a one-dimensional non-empty array")
    if not np.all(np.isfinite(k)):
        raise ValueError("strikes must be finite")

    s = float(sigma) * np.sqrt(float(T))
    d = (float(X0) - k) / s
    dens = np.exp(-0.5 * d * d) / np.sqrt(2.0 * np.pi)
    ndtr = __import__("scipy.special", fromlist=["ndtr"]).ndtr
    cdf = np.asarray(ndtr(d), dtype=float)
    price = (float(X0) - k) * cdf + s * dens
    return price, cdf, dens / s

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
T, X0 = 2.0, 100.0
strikes = np.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
sigma = np.sqrt(1.0 - 0.09) * 45.613268929246582
def flat_terms(t):
    return [float(x) for part in t for x in np.asarray(part).ravel()]
""",
            "call": "flat_terms(normal_call_terms(T, X0, strikes, sigma))",
            "gold_call": "flat_terms(_oracle_normal_call_terms(T, X0, strikes, sigma))",
        },
        {
            "setup": """import numpy as np
T, X0 = 0.25, 50.0
strikes = np.array([30.0, 50.0, 75.5])
sigma = 8.0
def flat_terms(t):
    return [float(x) for part in t for x in np.asarray(part).ravel()]
""",
            "call": "flat_terms(normal_call_terms(T, X0, strikes, sigma))",
            "gold_call": "flat_terms(_oracle_normal_call_terms(T, X0, strikes, sigma))",
        },
        {
            "setup": """import numpy as np
T, X0 = 5.0, -12.0
strikes = np.linspace(-40.0, 40.0, 9)
sigma = 3.5
def flat_terms(t):
    return [float(x) for part in t for x in np.asarray(part).ravel()]
""",
            "call": "flat_terms(normal_call_terms(T, X0, strikes, sigma))",
            "gold_call": "flat_terms(_oracle_normal_call_terms(T, X0, strikes, sigma))",
        },
        {
            "setup": """import numpy as np
strikes = np.array([90.0, 100.0])
def run_model_sigma():
    try:
        normal_call_terms(1.0, 100.0, strikes, -2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_sigma():
    try:
        _oracle_normal_call_terms(1.0, 100.0, strikes, -2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_sigma()",
            "gold_call": "run_oracle_sigma()",
        },
        {
            "setup": """import numpy as np
strikes = np.zeros((2, 3))
def run_model_shape():
    try:
        normal_call_terms(1.0, 100.0, strikes, 20.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_shape():
    try:
        _oracle_normal_call_terms(1.0, 100.0, strikes, 20.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_shape()",
            "gold_call": "run_oracle_shape()",
        },
    ]
