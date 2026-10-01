"""
Evaluate the additive forward representation of the put without early exercise by direct quadrature, split into its parts.

When exercise is only possible at maturity, the stopping indicator is identically one and the forward drift at maturity u is the discounted expectation of the drift rate mu - r G at time u under the lognormal law of S_u. Integrating that forward drift over [0, T] and adding the current gain gives the representation of the European put. The drift rate is the one of step 01, with the Gaussian-kernel local time. Every term is a one-dimensional expectation under a lognormal law followed by a time integral, so the value is computed by quadrature, with no grid in the asset direction. The integration must resolve the short-time behaviour at the strike, where the law of S_u is much narrower than the kernel, and the long-time behaviour, where the kernel is much narrower than the law, to a relative accuracy of 1e-9.

Returns
-------
np.ndarray, float, shape (3,): the representation value, the time integral of the part of the forward drift without local time, and the time integral of its local-time part.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def european_additive_value(spot: float, strike: float, rate: float, vol: float,
                            maturity: float, eps: float) -> np.ndarray:
    '''Additive representation of the European put by quadrature.

    Parameters
    ----------
    spot : float
        Current price S_0 > 0.
    strike : float
        Strike K > 0.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b > 0.
    maturity : float
        Maturity T > 0.
    eps : float
        Standard deviation of the Gaussian kernel in price units,
        0 < eps < strike / 10.

    Returns
    -------
    parts : np.ndarray
        Shape (3,) float array: [value, integral without local time,
        integral of the local-time part]. The value equals (K - S_0)^+ plus
        the two integrals.

    Raises
    ------
    ValueError
        If any argument is outside its stated domain.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros(3, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_european_additive_value(spot: float, strike: float, rate: float, vol: float,
                                    maturity: float, eps: float) -> np.ndarray:
    import numpy as np
    from math import erf, exp, log, pi, sqrt
    from scipy import integrate

    for name, value in (("spot", spot), ("strike", strike), ("vol", vol),
                        ("maturity", maturity), ("eps", eps)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(rate, bool) or not np.isfinite(float(rate)):
        raise ValueError("rate must be finite")
    s0, k, r, b, t_mat, e = (float(x) for x in (spot, strike, rate, vol, maturity, eps))
    if e >= k / 10.0:
        raise ValueError("eps must be below strike / 10")

    mu = r - 0.5 * b * b
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    weights = weights / np.sqrt(2.0 * np.pi)
    norm_cdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def kernel_expectation(u):
        # E[ phi_eps(S_u - K) ]: integrate over whichever of the two laws is wider.
        if u <= 0.0:
            return exp(-(s0 - k) ** 2 / (2.0 * e * e)) / (sqrt(2.0 * pi) * e)
        sd = b * sqrt(u)
        if s0 * sd < e:
            s = s0 * np.exp(mu * u + sd * nodes)
            return float(np.sum(weights * np.exp(-(s - k) ** 2 / (2.0 * e * e)))) / (sqrt(2.0 * pi) * e)
        s = k + e * nodes
        x = np.log(s / s0)
        density = np.exp(-(x - mu * u) ** 2 / (2.0 * sd * sd)) / (s * sd * sqrt(2.0 * pi))
        return float(np.sum(weights * density))

    def drift_part(u):
        if u <= 0.0:
            below = 1.0 if s0 < k else (0.5 if s0 == k else 0.0)
        else:
            below = norm_cdf(-(log(s0 / k) + mu * u) / (b * sqrt(u)))
        return -r * k * exp(-r * u) * below

    def local_time_part(u):
        return 0.5 * b * b * k * k * exp(-r * u) * kernel_expectation(u)

    # u = v^2 resolves the square-root behaviour at the start.
    opts = dict(limit=800, epsabs=1e-13, epsrel=1e-12)
    top = sqrt(t_mat)
    d_int, _ = integrate.quad(lambda v: 2.0 * v * drift_part(v * v), 0.0, top, **opts)
    l_int, _ = integrate.quad(lambda v: 2.0 * v * local_time_part(v * v), 0.0, top, **opts)
    return np.array([max(k - s0, 0.0) + d_int + l_int, d_int, l_int], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the at-the-money benchmark contract.
        {
            "setup": """import numpy as np
""",
            "call": "european_additive_value(100.0, 100.0, 0.05, 0.2, 1.0, 0.5)",
            "gold_call": "_oracle_european_additive_value(100.0, 100.0, 0.05, 0.2, 1.0, 0.5)",
        },
        # --- Pinned consistency: the value is the gain plus the two parts.
        {
            "setup": """import numpy as np
def gap(fn):
    v = fn(90.0, 100.0, 0.05, 0.2, 1.0, 0.5)
    return np.array([v[0] - 10.0 - v[1] - v[2], v[0]])
EXPECTED = np.array([0.0, _oracle_european_additive_value(90.0, 100.0, 0.05, 0.2, 1.0, 0.5)[0]])
""",
            "call": "np.round(gap(european_additive_value) * np.array([1e6, 1.0]), 3) + 0.0",
            "gold_call": "np.round(EXPECTED * np.array([1e6, 1.0]), 3) + 0.0",
        },
        # --- Pinned limit: a narrow kernel recovers the Black-Scholes put to
        # about 1e-3 (the leading correction is -eps/sqrt(2 pi) at the money).
        {
            "setup": """import numpy as np
from math import erf, exp, log, sqrt
N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
d1 = (0.05 + 0.02) / 0.2; d2 = d1 - 0.2
bs = 100 * exp(-0.05) * N(-d2) - 100 * N(-d1)
def close(fn):
    v = fn(100.0, 100.0, 0.05, 0.2, 1.0, 0.002)[0]
    return np.array([float(abs(v - bs) < 2e-3), v])
""",
            "call": "close(european_additive_value)",
            "gold_call": "np.array([1.0, _oracle_european_additive_value(100.0, 100.0, 0.05, 0.2, 1.0, 0.002)[0]])",
        },
        # --- Boundary: deep in the money, where the local-time part is small.
        {
            "setup": """import numpy as np
""",
            "call": "european_additive_value(60.0, 100.0, 0.05, 0.2, 1.0, 0.5)",
            "gold_call": "_oracle_european_additive_value(60.0, 100.0, 0.05, 0.2, 1.0, 0.5)",
        },
        # --- Edge: a wide kernel with a short maturity.
        {
            "setup": """import numpy as np
""",
            "call": "european_additive_value(52.0, 50.0, 0.03, 0.35, 0.1, 2.0)",
            "gold_call": "_oracle_european_additive_value(52.0, 50.0, 0.03, 0.35, 0.1, 2.0)",
        },
        # --- Invalid: kernel wider than a tenth of the strike ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        european_additive_value(100.0, 100.0, 0.05, 0.2, 1.0, 20.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_european_additive_value(100.0, 100.0, 0.05, 0.2, 1.0, 20.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive maturity ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        european_additive_value(100.0, 100.0, 0.05, 0.2, 0.0, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_european_additive_value(100.0, 100.0, 0.05, 0.2, 0.0, 0.5)
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
