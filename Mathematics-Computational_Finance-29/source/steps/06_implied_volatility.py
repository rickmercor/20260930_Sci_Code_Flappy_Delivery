"""
Invert the zero-rate Black-Scholes call formula for the volatility that reproduces price at spot s0, strike strike and tenor tau, by bisection on the interval [1e-4, 5] with exactly 200 halvings, returning the midpoint of the final bracket. Reject prices outside the no-arbitrage bounds.

Implied volatility is the quoting convention in which the source reports every smile; the model prices of equation (9) are mapped back to it before any comparison across strikes.

Returns
-------
A single float, the annualised implied volatility.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def implied_volatility(price, s0, strike, tau):
    """Return the Black-Scholes implied volatility of a call price at zero rate."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_implied_volatility(price, s0, strike, tau):
    import numpy as np
    from math import erf, sqrt, log
    if s0 <= 0.0 or strike <= 0.0 or tau <= 0.0:
        raise ValueError("s0, strike and tau must be positive")
    if not (max(s0 - strike, 0.0) <= price <= s0):
        raise ValueError("price violates the zero-rate no-arbitrage bounds")
    def ncdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    def bs_call(sig):
        d1 = (log(s0 / strike) + 0.5 * sig * sig * tau) / (sig * sqrt(tau))
        return s0 * ncdf(d1) - strike * ncdf(d1 - sig * sqrt(tau))
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_call(mid) > price:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "implied_volatility(5.081956382691, 100.0, 94.98, 5.0/365.0)",
         "gold_call": "_oracle_implied_volatility(5.081956382691, 100.0, 94.98, 5.0/365.0)"},
        {"setup": "import numpy as np",
         "call": "implied_volatility(3.9877611676744, 100.0, 100.0, 0.25)",
         "gold_call": "_oracle_implied_volatility(3.9877611676744, 100.0, 100.0, 0.25)"},
        {"setup": "import numpy as np",
         "call": "implied_volatility(0.0102897587, 100.0, 105.28, 5.0/365.0)",
         "gold_call": "_oracle_implied_volatility(0.0102897587, 100.0, 105.28, 5.0/365.0)"},
    ]
