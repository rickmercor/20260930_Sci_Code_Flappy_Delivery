"""
Compute the Margrabe price of the option to exchange asset 2 for asset 1 when the log price ratio is Gaussian with constant variance rate.

With a constant variance rate s^2 of ln(S1/S2), the exchange value is S1 N(d1) - S2 N(d2), with d1 = (ln(S1/S2) + s^2 tau / 2) / (s sqrt(tau)) and d2 = d1 - s sqrt(tau). It does not depend on the interest rate. It is the limit of the full model when stochastic variance, liquidity loading and regime switching are all removed, and it serves as the reference that the transform pricing must reproduce in that limit.

Returns
-------
float: the Margrabe exchange value, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def margrabe_price(s1: float, s2: float, variance_rate: float, tau: float) -> float:
    '''Margrabe value of the option to exchange asset 2 for asset 1.

    Parameters
    ----------
    s1 : float
        Price of asset 1, s1 > 0.
    s2 : float
        Price of asset 2, s2 > 0.
    variance_rate : float
        Constant variance rate of ln(S1/S2), > 0.
    tau : float
        Time to maturity, tau > 0.

    Returns
    -------
    value : float
        Exchange value, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is not a finite positive number.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_margrabe_price(s1: float, s2: float, variance_rate: float, tau: float) -> float:
    import numpy as np
    from math import erf, log, sqrt

    for name, value in (("s1", s1), ("s2", s2), ("variance_rate", variance_rate), ("tau", tau)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    s = sqrt(float(variance_rate) * float(tau))
    d1 = (log(float(s1) / float(s2)) + 0.5 * s * s) / s
    norm_cdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return float(float(s1) * norm_cdf(d1) - float(s2) * norm_cdf(d1 - s))

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned identity: exchanging 2 for 1 minus exchanging 1 for 2 is
        # worth S1 - S2 for any variance and maturity.
        {
            "setup": """import numpy as np
def parity(fn):
    return fn(100.0, 95.0, 0.06, 1.3) - fn(95.0, 100.0, 0.06, 1.3)
""",
            "call": "parity(margrabe_price)",
            "gold_call": "5.0",
        },
        # --- Normal: the task's constant variance rate in regime 2.
        {
            "setup": """import numpy as np
""",
            "call": "margrabe_price(100.0, 95.0, 0.06, 1.0)",
            "gold_call": "_oracle_margrabe_price(100.0, 95.0, 0.06, 1.0)",
        },
        # --- Boundary: equal prices, where the value is S (2 N(s sqrt(tau)/2) - 1).
        {
            "setup": """import numpy as np
from math import erf, sqrt
EXPECTED = 50.0 * (2 * 0.5 * (1 + erf(0.5 * sqrt(0.09 * 2.0) / sqrt(2))) - 1)
""",
            "call": "margrabe_price(50.0, 50.0, 0.09, 2.0)",
            "gold_call": "EXPECTED",
        },
        # --- Edge: tiny variance, where the value approaches (S1 - S2)^+.
        {
            "setup": """import numpy as np
""",
            "call": "margrabe_price(120.0, 80.0, 1e-8, 0.25)",
            "gold_call": "_oracle_margrabe_price(120.0, 80.0, 1e-8, 0.25)",
        },
        # --- Invalid: non-positive variance rate ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        margrabe_price(100.0, 95.0, 0.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_margrabe_price(100.0, 95.0, 0.0, 1.0)
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
        margrabe_price(100.0, 95.0, 0.06, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_margrabe_price(100.0, 95.0, 0.06, -1.0)
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
