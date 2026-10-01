"""
Compute the posterior probability that the on-region mean exceeds the background, under the prescribed non-informative prior of the reference Bayesian on/off analysis.

In the Bayesian description of the on/off channel, the on-region mean mu_on = s + b and the background mean b are treated as random variables. Both are assigned the same non-informative prior independently. Use the prior recommended by the reference comparison of Bayesian on/off criteria and compute the resulting posterior probability. Accurate numerical quadrature is acceptable when it agrees with the prescribed posterior probability within the test tolerance.

Returns
-------
float, the posterior probability P(mu_on > b | n, m), represented in [0, 1]; a value may round to 0 or 1 in IEEE double precision
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_signal_probability(n: float, m: float, tau: float) -> float:
    '''Posterior probability P(mu_on > b | n, m) of the on/off channel.

    Parameters
    ----------
    n : float
        On-region count (real-valued Asimov counts allowed); must be
        finite and satisfy 0 <= n <= 1e12.
    m : float
        Control-region count; must be finite and satisfy 0 <= m <= 1e12.
    tau : float
        Exposure ratio; must be finite and satisfy 0 < tau <= 1e12.

    Raises
    ------
    ValueError
        If any argument is non-finite, if n < 0 or m < 0, if tau <= 0,
        or if any of n, m, tau exceeds 1e12, which is the supported
        domain of the counting model.

    Returns
    -------
    p : float
        The posterior probability that the on-region mean exceeds the
        background, under the prescribed non-informative prior, as a
        native Python float in [0, 1]. The mathematical probability is
        strictly between 0 and 1; IEEE double precision may round it
        to an endpoint.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import special


def _oracle_compute_signal_probability(n: float, m: float, tau: float) -> float:
    vals = [n, m, tau]
    if not all(np.isfinite(float(v)) for v in vals):
        raise ValueError("all arguments must be finite")
    n, m, tau = (float(v) for v in vals)
    if n < 0.0 or m < 0.0:
        raise ValueError("n and m must be non-negative")
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive")
    if max(n, m, tau) > 1.0e12:
        raise ValueError("n, m and tau must not exceed 1e12")
    sigma = 0.5
    x = tau / (1.0 + tau)
    a1 = m + 1.0 - sigma
    a2 = n - sigma + 1.0
    return float(special.betainc(a1, a2, x))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: normal, integer counts ---
        {
            "setup": """import numpy as np
n, m, tau = 7.0, 4.0, 2.0
""",
            "call": "compute_signal_probability(n, m, tau)",
            "gold_call": "_oracle_compute_signal_probability(n, m, tau)",
        },
        # --- Valid: boundary, real-valued Asimov counts near threshold ---
        {
            "setup": """import numpy as np
n, m, tau = 7.8340219416, 8.5375650697, 2.2745029760
""",
            "call": "compute_signal_probability(n, m, tau)",
            "gold_call": "_oracle_compute_signal_probability(n, m, tau)",
        },
        # --- Valid: edge, zero counts in both regions ---
        {
            "setup": """import numpy as np
n, m, tau = 0.0, 0.0, 1.5
""",
            "call": "compute_signal_probability(n, m, tau)",
            "gold_call": "_oracle_compute_signal_probability(n, m, tau)",
        },
        # --- Invalid: count beyond the supported domain ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_signal_probability(1e13, 2.0, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_signal_probability(1e13, 2.0, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative count ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_signal_probability(-1.0, 2.0, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_signal_probability(-1.0, 2.0, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: tau = 0 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_signal_probability(3.0, 2.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_signal_probability(3.0, 2.0, 0.0)
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
