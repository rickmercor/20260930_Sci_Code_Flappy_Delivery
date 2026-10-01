"""
Compute the constrained (profiled) maximum-likelihood estimate of the background rate b for a specified signal value s in the two-measurement on/off model.

The on/off model has independent Poisson observations with means s + b and tau*b. For a supplied fixed signal value s, the profiled background is the nonnegative background value that maximizes the joint likelihood. Inputs are restricted to the documented domain with counts, signal, and exposure ratio at most 1e12. Return the profiled estimate accurately throughout that domain, including cases with a large hypothesized signal and small background.

Returns
-------
float, the profiled background estimate bhh(s) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_profiled_background(n: float, m: float, tau: float, s: float) -> float:
    '''Profiled background estimate bhh(s) of the on/off model.

    Parameters
    ----------
    n : float
        On-region count (real-valued Asimov counts allowed); must be
        finite and satisfy 0 <= n <= 1e12.
    m : float
        Control-region count; must be finite and satisfy 0 <= m <= 1e12.
    tau : float
        Exposure ratio; must be finite and satisfy 0 < tau <= 1e12.
    s : float
        Hypothesised signal value; must be finite and satisfy
        0 <= s <= 1e12.

    Raises
    ------
    ValueError
        If any argument is non-finite, if n < 0 or m < 0 or s < 0, if
        tau <= 0, or if any of n, m, tau, s exceeds 1e12, which is the
        supported domain of the counting model.

    Returns
    -------
    bhh : float
        The profiled background estimate bhh(s), the nonnegative
        constrained maximum-likelihood estimate, as a
        native Python float.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_compute_profiled_background(n: float, m: float, tau: float, s: float) -> float:
    vals = [n, m, tau, s]
    if not all(np.isfinite(float(v)) for v in vals):
        raise ValueError("all arguments must be finite")
    n, m, tau, s = (float(v) for v in vals)
    if n < 0.0 or m < 0.0:
        raise ValueError("n and m must be non-negative")
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive")
    if s < 0.0:
        raise ValueError("s must be non-negative")
    if max(n, m, tau, s) > 1.0e12:
        raise ValueError("n, m, tau and s must not exceed 1e12")
    a = n + m - (1.0 + tau) * s
    c_root = 2.0 * math.sqrt(1.0 + tau) * math.sqrt(s) * math.sqrt(m)
    if c_root == 0.0:
        return float(a / (1.0 + tau)) if a > 0.0 else 0.0
    disc = math.hypot(a, c_root)
    num = (a + disc) if a >= 0.0 else (c_root * c_root) / (disc - a)
    return float(num / (2.0 * (1.0 + tau)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: normal, s = 0 reduces to (n + m)/(1 + tau) ---
        {
            "setup": """import numpy as np
n, m, tau, s = 7.8340219416, 8.5375650697, 2.2745029760, 0.0
""",
            "call": "compute_profiled_background(n, m, tau, s)",
            "gold_call": "_oracle_compute_profiled_background(n, m, tau, s)",
        },
        # --- Valid: boundary, discriminates the root branch (a < 0) ---
        {
            "setup": """import numpy as np
n, m, tau, s = 2.0, 1.0, 1.5, 4.0
""",
            "call": "compute_profiled_background(n, m, tau, s)",
            "gold_call": "_oracle_compute_profiled_background(n, m, tau, s)",
        },
        # --- Valid: edge, m = 0 with positive s ---
        {
            "setup": """import numpy as np
n, m, tau, s = 3.0, 0.0, 2.0, 1.0
""",
            "call": "compute_profiled_background(n, m, tau, s)",
            "gold_call": "_oracle_compute_profiled_background(n, m, tau, s)",
        },
        # --- Valid: edge, large signal where naive cancellation fails ---
        {
            "setup": """import numpy as np
n, m, tau, s = 2.0, 1.0, 1.5, 1000000000000.0
""",
            "call": "compute_profiled_background(n, m, tau, s)",
            "gold_call": "_oracle_compute_profiled_background(n, m, tau, s)",
        },
        # --- Invalid: tau = 0 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_profiled_background(3.0, 2.0, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_profiled_background(3.0, 2.0, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: s beyond the supported domain ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_profiled_background(2.0, 1.0, 1.5, 1e16)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_profiled_background(2.0, 1.0, 1.5, 1e16)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative s ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_profiled_background(3.0, 2.0, 1.0, -0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_profiled_background(3.0, 2.0, 1.0, -0.5)
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
