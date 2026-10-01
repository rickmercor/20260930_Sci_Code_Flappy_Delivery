"""
Compute the auxiliary statistic u(0) that enters the higher-order correction of the signed likelihood-ratio root, in the form specific to the two-measurement on/off model.

Higher-order asymptotic theory refines the signed root by means of a model-dependent auxiliary statistic u. Compute the auxiliary statistic prescribed for the two-measurement Poisson model. Inputs are restricted to counts and exposure ratio of at most 1e12, as documented in the signature.

Returns
-------
float, the auxiliary statistic u(0) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_auxiliary_statistic(n: float, m: float, tau: float) -> float:
    '''Auxiliary statistic u(0) of the two-measurement on/off model.

    Parameters
    ----------
    n : float
        On-region count (real-valued Asimov counts allowed); must be
        finite and satisfy 0 < n <= 1e12.
    m : float
        Control-region count; must be finite and satisfy 0 < m <= 1e12.
    tau : float
        Exposure ratio; must be finite and satisfy 0 < tau <= 1e12.

    Raises
    ------
    ValueError
        If any argument is non-finite, if n <= 0 or m <= 0 or tau <= 0,
        or if any of n, m, tau exceeds 1e12, which is the supported
        domain of the counting model.

    Returns
    -------
    u0 : float
        The auxiliary statistic u(0) for a test of s = 0, in the form
        prescribed for the two-measurement model, as a native Python
        float.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_compute_auxiliary_statistic(n: float, m: float, tau: float) -> float:
    vals = [n, m, tau]
    if not all(np.isfinite(float(v)) for v in vals):
        raise ValueError("all arguments must be finite")
    n, m, tau = (float(v) for v in vals)
    if n <= 0.0 or m <= 0.0:
        raise ValueError("n and m must be strictly positive")
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive")
    if max(n, m, tau) > 1.0e12:
        raise ValueError("n, m and tau must not exceed 1e12")
    scale = math.sqrt(n) * math.sqrt(m / (n + m))
    return float(scale * math.log(n * tau / m))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: normal, positive excess ---
        {
            "setup": """import numpy as np
n, m, tau = 5.0565705494, 0.7883573968, 2.2496124098
""",
            "call": "compute_auxiliary_statistic(n, m, tau)",
            "gold_call": "_oracle_compute_auxiliary_statistic(n, m, tau)",
        },
        # --- Valid: boundary, n*tau = m gives u(0) = 0 exactly ---
        {
            "setup": """import numpy as np
n, m, tau = 2.0, 4.0, 2.0
""",
            "call": "compute_auxiliary_statistic(n, m, tau)",
            "gold_call": "_oracle_compute_auxiliary_statistic(n, m, tau)",
        },
        # --- Valid: edge, small counts where the correction matters most ---
        {
            "setup": """import numpy as np
n, m, tau = 1.1, 0.15, 0.5
""",
            "call": "compute_auxiliary_statistic(n, m, tau)",
            "gold_call": "_oracle_compute_auxiliary_statistic(n, m, tau)",
        },
        # --- Valid: edge, large counts near the domain limit ---
        {
            "setup": """import numpy as np
n, m, tau = 1000000000000.0, 500000000000.0, 1.5
""",
            "call": "compute_auxiliary_statistic(n, m, tau)",
            "gold_call": "_oracle_compute_auxiliary_statistic(n, m, tau)",
        },
        # --- Invalid: n = 0 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_auxiliary_statistic(0.0, 2.0, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_auxiliary_statistic(0.0, 2.0, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: count beyond the supported domain ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_auxiliary_statistic(1e200, 1e200, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_auxiliary_statistic(1e200, 1e200, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: tau <= 0 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_auxiliary_statistic(3.0, 2.0, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_auxiliary_statistic(3.0, 2.0, -1.0)
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
