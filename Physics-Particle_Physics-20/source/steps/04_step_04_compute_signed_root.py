"""
Compute the signed likelihood-ratio root r(0) for a test of s = 0 in the two-measurement on/off model.

The signed likelihood-ratio root tests the zero-signal hypothesis in the two-measurement on/off model. Positive values correspond to an observed excess, negative values to a deficit, and exact balance has value zero. Inputs are restricted to the documented domain with positive counts and exposure ratio at most 1e12. The returned root must retain numerical accuracy when large counts are nearly balanced as well as for the small-count search channels.

Returns
-------
float, the signed likelihood-ratio root r(0) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_signed_root(n: float, m: float, tau: float) -> float:
    '''Signed likelihood-ratio root r(0) of the on/off model.

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
    r0 : float
        The signed root r(0), positive when n > m/tau and negative when
        n < m/tau, as a native Python float.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _poisson_deviance(a: float, e: float) -> float:
    if a == 0.0:
        return 2.0 * e
    x = (a - e) / e
    if abs(x) < 1.0e-4:
        return 2.0 * e * (x * x / 2.0 - x ** 3 / 6.0 + x ** 4 / 12.0 - x ** 5 / 20.0)
    return 2.0 * (a * math.log(a / e) - (a - e))


def _oracle_compute_signed_root(n: float, m: float, tau: float) -> float:
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
    b00 = _oracle_compute_profiled_background(n, m, tau, 0.0)
    q = _poisson_deviance(n, b00) + _poisson_deviance(m, tau * b00)
    if q < 0.0:
        q = 0.0
    d = n - m / tau
    return 0.0 if d == 0.0 else float(math.copysign(math.sqrt(q), d))

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
            "call": "compute_signed_root(n, m, tau)",
            "gold_call": "_oracle_compute_signed_root(n, m, tau)",
        },
        # --- Valid: boundary, n exactly equal to m/tau gives r(0) = 0 ---
        {
            "setup": """import numpy as np
n, m, tau = 2.0, 4.0, 2.0
""",
            "call": "compute_signed_root(n, m, tau)",
            "gold_call": "_oracle_compute_signed_root(n, m, tau)",
        },
        # --- Valid: edge, deficit (n < m/tau) gives a negative root ---
        {
            "setup": """import numpy as np
n, m, tau = 1.5, 6.0, 2.0
""",
            "call": "compute_signed_root(n, m, tau)",
            "gold_call": "_oracle_compute_signed_root(n, m, tau)",
        },
        # --- Valid: edge, large nearly-equal counts (cancellation test) ---
        {
            "setup": """import numpy as np
n, m, tau = 100000001.0, 100000000.0, 1.0
""",
            "call": "compute_signed_root(n, m, tau)",
            "gold_call": "_oracle_compute_signed_root(n, m, tau)",
        },
        # --- Invalid: m = 0 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_signed_root(3.0, 0.0, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_signed_root(3.0, 0.0, 1.5)
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
        compute_signed_root(1e13, 1e12, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_signed_root(1e13, 1e12, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite input ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_signed_root(float('nan'), 2.0, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_signed_root(float('nan'), 2.0, 1.5)
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
