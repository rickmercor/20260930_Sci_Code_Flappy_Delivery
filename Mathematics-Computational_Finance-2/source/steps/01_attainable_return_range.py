"""
Return the problem statement's interval [y_lo, y_hi] for the return box [xmin, xmax] and per-asset cap `cap`.

The interval carries the tangent points of the surrogate utility.

Returns
-------
np.ndarray of shape (2,), dtype float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def attainable_return_range(xmin: np.ndarray, xmax: np.ndarray,
                            cap: float) -> np.ndarray:
    '''Return [y_lo, y_hi].

    Parameters
    ----------
    xmin : np.ndarray
        Shape (n,). Lower bounds of the return box.
    xmax : np.ndarray
        Shape (n,). Upper bounds of the return box.
    cap : float
        Per-asset weight cap.

    Returns
    -------
    y_range : np.ndarray
        Shape (2,), dtype float.

    Raises
    ------
    ValueError
        If xmin and xmax are not finite one-dimensional arrays of a common
        length n >= 1 with -1 < xmin < xmax elementwise, or if cap is not a
        finite scalar with cap > 0 and n * cap >= 1.
    '''
    return y_range  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _ref_check_box(xmin, xmax):
    """Validate the box bounds and return them as float arrays."""
    xmin = np.asarray(xmin, dtype=float)
    xmax = np.asarray(xmax, dtype=float)
    if xmin.ndim != 1 or xmax.ndim != 1:
        raise ValueError("xmin and xmax must be one-dimensional arrays")
    if xmin.size != xmax.size or xmin.size < 1:
        raise ValueError("xmin and xmax must have the same length n >= 1")
    if not (np.all(np.isfinite(xmin)) and np.all(np.isfinite(xmax))):
        raise ValueError("xmin and xmax must be finite")
    if np.any(xmax <= xmin):
        raise ValueError("the box must be nonempty in every coordinate")
    if np.any(xmin <= -1.0):
        raise ValueError("every lower bound must exceed -1")
    return xmin, xmax


def _oracle_attainable_return_range(xmin: np.ndarray, xmax: np.ndarray,
                                    cap: float) -> np.ndarray:
    """Reference implementation."""
    xmin, xmax = _ref_check_box(xmin, xmax)
    if not np.isscalar(cap) or not np.isfinite(float(cap)):
        raise ValueError("cap must be a finite scalar")
    cap = float(cap)
    n = xmin.size
    if cap <= 0.0 or n * cap < 1.0:
        raise ValueError("the admissible portfolio set must be nonempty")

    endpoints = []
    for values, largest in ((xmin, False), (xmax, True)):
        order = np.argsort(-values if largest else values, kind="stable")
        remaining = 1.0
        total = 0.0
        for idx in order:
            if remaining <= 0.0:
                break
            take = cap if cap < remaining else remaining
            total += take * values[idx]
            remaining -= take
        endpoints.append(total)
    return np.array(endpoints, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    prod = """import numpy as np
i = np.arange(1000)
xmin = -(150 + (37 * i) % 121) / 1000.0
xmax = (120 + (53 * i) % 101) / 1000.0
"""
    proj = """import numpy as np
i = np.arange(6)
xmin = -(150 + (37 * i) % 121) / 1000.0
xmax = (120 + (53 * i) % 101) / 1000.0
"""
    invalid = """
def run_model():
    try:
        attainable_return_range(xmin.copy(), xmax.copy(), cap)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_attainable_return_range(xmin, xmax, cap)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        # --- Valid: the production configuration, cap forcing 100 assets ---
        {"setup": prod + "cap = 0.01\n",
         "call": "attainable_return_range(xmin.copy(), xmax.copy(), cap)",
         "gold_call": "_oracle_attainable_return_range(xmin, xmax, cap)"},
        # --- Valid: cap = 1, the uncapped simplex ---
        {"setup": prod + "cap = 1.0\n",
         "call": "attainable_return_range(xmin.copy(), xmax.copy(), cap)",
         "gold_call": "_oracle_attainable_return_range(xmin, xmax, cap)"},
        # --- Valid: boundary case cap = 1/n, equal weight forced ---
        {"setup": prod + "cap = 0.001\n",
         "call": "attainable_return_range(xmin.copy(), xmax.copy(), cap)",
         "gold_call": "_oracle_attainable_return_range(xmin, xmax, cap)"},
        # --- Valid: the six-asset projection used later for validation ---
        {"setup": proj + "cap = 0.25\n",
         "call": "attainable_return_range(xmin.copy(), xmax.copy(), cap)",
         "gold_call": "_oracle_attainable_return_range(xmin, xmax, cap)"},
        # --- Valid: budget not an integer multiple of the cap ---
        {"setup": """import numpy as np
xmin = np.array([-0.42, -0.11, -0.37, -0.05, -0.28])
xmax = np.array([0.09, 0.31, 0.14, 0.27, 0.02])
cap = 0.3
""",
         "call": "attainable_return_range(xmin.copy(), xmax.copy(), cap)",
         "gold_call": "_oracle_attainable_return_range(xmin, xmax, cap)"},
        # --- Valid: repeated bounds, extremal weights not unique ---
        {"setup": """import numpy as np
xmin = np.array([-0.25, -0.25, -0.25, -0.10, -0.25, -0.10])
xmax = np.array([0.20, 0.20, 0.05, 0.20, 0.05, 0.20])
cap = 0.35
""",
         "call": "attainable_return_range(xmin.copy(), xmax.copy(), cap)",
         "gold_call": "_oracle_attainable_return_range(xmin, xmax, cap)"},
        # --- Valid: n = 1 edge case ---
        {"setup": """import numpy as np
xmin = np.array([-0.4])
xmax = np.array([0.6])
cap = 1.0
""",
         "call": "attainable_return_range(xmin.copy(), xmax.copy(), cap)",
         "gold_call": "_oracle_attainable_return_range(xmin, xmax, cap)"},
        # --- Invalid: cap too small, admissible set empty ---
        {"setup": "import numpy as np\n"
                  "xmin = np.array([-0.2, -0.3, -0.25])\n"
                  "xmax = np.array([0.2, 0.3, 0.25])\n"
                  "cap = 0.3\n" + invalid,
         "call": "run_model()", "gold_call": "run_gold()"},
        # --- Invalid: nonpositive cap ---
        {"setup": "import numpy as np\n"
                  "xmin = np.array([-0.2, -0.3, -0.25])\n"
                  "xmax = np.array([0.2, 0.3, 0.25])\n"
                  "cap = 0.0\n" + invalid,
         "call": "run_model()", "gold_call": "run_gold()"},
        # --- Invalid: a lower bound below -1 ---
        {"setup": "import numpy as np\n"
                  "xmin = np.array([-1.2, -0.3])\n"
                  "xmax = np.array([0.2, 0.3])\n"
                  "cap = 0.75\n" + invalid,
         "call": "run_model()", "gold_call": "run_gold()"},
        # --- Invalid: an empty coordinate of the box ---
        {"setup": "import numpy as np\n"
                  "xmin = np.array([-0.2, 0.1, -0.3])\n"
                  "xmax = np.array([0.2, 0.1, 0.3])\n"
                  "cap = 0.5\n" + invalid,
         "call": "run_model()", "gold_call": "run_gold()"},
    ]
