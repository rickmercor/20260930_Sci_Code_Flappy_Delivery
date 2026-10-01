"""
Return the surrogate utility's affine pieces for M tangent points placed on y_range as in the problem statement.

Row 0 holds the slopes and row 1 the intercepts, in increasing order of tangent point.

Returns
-------
np.ndarray of shape (2, M), dtype float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def tangent_coefficients(y_range: np.ndarray, M: int) -> np.ndarray:
    '''Return the affine pieces of the surrogate utility.

    Parameters
    ----------
    y_range : np.ndarray
        Shape (2,). Interval carrying the tangent points.
    M : int
        Number of tangent points.

    Returns
    -------
    coeffs : np.ndarray
        Shape (2, M), dtype float.

    Raises
    ------
    ValueError
        If y_range is not a finite array of shape (2,) with
        -1 < y_range[0] < y_range[1], or if M is not an integer with M >= 2.
    '''
    return coeffs  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_tangent_coefficients(y_range: np.ndarray, M: int) -> np.ndarray:
    """Reference implementation."""
    y_range = np.asarray(y_range, dtype=float)
    if y_range.shape != (2,):
        raise ValueError("y_range must have shape (2,)")
    if not np.all(np.isfinite(y_range)):
        raise ValueError("y_range must be finite")
    y_lo, y_hi = float(y_range[0]), float(y_range[1])
    if y_lo >= y_hi:
        raise ValueError("y_range must be a nondegenerate interval")
    if y_lo <= -1.0:
        raise ValueError("the utility must be finite on the interval")
    if isinstance(M, bool) or not isinstance(M, (int, np.integer)):
        raise ValueError("M must be an integer")
    M = int(M)
    if M < 2:
        raise ValueError("at least two tangent lines are required")

    points = np.linspace(y_lo, y_hi, M)
    alpha = 1.0 / (1.0 + points)
    beta = np.log1p(points) - alpha * points
    return np.vstack([alpha, beta])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    invalid = """
def run_model():
    try:
        tangent_coefficients(y_range.copy(), M)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_tangent_coefficients(y_range, M)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        # --- Valid: the production interval and hyperplane count ---
        {"setup": "import numpy as np\ny_range = np.array([-0.26444, 0.21543])\nM = 9\n",
         "call": "tangent_coefficients(y_range.copy(), M)",
         "gold_call": "_oracle_tangent_coefficients(y_range, M)"},
        # --- Valid: the interval of the six-asset projection ---
        {"setup": "import numpy as np\ny_range = np.array([-0.2215, 0.166])\nM = 9\n",
         "call": "tangent_coefficients(y_range.copy(), M)",
         "gold_call": "_oracle_tangent_coefficients(y_range, M)"},
        # --- Valid: boundary case M = 2, endpoints only ---
        {"setup": "import numpy as np\ny_range = np.array([-0.26444, 0.21543])\nM = 2\n",
         "call": "tangent_coefficients(y_range.copy(), M)",
         "gold_call": "_oracle_tangent_coefficients(y_range, M)"},
        # --- Valid: interval close to the domain limit, large curvature ---
        {"setup": "import numpy as np\ny_range = np.array([-0.97, -0.02])\nM = 15\n",
         "call": "tangent_coefficients(y_range.copy(), M)",
         "gold_call": "_oracle_tangent_coefficients(y_range, M)"},
        # --- Valid: interval straddling zero, one intercept vanishes ---
        {"setup": "import numpy as np\ny_range = np.array([-0.4, 0.4])\nM = 5\n",
         "call": "tangent_coefficients(y_range.copy(), M)",
         "gold_call": "_oracle_tangent_coefficients(y_range, M)"},
        # --- Valid: strictly positive interval, wide upper endpoint ---
        {"setup": "import numpy as np\ny_range = np.array([0.05, 6.5])\nM = 7\n",
         "call": "tangent_coefficients(y_range.copy(), M)",
         "gold_call": "_oracle_tangent_coefficients(y_range, M)"},
        # --- Valid: narrow interval, closely spaced tangent points ---
        {"setup": "import numpy as np\ny_range = np.array([0.0999, 0.1001])\nM = 4\n",
         "call": "tangent_coefficients(y_range.copy(), M)",
         "gold_call": "_oracle_tangent_coefficients(y_range, M)"},
        # --- Invalid: fewer than two tangent lines ---
        {"setup": "import numpy as np\ny_range = np.array([-0.26444, 0.21543])\nM = 1\n" + invalid,
         "call": "run_model()", "gold_call": "run_gold()"},
        # --- Invalid: degenerate interval ---
        {"setup": "import numpy as np\ny_range = np.array([0.1, 0.1])\nM = 5\n" + invalid,
         "call": "run_model()", "gold_call": "run_gold()"},
        # --- Invalid: utility not finite on the interval ---
        {"setup": "import numpy as np\ny_range = np.array([-1.5, 0.2])\nM = 6\n" + invalid,
         "call": "run_model()", "gold_call": "run_gold()"},
        # --- Invalid: wrong shape for the interval ---
        {"setup": "import numpy as np\ny_range = np.array([-0.26444, 0.0, 0.21543])\nM = 5\n" + invalid,
         "call": "run_model()", "gold_call": "run_gold()"},
    ]
