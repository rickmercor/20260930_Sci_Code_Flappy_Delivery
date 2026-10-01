"""
Compute the L1 coefficients that approximate the Caputo derivative of order alpha at one time level of a uniform grid.

The L1 formula replaces u, inside the Caputo integral of order alpha, by its piecewise-linear interpolant on the uniform grid tau_m = m delta. This turns the derivative at level n into delta**(-alpha) * sum_{j=0}^{n} l_{j,n} u(tau_{n-j}). The coefficients returned here include the factor 1/Gamma(2 - alpha) but not delta**(-alpha). At alpha = 1 they must be exactly those of the backward difference, (1, -1, 0, ..., 0).

Returns
-------
np.ndarray, float, shape (n + 1,): the coefficients l_{0,n}, ..., l_{n,n}, including the factor 1/Gamma(2 - alpha) but not delta**(-alpha).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def l1_caputo_coefficients(alpha: float, n_level: int) -> np.ndarray:
    '''L1 coefficients of the Caputo derivative at time level n.

    Parameters
    ----------
    alpha : float
        Order of the Caputo derivative, 0 < alpha <= 1.
    n_level : int
        Index n >= 1 of the time level at which the derivative is taken.

    Returns
    -------
    coefficients : np.ndarray
        Shape (n_level + 1,) float array; entry j multiplies u(tau_{n-j}).

    Raises
    ------
    ValueError
        If alpha is outside (0, 1] or n_level is not an integer >= 1.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros(int(n_level) + 1, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_l1_caputo_coefficients(alpha: float, n_level: int) -> np.ndarray:
    import numpy as np
    from math import gamma

    if isinstance(alpha, bool) or not np.isfinite(float(alpha)) or not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    if isinstance(n_level, bool) or not isinstance(n_level, (int, np.integer)) or int(n_level) < 1:
        raise ValueError("n_level must be an integer >= 1")
    alpha = float(alpha)
    n = int(n_level)
    p = 1.0 - alpha

    # j**(1-alpha) with 0**(1-alpha) read as its limit 0 for all alpha in (0,1].
    j = np.arange(0, n + 2, dtype=float)
    pw = np.where(j > 0.0, j ** p, 0.0)

    coeff = np.empty(n + 1)
    coeff[0] = 1.0
    if n >= 2:
        k = np.arange(1, n)
        coeff[1:n] = pw[k - 1] - 2.0 * pw[k] + pw[k + 1]
    coeff[n] = pw[n - 1] - pw[n]
    return coeff / gamma(2.0 - alpha)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned exactness: L1 interpolates linearly between levels, so for
        # u(t) = 3 + 2 t the discrete derivative equals the exact Caputo
        # derivative 2 t**(1 - alpha) / Gamma(2 - alpha); the constant drops
        # out because the coefficients sum to zero. A unit step keeps the
        # factor delta**(-alpha) at one, so round-off is not amplified.
        {
            "setup": """import numpy as np
from math import gamma
alpha, n, delta = 0.8, 5, 1.0
u = 3.0 + 2.0 * delta * np.arange(n, -1, -1)   # u(tau_{n-j}) for j = 0..n
EXPECTED = 2.0 * (n * delta) ** (1.0 - alpha) / gamma(2.0 - alpha)
def apply(fn):
    return float(delta ** (-alpha) * np.dot(fn(alpha, n), u))
""",
            "call": "apply(l1_caputo_coefficients)",
            "gold_call": "EXPECTED",
        },
        # --- Pinned limit: alpha = 1 must give the backward difference.
        {
            "setup": """import numpy as np
EXPECTED = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
""",
            "call": "l1_caputo_coefficients(1.0, 5)",
            "gold_call": "EXPECTED",
        },
        # --- Normal: the benchmark order at the final level of a 400-step grid.
        {
            "setup": """import numpy as np
""",
            "call": "l1_caputo_coefficients(0.8, 400)",
            "gold_call": "_oracle_l1_caputo_coefficients(0.8, 400)",
        },
        # --- Boundary: the first level, where the history has one term.
        {
            "setup": """import numpy as np
""",
            "call": "l1_caputo_coefficients(0.35, 1)",
            "gold_call": "_oracle_l1_caputo_coefficients(0.35, 1)",
        },
        # --- Edge: a very weak derivative order with a long history.
        {
            "setup": """import numpy as np
""",
            "call": "l1_caputo_coefficients(0.05, 37)",
            "gold_call": "_oracle_l1_caputo_coefficients(0.05, 37)",
        },
        # --- Invalid: order above one ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        l1_caputo_coefficients(1.2, 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_l1_caputo_coefficients(1.2, 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: level zero ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        l1_caputo_coefficients(0.5, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_l1_caputo_coefficients(0.5, 0)
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
