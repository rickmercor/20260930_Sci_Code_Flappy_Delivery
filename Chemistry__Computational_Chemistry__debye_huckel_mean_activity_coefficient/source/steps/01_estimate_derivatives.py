"""
Estimate concentration derivatives on a nonuniform time grid.

SISR fits candidate mechanisms against time derivatives, so uneven sampling must be handled explicitly.

Returns
-------
np.ndarray with the same shape as concentrations.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_derivatives(times: np.ndarray, concentrations: np.ndarray) -> np.ndarray:
    """Estimate concentration derivatives on a nonuniform time grid.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing one-dimensional sampling times.
    concentrations : np.ndarray
        Concentrations with shape (n_times, n_species).

    Returns
    -------
    np.ndarray
        Derivative estimates with the same shape as concentrations.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_estimate_derivatives(times: np.ndarray, concentrations: np.ndarray) -> np.ndarray:
    """Reference nonuniform finite-difference implementation."""
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)

    if t.ndim != 1 or t.size < 3:
        raise ValueError("times must be one-dimensional with at least three entries")
    if c.ndim != 2 or c.shape[0] != t.size or c.shape[1] < 1:
        raise ValueError("concentrations must have shape (n_times, n_species)")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(c)):
        raise ValueError("inputs must be finite")

    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("times must be strictly increasing")

    derivative = np.empty_like(c, dtype=float)
    derivative[0] = (c[1] - c[0]) / dt[0]
    derivative[-1] = (c[-1] - c[-2]) / dt[-1]

    h_minus = t[1:-1] - t[:-2]
    h_plus = t[2:] - t[1:-1]

    left = -h_plus / (h_minus * (h_minus + h_plus))
    center = (h_plus - h_minus) / (h_minus * h_plus)
    right = h_minus / (h_plus * (h_minus + h_plus))

    derivative[1:-1] = (
        left[:, None] * c[:-2]
        + center[:, None] * c[1:-1]
        + right[:, None] * c[2:]
    )
    return derivative

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
times = np.array([0.0, 0.4, 1.1, 2.0])
concentrations = np.column_stack((times**2, 2.0 * times + 1.0))""",
            "call": "estimate_derivatives(times, concentrations)",
            "gold_call": "_oracle_estimate_derivatives(times, concentrations)",
        },
        {
            "setup": """import numpy as np
times = np.array([0.0, 0.1, 0.9])
concentrations = np.array([[1.0], [0.8], [0.2]])""",
            "call": "estimate_derivatives(times, concentrations)",
            "gold_call": "_oracle_estimate_derivatives(times, concentrations)",
        },
        {
            "setup": """import numpy as np
times = np.array([2.0, 2.3, 2.9, 4.2, 7.0])
concentrations = np.column_stack((np.exp(-times), np.sin(times), np.ones_like(times)))""",
            "call": "estimate_derivatives(times, concentrations)",
            "gold_call": "_oracle_estimate_derivatives(times, concentrations)",
        },
    ]
