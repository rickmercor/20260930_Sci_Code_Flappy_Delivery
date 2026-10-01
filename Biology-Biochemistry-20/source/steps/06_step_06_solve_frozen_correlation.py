"""
Solve the self-consistency condition of a time-independent collective state for the equal-time correlation of the disorder-induced part of a gene's regulatory input.

When the collective state carries no time dependence, the two ends of every lag coincide, so the correlation of the Gaussian input must reproduce itself through the population-averaged squared response.

Returns
-------
float: self-consistent equal-time correlation of the time-independent state, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_frozen_correlation(fields: "np.ndarray", weights: "np.ndarray", beta: float,
                             lower: float, upper: float, nodes: int = 241) -> float:
    """Return the equal-time correlation of a time-independent collective state.

    The state is time-independent when the equal-time residual

    ``average_correlation_overlap(np.array([d]), d, fields, weights, beta, nodes)[0] - d``

    vanishes. The root is located in ``[lower, upper]`` with Brent's method to
    double precision. The residual must not take the same non-zero sign at the
    two ends; if it is exactly zero at an end, that end is returned, and
    ``lower`` wins when it is exactly zero at both.

    Parameters
    ----------
    fields : np.ndarray
        Non-empty one-dimensional array of finite constant regulatory inputs.
    weights : np.ndarray
        Nonnegative population shares of the same length as ``fields``, summing
        to one within ``1e-9``.
    beta : float
        Positive finite response gain.
    lower : float
        Nonnegative finite lower end of the search bracket.
    upper : float
        Finite upper end of the search bracket, greater than ``lower``.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3.

    Returns
    -------
    float
        The self-consistent equal-time correlation, a native Python float.

    Raises
    ------
    ValueError
        If ``lower`` is negative or not finite, if ``upper`` is not finite or not
        greater than ``lower``, if the residual has the same non-zero sign at
        both ends, or if ``fields``, ``weights``, ``beta`` or ``nodes`` fails the
        contract of ``average_correlation_overlap``.
    """
    return correlation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_solve_frozen_correlation(fields: "np.ndarray", weights: "np.ndarray", beta: float,
                                     lower: float, upper: float, nodes: int = 241) -> float:
    """Reference implementation (Brent's method on the equal-time residual)."""
    import numpy as np

    low = _epi_require_scalar(lower, "lower", 0.0, False)
    high = _epi_require_scalar(upper, "upper", -np.inf, False)
    if not high > low:
        raise ValueError("upper must be greater than lower")

    def _residual(value):
        overlap = _oracle_average_correlation_overlap(np.array([value]), value, fields,
                                                      weights, beta, nodes)
        return float(overlap[0]) - value

    low_value = _residual(low)
    high_value = _residual(high)
    if low_value * high_value > 0.0:
        raise ValueError("the residual does not change sign across the bracket")
    if low_value == 0.0 or high_value == 0.0:
        return low if low_value == 0.0 else high
    return float(brentq(_residual, low, high, xtol=1e-15,
                        rtol=4.0 * float(np.finfo(float).eps)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    ensemble = (
        "import numpy as np\n"
        "fields = np.array([-0.81857, -0.591251, 0.336601, -0.42859, 0.525256, -0.31618, 0.601964, 0.848877])\n"
        "shares = np.array([0.14, 0.11946, 0.10054, 0.13068, 0.13932, 0.09492, 0.11508, 0.16])\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(action):\n"
        "    try:\n"
        "        action()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # --- Normal: wide bracket for a sharp-gain mixture ---
        {
            "setup": ensemble,
            "call": "solve_frozen_correlation(fields, shares, 4.0, 0.0, 1.0, 81)",
            "gold_call": "_oracle_solve_frozen_correlation(fields, shares, 4.0, 0.0, 1.0, 81)",
        },
        # --- Boundary: narrow bracket around the same root ---
        {
            "setup": ensemble,
            "call": "solve_frozen_correlation(fields, shares, 4.0, 0.8, 0.85, 81)",
            "gold_call": "_oracle_solve_frozen_correlation(fields, shares, 4.0, 0.8, 0.85, 81)",
        },
        # --- Boundary: a population without constant input, residual exactly zero at the lower end ---
        {
            "setup": "import numpy as np\nfields = np.array([0.0])\nshares = np.array([1.0])\n",
            "call": "solve_frozen_correlation(fields, shares, 0.6, 0.0, 1.0, 41)",
            "gold_call": "_oracle_solve_frozen_correlation(fields, shares, 0.6, 0.0, 1.0, 41)",
        },
        # --- Edge: weak gain drives the correlation close to zero ---
        {
            "setup": "import numpy as np\nfields = np.array([0.35])\nshares = np.array([1.0])\n",
            "call": "solve_frozen_correlation(fields, shares, 0.4, 0.0, 1.0, 81)",
            "gold_call": "_oracle_solve_frozen_correlation(fields, shares, 0.4, 0.0, 1.0, 81)",
        },
        # --- Normal: a strongly saturated two-population mixture ---
        {
            "setup": "import numpy as np\nfields = np.array([-1.2, 1.9])\nshares = np.array([0.4, 0.6])\n",
            "call": "solve_frozen_correlation(fields, shares, 2.0, 0.0, 1.0, 121)",
            "gold_call": "_oracle_solve_frozen_correlation(fields, shares, 2.0, 0.0, 1.0, 121)",
        },
        # --- Invalid: bracket that misses the root ---
        {
            "setup": status + "fields = np.array([-0.81857, 0.848877])\nshares = np.array([0.5, 0.5])\n",
            "call": "_status(lambda: solve_frozen_correlation(fields, shares, 4.0, 0.95, 1.0, 41))",
            "gold_call": "_status(lambda: _oracle_solve_frozen_correlation(fields, shares, 4.0, 0.95, 1.0, 41))",
        },
        # --- Invalid: inverted bracket ---
        {
            "setup": status + "fields = np.array([0.5, -0.5])\nshares = np.array([0.5, 0.5])\n",
            "call": "_status(lambda: solve_frozen_correlation(fields, shares, 4.0, 0.8, 0.2, 41))",
            "gold_call": "_status(lambda: _oracle_solve_frozen_correlation(fields, shares, 4.0, 0.8, 0.2, 41))",
        },
    ]
