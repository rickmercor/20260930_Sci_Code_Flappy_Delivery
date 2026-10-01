"""
Find the lowest interior maximum of the effective potential that governs the autocorrelation of the disorder-induced input, for a given equal-time correlation.

The autocorrelation obeys Newtonian motion in an effective potential whose slope is the equal-time value's pair average of the response minus the lag value, and a decaying autocorrelation can only come to rest on a hilltop of that potential.

Returns
-------
float: lag value of the lowest interior hilltop of the effective potential, or nan when there is none.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_correlation_hilltop(delta_zero: float, fields: "np.ndarray", weights: "np.ndarray",
                               beta: float, nodes: int = 241, scan_points: int = 17) -> float:
    """Return the lowest lag value at which the potential slope turns from rising to falling.

    The slope of the effective potential at lag value ``d`` is

    ``slope(d) = average_correlation_overlap(np.array([d]), delta_zero, fields, weights, beta, nodes)[0] - d``.

    It is sampled at ``scan_points`` equally spaced lag values from ``0`` to
    ``delta_zero`` inclusive. For the first consecutive pair of samples with
    ``slope > 0`` followed by ``slope <= 0``, the root inside that pair is refined
    to double precision with Brent's method and returned (a later sample whose
    slope is exactly zero is itself that root). If no such pair exists the
    potential has no interior hilltop and ``nan`` is returned.

    Parameters
    ----------
    delta_zero : float
        Positive finite equal-time correlation.
    fields : np.ndarray
        Non-empty one-dimensional array of finite constant regulatory inputs.
    weights : np.ndarray
        Nonnegative population shares of the same length as ``fields``, summing
        to one within ``1e-9``.
    beta : float
        Positive finite response gain.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3.
    scan_points : int
        Number of equally spaced scan samples, at least 3 (booleans are rejected).

    Returns
    -------
    float
        The hilltop lag value, or ``nan`` when there is none.

    Raises
    ------
    ValueError
        If ``delta_zero`` is not positive and finite, if ``scan_points`` is not an
        integer of at least 3, or if ``fields``, ``weights``, ``beta`` or ``nodes``
        fails the contract of ``average_correlation_overlap``.
    """
    return hilltop

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_locate_correlation_hilltop(delta_zero: float, fields: "np.ndarray",
                                       weights: "np.ndarray", beta: float, nodes: int = 241,
                                       scan_points: int = 17) -> float:
    """Reference implementation (vectorised slope scan, then Brent refinement)."""
    import numpy as np

    equal_time = _epi_require_scalar(delta_zero, "delta_zero", 0.0, True)
    samples = _epi_require_count(scan_points, "scan_points", 3)

    def _slope(value):
        return float(_oracle_average_correlation_overlap(np.array([value]), equal_time, fields,
                                                         weights, beta, nodes)[0]) - value

    lags = np.linspace(0.0, equal_time, samples)
    lags[-1] = equal_time
    # The scan uses the same scalar evaluation as the refinement, so the bracket signs agree.
    slopes = [_slope(float(lag)) for lag in lags]
    for index in range(samples - 1):
        if slopes[index] > 0.0 >= slopes[index + 1]:
            # Brent's method returns an endpoint whose slope is exactly zero as the root.
            return float(brentq(_slope, lags[index], lags[index + 1], xtol=1e-15,
                                rtol=4.0 * float(np.finfo(float).eps)))
    return float("nan")

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
        # --- Normal: equal-time correlation where a clear hilltop exists ---
        {
            "setup": ensemble,
            "call": "locate_correlation_hilltop(0.81, fields, shares, 4.0, 81)",
            "gold_call": "_oracle_locate_correlation_hilltop(0.81, fields, shares, 4.0, 81)",
        },
        # --- Boundary: equal-time correlation near where the hilltop is born ---
        {
            "setup": ensemble,
            "call": "locate_correlation_hilltop(0.797, fields, shares, 4.0, 81, 33)",
            "gold_call": "_oracle_locate_correlation_hilltop(0.797, fields, shares, 4.0, 81, 33)",
        },
        # --- Edge: small equal-time correlation, no hilltop, reported as a flag ---
        {
            "setup": ensemble,
            "call": "float(np.isnan(locate_correlation_hilltop(0.3, fields, shares, 4.0, 61)))",
            "gold_call": "float(np.isnan(_oracle_locate_correlation_hilltop(0.3, fields, shares, 4.0, 61)))",
        },
        # --- Normal: feedback-free inputs, a low-lying hilltop ---
        {
            "setup": "import numpy as np\nfields = np.array([-0.32, -0.1, 0.04, 0.11, 0.35])\nshares = np.array([0.14, 0.22, 0.27, 0.21, 0.16])\n",
            "call": "locate_correlation_hilltop(0.7, fields, shares, 4.0, 81)",
            "gold_call": "_oracle_locate_correlation_hilltop(0.7, fields, shares, 4.0, 81)",
        },
        # --- Edge: zero constant input, the slope starts at zero so no interior hilltop is reported ---
        {
            "setup": "import numpy as np\nfields = np.array([0.0])\nshares = np.array([1.0])\n",
            "call": "float(np.isnan(locate_correlation_hilltop(0.5, fields, shares, 4.0, 61)))",
            "gold_call": "float(np.isnan(_oracle_locate_correlation_hilltop(0.5, fields, shares, 4.0, 61)))",
        },
        # --- Invalid: zero equal-time correlation ---
        {
            "setup": status + "fields = np.array([0.5, -0.5])\nshares = np.array([0.5, 0.5])\n",
            "call": "_status(lambda: locate_correlation_hilltop(0.0, fields, shares, 4.0, 21))",
            "gold_call": "_status(lambda: _oracle_locate_correlation_hilltop(0.0, fields, shares, 4.0, 21))",
        },
        # --- Invalid: too few scan samples ---
        {
            "setup": status + "fields = np.array([0.5, -0.5])\nshares = np.array([0.5, 0.5])\n",
            "call": "_status(lambda: locate_correlation_hilltop(0.7, fields, shares, 4.0, 21, 2))",
            "gold_call": "_status(lambda: _oracle_locate_correlation_hilltop(0.7, fields, shares, 4.0, 21, 2))",
        },
    ]
