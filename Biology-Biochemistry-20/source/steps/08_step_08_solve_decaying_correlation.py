"""
Solve for the equal-time correlation and the infinite-lag limit of a decaying autocorrelation, the solution that starts at rest and climbs asymptotically onto a hilltop of the effective potential.

Released from rest at its equal-time value, the autocorrelation conserves its Newtonian energy, so it can approach a hilltop only when the potential there equals the potential at the starting point; the hilltop is then the static part of the correlation.

Returns
-------
np.ndarray: float array [equal-time correlation, infinite-lag limit] of the decaying autocorrelation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_decaying_correlation(fields: "np.ndarray", weights: "np.ndarray", beta: float,
                               lower: float, upper: float, nodes: int = 241,
                               scan_points: int = 17) -> "np.ndarray":
    """Return the equal-time correlation and infinite-lag limit of the decaying autocorrelation.

    For an equal-time correlation ``d0`` the effective potential at lag value
    ``d`` is ``V(d) = average_potential_overlap(np.array([d]), d0, ...)[0] - d**2 / 2``.
    With ``h = locate_correlation_hilltop(d0, fields, weights, beta, nodes, scan_points)``
    the energy gap is ``V(d0) - V(h)``; when ``h`` is ``nan`` (no hilltop) the gap is
    taken as ``+1``. The gap must be positive at ``lower`` and negative at
    ``upper``; its sign change is located with Brent's method to double
    precision, giving ``d0``. The result is ``[d0, h(d0)]``.

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
        Positive finite lower end of the search bracket.
    upper : float
        Finite upper end of the search bracket, greater than ``lower``.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3.
    scan_points : int
        Number of hilltop scan samples, at least 3.

    Returns
    -------
    np.ndarray
        Float array ``[equal_time_correlation, infinite_lag_limit]``.

    Raises
    ------
    ValueError
        If ``lower`` is not positive and finite, if ``upper`` is not finite or not
        greater than ``lower``, if the gap is not positive at ``lower`` and
        negative at ``upper``, or if an argument fails the contract of
        ``locate_correlation_hilltop`` or ``average_potential_overlap``.
    """
    return solution

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_solve_decaying_correlation(fields: "np.ndarray", weights: "np.ndarray", beta: float,
                                       lower: float, upper: float, nodes: int = 241,
                                       scan_points: int = 17) -> "np.ndarray":
    """Reference implementation (Brent's method on the energy gap)."""
    import numpy as np

    low = _epi_require_scalar(lower, "lower", 0.0, True)
    high = _epi_require_scalar(upper, "upper", -np.inf, False)
    if not high > low:
        raise ValueError("upper must be greater than lower")

    def _gap(equal_time):
        top = _oracle_locate_correlation_hilltop(equal_time, fields, weights, beta, nodes,
                                                 scan_points)
        if np.isnan(top):
            return 1.0
        lags = np.array([top, equal_time])
        potential = (_oracle_average_potential_overlap(lags, equal_time, fields, weights, beta,
                                                       nodes) - 0.5 * lags ** 2)
        return float(potential[1] - potential[0])

    if not (_gap(low) > 0.0 > _gap(high)):
        raise ValueError("the energy gap must be positive at lower and negative at upper")
    equal_time = float(brentq(_gap, low, high, xtol=1e-15,
                              rtol=4.0 * float(np.finfo(float).eps)))
    top = _oracle_locate_correlation_hilltop(equal_time, fields, weights, beta, nodes,
                                             scan_points)
    return np.array([equal_time, top])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _digest(values, rows, cols):\n"
        "    array = np.asarray(values, dtype=float)\n"
        "    if cols == 0:\n"
        "        if array.ndim != 1 or array.shape[0] != rows:\n"
        "            return -1.0\n"
        "    elif array.shape != (rows, cols):\n"
        "        return -1.0\n"
        "    flat = array.ravel()\n"
        "    phase = np.cos(np.arange(flat.size, dtype=float) + 1.0)\n"
        "    return float(flat.size) + float(np.sum(np.abs(flat)) + flat @ phase)\n"
    )
    ensemble = (
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
        # --- Normal: sharp-gain mixture, wide bracket crossing the hilltop's birth ---
        {
            "setup": digest + ensemble,
            "call": "_digest(solve_decaying_correlation(fields, shares, 4.0, 0.41, 0.82, 61), 2, 0)",
            "gold_call": "_digest(_oracle_solve_decaying_correlation(fields, shares, 4.0, 0.41, 0.82, 61), 2, 0)",
        },
        # --- Boundary: bracket tight around the root, both ends carrying a hilltop ---
        {
            "setup": digest + ensemble,
            "call": "_digest(solve_decaying_correlation(fields, shares, 4.0, 0.796, 0.81, 61), 2, 0)",
            "gold_call": "_digest(_oracle_solve_decaying_correlation(fields, shares, 4.0, 0.796, 0.81, 61), 2, 0)",
        },
        # --- Edge: bare inputs without feedback, a much lower static correlation ---
        {
            "setup": digest + "fields = np.array([-0.32, -0.1, 0.04, 0.11, 0.35])\nshares = np.array([0.14, 0.22, 0.27, 0.21, 0.16])\n",
            "call": "_digest(solve_decaying_correlation(fields, shares, 4.0, 0.4, 0.786, 61), 2, 0)",
            "gold_call": "_digest(_oracle_solve_decaying_correlation(fields, shares, 4.0, 0.4, 0.786, 61), 2, 0)",
        },
        # --- Invalid: bracket entirely below the root, gap positive at both ends ---
        {
            "setup": status + ensemble,
            "call": "_status(lambda: solve_decaying_correlation(fields, shares, 4.0, 0.3, 0.5, 41))",
            "gold_call": "_status(lambda: _oracle_solve_decaying_correlation(fields, shares, 4.0, 0.3, 0.5, 41))",
        },
        # --- Invalid: non-positive lower end ---
        {
            "setup": status + ensemble,
            "call": "_status(lambda: solve_decaying_correlation(fields, shares, 4.0, 0.0, 0.8, 41))",
            "gold_call": "_status(lambda: _oracle_solve_decaying_correlation(fields, shares, 4.0, 0.0, 0.8, 41))",
        },
    ]
