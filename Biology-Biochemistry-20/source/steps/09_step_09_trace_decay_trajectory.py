"""
Trace the decaying autocorrelation of the disorder-induced input as a function of lag time, from its equal-time value down onto the hilltop of the effective potential.

Near the hilltop, the autocorrelation approaches its static part exponentially, turning the physical orbit into a two-endpoint problem: it is released from rest at zero lag but reaches the hilltop only asymptotically.

Returns
-------
np.ndarray: autocorrelation of the disorder-induced input on the uniform lag-time grid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def trace_decay_trajectory(delta_zero: float, delta_inf: float, fields: "np.ndarray",
                           weights: "np.ndarray", beta: float, t_max: float, n_tau: int,
                           nodes: int = 241, degree: int = 64) -> "np.ndarray":
    """Return the decaying autocorrelation sampled on a uniform grid of lag times.

    The acceleration of the autocorrelation ``D`` is ``-slope(D)`` with
    ``slope(D) = average_correlation_overlap(np.array([D]), delta_zero, ...)[0] - D``.
    Represent this slope by its degree-``degree`` interpolant at the first-kind
    Chebyshev nodes on ``[delta_inf, delta_zero]``. Let ``top`` be the root of
    that interpolant in the interval extending ``0.001 * (delta_zero-delta_inf)``
    to either side of ``delta_inf``.

    Return the unique physical branch of ``D'' = -slope(D)`` that is even in lag
    (so ``D'(0)=0``), decreases for positive lag, and approaches ``top`` along
    its decaying exponential mode as lag tends to infinity. Its turning point
    must lie above the midpoint between ``delta_inf`` and ``delta_zero``. Entry
    ``k`` is this solution at ``tau_k = k * t_max / (n_tau - 1)``. Any stable
    numerical method that satisfies these boundary conditions may be used.

    Parameters
    ----------
    delta_zero : float
        Finite equal-time correlation, greater than ``delta_inf``.
    delta_inf : float
        Nonnegative finite hilltop lag value of the effective potential.
    fields : np.ndarray
        Non-empty one-dimensional array of finite constant regulatory inputs.
    weights : np.ndarray
        Nonnegative population shares of the same length as ``fields``, summing
        to one within ``1e-9``.
    beta : float
        Positive finite response gain.
    t_max : float
        Positive finite largest lag time of the grid.
    n_tau : int
        Number of grid points, at least 2.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3.
    degree : int
        Chebyshev interpolation degree, at least 4.

    Returns
    -------
    np.ndarray
        Float array of length ``n_tau``, starting at the turning point near
        ``delta_zero`` and decreasing towards ``delta_inf``.

    Raises
    ------
    ValueError
        If ``delta_inf`` is negative or not finite, if ``delta_zero`` is not
        finite or not greater than ``delta_inf``, if ``t_max`` is not positive and
        finite, if ``n_tau`` or ``degree`` is too small, if the interpolated slope
        does not fall from positive to negative across the bracket around
        ``delta_inf`` or ``1 -`` the gain overlap at its root is not positive (no
        hilltop there), if the integration never comes to rest or comes to rest
        below the midpoint of ``[delta_inf, delta_zero]``, or if an argument fails
        the contract of the overlap steps.
    """
    return profile

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial import chebyshev
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def _epi_chebyshev_fit(function, lower: float, upper: float, degree: int) -> "np.ndarray":
    """Return Chebyshev coefficients interpolating a vectorised function on [lower, upper]."""
    half, middle = 0.5 * (upper - lower), 0.5 * (upper + lower)
    return chebyshev.chebinterpolate(lambda t: function(half * np.asarray(t) + middle), degree)

def _epi_chebyshev_eval(coefficients: "np.ndarray", values, lower: float, upper: float):
    """Evaluate Chebyshev coefficients fitted on [lower, upper] at the given values."""
    return chebyshev.chebval((2.0 * np.asarray(values) - lower - upper) / (upper - lower),
                             coefficients)

def _oracle_trace_decay_trajectory(delta_zero: float, delta_inf: float, fields: "np.ndarray",
                                   weights: "np.ndarray", beta: float, t_max: float,
                                   n_tau: int, nodes: int = 241,
                                   degree: int = 64) -> "np.ndarray":
    """Reference implementation (backward integration along the unstable manifold)."""
    import numpy as np

    top = _epi_require_scalar(delta_inf, "delta_inf", 0.0, False)
    start = _epi_require_scalar(delta_zero, "delta_zero", top, True)
    horizon = _epi_require_scalar(t_max, "t_max", 0.0, True)
    samples = _epi_require_count(n_tau, "n_tau", 2)
    order = _epi_require_count(degree, "degree", 4)

    def _slope(values):
        lags = np.atleast_1d(np.asarray(values, dtype=float))
        return _oracle_average_correlation_overlap(lags, start, fields, weights, beta,
                                                   nodes) - lags

    coefficients = _epi_chebyshev_fit(_slope, top, start, order)
    width = start - top

    def _fitted(value):
        return float(_epi_chebyshev_eval(coefficients, value, top, start))

    # Start on the interpolant's own hilltop, so the release is consistent with the force used.
    low, high = top - 1e-3 * width, top + 1e-3 * width
    if not (_fitted(low) > 0.0 > _fitted(high)):
        raise ValueError("the interpolated slope does not fall through zero near delta_inf")
    summit = float(brentq(_fitted, low, high, xtol=1e-15, rtol=4.0 * float(np.finfo(float).eps)))
    fluctuation = 1.0 - float(_oracle_average_gain_overlap(np.array([max(summit, 0.0)]), start,
                                                           fields, weights, beta, nodes)[0])
    if not fluctuation > 0.0:
        raise ValueError("delta_inf is not a hilltop: the fluctuation potential is not positive")
    kappa = np.sqrt(fluctuation)
    eps = 1e-11 * width

    def _rest(s, state):
        return state[1]
    _rest.terminal, _rest.direction = True, -1
    solution = solve_ivp(lambda s, state: [state[1], -_fitted(state[0])], (0.0, 1.0e4),
                         [summit + eps, kappa * eps], method="DOP853", rtol=1e-12, atol=1e-15,
                         events=_rest, dense_output=True)
    if solution.t_events[0].size == 0:
        raise ValueError("the backward integration never came to rest")
    if float(solution.y_events[0][0][0]) < top + 0.5 * width:
        raise ValueError("the motion came to rest before approaching delta_zero")
    s_turn = float(solution.t_events[0][0])

    tau = np.linspace(0.0, horizon, samples)
    reversed_time = s_turn - tau
    profile = summit + eps * np.exp(kappa * np.minimum(reversed_time, 0.0))
    inside = reversed_time >= 0.0
    profile[inside] = solution.sol(reversed_time[inside])[0]
    return profile

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
        # --- Normal: the sharp-gain mixture's decaying state on a coarse lag grid ---
        {
            "setup": digest + ensemble,
            "call": "_digest(trace_decay_trajectory(0.797285062533527, 0.6047599874259055, fields, shares, 4.0, 30.0, 61, 81, 24), 61, 0)",
            "gold_call": "_digest(_oracle_trace_decay_trajectory(0.797285062533527, 0.6047599874259055, fields, shares, 4.0, 30.0, 61, 81, 24), 61, 0)",
            "tol": 2e-5,
        },
        # --- Boundary: two grid points only, the turning point and the far tail ---
        {
            "setup": digest + ensemble,
            "call": "_digest(trace_decay_trajectory(0.7972364825659272, 0.6049702299294569, fields, shares, 4.0, 200.0, 2, 61, 16), 2, 0)",
            "gold_call": "_digest(_oracle_trace_decay_trajectory(0.7972364825659272, 0.6049702299294569, fields, shares, 4.0, 200.0, 2, 61, 16), 2, 0)",
            "tol": 2e-7,
        },
        # --- Edge: bare inputs without feedback, a deep decay onto a low static part ---
        {
            "setup": digest + "fields = np.array([-0.32, -0.1, 0.04, 0.11, 0.35])\nshares = np.array([0.14, 0.22, 0.27, 0.21, 0.16])\n",
            "call": "_digest(trace_decay_trajectory(0.687436167415904, 0.194989979757067, fields, shares, 4.0, 12.0, 25, 81, 32), 25, 0)",
            "gold_call": "_digest(_oracle_trace_decay_trajectory(0.687436167415904, 0.194989979757067, fields, shares, 4.0, 12.0, 25, 81, 32), 25, 0)",
            "tol": 1e-5,
        },
        # --- Boundary: a low odd-sized interpolant on a non-round lag grid ---
        {
            "setup": digest + ensemble,
            "call": "_digest(trace_decay_trajectory(0.7972364825659272, 0.6049702299294569, fields, shares, 4.0, 7.3, 38, 61, 15), 38, 0)",
            "gold_call": "_digest(_oracle_trace_decay_trajectory(0.7972364825659272, 0.6049702299294569, fields, shares, 4.0, 7.3, 38, 61, 15), 38, 0)",
            "tol": 1e-5,
        },
        # --- Invalid: the proposed hilltop sits above the equal-time value ---
        {
            "setup": status + ensemble,
            "call": "_status(lambda: trace_decay_trajectory(0.6, 0.7, fields, shares, 4.0, 10.0, 11, 41, 16))",
            "gold_call": "_status(lambda: _oracle_trace_decay_trajectory(0.6, 0.7, fields, shares, 4.0, 10.0, 11, 41, 16))",
        },
        # --- Invalid: a lag value inside the potential's valley, where the slope rises instead of falling ---
        {
            "setup": status + ensemble,
            "call": "_status(lambda: trace_decay_trajectory(0.797, 0.78, fields, shares, 4.0, 10.0, 11, 41, 16))",
            "gold_call": "_status(lambda: _oracle_trace_decay_trajectory(0.797, 0.78, fields, shares, 4.0, 10.0, 11, 41, 16))",
        },
    ]
