"""
Find the range along a pixel's line of sight at which the source meets an asserted barycentric distance history at the moment the observed light left it.

A pixel fixes only a direction from the observer, so asserting how far the source is from the barycentre as a function of time places it on the line of sight once the light-travel time is accounted for.

Returns
-------
np.ndarray: [rho, t_emit], the range along the line of sight and the emission epoch.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_light_time_range(
    direction: "np.ndarray",
    observer: "np.ndarray",
    t_obs: float,
    radial_fn: "Callable[[float], np.ndarray]",
    light_speed: float,
) -> "np.ndarray":
    """Return the range and emission epoch at which a line of sight meets a distance history.

    Light received at epoch ``t_obs`` by an observer at barycentric position
    ``observer``, arriving from the unit vector ``direction``, left a source
    at barycentric position ``observer + rho * direction`` at the emission
    epoch ``t_emit = t_obs - rho / light_speed``. ``radial_fn(t)`` returns
    ``[r, rdot]``, an asserted barycentric distance of the source and its
    rate of change at epoch ``t`` on the same time axis as ``t_obs``. Return
    ``[rho, t_emit]`` for the range ``rho > 0`` at which
    ``|observer + rho * direction|`` equals ``radial_fn(t_emit)[0]``.
    Select the smallest positive range if a synthetic distance history has
    more than one mathematical intersection. Supplied histories have a
    single outward crossing before any more distant intersection, and the
    observer lies inside the asserted distance at the observation epoch.
    ``rho`` must have relative accuracy ``1e-13``.

    Parameters
    ----------
    direction : "np.ndarray"
        Unit vector of shape ``(3,)`` (norm within ``1e-9`` of one).
    observer : "np.ndarray"
        Barycentric observer position of shape ``(3,)``.
    t_obs : float
        Observation epoch.
    radial_fn : callable
        Function of one epoch returning two finite numbers ``[r, rdot]``
        with ``r > 0``.
    light_speed : float
        Positive speed of light in the units of the other arguments.

    Returns
    -------
    np.ndarray
        Float array ``[rho, t_emit]``.

    Raises
    ------
    ValueError
        If ``direction`` or ``observer`` is not three finite numbers,
        ``direction`` is not a unit vector, ``t_obs`` is not a finite real
        number, ``light_speed`` is not a finite positive number,
        ``radial_fn`` is not callable or returns anything other than two
        finite numbers with a positive distance, or
        ``|observer| >= radial_fn(t_obs)[0]``.
    """
    return range_and_epoch

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_light_time_range(
    direction: "np.ndarray",
    observer: "np.ndarray",
    t_obs: float,
    radial_fn: "Callable[[float], np.ndarray]",
    light_speed: float,
) -> "np.ndarray":
    """Reference implementation (bracketed Newton on the squared-distance mismatch)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    def _vector(value, name):
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must hold three finite numbers") from None
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must hold three finite numbers")
        return array

    unit = _vector(direction, "direction")
    origin = _vector(observer, "observer")
    if abs(float(np.linalg.norm(unit)) - 1.0) > 1e-9:
        raise ValueError("direction must be a unit vector")
    if not _is_number(t_obs):
        raise ValueError("t_obs must be a finite real number")
    if not (_is_number(light_speed) and light_speed > 0.0):
        raise ValueError("light_speed must be a finite positive number")
    if not _is_function(radial_fn):
        raise ValueError("radial_fn must be callable")
    t_obs, light_speed = float(t_obs), float(light_speed)

    def _track(epoch):
        values = np.asarray(radial_fn(epoch), dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)) or values[0] <= 0.0:
            raise ValueError("radial_fn must return two finite numbers with r > 0")
        return values

    def _mismatch(rho):
        # |x + rho d|^2 - r(t - rho / c)^2 and its derivative in rho.
        radius, rate = _track(t_obs - rho / light_speed)
        point = origin + rho * unit
        value = float(point @ point) - radius * radius
        slope = 2.0 * float(point @ unit) + 2.0 * radius * rate / light_speed
        return value, slope

    if float(np.linalg.norm(origin)) >= _track(t_obs)[0]:
        raise ValueError("the observer must lie inside the asserted distance")
    low, high = 0.0, float(np.linalg.norm(origin)) + _track(t_obs)[0]
    for _ in range(200):
        if _mismatch(high)[0] > 0.0:
            break
        low, high = high, 2.0 * high
    else:
        raise ValueError("no range along the line of sight meets the distance history")
    rho = 0.5 * (low + high)
    for _ in range(200):
        value, slope = _mismatch(rho)
        if value < 0.0:
            low = rho
        else:
            high = rho
        trial = rho - value / slope if slope > 0.0 else 0.5 * (low + high)
        if not low < trial < high:
            trial = 0.5 * (low + high)
        if abs(trial - rho) <= 1e-15 * trial or high - low <= 1e-15 * trial:
            rho = trial
            break
        rho = trial
    return np.array([float(rho), t_obs - float(rho) / light_speed])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    tracks = (
        "import numpy as np\n"
        "def _line(r0, rate):\n"
        "    return lambda t: np.array([r0 + rate * t, rate])\n"
        "def _bent(r0, rate, accel):\n"
        "    return lambda t: np.array([r0 + rate * t + 0.5 * accel * t * t, rate + accel * t])\n"
        "def _unit(v):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    return v / np.linalg.norm(v)\n"
        "def _pair(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (2,):\n"
        "        return -1.0e3\n"
        "    return float(a[0] / 10.0 + a[1])\n"
        "c_au = 173.1446326846693\n"
        "look = _unit([-0.15, -0.78, -0.21])\n"
        "obs = np.array([-0.5326, 0.7595, 0.3293])\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": tracks,
            "call": "_pair(solve_light_time_range(look.copy(), obs.copy(), -7.35, _line(32.0, -5.5e-4), c_au))",
            "gold_call": "_pair(_oracle_solve_light_time_range(look.copy(), obs.copy(), -7.35, _line(32.0, -5.5e-4), c_au))",
        },
        {
            "setup": tracks,
            "call": "float(10.0 * (solve_light_time_range(look.copy(), obs.copy(), 4.2, _line(32.0, -5.5e-4), c_au)[1] - 4.2))",
            "gold_call": "float(10.0 * (_oracle_solve_light_time_range(look.copy(), obs.copy(), 4.2, _line(32.0, -5.5e-4), c_au)[1] - 4.2))",
        },
        {
            "setup": tracks,
            "call": "_pair(solve_light_time_range(_unit([0.4, 0.9, -0.1]), obs.copy(), 1.0, _line(6.0, 0.0), c_au))",
            "gold_call": "_pair(_oracle_solve_light_time_range(_unit([0.4, 0.9, -0.1]), obs.copy(), 1.0, _line(6.0, 0.0), c_au))",
        },
        {
            "setup": tracks,
            "call": "_pair(solve_light_time_range(_unit([0.3, -0.2, 0.93]), np.array([0.0, 0.0, 1.5]), 2.0, _bent(3.0, 0.2, -0.05), 5.0))",
            "gold_call": "_pair(_oracle_solve_light_time_range(_unit([0.3, -0.2, 0.93]), np.array([0.0, 0.0, 1.5]), 2.0, _bent(3.0, 0.2, -0.05), 5.0))",
        },
        {
            "setup": tracks,
            "call": "_pair(solve_light_time_range(_unit([-0.2, 0.1, -0.97]), np.array([0.0, 0.0, 1.5]), 0.0, _bent(3.0, -0.3, 0.04), 4.0))",
            "gold_call": "_pair(_oracle_solve_light_time_range(_unit([-0.2, 0.1, -0.97]), np.array([0.0, 0.0, 1.5]), 0.0, _bent(3.0, -0.3, 0.04), 4.0))",
        },
        {
            "setup": tracks + status,
            "call": "_status(lambda: solve_light_time_range(np.array([0.3, 0.3, 0.3]), obs.copy(), 0.0, _line(32.0, 0.0), c_au))",
            "gold_call": "_status(lambda: _oracle_solve_light_time_range(np.array([0.3, 0.3, 0.3]), obs.copy(), 0.0, _line(32.0, 0.0), c_au))",
        },
        {
            "setup": tracks + status,
            "call": "_status(lambda: solve_light_time_range(look.copy(), np.array([2.0, 0.0, 0.0]), 0.0, _line(1.5, 0.0), c_au))",
            "gold_call": "_status(lambda: _oracle_solve_light_time_range(look.copy(), np.array([2.0, 0.0, 0.0]), 0.0, _line(1.5, 0.0), c_au))",
        },
        {
            "setup": tracks + status,
            "call": "_status(lambda: solve_light_time_range(look.copy(), obs.copy(), 0.0, _line(32.0, 0.0), -1.0))",
            "gold_call": "_status(lambda: _oracle_solve_light_time_range(look.copy(), obs.copy(), 0.0, _line(32.0, 0.0), -1.0))",
        },
    ]
