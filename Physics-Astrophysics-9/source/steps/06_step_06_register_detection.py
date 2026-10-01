"""
Map one pixel of one exposure onto the barycentric ecliptic canvas at the reference epoch under an asserted initial condition.

Treating every pixel as a possible detection of an object on an asserted orbit, and carrying that object to a common epoch, lets exposures taken days apart be summed on one canvas whatever the observer's own motion.

Returns
-------
np.ndarray: [longitude, latitude] in degrees on the barycentric ecliptic canvas at t_ref.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def register_detection(
    right_ascension: float,
    declination: float,
    t_obs: float,
    observer: "np.ndarray",
    initial_condition: tuple,
    obliquity: float,
    light_speed: float,
    radial_fn: "Callable[[float], np.ndarray]",
    light_time_fn: "Callable[..., np.ndarray]",
    state_fn: "Callable[..., np.ndarray]",
    propagate_fn: "Callable[[np.ndarray, float], np.ndarray]",
) -> "np.ndarray":
    """Return the canvas direction at which one pixel registers under an initial condition.

    The canvas holds barycentric ecliptic directions at the reference epoch
    ``t_ref``, and every epoch is measured from ``t_ref``. The pixel is seen
    at epoch ``t_obs`` at ICRF right ascension and declination (degrees;
    astrometric, that is light-time retarded and free of aberration) by an
    observer at barycentric ICRF position ``observer``. The ecliptic frame
    shares the ICRF ``x`` axis and has its ``z`` axis along the north
    ecliptic pole, whose ICRF direction is ``(0, -sin(eps), cos(eps))`` with
    ``eps = obliquity`` degrees. ``initial_condition = (r_ref, vr_ref,
    vt_ref, inclination, kappa)`` asserts, at ``t_ref``, the barycentric
    distance, radial velocity and speed perpendicular to the radius vector,
    the inclination of the orbit to the ecliptic in degrees, and the sign of
    the rate of ecliptic latitude; lengths and times share the units of
    ``observer``, ``light_speed`` and ``t_obs``.

    Register the pixel as the source on the unique orbit that has the
    distance history ``radial_fn``, the angular momentum and ``kappa`` of the
    initial condition, and light reaching the observer from this pixel's
    direction, and return that orbit's barycentric ecliptic longitude in
    ``[0, 360)`` and latitude at ``t_ref``, in degrees. Every vector passed
    to the callables is in the ecliptic frame:

    - ``radial_fn(t) -> [r, rdot]`` is the distance history shared by all
      orbits of the initial condition;
    - ``light_time_fn(direction, observer, t_obs, radial_fn, light_speed)
      -> [rho, t_emit]`` follows ``solve_light_time_range``;
    - ``state_fn(direction, observer, rho, radial_rate, angular_momentum,
      inclination, kappa) -> state`` follows ``construct_emission_state``
      (inclination in radians);
    - ``propagate_fn(state, dt) -> state`` follows
      ``propagate_two_body_state`` with its gravitational parameter fixed.

    Parameters
    ----------
    right_ascension, declination : float
        Pixel direction in degrees; the declination lies in ``[-90, 90]``.
    t_obs : float
        Observation epoch relative to ``t_ref``.
    observer : "np.ndarray"
        Barycentric ICRF observer position, shape ``(3,)``.
    initial_condition : tuple
        ``(r_ref > 0, vr_ref, vt_ref > 0, inclination in [0, 180],
        kappa = +1 or -1)``.
    obliquity : float
        Angle between the ICRF equator and the ecliptic, in degrees.
    light_speed : float
        Positive speed of light.
    radial_fn, light_time_fn, state_fn, propagate_fn : callable
        The callables described above.

    Returns
    -------
    np.ndarray
        Float array ``[longitude, latitude]`` in degrees.

    Raises
    ------
    ValueError
        If any number is not finite and real, the declination is outside
        ``[-90, 90]``, ``observer`` is not three finite numbers,
        ``initial_condition`` violates its stated ranges (booleans are
        rejected for ``kappa``), ``light_speed`` is not positive, a callable
        is missing, a callable returns an array of the wrong shape or with
        non-finite entries, or a callable raises ``ValueError``.
    """
    return canvas_direction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_register_detection(
    right_ascension: float,
    declination: float,
    t_obs: float,
    observer: "np.ndarray",
    initial_condition: tuple,
    obliquity: float,
    light_speed: float,
    radial_fn: "Callable[[float], np.ndarray]",
    light_time_fn: "Callable[..., np.ndarray]",
    state_fn: "Callable[..., np.ndarray]",
    propagate_fn: "Callable[[np.ndarray, float], np.ndarray]",
) -> "np.ndarray":
    """Reference implementation (light-time placement, invariant velocity, propagation to t_ref)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    def _finite(value, shape, name):
        array = np.asarray(value, dtype=float)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must return {shape} finite numbers")
        return array

    for name, value in (("right_ascension", right_ascension), ("declination", declination),
                        ("t_obs", t_obs), ("obliquity", obliquity), ("light_speed", light_speed)):
        if not _is_number(value):
            raise ValueError(f"{name} must be a finite real number")
    if not -90.0 <= declination <= 90.0:
        raise ValueError("declination must lie in [-90, 90]")
    if light_speed <= 0.0:
        raise ValueError("light_speed must be positive")
    try:
        origin_icrf = np.asarray(observer, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("observer must hold three finite numbers") from None
    if origin_icrf.shape != (3,) or not np.all(np.isfinite(origin_icrf)):
        raise ValueError("observer must hold three finite numbers")
    try:
        r_ref, vr_ref, vt_ref, inclination, kappa = initial_condition
    except (TypeError, ValueError):
        raise ValueError("initial_condition must hold five entries") from None
    if not all(_is_number(v) for v in (r_ref, vr_ref, vt_ref, inclination)):
        raise ValueError("initial_condition entries must be finite real numbers")
    if r_ref <= 0.0 or vt_ref <= 0.0 or not 0.0 <= inclination <= 180.0:
        raise ValueError("initial_condition is outside its stated ranges")
    if not (_is_integer(kappa) and kappa in (1, -1)):
        raise ValueError("kappa must be +1 or -1")
    if not all(callable(fn) for fn in (radial_fn, light_time_fn, state_fn, propagate_fn)):
        raise ValueError("radial_fn, light_time_fn, state_fn and propagate_fn must be callable")
    eps = np.radians(float(obliquity))
    to_ecliptic = np.array([[1.0, 0.0, 0.0],
                            [0.0, np.cos(eps), np.sin(eps)],
                            [0.0, -np.sin(eps), np.cos(eps)]])
    ra, dec = np.radians(float(right_ascension)), np.radians(float(declination))
    direction = to_ecliptic @ np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    origin = to_ecliptic @ origin_icrf
    rho, t_emit = _finite(light_time_fn(direction, origin, float(t_obs), radial_fn, float(light_speed)),
                          (2,), "light_time_fn")
    # The asserted radial motion, read at the emission epoch, fixes the sign of v_r there.
    radial_rate = _finite(radial_fn(t_emit), (2,), "radial_fn")[1]
    emission = _finite(state_fn(direction, origin, float(rho), float(radial_rate),
                                float(r_ref) * float(vt_ref), np.radians(float(inclination)),
                                int(kappa)), (6,), "state_fn")
    state = _finite(propagate_fn(emission, -float(t_emit)), (6,), "propagate_fn")
    x, y, z = state[:3]
    longitude = float(np.degrees(np.arctan2(y, x)) % 360.0)
    if longitude >= 360.0:
        longitude -= 360.0
    latitude = float(np.degrees(np.arctan2(z, np.hypot(x, y))))
    return np.array([longitude, latitude])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    fakes = (
        "import numpy as np\n"
        "def _track(t):\n"
        "    return np.array([40.0 - 4.0e-4 * t + 2.5e-5 * t * t, -4.0e-4 + 5.0e-5 * t])\n"
        "def _light(direction, observer, t_obs, radial_fn, light_speed):\n"
        "    rho = (radial_fn(t_obs)[0] - 0.8 + 0.01 * float(direction @ np.array([1.0, 2.0, 3.0]))\n"
        "           + 1.0e-3 * float(observer @ np.array([3.0, -1.0, 2.0])))\n"
        "    return np.array([rho, t_obs - rho / light_speed])\n"
        "def _state(direction, observer, rho, radial_rate, angular_momentum, inclination, kappa):\n"
        "    position = observer + rho * direction\n"
        "    r = np.linalg.norm(position)\n"
        "    tilt = np.array([np.cos(inclination), 0.3 * kappa, np.sin(inclination)])\n"
        "    velocity = radial_rate * position / r + angular_momentum / r * tilt + 30.0 * radial_rate * np.array([0.0, 1.0, 0.5])\n"
        "    return np.concatenate([position, velocity])\n"
        "def _propagate(state, dt):\n"
        "    p, v = np.asarray(state[:3], dtype=float), np.asarray(state[3:], dtype=float)\n"
        "    return np.concatenate([p + v * dt - 2.0e-7 * dt * dt * p, v - 4.0e-7 * dt * p])\n"
        "def _refuse(direction, observer, rho, radial_rate, angular_momentum, inclination, kappa):\n"
        "    raise ValueError('no orbit of this inclination reaches the position')\n"
        "def _sky(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (2,):\n"
        "        return -1.0e3\n"
        "    return float((a[0] - 200.0) / 10.0 + a[1])\n"
        "obs = np.array([-0.5326, 0.7595, 0.3293])\n"
        "ic = (40.0, -4.0e-4, 3.0e-3, 9.2, 1)\n"
        "c_au = 173.1446326846693\n"
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
            "setup": fakes,
            "call": "_sky(register_detection(218.43041933, -12.04702376, -7.3518, obs.copy(), ic, 23.4392911, c_au, _track, _light, _state, _propagate))",
            "gold_call": "_sky(_oracle_register_detection(218.43041933, -12.04702376, -7.3518, obs.copy(), ic, 23.4392911, c_au, _track, _light, _state, _propagate))",
        },
        {
            "setup": fakes,
            "call": "_sky(register_detection(218.52710154, -12.04113391, 7.1852, np.array([-0.7222, 0.6161, 0.2671]), ic, 23.4392911, c_au, _track, _light, _state, _propagate))",
            "gold_call": "_sky(_oracle_register_detection(218.52710154, -12.04113391, 7.1852, np.array([-0.7222, 0.6161, 0.2671]), ic, 23.4392911, c_au, _track, _light, _state, _propagate))",
        },
        {
            "setup": fakes,
            "call": "_sky(register_detection(218.43041933, -12.04702376, -7.3518, obs.copy(), (40.0, -4.0e-4, 3.0e-3, 9.2, -1), 23.4392911, c_au, _track, _light, _state, _propagate))",
            "gold_call": "_sky(_oracle_register_detection(218.43041933, -12.04702376, -7.3518, obs.copy(), (40.0, -4.0e-4, 3.0e-3, 9.2, -1), 23.4392911, c_au, _track, _light, _state, _propagate))",
        },
        {
            "setup": fakes,
            "call": "_sky(register_detection(215.0, -10.0, 3.0, obs.copy(), ic, 0.0, c_au, _track, _light, _state, _propagate))",
            "gold_call": "_sky(_oracle_register_detection(215.0, -10.0, 3.0, obs.copy(), ic, 0.0, c_au, _track, _light, _state, _propagate))",
        },
        {
            "setup": fakes,
            "call": "float(register_detection(359.99, 0.0, 0.5, np.array([0.0, 0.0, 0.0]), (40.0, 0.0, 3.0e-3, 30.0, 1), 0.0, c_au, _track, _light, _state, _propagate)[0] / 100.0)",
            "gold_call": "float(_oracle_register_detection(359.99, 0.0, 0.5, np.array([0.0, 0.0, 0.0]), (40.0, 0.0, 3.0e-3, 30.0, 1), 0.0, c_au, _track, _light, _state, _propagate)[0] / 100.0)",
        },
        {
            "setup": fakes + status,
            "call": "_status(lambda: register_detection(218.43, -12.05, -7.35, obs.copy(), (40.0, -4.0e-4, 3.0e-3, 9.2, 0), 23.4392911, c_au, _track, _light, _state, _propagate))",
            "gold_call": "_status(lambda: _oracle_register_detection(218.43, -12.05, -7.35, obs.copy(), (40.0, -4.0e-4, 3.0e-3, 9.2, 0), 23.4392911, c_au, _track, _light, _state, _propagate))",
        },
        {
            "setup": fakes + status,
            "call": "_status(lambda: register_detection(218.43, 95.0, -7.35, obs.copy(), ic, 23.4392911, c_au, _track, _light, _state, _propagate))",
            "gold_call": "_status(lambda: _oracle_register_detection(218.43, 95.0, -7.35, obs.copy(), ic, 23.4392911, c_au, _track, _light, _state, _propagate))",
        },
        {
            "setup": fakes + status,
            "call": "_status(lambda: register_detection(218.43, -12.05, -7.35, obs.copy(), ic, 23.4392911, c_au, _track, _light, _refuse, _propagate))",
            "gold_call": "_status(lambda: _oracle_register_detection(218.43, -12.05, -7.35, obs.copy(), ic, 23.4392911, c_au, _track, _light, _refuse, _propagate))",
        },
    ]
