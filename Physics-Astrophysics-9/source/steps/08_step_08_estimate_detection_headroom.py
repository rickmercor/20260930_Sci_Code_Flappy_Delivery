"""
Register every exposure of a faint object under one trial initial condition, stack them, and return how many magnitudes fainter the object could be while its stack maximum still reaches a candidate threshold.

A trial orbit that differs slightly from an object's true orbit registers its exposures slightly apart on the canvas, and the significance lost at the stack maximum sets how faint an object on that track can be and still be flagged.

Returns
-------
float: the magnitude headroom 2.5 * log10(nu_max / threshold).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_detection_headroom(
    times: "np.ndarray",
    observers: "np.ndarray",
    right_ascensions: "np.ndarray",
    declinations: "np.ndarray",
    flux_errors: "np.ndarray",
    initial_condition: tuple,
    flux: float,
    threshold: float,
    psf_sigma: float = 0.042,
    gm: float = 0.01720209895 ** 2,
    light_speed: float = 173.1446326846693,
    obliquity: float = 23.4392911,
    au_km: float = 149597870.7,
) -> float:
    """Return the magnitude headroom with which one initial condition's stack flags an object.

    Exposure ``k`` was taken at ``times[k]`` (TDB days from the canvas
    epoch ``t_ref``) from the barycentric ICRF position ``observers[k]``
    (au); the object's astrometric position in it is
    (``right_ascensions[k]``, ``declinations[k]``) in degrees, and
    ``flux_errors[k]`` is its PSF-fit flux uncertainty in the unit of
    ``flux``. ``initial_condition = (r_ref, vr_ref, vt_ref, inclination,
    kappa)`` asserts, at ``t_ref``, the barycentric distance (au), radial
    velocity and speed perpendicular to the radius vector (km/s, converted
    with ``au_km`` kilometres per au and 86400 s per day), the orbital
    inclination to the ecliptic (degrees) and the sign of the ecliptic
    latitude rate. Motion is two-body about a centre of gravitational
    parameter ``gm`` (au^3/day^2) fixed at the barycentre, light travels at
    ``light_speed`` (au/day), and the ecliptic is the ICRF frame tilted by
    ``obliquity`` degrees about its ``x`` axis.

    Register every exposure's object position on the barycentric ecliptic
    canvas at ``t_ref`` under the initial condition, stack the exposures
    with the inverse-variance-weighted PSF-matched filter of a circular
    Gaussian canvas PSF of standard deviation ``psf_sigma`` arcseconds, and
    return ``2.5 * log10(nu_max / threshold)``, where ``nu_max`` is the
    expected significance at the stack maximum for an object of constant
    flux ``flux``: the number of magnitudes by which the object could be
    fainter and still reach ``threshold`` (negative if it already falls
    short).

    Parameters
    ----------
    times : "np.ndarray"
        Exposure epochs, shape ``(N,)``.
    observers : "np.ndarray"
        Observer positions, shape ``(N, 3)``.
    right_ascensions, declinations : "np.ndarray"
        Object positions in degrees, shape ``(N,)`` each.
    flux_errors : "np.ndarray"
        Positive PSF-fit flux uncertainties, shape ``(N,)``.
    initial_condition : tuple
        ``(r_ref > 0, vr_ref, vt_ref > 0, inclination in [0, 180],
        kappa = +1 or -1)``.
    flux, threshold : float
        Positive object flux and positive candidate significance threshold.
    psf_sigma, gm, light_speed, au_km : float
        Positive PSF width (arcsec), gravitational parameter, speed of light
        and kilometres per au.
    obliquity : float
        Finite obliquity in degrees.

    Returns
    -------
    float
        The headroom ``2.5 * log10(nu_max / threshold)`` in magnitudes.

    Raises
    ------
    ValueError
        If the arrays are not finite with the stated shapes and a common
        ``N >= 1``, any scalar is outside its stated domain,
        ``initial_condition`` violates its stated ranges, or any stage
        rejects its input (for example when an exposure cannot be reached by
        an orbit of the asserted inclination).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_detection_headroom(
    times: "np.ndarray",
    observers: "np.ndarray",
    right_ascensions: "np.ndarray",
    declinations: "np.ndarray",
    flux_errors: "np.ndarray",
    initial_condition: tuple,
    flux: float,
    threshold: float,
    psf_sigma: float = 0.042,
    gm: float = 0.01720209895 ** 2,
    light_speed: float = 173.1446326846693,
    obliquity: float = 23.4392911,
    au_km: float = 149597870.7,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _array(value, shape_tail, name):
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must hold finite numbers") from None
        if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
            raise ValueError(f"{name} has the wrong shape")
        if array.shape[0] == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must hold finite numbers")
        return array

    epochs = _array(times, (), "times")
    stations = _array(observers, (3,), "observers")
    ras = _array(right_ascensions, (), "right_ascensions")
    decs = _array(declinations, (), "declinations")
    errors = _array(flux_errors, (), "flux_errors")
    if not len(epochs) == len(stations) == len(ras) == len(decs) == len(errors):
        raise ValueError("all per-exposure arrays must share one length")
    for name, value in (("flux", flux), ("threshold", threshold), ("psf_sigma", psf_sigma),
                        ("gm", gm), ("light_speed", light_speed), ("au_km", au_km)):
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    if not _is_number(obliquity):
        raise ValueError("obliquity must be a finite real number")
    try:
        r_ref, vr_ref, vt_ref, inclination, kappa = initial_condition
    except (TypeError, ValueError):
        raise ValueError("initial_condition must hold five entries") from None
    if not all(_is_number(v) for v in (vr_ref, vt_ref)):
        raise ValueError("the asserted velocities must be finite real numbers")
    per_day = 86400.0 / float(au_km)
    condition = (r_ref, float(vr_ref) * per_day, float(vt_ref) * per_day, inclination, kappa)

    def _anomaly(r0, vr0, alpha, mu, dt):
        return _oracle_solve_universal_anomaly(r0, vr0, alpha, mu, dt)

    def _radial(t):
        return _oracle_evaluate_radial_track(r_ref, condition[1], condition[2], gm, t, _anomaly)

    def _light(direction, observer, t_obs, radial_fn, speed):
        return _oracle_solve_light_time_range(direction, observer, t_obs, radial_fn, speed)

    def _state(direction, observer, rho, radial_rate, angular_momentum, tilt, sense):
        return _oracle_construct_emission_state(direction, observer, rho, radial_rate,
                                                angular_momentum, tilt, sense)

    def _propagate(state, dt):
        return _oracle_propagate_two_body_state(state, gm, dt, _anomaly)

    canvas = np.array([
        _oracle_register_detection(ra, dec, t, station, condition, obliquity, light_speed,
                                   _radial, _light, _state, _propagate)
        for ra, dec, t, station in zip(ras, decs, epochs, stations)
    ])
    peak = _oracle_maximize_stacked_significance(canvas[:, 0], canvas[:, 1], errors, flux, psf_sigma)
    return float(2.5 * np.log10(peak[0] / float(threshold)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only end-to-end test specifications."""
    survey = (
        "import numpy as np\n"
        "times = np.array([-7.3518, -5.9046, -4.6189, -3.0772, -1.4653, 0.5207, 1.9631, 3.7094, 5.2418, 7.1852])\n"
        "observers = np.array([[-0.532560188, 0.759515749, 0.329298322], [-0.553041423, 0.747239391, 0.323958665],\n"
        "                      [-0.571049125, 0.735867417, 0.319053923], [-0.592159359, 0.721842746, 0.312948320],\n"
        "                      [-0.613876126, 0.706713902, 0.306388170], [-0.639985759, 0.687238631, 0.297942069],\n"
        "                      [-0.658489175, 0.672538484, 0.291566922], [-0.680427115, 0.654311486, 0.283678731],\n"
        "                      [-0.699178130, 0.637757483, 0.276514388], [-0.722214969, 0.616087644, 0.267111801]])\n"
        "ras = np.array([218.43041933, 218.44465668, 218.45644418, 218.46949490, 218.48209218,\n"
        "                218.49570749, 218.50429262, 218.51355724, 218.52032553, 218.52710154])\n"
        "decs = np.array([-12.04702376, -12.04793270, -12.04851262, -12.04876521, -12.04872461,\n"
        "                 -12.04804181, -12.04711698, -12.04566106, -12.04393920, -12.04113391])\n"
        "errs = np.array([0.029, 0.033, 0.028, 0.034, 0.030, 0.029, 0.033, 0.031, 0.036, 0.030])\n"
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
            "setup": survey,
            "call": "estimate_detection_headroom(times.copy(), observers.copy(), ras.copy(), decs.copy(), errs.copy(), (32.006, -0.956, 5.6565, 9.194, 1), 0.100, 7.0)",
            "gold_call": "_oracle_estimate_detection_headroom(times.copy(), observers.copy(), ras.copy(), decs.copy(), errs.copy(), (32.006, -0.956, 5.6565, 9.194, 1), 0.100, 7.0)",
        },
        {
            "setup": survey,
            "call": "estimate_detection_headroom(times.copy(), observers.copy(), ras.copy(), decs.copy(), errs.copy(), (32.003, -0.960, 5.6567, 9.197, 1), 0.100, 7.0)",
            "gold_call": "_oracle_estimate_detection_headroom(times.copy(), observers.copy(), ras.copy(), decs.copy(), errs.copy(), (32.003, -0.960, 5.6567, 9.197, 1), 0.100, 7.0)",
        },
        {
            "setup": survey,
            "call": "estimate_detection_headroom(times[::3].copy(), observers[::3].copy(), ras[::3].copy(), decs[::3].copy(), errs[::3].copy(), (32.006, -0.956, 5.6565, 9.194, 1), 0.12, 5.0, psf_sigma=0.05)",
            "gold_call": "_oracle_estimate_detection_headroom(times[::3].copy(), observers[::3].copy(), ras[::3].copy(), decs[::3].copy(), errs[::3].copy(), (32.006, -0.956, 5.6565, 9.194, 1), 0.12, 5.0, psf_sigma=0.05)",
        },
        {
            "setup": survey,
            "call": "estimate_detection_headroom(times.copy(), observers.copy(), ras.copy(), decs.copy(), errs.copy(), (32.006, -0.956, 5.6565, 9.194, -1), 0.100, 7.0)",
            "gold_call": "_oracle_estimate_detection_headroom(times.copy(), observers.copy(), ras.copy(), decs.copy(), errs.copy(), (32.006, -0.956, 5.6565, 9.194, -1), 0.100, 7.0)",
        },
        {
            "setup": survey + status,
            "call": "_status(lambda: estimate_detection_headroom(times.copy(), observers.copy(), ras.copy(), decs.copy(), errs.copy(), (32.006, -0.956, 5.6565, 9.194, 1), 0.100, 0.0))",
            "gold_call": "_status(lambda: _oracle_estimate_detection_headroom(times.copy(), observers.copy(), ras.copy(), decs.copy(), errs.copy(), (32.006, -0.956, 5.6565, 9.194, 1), 0.100, 0.0))",
        },
        {
            "setup": survey + status,
            "call": "_status(lambda: estimate_detection_headroom(times.copy(), observers.copy(), ras.copy(), decs.copy(), errs.copy(), (32.006, -0.956, 5.6565, 2.0, 1), 0.100, 7.0))",
            "gold_call": "_status(lambda: _oracle_estimate_detection_headroom(times.copy(), observers.copy(), ras.copy(), decs.copy(), errs.copy(), (32.006, -0.956, 5.6565, 2.0, 1), 0.100, 7.0))",
        },
    ]
