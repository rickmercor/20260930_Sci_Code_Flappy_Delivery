#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
def solve_universal_anomaly(r0: float, vr0: float, alpha: float, mu: float, dt: float) -> float:
    """Reference implementation (bracketed Newton on the universal Kepler equation)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _stumpff(z):
        # C(z) and S(z); the power series avoids cancellation near z = 0.
        if abs(z) < 0.1:
            c_term, s_term, c_sum, s_sum = 0.5, 1.0 / 6.0, 0.0, 0.0
            for k in range(10):
                c_sum += c_term
                s_sum += s_term
                c_term *= -z / ((2 * k + 3) * (2 * k + 4))
                s_term *= -z / ((2 * k + 4) * (2 * k + 5))
            return c_sum, s_sum
        if z > 0.0:
            root = np.sqrt(z)
            return 2.0 * np.sin(0.5 * root) ** 2 / z, (root - np.sin(root)) / root ** 3
        root = np.sqrt(-z)
        return 2.0 * np.sinh(0.5 * root) ** 2 / -z, (np.sinh(root) - root) / root ** 3

    names = ("r0", "vr0", "alpha", "mu", "dt")
    values = (r0, vr0, alpha, mu, dt)
    for name, value in zip(names, values):
        if not _is_number(value):
            raise ValueError(f"{name} must be a finite real number")
    r0, vr0, alpha, mu, dt = (float(value) for value in values)
    if r0 <= 0.0 or mu <= 0.0:
        raise ValueError("r0 and mu must be positive")
    if alpha > 2.0 / r0 - vr0 * vr0 / mu:
        raise ValueError("alpha leaves no real speed perpendicular to the radius")
    if dt == 0.0:
        return 0.0
    root_mu = np.sqrt(mu)
    lead = r0 * vr0 / root_mu
    bend = 1.0 - alpha * r0

    def _kepler(chi):
        # Residual of the universal Kepler equation; its derivative is the distance.
        c, s = _stumpff(alpha * chi * chi)
        value = lead * chi * chi * c + bend * chi ** 3 * s + r0 * chi - root_mu * dt
        slope = lead * chi * (1.0 - alpha * chi * chi * s) + bend * chi * chi * c + r0
        return value, slope

    # The residual rises monotonically in chi, so expand a bracket on the side of dt.
    guess = root_mu * dt / r0
    low, high = (0.0, guess) if dt > 0.0 else (guess, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(200):
            edge = high if dt > 0.0 else low
            value = _kepler(edge)[0]
            if not np.isfinite(value) or (dt > 0.0 and value >= 0.0) or (dt < 0.0 and value <= 0.0):
                break
            if dt > 0.0:
                low, high = high, 2.0 * high
            else:
                high, low = low, 2.0 * low
    chi = 0.5 * (low + high)
    for _ in range(200):
        with np.errstate(over="ignore", invalid="ignore"):
            value, slope = _kepler(chi)
        if value == 0.0:
            break
        # An overflowing residual lies beyond the root on the side of its sign.
        if (value < 0.0) if np.isfinite(value) else (chi < 0.0):
            low = chi
        else:
            high = chi
        trial = chi - value / slope if np.isfinite(value) else 0.5 * (low + high)
        if not low < trial < high:
            trial = 0.5 * (low + high)
        converged = abs(trial - chi) <= 1e-15 * abs(trial) or high - low <= 1e-15 * abs(trial)
        chi = trial
        if converged:
            break
    return float(chi)

import numpy as np
def propagate_two_body_state(
    state: "np.ndarray",
    mu: float,
    dt: float,
    anomaly_fn: "Callable[..., float]",
) -> "np.ndarray":
    """Reference implementation (Lagrange f, g and their derivatives)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _stumpff(z):
        # C(z) and S(z); the power series avoids cancellation near z = 0.
        if abs(z) < 0.1:
            c_term, s_term, c_sum, s_sum = 0.5, 1.0 / 6.0, 0.0, 0.0
            for k in range(10):
                c_sum += c_term
                s_sum += s_term
                c_term *= -z / ((2 * k + 3) * (2 * k + 4))
                s_term *= -z / ((2 * k + 4) * (2 * k + 5))
            return c_sum, s_sum
        if z > 0.0:
            root = np.sqrt(z)
            return 2.0 * np.sin(0.5 * root) ** 2 / z, (root - np.sin(root)) / root ** 3
        root = np.sqrt(-z)
        return 2.0 * np.sinh(0.5 * root) ** 2 / -z, (np.sinh(root) - root) / root ** 3

    def _is_function(value):
        return callable(value)

    try:
        vector = np.asarray(state, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("state must hold six finite numbers") from None
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError("state must hold six finite numbers")
    if not (_is_number(mu) and mu > 0.0):
        raise ValueError("mu must be a finite positive number")
    if not _is_number(dt):
        raise ValueError("dt must be a finite real number")
    if not _is_function(anomaly_fn):
        raise ValueError("anomaly_fn must be callable")
    position, velocity = vector[:3], vector[3:]
    r0 = float(np.linalg.norm(position))
    if r0 == 0.0:
        raise ValueError("the position must be nonzero")
    mu, dt = float(mu), float(dt)
    vr0 = float(position @ velocity) / r0
    alpha = 2.0 / r0 - float(velocity @ velocity) / mu
    chi = anomaly_fn(r0, vr0, alpha, mu, dt)
    if not _is_number(chi):
        raise ValueError("anomaly_fn must return a finite real number")
    chi = float(chi)
    c, s = _stumpff(alpha * chi * chi)
    f = 1.0 - chi * chi * c / r0
    g = dt - chi ** 3 * s / np.sqrt(mu)
    new_position = f * position + g * velocity
    r = float(np.linalg.norm(new_position))
    f_dot = np.sqrt(mu) / (r * r0) * (alpha * chi ** 3 * s - chi)
    g_dot = 1.0 - chi * chi * c / r
    new_velocity = f_dot * position + g_dot * velocity
    return np.concatenate([new_position, new_velocity])

import numpy as np
def evaluate_radial_track(
    r_ref: float,
    vr_ref: float,
    vt_ref: float,
    mu: float,
    dt: float,
    anomaly_fn: "Callable[..., float]",
) -> "np.ndarray":
    """Reference implementation (squared distance from the Lagrange coefficients)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _stumpff(z):
        # C(z) and S(z); the power series avoids cancellation near z = 0.
        if abs(z) < 0.1:
            c_term, s_term, c_sum, s_sum = 0.5, 1.0 / 6.0, 0.0, 0.0
            for k in range(10):
                c_sum += c_term
                s_sum += s_term
                c_term *= -z / ((2 * k + 3) * (2 * k + 4))
                s_term *= -z / ((2 * k + 4) * (2 * k + 5))
            return c_sum, s_sum
        if z > 0.0:
            root = np.sqrt(z)
            return 2.0 * np.sin(0.5 * root) ** 2 / z, (root - np.sin(root)) / root ** 3
        root = np.sqrt(-z)
        return 2.0 * np.sinh(0.5 * root) ** 2 / -z, (np.sinh(root) - root) / root ** 3

    def _is_function(value):
        return callable(value)

    names = ("r_ref", "vr_ref", "vt_ref", "mu", "dt")
    values = (r_ref, vr_ref, vt_ref, mu, dt)
    for name, value in zip(names, values):
        if not _is_number(value):
            raise ValueError(f"{name} must be a finite real number")
    r0, vr0, vt0, mu, dt = (float(value) for value in values)
    if r0 <= 0.0 or mu <= 0.0:
        raise ValueError("r_ref and mu must be positive")
    if vt0 < 0.0:
        raise ValueError("vt_ref must be nonnegative")
    if not _is_function(anomaly_fn):
        raise ValueError("anomaly_fn must be callable")
    speed_sq = vr0 * vr0 + vt0 * vt0
    alpha = 2.0 / r0 - speed_sq / mu
    chi = anomaly_fn(r0, vr0, alpha, mu, dt)
    if not _is_number(chi):
        raise ValueError("anomaly_fn must return a finite real number")
    chi = float(chi)
    c, s = _stumpff(alpha * chi * chi)
    f = 1.0 - chi * chi * c / r0
    g = dt - chi ** 3 * s / np.sqrt(mu)
    # |f r0 + g v0|^2 needs only r0, v0 and the radial velocity r0 . v0 / r0.
    r = np.sqrt(r0 * r0 * f * f + speed_sq * g * g + 2.0 * f * g * r0 * vr0)
    if not (np.isfinite(r) and r > 0.0):
        raise ValueError("the distance history reaches the attracting centre at dt")
    f_dot = np.sqrt(mu) / (r * r0) * (alpha * chi ** 3 * s - chi)
    g_dot = 1.0 - chi * chi * c / r
    r_dot = (r0 * r0 * f * f_dot + speed_sq * g * g_dot + (f * g_dot + f_dot * g) * r0 * vr0) / r
    return np.array([float(r), float(r_dot)])

import numpy as np
def solve_light_time_range(
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

import numpy as np
def construct_emission_state(
    direction: "np.ndarray",
    observer: "np.ndarray",
    rho: float,
    radial_rate: float,
    angular_momentum: float,
    inclination: float,
    kappa: int,
) -> "np.ndarray":
    """Reference implementation (heading of the tangential velocity from the z-angular momentum)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

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
    if not (_is_number(rho) and rho > 0.0):
        raise ValueError("rho must be a finite positive number")
    if not _is_number(radial_rate):
        raise ValueError("radial_rate must be a finite real number")
    if not (_is_number(angular_momentum) and angular_momentum > 0.0):
        raise ValueError("angular_momentum must be a finite positive number")
    if not (_is_number(inclination) and 0.0 <= inclination <= np.pi):
        raise ValueError("inclination must lie in [0, pi]")
    if not (_is_integer(kappa) and kappa in (1, -1)):
        raise ValueError("kappa must be +1 or -1")
    point = origin + float(rho) * unit
    radius = float(np.linalg.norm(point))
    in_plane = float(np.hypot(point[0], point[1]))
    if in_plane == 0.0:
        raise ValueError("the position must not lie on the z axis")
    cos_lat, sin_lat = in_plane / radius, float(point[2]) / radius
    cos_lon, sin_lon = float(point[0]) / in_plane, float(point[1]) / in_plane
    # The z-angular momentum h cos(i) equals r * v_t * cos(psi) * cos(latitude),
    # where psi is the heading of the tangential velocity from east toward north.
    cos_psi = np.cos(inclination) / cos_lat
    if abs(cos_psi) > 1.0 + 1e-12:
        raise ValueError("no orbit of this inclination reaches the latitude of the position")
    cos_psi = float(np.clip(cos_psi, -1.0, 1.0))
    # Evaluate the latitude component without subtracting cos(psi)^2
    # from one: rounding of cos(latitude) can create a false vertical
    # velocity even for an exactly equatorial orbit.
    sin_inc = float(np.sin(inclination))
    latitude_gap = (sin_inc - abs(sin_lat)) * (sin_inc + abs(sin_lat))
    sin_psi = int(kappa) * np.sqrt(max(0.0, latitude_gap)) / cos_lat
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    tangential = angular_momentum / radius
    velocity = (float(radial_rate) * point / radius
                + tangential * (cos_psi * east + sin_psi * north))
    return np.concatenate([point, velocity]).astype(float)

import numpy as np
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

import numpy as np
def maximize_stacked_significance(
    longitudes: "np.ndarray",
    latitudes: "np.ndarray",
    flux_errors: "np.ndarray",
    flux: float,
    psf_sigma: float,
) -> "np.ndarray":
    """Reference implementation (multi-start mean shift with the exact spherical angular response)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _series(value, name):
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a nonempty 1-D array of finite numbers") from None
        if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a nonempty 1-D array of finite numbers")
        return array

    lon = _series(longitudes, "longitudes")
    lat = _series(latitudes, "latitudes")
    err = _series(flux_errors, "flux_errors")
    if not lon.size == lat.size == err.size:
        raise ValueError("longitudes, latitudes and flux_errors must have equal lengths")
    if np.any(np.abs(lat) > 90.0) or np.any(err <= 0.0):
        raise ValueError("latitudes must lie in [-90, 90] and flux errors must be positive")
    if not (_is_number(flux) and flux > 0.0 and _is_number(psf_sigma) and psf_sigma > 0.0):
        raise ValueError("flux and psf_sigma must be finite positive numbers")
    arcsec = np.pi / 648000.0
    lam, bet = np.radians(lon), np.radians(lat)
    units = np.column_stack([np.cos(bet) * np.cos(lam), np.cos(bet) * np.sin(lam), np.sin(bet)])
    weights = 1.0 / err ** 2
    # An exposure registered a distance d away enters the matched filter through
    # the PSF autocorrelation, a Gaussian of variance 2 sigma^2 per axis.
    width = 4.0 * float(psf_sigma) ** 2

    def _separations(center):
        return np.arctan2(np.linalg.norm(np.cross(units, center), axis=1), units @ center) / arcsec

    def _climb(center):
        # Riemannian mean shift uses exact angular separations and the
        # spherical logarithm, avoiding the tan(distance) approximation.
        for _ in range(20000):
            east = np.cross([0.0, 0.0, 1.0], center)
            east = east / np.linalg.norm(east) if np.linalg.norm(east) > 1e-12 else np.array([0.0, 1.0, 0.0])
            north = np.cross(center, east)
            sine = np.linalg.norm(np.cross(units, center), axis=1)
            angle = np.arctan2(sine, units @ center)
            kernel = weights * np.exp(-(angle / arcsec) ** 2 / width)
            if kernel.sum() == 0.0:
                break
            factor = np.ones_like(angle)
            away = sine > 1e-15
            factor[away] = angle[away] / sine[away]
            tangent = np.column_stack([units @ east, units @ north]) * factor[:, None]
            step = kernel @ tangent / kernel.sum()
            length = float(np.linalg.norm(step))
            if length / arcsec <= 1e-10:
                break
            direction = (step[0] * east + step[1] * north) / length
            moved = np.cos(length) * center + np.sin(length) * direction
            moved /= np.linalg.norm(moved)
            shift = float(np.arctan2(np.linalg.norm(np.cross(moved, center)), moved @ center)) / arcsec
            center = moved
            if shift <= 1e-10:
                break
        return center

    mean = weights @ units
    starts = ([mean / np.linalg.norm(mean)] if np.linalg.norm(mean) > 0.0 else []) + list(units)
    best, best_signal = None, -np.inf
    for start in starts:
        peak = _climb(np.array(start, dtype=float))
        signal = float(weights @ np.exp(-_separations(peak) ** 2 / width))
        if best is None or signal > best_signal * (1.0 + 1e-12):
            best, best_signal = peak, signal
    significance = float(flux) * best_signal / np.sqrt(weights.sum())
    longitude = float(np.degrees(np.arctan2(best[1], best[0])) % 360.0)
    if longitude >= 360.0:
        longitude -= 360.0
    latitude = float(np.degrees(np.arctan2(best[2], np.hypot(best[0], best[1]))))
    return np.array([significance, longitude, latitude, float(np.max(_separations(best)))])

import numpy as np
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
        return solve_universal_anomaly(r0, vr0, alpha, mu, dt)

    def _radial(t):
        return evaluate_radial_track(r_ref, condition[1], condition[2], gm, t, _anomaly)

    def _light(direction, observer, t_obs, radial_fn, speed):
        return solve_light_time_range(direction, observer, t_obs, radial_fn, speed)

    def _state(direction, observer, rho, radial_rate, angular_momentum, tilt, sense):
        return construct_emission_state(direction, observer, rho, radial_rate,
                                                angular_momentum, tilt, sense)

    def _propagate(state, dt):
        return propagate_two_body_state(state, gm, dt, _anomaly)

    canvas = np.array([
        register_detection(ra, dec, t, station, condition, obliquity, light_speed,
                                   _radial, _light, _state, _propagate)
        for ra, dec, t, station in zip(ras, decs, epochs, stations)
    ])
    peak = maximize_stacked_significance(canvas[:, 0], canvas[:, 1], errors, flux, psf_sigma)
    return float(2.5 * np.log10(peak[0] / float(threshold)))
SCICODE_GOLD_EOF
