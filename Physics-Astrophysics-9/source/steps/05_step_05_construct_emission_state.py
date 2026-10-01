"""
Build the barycentric position and velocity, at the moment its light left it, of the source placed along a pixel's line of sight on the orbit that carries a prescribed radial velocity, angular-momentum magnitude, inclination and sense of latitude motion.

The asserted energy, angular momentum and its inclination are conserved along a Keplerian orbit, so at any position they fix the velocity up to the sign of the out-of-plane motion, and some positions cannot be reached by any orbit of the asserted inclination.

Returns
-------
np.ndarray: shape (6,), the source position and velocity at the emission epoch.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def construct_emission_state(
    direction: "np.ndarray",
    observer: "np.ndarray",
    rho: float,
    radial_rate: float,
    angular_momentum: float,
    inclination: float,
    kappa: int,
) -> "np.ndarray":
    """Return the emission-epoch state of the source a range ``rho`` along a line of sight.

    ``direction`` (a unit vector) and ``observer`` are barycentric ecliptic
    vectors whose ``z`` axis points to the north ecliptic pole, and the
    source lies at ``position = observer + rho * direction``. Return that
    position together with the velocity there of the Keplerian orbit whose
    radial velocity at ``position`` is ``radial_rate``, whose specific
    angular momentum ``r x v`` has magnitude ``angular_momentum`` and makes
    the angle ``inclination`` (radians) with the ``+z`` axis, and along which
    the ecliptic latitude ``asin(z / |r|)`` is increasing if ``kappa = +1``
    and decreasing if ``kappa = -1`` (``kappa`` makes no difference where
    the orbit is at its extreme latitude). Lengths and times follow the
    units of the inputs.

    Parameters
    ----------
    direction : "np.ndarray"
        Unit vector of shape ``(3,)`` (norm within ``1e-9`` of one).
    observer : "np.ndarray"
        Barycentric observer position of shape ``(3,)``.
    rho : float
        Positive range along the line of sight.
    radial_rate : float
        Velocity component along the barycentric radius vector.
    angular_momentum : float
        Positive magnitude of the specific angular momentum.
    inclination : float
        Angle between the angular momentum and ``+z``, in ``[0, pi]``.
    kappa : int
        ``+1`` or ``-1``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(6,)``: position then velocity.

    Raises
    ------
    ValueError
        If ``direction`` or ``observer`` is not three finite numbers,
        ``direction`` is not a unit vector, ``rho`` is not a finite positive
        number, the position lies on the ``z`` axis, ``radial_rate`` is not
        finite, ``angular_momentum`` is not a finite positive number,
        ``inclination`` is not a finite number in ``[0, pi]``, ``kappa`` is
        not the integer ``+1`` or ``-1`` (booleans are rejected), or no orbit
        of this inclination passes through the position because the
        magnitude of its latitude exceeds ``min(inclination, pi -
        inclination)`` beyond rounding.
    """
    return emission_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_construct_emission_state(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    probe = (
        "import numpy as np\n"
        "def _aim(target, observer):\n"
        "    offset = np.asarray(target, dtype=float) - np.asarray(observer, dtype=float)\n"
        "    rho = float(np.linalg.norm(offset))\n"
        "    return offset / rho, rho\n"
        "def _probe(v, scale=1.0):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    if v.shape != (6,):\n"
        "        return -1.0e3\n"
        "    return float(scale * (v[3:] @ np.array([1.0, -2.0, 3.0])) + 0.1 * (v[:3] @ np.array([0.5, 0.25, -1.0])))\n"
        "obs_near = np.array([0.1, -0.2, 0.05])\n"
        "obs_far = np.array([-0.5326, 0.7595, 0.3293])\n"
        "u_near, rho_near = _aim([0.6, 0.8, 0.3], obs_near)\n"
        "u_far, rho_far = _aim([-5.02, -25.1, 1.61], obs_far)\n"
        "u_flat, rho_flat = _aim([1.3, -0.4, 0.0], [0.2, 0.1, 0.0])\n"
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
            "setup": probe,
            "call": "_probe(construct_emission_state(u_near.copy(), obs_near.copy(), rho_near, 0.2, 1.1, 0.9, 1))",
            "gold_call": "_probe(_oracle_construct_emission_state(u_near.copy(), obs_near.copy(), rho_near, 0.2, 1.1, 0.9, 1))",
        },
        {
            "setup": probe,
            "call": "_probe(construct_emission_state(u_near.copy(), obs_near.copy(), rho_near, 0.2, 1.1, 0.9, -1))",
            "gold_call": "_probe(_oracle_construct_emission_state(u_near.copy(), obs_near.copy(), rho_near, 0.2, 1.1, 0.9, -1))",
        },
        {
            "setup": probe,
            "call": "_probe(construct_emission_state(u_near.copy(), obs_near.copy(), rho_near, -0.1, 0.8, 2.6, 1))",
            "gold_call": "_probe(_oracle_construct_emission_state(u_near.copy(), obs_near.copy(), rho_near, -0.1, 0.8, 2.6, 1))",
        },
        {
            "setup": probe,
            "call": "_probe(construct_emission_state(u_far.copy(), obs_far.copy(), rho_far, -5.52e-4, 0.0833, 0.16, 1), 1.0e3)",
            "gold_call": "_probe(_oracle_construct_emission_state(u_far.copy(), obs_far.copy(), rho_far, -5.52e-4, 0.0833, 0.16, 1), 1.0e3)",
        },
        {
            "setup": probe,
            "call": "1.0e3 * float(construct_emission_state(u_far.copy(), obs_far.copy(), rho_far, -5.52e-4, 0.0833, 0.16, 1)[5])",
            "gold_call": "1.0e3 * float(_oracle_construct_emission_state(u_far.copy(), obs_far.copy(), rho_far, -5.52e-4, 0.0833, 0.16, 1)[5])",
        },
        {
            "setup": probe,
            "call": "_probe(construct_emission_state(u_flat.copy(), np.array([0.2, 0.1, 0.0]), rho_flat, 0.05, 0.9, 0.0, -1))",
            "gold_call": "_probe(_oracle_construct_emission_state(u_flat.copy(), np.array([0.2, 0.1, 0.0]), rho_flat, 0.05, 0.9, 0.0, -1))",
        },
        {
            "setup": probe + status,
            "call": "_status(lambda: construct_emission_state(u_near.copy(), obs_near.copy(), rho_near, 0.2, 1.1, 0.2, 1))",
            "gold_call": "_status(lambda: _oracle_construct_emission_state(u_near.copy(), obs_near.copy(), rho_near, 0.2, 1.1, 0.2, 1))",
        },
        {
            "setup": probe + status,
            "call": "_status(lambda: construct_emission_state(u_near.copy(), obs_near.copy(), rho_near, 0.2, 1.1, 0.9, 0))",
            "gold_call": "_status(lambda: _oracle_construct_emission_state(u_near.copy(), obs_near.copy(), rho_near, 0.2, 1.1, 0.9, 0))",
        },
        {
            "setup": probe + status,
            "call": "_status(lambda: construct_emission_state(u_near.copy(), obs_near.copy(), rho_near, 0.2, 1.1, 0.9, True))",
            "gold_call": "_status(lambda: _oracle_construct_emission_state(u_near.copy(), obs_near.copy(), rho_near, 0.2, 1.1, 0.9, True))",
        },
        {
            "setup": probe + status,
            "call": "_status(lambda: construct_emission_state(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.5]), 1.5, 0.2, 1.1, 0.9, 1))",
            "gold_call": "_status(lambda: _oracle_construct_emission_state(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.5]), 1.5, 0.2, 1.1, 0.9, 1))",
        },
        {
            "setup": probe + status,
            "call": "_status(lambda: construct_emission_state(u_near.copy(), obs_near.copy(), -1.0, 0.2, 1.1, 0.9, 1))",
            "gold_call": "_status(lambda: _oracle_construct_emission_state(u_near.copy(), obs_near.copy(), -1.0, 0.2, 1.1, 0.9, 1))",
        },
        {
            "setup": probe + status,
            "call": "_status(lambda: construct_emission_state(2.0 * u_near.copy(), obs_near.copy(), rho_near, 0.2, 1.1, 0.9, 1))",
            "gold_call": "_status(lambda: _oracle_construct_emission_state(2.0 * u_near.copy(), obs_near.copy(), rho_near, 0.2, 1.1, 0.9, 1))",
        },
    ]
