"""
Evaluate the expected significance of an inverse-variance-weighted PSF-matched-filter stack of registered exposures at its maximum, where it lies, and how far the worst-registered exposure sits from it.

Stacking pre-filtered exposures sums their signals and their noise variances, so exposures that register slightly apart on the canvas lose significance at the stack maximum and set the depth that a trial orbit actually reaches.

Returns
-------
np.ndarray: [significance at the maximum, longitude (deg), latitude (deg), largest offset (arcsec)].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def maximize_stacked_significance(
    longitudes: "np.ndarray",
    latitudes: "np.ndarray",
    flux_errors: "np.ndarray",
    flux: float,
    psf_sigma: float,
) -> "np.ndarray":
    """Return the expected significance at the maximum of a matched-filter stack.

    Exposure ``k`` of a point source of constant flux ``flux`` registers on
    the canvas at the direction (``longitudes[k]``, ``latitudes[k]``), in
    degrees, and has PSF-fit flux uncertainty ``flux_errors[k]`` in the same
    flux unit. The canvas PSF is a circular Gaussian of standard deviation
    ``psf_sigma`` arcseconds. The stack is the inverse-variance-weighted
    PSF-matched filter: every exposure is cross-correlated with the PSF, the
    filtered exposures and their filtered variances are summed over
    exposures, and the significance at a canvas direction is the summed
    filtered signal divided by the square root of the summed filtered
    variance. Return the expected (noise-free) significance at its global
    maximum over canvas directions, the direction of that maximum, and the
    largest great-circle distance between the maximum and a registered
    exposure. Locate the maximum to ``1e-9`` arcsec.

    Parameters
    ----------
    longitudes, latitudes : "np.ndarray"
        Registered directions in degrees, nonempty 1-D arrays of equal
        length; latitudes lie in ``[-90, 90]``.
    flux_errors : "np.ndarray"
        Positive PSF-fit flux uncertainties, one per exposure.
    flux : float
        Positive source flux.
    psf_sigma : float
        Positive PSF standard deviation in arcseconds.

    Returns
    -------
    np.ndarray
        Float array ``[significance, longitude, latitude, largest_offset]``:
        the significance at the maximum, the maximum's longitude in
        ``[0, 360)`` and latitude in degrees, and the largest distance in
        arcseconds from the maximum to a registered exposure.

    Raises
    ------
    ValueError
        If an array is not a nonempty 1-D array of finite numbers, the
        lengths differ, a latitude lies outside ``[-90, 90]``, a flux error
        is not positive, or ``flux`` or ``psf_sigma`` is not a finite
        positive number.
    """
    return peak

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_maximize_stacked_significance(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    drift = (
        "import numpy as np\n"
        "s = -0.06 + 0.017 * np.arange(8)\n"
        "j = np.array([0.004, -0.003, 0.002, 0.0, -0.002, 0.003, -0.004, 0.001])\n"
        "ux, uy = 0.7 / np.hypot(0.7, 0.71), 0.71 / np.hypot(0.7, 0.71)\n"
        "lat = 2.9 + (s * uy + j * ux) / 3600.0\n"
        "lon = 218.2 + (s * ux - j * uy) / 3600.0 / np.cos(np.radians(2.9))\n"
        "err = np.array([0.03, 0.034, 0.028, 0.031, 0.036, 0.029, 0.033, 0.032])\n"
        "def _pick(a, k):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (4,):\n"
        "        return -1.0e3\n"
        "    return float(a[k])\n"
        "def _lone(a):\n"
        "    return _pick(a, 0) + _pick(a, 3)\n"
    )
    twin = (
        "import numpy as np\n"
        "lon2 = np.array([10.0, 10.0000030, 9.9999975, 10.0008, 10.0008025, 10.0007980, 10.0008010, 10.0007995])\n"
        "lat2 = np.array([-5.0, -5.0000020, -5.0000015, -5.0003, -5.0003010, -5.0002990, -5.0003025, -5.0002985])\n"
        "err2 = np.array([0.02, 0.022, 0.021, 0.04, 0.041, 0.039, 0.042, 0.04])\n"
        "wrap_lon = np.array([359.999995, 0.0000010, 359.9999990, 0.0000025])\n"
        "wrap_lat = np.array([0.40, 0.4000012, 0.3999990, 0.4000021])\n"
        "def _pick(a, k):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (4,):\n"
        "        return -1.0e3\n"
        "    return float(a[k])\n"
        "def _heavier(a):\n"
        "    return _pick(a, 0) + 360.0 * (_pick(a, 1) - 10.0)\n"
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
            "setup": drift,
            "call": "_pick(maximize_stacked_significance(lon.copy(), lat.copy(), err.copy(), 0.1, 0.042), 0)",
            "gold_call": "_pick(_oracle_maximize_stacked_significance(lon.copy(), lat.copy(), err.copy(), 0.1, 0.042), 0)",
        },
        {
            "setup": drift,
            "call": "360.0 * (_pick(maximize_stacked_significance(lon.copy(), lat.copy(), err.copy(), 0.1, 0.042), 1) - 218.2)",
            "gold_call": "360.0 * (_pick(_oracle_maximize_stacked_significance(lon.copy(), lat.copy(), err.copy(), 0.1, 0.042), 1) - 218.2)",
        },
        {
            "setup": drift,
            "call": "360.0 * (_pick(maximize_stacked_significance(lon.copy(), lat.copy(), err.copy(), 0.1, 0.042), 2) - 2.9)",
            "gold_call": "360.0 * (_pick(_oracle_maximize_stacked_significance(lon.copy(), lat.copy(), err.copy(), 0.1, 0.042), 2) - 2.9)",
        },
        {
            "setup": drift,
            "call": "10.0 * _pick(maximize_stacked_significance(lon.copy(), lat.copy(), err.copy(), 0.1, 0.042), 3)",
            "gold_call": "10.0 * _pick(_oracle_maximize_stacked_significance(lon.copy(), lat.copy(), err.copy(), 0.1, 0.042), 3)",
            # The result is 10 times an offset in arcsec. Allow the
            # documented 1e-9 arcsec position error plus rounding.
            "tol": 1e-8,
        },
        {
            "setup": drift,
            "call": "_pick(maximize_stacked_significance(lon.copy(), lat.copy(), err.copy(), 0.1, 5.0), 0)",
            "gold_call": "_pick(_oracle_maximize_stacked_significance(lon.copy(), lat.copy(), err.copy(), 0.1, 5.0), 0)",
        },
        {
            "setup": drift,
            "call": "_lone(maximize_stacked_significance(np.array([218.2]), np.array([2.9]), np.array([0.03]), 0.1, 0.042))",
            "gold_call": "_lone(_oracle_maximize_stacked_significance(np.array([218.2]), np.array([2.9]), np.array([0.03]), 0.1, 0.042))",
        },
        {
            "setup": twin,
            "call": "_heavier(maximize_stacked_significance(lon2.copy(), lat2.copy(), err2.copy(), 0.1, 0.042))",
            "gold_call": "_heavier(_oracle_maximize_stacked_significance(lon2.copy(), lat2.copy(), err2.copy(), 0.1, 0.042))",
        },
        {
            "setup": twin,
            "call": "_pick(maximize_stacked_significance(wrap_lon.copy(), wrap_lat.copy(), np.array([0.03, 0.03, 0.035, 0.03]), 0.1, 0.042), 1) / 100.0",
            "gold_call": "_pick(_oracle_maximize_stacked_significance(wrap_lon.copy(), wrap_lat.copy(), np.array([0.03, 0.03, 0.035, 0.03]), 0.1, 0.042), 1) / 100.0",
        },
        {
            "setup": "import numpy as np\nlon = np.array([0.0, 60.0 / 3600.0])\nlat = np.zeros(2)\nerr = np.array([1.0, 0.5])\n",
            "call": "3600.0 * float(maximize_stacked_significance(lon.copy(), lat.copy(), err.copy(), 0.1, 90.0)[1])",
            "gold_call": "3600.0 * float(_oracle_maximize_stacked_significance(lon.copy(), lat.copy(), err.copy(), 0.1, 90.0)[1])",
            # The comparison is in arcsec. Allow the documented 1e-9
            # positional error plus reference rounding.
            "tol": 3e-11,
        },
        {
            "setup": drift + status,
            "call": "_status(lambda: maximize_stacked_significance(lon.copy(), lat.copy(), np.where(err.copy() > 0.035, 0.0, err.copy()), 0.1, 0.042))",
            "gold_call": "_status(lambda: _oracle_maximize_stacked_significance(lon.copy(), lat.copy(), np.where(err.copy() > 0.035, 0.0, err.copy()), 0.1, 0.042))",
        },
        {
            "setup": drift + status,
            "call": "_status(lambda: maximize_stacked_significance(lon.copy(), lat[:5].copy(), err.copy(), 0.1, 0.042))",
            "gold_call": "_status(lambda: _oracle_maximize_stacked_significance(lon.copy(), lat[:5].copy(), err.copy(), 0.1, 0.042))",
        },
    ]
