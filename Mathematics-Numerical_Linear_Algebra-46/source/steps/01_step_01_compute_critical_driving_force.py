"""
Calibrate the bond fracture energy of a spherical-kernel nonlocal model against Griffith's energy release rate and return the resulting normalization constant and critical crack driving force.

When damage is carried by individual bonds of a nonlocal horizon, the energy dissipated by a crack depends on how much kernel weight crosses the crack plane, so the normalization of the bond fracture energy is a property of the kernel profile.

Returns
-------
tuple[float, float]: the normalization constant c_0 and the critical crack driving force Y_c.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_critical_driving_force(
    profile_breaks: "np.ndarray",
    profile_coeffs: "np.ndarray",
    fracture_energy: float,
    horizon: float,
) -> tuple:
    """Return the fracture-energy normalization constant and critical driving force.

    The radial kernel profile ``p(rho)``, with ``rho = r / horizon`` in
    ``[0, 1]``, is piecewise polynomial: on the interval
    ``[profile_breaks[j], profile_breaks[j + 1]]`` it equals
    ``sum_m profile_coeffs[j, m] * rho**m``. The kernel is
    ``omega(dX) = C * p(|dX| / horizon)`` for any constant ``C > 0``.

    The bond-wise fracture energy of the body ``Omega`` is

        E_Gamma = int_Omega int_{H(X)} (omega(X' - X) / omega_0)
                  * G_c * s(X, X') / (c_0 * horizon) dV' dV,

    where ``H(X)`` is the ball of radius ``horizon`` around ``X``,
    ``omega_0 = int_{H(X)} omega dV'`` is the kernel integral of a full
    neighbourhood, and the bond phase field ``s`` equals one on every bond
    that crosses a planar crack in an infinite body and zero on every other
    bond. ``c_0`` is the constant for which ``E_Gamma`` equals ``G_c`` per
    unit crack area. The critical crack driving force is
    ``Y_c = G_c / (2 * c_0 * horizon)``.

    Parameters
    ----------
    profile_breaks : np.ndarray
        Strictly increasing 1D array starting at 0 and ending at 1.
    profile_coeffs : np.ndarray
        2D array with one row of ascending-power coefficients per interval.
    fracture_energy : float
        Griffith energy release rate ``G_c`` (positive).
    horizon : float
        Horizon radius ``delta`` (positive).

    Returns
    -------
    tuple
        ``(c_0, Y_c)`` as two floats.

    Raises
    ------
    ValueError
        If the breaks are not a strictly increasing 1D array of finite
        numbers from 0 to 1, if ``profile_coeffs`` is not a finite 2D array
        with one row per interval, if either moment
        ``int_0^1 p(rho) rho**2 drho`` or ``int_0^1 p(rho) rho**3 drho`` is
        not positive, or if ``fracture_energy`` or ``horizon`` is not a
        finite positive number.
    """
    return 0.0, 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_critical_driving_force(
    profile_breaks: "np.ndarray",
    profile_coeffs: "np.ndarray",
    fracture_energy: float,
    horizon: float,
) -> tuple:
    """Reference implementation (exact moments of the piecewise-polynomial profile)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _moment(breaks, coeffs, power):
        # int_0^1 p(rho) rho**power drho, integrated exactly piece by piece.
        total = 0.0
        for j in range(coeffs.shape[0]):
            low, high = breaks[j], breaks[j + 1]
            for m, a in enumerate(coeffs[j]):
                q = m + power + 1
                total += a * (high ** q - low ** q) / q
        return total

    if not (_is_number(fracture_energy) and fracture_energy > 0.0):
        raise ValueError("fracture_energy must be a finite positive number")
    if not (_is_number(horizon) and horizon > 0.0):
        raise ValueError("horizon must be a finite positive number")
    try:
        breaks = np.asarray(profile_breaks, dtype=float)
        coeffs = np.asarray(profile_coeffs, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("profile arrays must be numeric") from None
    if breaks.ndim != 1 or breaks.size < 2 or not np.all(np.isfinite(breaks)):
        raise ValueError("profile_breaks must be a finite 1D array of length >= 2")
    if breaks[0] != 0.0 or breaks[-1] != 1.0 or np.any(np.diff(breaks) <= 0.0):
        raise ValueError("profile_breaks must increase strictly from 0 to 1")
    if coeffs.ndim != 2 or coeffs.shape[0] != breaks.size - 1 or coeffs.shape[1] < 1:
        raise ValueError("profile_coeffs needs one coefficient row per interval")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("profile_coeffs must be finite")

    # A point at height z = xi * delta above the crack sees the cap of its
    # neighbourhood below the plane; the cap weight fraction is
    # f(xi) = int_xi^1 p rho (rho - xi) drho / (2 int_0^1 p rho^2 drho).
    # Both sides of the plane contribute, so c_0 = 2 int_0^1 f(xi) dxi, and
    # exchanging the order of integration leaves a ratio of two moments.
    second = _moment(breaks, coeffs, 2)
    third = _moment(breaks, coeffs, 3)
    if not (second > 0.0 and third > 0.0):
        raise ValueError("the profile moments must be positive")
    c0 = third / (2.0 * second)
    critical = float(fracture_energy) / (2.0 * c0 * float(horizon))
    return float(c0), float(critical)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return deterministic numerical comparisons with isolated inputs."""
    fixture_0 = (
        'import numpy as np\n'
        'def _pair(result):\n'
        '    c0, yc = result\n'
        '    return np.array([c0, 0.1 * yc], dtype=float)\n'
    )
    fixture_1 = (
        'import numpy as np\n'
    )
    fixture_2 = (
        'import numpy as np\n'
        'def _status(fn):\n'
        '    try:\n'
        '        fn()\n'
        '        return 0\n'
        '    except ValueError:\n'
        '        return 1\n'
        '    except Exception:\n'
        '        return 2\n'
    )
    return [
        {
            "setup": fixture_0,
            'call': '_pair(compute_critical_driving_force(np.array([0.0, 1.0]), np.array([[1.0]]), 2.0, 0.5))',
            'gold_call': '_pair(_oracle_compute_critical_driving_force(np.array([0.0, 1.0]), np.array([[1.0]]), 2.0, 0.5))',
        },
        {
            "setup": fixture_0,
            'call': '_pair(compute_critical_driving_force(np.array([0.0, 1.0]), np.array([[3.0, -3.0]]), 1.5, 0.8))',
            'gold_call': '_pair(_oracle_compute_critical_driving_force(np.array([0.0, 1.0]), np.array([[3.0, -3.0]]), 1.5, 0.8))',
        },
        {
            "setup": fixture_0,
            'call': '_pair(compute_critical_driving_force(np.array([0.0, 0.3, 1.0]), np.array([[1.0, 0.0, 0.0], [1.3, -1.0, 0.0]]), 4.0, 2.0))',
            'gold_call': '_pair(_oracle_compute_critical_driving_force(np.array([0.0, 0.3, 1.0]), np.array([[1.0, 0.0, 0.0], [1.3, -1.0, 0.0]]), 4.0, 2.0))',
        },
        {
            "setup": fixture_0,
            'call': '_pair(compute_critical_driving_force(np.array([0.0, 1.0]), np.array([[1.0, 0.0, -10.0, 20.0, -15.0, 4.0]]), 0.7, 1.25))',
            'gold_call': '_pair(_oracle_compute_critical_driving_force(np.array([0.0, 1.0]), np.array([[1.0, 0.0, -10.0, 20.0, -15.0, 4.0]]), 0.7, 1.25))',
        },
        {
            "setup": fixture_0,
            'call': '_pair(compute_critical_driving_force(np.array([0.0, 0.25, 0.6, 1.0]), np.array([[0.0, 4.0], [1.0, 0.0], [2.5, -2.5]]), 3.0, 0.4))',
            'gold_call': '_pair(_oracle_compute_critical_driving_force(np.array([0.0, 0.25, 0.6, 1.0]), np.array([[0.0, 4.0], [1.0, 0.0], [2.5, -2.5]]), 3.0, 0.4))',
        },
        {
            "setup": fixture_1,
            'call': 'compute_critical_driving_force(np.array([0.0, 1.0]), np.array([[0.0, 0.0, 0.0, 5.0]]), 1.0, 1.0)[0]',
            'gold_call': '_oracle_compute_critical_driving_force(np.array([0.0, 1.0]), np.array([[0.0, 0.0, 0.0, 5.0]]), 1.0, 1.0)[0]',
        },
        {
            "setup": fixture_2,
            'call': '_status(lambda: compute_critical_driving_force(np.array([0.0, 0.6, 0.4, 1.0]), np.ones((3, 1)), 1.0, 1.0))',
            'gold_call': '_status(lambda: _oracle_compute_critical_driving_force(np.array([0.0, 0.6, 0.4, 1.0]), np.ones((3, 1)), 1.0, 1.0))',
        },
        {
            "setup": fixture_2,
            'call': '_status(lambda: compute_critical_driving_force(np.array([0.0, 1.0]), np.array([[0.0, 1.0, -2.0]]), 1.0, 1.0))',
            'gold_call': '_status(lambda: _oracle_compute_critical_driving_force(np.array([0.0, 1.0]), np.array([[0.0, 1.0, -2.0]]), 1.0, 1.0))',
        },
        {
            "setup": fixture_2,
            'call': '_status(lambda: compute_critical_driving_force(np.array([0.0, 0.5, 1.0]), np.ones((1, 2)), 1.0, 1.0))',
            'gold_call': '_status(lambda: _oracle_compute_critical_driving_force(np.array([0.0, 0.5, 1.0]), np.ones((1, 2)), 1.0, 1.0))',
        },
        {
            "setup": fixture_2,
            'call': '_status(lambda: compute_critical_driving_force(np.array([0.0, 1.0]), np.array([[1.0]]), 0.0, 1.0))',
            'gold_call': '_status(lambda: _oracle_compute_critical_driving_force(np.array([0.0, 1.0]), np.array([[1.0]]), 0.0, 1.0))',
        },
    ]
