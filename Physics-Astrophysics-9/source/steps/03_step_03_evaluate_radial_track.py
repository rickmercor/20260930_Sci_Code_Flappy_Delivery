"""
Evaluate the barycentric distance and its rate of change at an arbitrary epoch for every two-body orbit that shares an asserted distance, radial velocity and tangential speed at the reference epoch.

In Keplerian motion the distance history is fixed by the distance, radial velocity and speed at one epoch and does not depend on the orientation of the orbit, so it can be computed once for all sky positions of a trial orbit.

Returns
-------
np.ndarray: [r, rdot], the distance and its rate of change after dt.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_radial_track(
    r_ref: float,
    vr_ref: float,
    vt_ref: float,
    mu: float,
    dt: float,
    anomaly_fn: "Callable[..., float]",
) -> "np.ndarray":
    """Return the distance and distance rate a time ``dt`` after the reference epoch.

    Every two-body orbit that, at the reference epoch, lies at distance
    ``r_ref`` from the attracting centre (gravitational parameter ``mu``) with
    radial velocity ``vr_ref`` and speed ``vt_ref`` perpendicular to its
    radius vector shares one distance history ``r(t)``, whatever the
    orientation of the orbit. Return ``r`` and ``dr/dt`` at
    ``reference epoch + dt``, with relative accuracy ``1e-12``. The universal
    anomaly is ``anomaly_fn(r0, vr0, alpha, mu, dt)``, which follows the
    contract of ``solve_universal_anomaly`` (``alpha = 2 / r0 - v0**2 / mu``).

    Parameters
    ----------
    r_ref : float
        Positive distance at the reference epoch.
    vr_ref : float
        Radial velocity at the reference epoch.
    vt_ref : float
        Nonnegative speed perpendicular to the radius vector at the
        reference epoch.
    mu : float
        Positive gravitational parameter.
    dt : float
        Time from the reference epoch; negative values run backwards.
    anomaly_fn : callable
        Universal-anomaly solver with the signature of
        ``solve_universal_anomaly``.

    Returns
    -------
    np.ndarray
        Float array ``[r, rdot]``.

    Raises
    ------
    ValueError
        If any number is not finite and real (booleans are rejected),
        ``r_ref`` or ``mu`` is not positive, ``vt_ref`` is negative,
        ``anomaly_fn`` is not callable, ``anomaly_fn`` returns anything
        other than a finite real number, or the distance history reaches the
        attracting centre at ``dt``.
    """
    return track

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_evaluate_radial_track(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    kepler = (
        "import numpy as np\n"
        "def _cs(z):\n"
        "    if abs(z) < 1e-6:\n"
        "        return 0.5 - z / 24.0, 1.0 / 6.0 - z / 120.0\n"
        "    if z > 0:\n"
        "        q = np.sqrt(z)\n"
        "        return (1.0 - np.cos(q)) / z, (q - np.sin(q)) / q ** 3\n"
        "    q = np.sqrt(-z)\n"
        "    return (np.cosh(q) - 1.0) / -z, (np.sinh(q) - q) / q ** 3\n"
        "def _chi(r0, vr0, alpha, mu, dt):\n"
        "    sm = np.sqrt(mu)\n"
        "    x = sm * dt / r0\n"
        "    for _ in range(60):\n"
        "        c, s = _cs(alpha * x * x)\n"
        "        f = r0 * vr0 / sm * x * x * c + (1.0 - alpha * r0) * x ** 3 * s + r0 * x - sm * dt\n"
        "        d = r0 * vr0 / sm * x * (1.0 - alpha * x * x * s) + (1.0 - alpha * r0) * x * x * c + r0\n"
        "        x -= f / d\n"
        "    return float(x)\n"
        "def _pair(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (2,):\n"
        "        return -1.0e3\n"
        "    return float(a[0] + 3.0 * a[1])\n"
        "mu_sun = 0.01720209895 ** 2\n"
        "kms = 86400.0 / 149597870.7\n"
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
            "setup": kepler,
            "call": "_pair(evaluate_radial_track(1.0, 0.2, 1.1, 1.0, 1.3, _chi))",
            "gold_call": "_pair(_oracle_evaluate_radial_track(1.0, 0.2, 1.1, 1.0, 1.3, _chi))",
        },
        {
            "setup": kepler,
            "call": "_pair(evaluate_radial_track(1.0, 0.0, 1.2, 1.0, -0.8, _chi))",
            "gold_call": "_pair(_oracle_evaluate_radial_track(1.0, 0.0, 1.2, 1.0, -0.8, _chi))",
        },
        {
            "setup": kepler,
            "call": "float(evaluate_radial_track(1.0, 0.0, 1.2, 1.0, 0.8, _chi)[1])",
            "gold_call": "float(_oracle_evaluate_radial_track(1.0, 0.0, 1.2, 1.0, 0.8, _chi)[1])",
        },
        {
            "setup": kepler,
            "call": "_pair(evaluate_radial_track(1.2, -0.3, 1.6, 1.0, 2.4, _chi))",
            "gold_call": "_pair(_oracle_evaluate_radial_track(1.2, -0.3, 1.6, 1.0, 2.4, _chi))",
        },
        {
            "setup": kepler,
            "call": "_pair(evaluate_radial_track(1.5, 0.3, 0.0, 1.0, 0.6, _chi))",
            "gold_call": "_pair(_oracle_evaluate_radial_track(1.5, 0.3, 0.0, 1.0, 0.6, _chi))",
        },
        {
            "setup": kepler,
            "call": "float(100.0 * (evaluate_radial_track(32.006, -0.956 * kms, 5.6565 * kms, mu_sun, -7.5365, _chi)[0] - 32.0))",
            "gold_call": "float(100.0 * (_oracle_evaluate_radial_track(32.006, -0.956 * kms, 5.6565 * kms, mu_sun, -7.5365, _chi)[0] - 32.0))",
        },
        {
            "setup": kepler,
            "call": "float(evaluate_radial_track(28.4, 1.3 * kms, 5.1 * kms, mu_sun, 6.2, _chi)[1] / kms)",
            "gold_call": "float(_oracle_evaluate_radial_track(28.4, 1.3 * kms, 5.1 * kms, mu_sun, 6.2, _chi)[1] / kms)",
        },
        {
            "setup": kepler,
            "call": "_pair(evaluate_radial_track(0.9, 0.25, 0.95, 1.0, 0.0, _chi))",
            "gold_call": "_pair(_oracle_evaluate_radial_track(0.9, 0.25, 0.95, 1.0, 0.0, _chi))",
        },
        {
            "setup": kepler + status,
            "call": "_status(lambda: evaluate_radial_track(1.0, 0.2, -0.5, 1.0, 1.0, _chi))",
            "gold_call": "_status(lambda: _oracle_evaluate_radial_track(1.0, 0.2, -0.5, 1.0, 1.0, _chi))",
        },
        {
            "setup": kepler + status,
            "call": "_status(lambda: evaluate_radial_track(-1.0, 0.2, 0.5, 1.0, 1.0, _chi))",
            "gold_call": "_status(lambda: _oracle_evaluate_radial_track(-1.0, 0.2, 0.5, 1.0, 1.0, _chi))",
        },
    ]
