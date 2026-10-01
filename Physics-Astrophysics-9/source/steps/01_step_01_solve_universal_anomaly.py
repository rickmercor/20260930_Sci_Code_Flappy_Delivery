"""
Solve the universal Kepler equation for the anomaly a two-body orbit of any conic type reaches a given time after its epoch.

Universal variables describe elliptic, parabolic and hyperbolic two-body motion with a single equation, so an asserted orbit can be advanced from its epoch without first classifying its conic.

Returns
-------
float: the universal anomaly chi reached after dt.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_universal_anomaly(r0: float, vr0: float, alpha: float, mu: float, dt: float) -> float:
    """Return the universal anomaly reached a time ``dt`` after an epoch.

    At the epoch the body is at distance ``r0`` from the attracting centre
    with radial velocity ``vr0``; ``alpha = 2 / r0 - v0**2 / mu`` is the
    reciprocal semi-major axis (positive for ellipses, zero for parabolas,
    negative for hyperbolas) and ``mu`` is the gravitational parameter, all
    in one consistent unit system. The universal anomaly ``chi`` is the
    variable for which the Lagrange coefficients at ``epoch + dt`` are
    ``f = 1 - chi**2 * C(alpha * chi**2) / r0`` and
    ``g = dt - chi**3 * S(alpha * chi**2) / sqrt(mu)``, where ``C`` and ``S``
    are the Stumpff functions. It vanishes at ``dt = 0`` and increases
    monotonically with ``dt``. Return it with relative accuracy ``1e-13``.

    Parameters
    ----------
    r0 : float
        Positive distance at the epoch.
    vr0 : float
        Radial velocity at the epoch.
    alpha : float
        Reciprocal semi-major axis; at most ``2 / r0 - vr0**2 / mu`` so that
        the speed perpendicular to the radius vector is real.
    mu : float
        Positive gravitational parameter.
    dt : float
        Time from the epoch; negative values run backwards.

    Returns
    -------
    float
        The universal anomaly ``chi``.

    Raises
    ------
    ValueError
        If any argument is not a finite real number (booleans are
        rejected), if ``r0`` or ``mu`` is not positive, or if ``alpha``
        exceeds ``2 / r0 - vr0**2 / mu``.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_universal_anomaly(r0: float, vr0: float, alpha: float, mu: float, dt: float) -> float:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    outer = (
        "import numpy as np\n"
        "mu = 0.01720209895 ** 2\n"
        "vr = -0.956 * 86400.0 / 149597870.7\n"
        "vt = 5.6565 * 86400.0 / 149597870.7\n"
        "alpha = 2.0 / 32.006 - (vr * vr + vt * vt) / mu\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "solve_universal_anomaly(1.2, 0.3, 0.8, 1.0, 2.7)",
            "gold_call": "_oracle_solve_universal_anomaly(1.2, 0.3, 0.8, 1.0, 2.7)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "solve_universal_anomaly(1.0, 0.4, -0.35, 1.0, 3.1)",
            "gold_call": "_oracle_solve_universal_anomaly(1.0, 0.4, -0.35, 1.0, 3.1)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "solve_universal_anomaly(1.5, -0.2, 0.5, 1.0, -2.2)",
            "gold_call": "_oracle_solve_universal_anomaly(1.5, -0.2, 0.5, 1.0, -2.2)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "solve_universal_anomaly(2.0, 0.5, 0.0, 1.0, 1.3)",
            "gold_call": "_oracle_solve_universal_anomaly(2.0, 0.5, 0.0, 1.0, 1.3)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "solve_universal_anomaly(1.0, 0.1, 0.02, 1.0, 1.5)",
            "gold_call": "_oracle_solve_universal_anomaly(1.0, 0.1, 0.02, 1.0, 1.5)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "solve_universal_anomaly(1.0, 0.0, 1.0, 1.0, 20.0) / 10.0",
            "gold_call": "_oracle_solve_universal_anomaly(1.0, 0.0, 1.0, 1.0, 20.0) / 10.0",
        },
        {
            "setup": outer,
            "call": "1.0e3 * solve_universal_anomaly(32.006, vr, alpha, mu, -7.5365)",
            "gold_call": "1.0e3 * _oracle_solve_universal_anomaly(32.006, vr, alpha, mu, -7.5365)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "solve_universal_anomaly(1.3, 0.2, 0.6, 1.0, 0.0)",
            "gold_call": "_oracle_solve_universal_anomaly(1.3, 0.2, 0.6, 1.0, 0.0)",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_universal_anomaly(0.0, 0.1, 0.5, 1.0, 1.0))",
            "gold_call": "_status(lambda: _oracle_solve_universal_anomaly(0.0, 0.1, 0.5, 1.0, 1.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_universal_anomaly(1.0, 0.5, 1.9, 1.0, 1.0))",
            "gold_call": "_status(lambda: _oracle_solve_universal_anomaly(1.0, 0.5, 1.9, 1.0, 1.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_universal_anomaly(1.0, 0.1, 0.5, 1.0, float('nan')))",
            "gold_call": "_status(lambda: _oracle_solve_universal_anomaly(1.0, 0.1, 0.5, 1.0, float('nan')))",
        },
    ]
