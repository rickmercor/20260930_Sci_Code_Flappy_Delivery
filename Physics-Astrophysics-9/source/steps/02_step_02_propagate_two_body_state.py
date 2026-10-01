"""
Advance a position and velocity along its two-body orbit by a given time with universal-variable Lagrange coefficients.

Once a pixel has been assigned a full state at the moment its light left the source, the state must be carried along the Keplerian orbit to the common reference epoch of the stack.

Returns
-------
np.ndarray: shape (6,), the propagated position and velocity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_two_body_state(
    state: "np.ndarray",
    mu: float,
    dt: float,
    anomaly_fn: "Callable[..., float]",
) -> "np.ndarray":
    """Return the two-body state reached a time ``dt`` after an epoch.

    ``state = (x, y, z, vx, vy, vz)`` is the position and velocity relative
    to an attracting centre of gravitational parameter ``mu`` at the epoch,
    in one consistent unit system. Advance it along its Keplerian orbit with
    the universal-variable Lagrange coefficients ``f`` and ``g`` and their
    time derivatives. The universal anomaly is
    ``anomaly_fn(r0, vr0, alpha, mu, dt)``, which follows the contract of
    ``solve_universal_anomaly`` (distance ``r0``, radial velocity ``vr0`` and
    reciprocal semi-major axis ``alpha = 2 / r0 - v0**2 / mu`` at the epoch).
    Return the new state with relative accuracy ``1e-12``.

    Parameters
    ----------
    state : "np.ndarray"
        Six finite numbers; the position must be nonzero.
    mu : float
        Positive gravitational parameter.
    dt : float
        Time from the epoch; negative values run backwards.
    anomaly_fn : callable
        Universal-anomaly solver with the signature of
        ``solve_universal_anomaly``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(6,)``: position then velocity after ``dt``.

    Raises
    ------
    ValueError
        If ``state`` is not six finite numbers or has a zero position,
        ``mu`` is not a finite positive number, ``dt`` is not a finite real
        number, ``anomaly_fn`` is not callable, or ``anomaly_fn`` returns
        anything other than a finite real number.
    """
    return new_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_propagate_two_body_state(
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
        "def _probe(v):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    if v.shape != (6,):\n"
        "        return -1.0e3\n"
        "    return float(v @ np.array([1.0, -2.0, 3.0, 0.5, -1.5, 2.5]))\n"
        "ell = np.array([1.1, 0.2, -0.1, -0.15, 0.9, 0.12])\n"
        "hyp = np.array([1.0, 0.0, 0.2, 0.3, 1.5, 0.1])\n"
        "kbo = np.array([-12.03, -29.66, 1.62, 3.13e-3, -1.28e-3, 5.2e-4])\n"
        "mu_sun = 0.01720209895 ** 2\n"
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
            "call": "_probe(propagate_two_body_state(ell.copy(), 1.0, 1.7, _chi))",
            "gold_call": "_probe(_oracle_propagate_two_body_state(ell.copy(), 1.0, 1.7, _chi))",
        },
        {
            "setup": kepler,
            "call": "_probe(propagate_two_body_state(hyp.copy(), 1.0, 2.0, _chi))",
            "gold_call": "_probe(_oracle_propagate_two_body_state(hyp.copy(), 1.0, 2.0, _chi))",
        },
        {
            "setup": kepler,
            "call": "_probe(propagate_two_body_state(ell.copy(), 1.0, -1.3, _chi))",
            "gold_call": "_probe(_oracle_propagate_two_body_state(ell.copy(), 1.0, -1.3, _chi))",
        },
        {
            "setup": kepler,
            "call": "_probe(propagate_two_body_state(hyp.copy(), 1.0, 0.0, _chi))",
            "gold_call": "_probe(_oracle_propagate_two_body_state(hyp.copy(), 1.0, 0.0, _chi))",
        },
        {
            "setup": kepler,
            "call": "float(propagate_two_body_state(ell.copy(), 1.0, 1.7, _chi)[1])",
            "gold_call": "float(_oracle_propagate_two_body_state(ell.copy(), 1.0, 1.7, _chi)[1])",
        },
        {
            "setup": kepler,
            "call": "1.0e3 * float(propagate_two_body_state(kbo.copy(), mu_sun, 7.54, _chi)[3])",
            "gold_call": "1.0e3 * float(_oracle_propagate_two_body_state(kbo.copy(), mu_sun, 7.54, _chi)[3])",
        },
        {
            "setup": kepler,
            "call": "float(propagate_two_body_state(kbo.copy(), mu_sun, -7.54, _chi)[2])",
            "gold_call": "float(_oracle_propagate_two_body_state(kbo.copy(), mu_sun, -7.54, _chi)[2])",
        },
        {
            "setup": kepler + status,
            "call": "_status(lambda: propagate_two_body_state(np.array([0.0, 0.0, 0.0, 0.1, 0.9, 0.0]), 1.0, 1.0, _chi))",
            "gold_call": "_status(lambda: _oracle_propagate_two_body_state(np.array([0.0, 0.0, 0.0, 0.1, 0.9, 0.0]), 1.0, 1.0, _chi))",
        },
        {
            "setup": kepler + status,
            "call": "_status(lambda: propagate_two_body_state(ell.copy(), 0.0, 1.0, _chi))",
            "gold_call": "_status(lambda: _oracle_propagate_two_body_state(ell.copy(), 0.0, 1.0, _chi))",
        },
        {
            "setup": kepler + status,
            "call": "_status(lambda: propagate_two_body_state(ell[:5].copy(), 1.0, 1.0, _chi))",
            "gold_call": "_status(lambda: _oracle_propagate_two_body_state(ell[:5].copy(), 1.0, 1.0, _chi))",
        },
    ]
