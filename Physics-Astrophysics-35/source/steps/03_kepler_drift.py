"""
Advance a relative position and velocity along their exact Keplerian orbit for a given time.

The Kepler part of a Wisdom-Holman map is advanced exactly. Under the potential -gm/|q| per unit mass, a relative position q and velocity v move along their conic section, and this step returns the position and velocity a time dt later.

The orbit may be elliptic, parabolic or hyperbolic, dt may be negative, and dt may span several orbital periods, so the method must hold for every conic type and must not assume that dt is small. The result must be accurate to near machine precision, because a simulation repeats this step thousands of times. No solution method is prescribed.

Returns
-------
np.ndarray of shape (2, 3), float: the propagated position (row 0) and velocity (row 1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def kepler_drift(q: np.ndarray, v: np.ndarray, gm: float, dt: float) -> np.ndarray:
    """Exact Keplerian propagation of a relative position and velocity.

    Parameters
    ----------
    q : np.ndarray
        Shape (3,), relative position (au); finite and not the zero vector.
    v : np.ndarray
        Shape (3,), relative velocity (au / yr); finite.
    gm : float
        Gravitational parameter of the central attraction (au^3 / yr^2), a
        finite number > 0.
    dt : float
        Time to advance (yr), any finite number (negative values propagate
        backwards).

    Returns
    -------
    qv : np.ndarray
        Shape (2, 3): qv[0] the position and qv[1] the velocity after dt.

    Raises
    ------
    ValueError
        If q or v does not have shape (3,) or contains a non-finite value; if
        q is the zero vector; if gm is not a finite number > 0; or if dt is
        not finite.
    """
    return qv  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_kepler_drift(q: np.ndarray, v: np.ndarray, gm: float, dt: float) -> np.ndarray:
    import numpy as np

    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    if q.shape != (3,) or v.shape != (3,):
        raise ValueError("q and v must both have shape (3,)")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(v))):
        raise ValueError("q and v must be finite")
    if not (isinstance(gm, (int, float, np.floating)) and np.isfinite(gm) and float(gm) > 0.0):
        raise ValueError("gm must be a finite number > 0")
    if not (isinstance(dt, (int, float, np.floating)) and np.isfinite(dt)):
        raise ValueError("dt must be a finite number")
    r0 = np.sqrt(q @ q)
    if r0 == 0.0:
        raise ValueError("q must be nonzero")
    gm, dt = float(gm), float(dt)
    if dt == 0.0:
        return np.stack([q.copy(), v.copy()])

    def stumpff(z):
        if z > 1e-6:
            s = np.sqrt(z)
            return (1.0 - np.cos(s)) / z, (s - np.sin(s)) / s ** 3
        if z < -1e-6:
            s = np.sqrt(-z)
            return (np.cosh(s) - 1.0) / (-z), (np.sinh(s) - s) / s ** 3
        # series for small |z|
        return (0.5 - z / 24.0 + z * z / 720.0 - z ** 3 / 40320.0,
                1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z ** 3 / 362880.0)

    sq = np.sqrt(gm)
    vr0 = (q @ v) / r0
    alpha = 2.0 / r0 - (v @ v) / gm            # reciprocal semimajor axis (< 0 for hyperbolic)
    # initial guess for the universal anomaly
    if alpha > 0.0:
        x = sq * alpha * dt
    else:
        x = np.sign(dt) * np.sqrt(abs(1.0 / alpha)) * np.log(1.0 + abs(dt) * sq / max(r0, 1e-300)) if alpha < 0.0 \
            else sq * dt / r0
    for _ in range(200):
        z = alpha * x * x
        C, S = stumpff(z)
        F = r0 * vr0 / sq * x * x * C + (1.0 - alpha * r0) * x ** 3 * S + r0 * x - sq * dt
        dF = r0 * vr0 / sq * x * (1.0 - alpha * x * x * S) + (1.0 - alpha * r0) * x * x * C + r0
        dx = F / dF
        x -= dx
        if abs(dx) <= 1e-15 * max(1.0, abs(x)):
            break
    else:
        raise ValueError("universal Kepler equation did not converge")
    z = alpha * x * x
    C, S = stumpff(z)
    f = 1.0 - x * x / r0 * C
    g = dt - x ** 3 / sq * S
    qn = f * q + g * v
    rn = np.sqrt(qn @ qn)
    fdot = sq / (rn * r0) * (alpha * x ** 3 * S - x)
    gdot = 1.0 - x * x / rn * C
    vn = fdot * q + gdot * v
    return np.stack([qn, vn])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: an inclined elliptic orbit advanced by a fraction of its period ---
        {
            "setup": """import numpy as np
q = np.array([1.0, 0.0, 0.0])
v = np.array([0.0, 5.0, 0.3])
gm = 4*np.pi**2
""",
            "call": "kepler_drift(q.copy(), v.copy(), gm, 0.21)",
            "gold_call": "_oracle_kepler_drift(q.copy(), v.copy(), gm, 0.21)",
        },
        # --- Edge: an orbit with e of about 0.9 advanced across several periods ---
        {
            "setup": """import numpy as np
q = np.array([0.1, 0.0, 0.0])
v = np.array([0.0, np.sqrt(4*np.pi**2*1.9/0.1), 0.0])
gm = 4*np.pi**2
""",
            "call": "kepler_drift(q.copy(), v.copy(), gm, 7.3)",
            "gold_call": "_oracle_kepler_drift(q.copy(), v.copy(), gm, 7.3)",
        },
        # --- Boundary: a hyperbolic flyby ---
        {
            "setup": """import numpy as np
q = np.array([1.0, 0.2, 0.0])
v = np.array([0.5, 9.5, 1.0])
gm = 4*np.pi**2
""",
            "call": "kepler_drift(q.copy(), v.copy(), gm, 0.8)",
            "gold_call": "_oracle_kepler_drift(q.copy(), v.copy(), gm, 0.8)",
        },
        # --- Valid: propagation backwards in time with a heavier central mass ---
        {
            "setup": """import numpy as np
q = np.array([0.3, -0.4, 0.1])
v = np.array([4.0, 6.0, -1.0])
gm = 4*np.pi**2*1.07
""",
            "call": "kepler_drift(q.copy(), v.copy(), gm, -0.37)",
            "gold_call": "_oracle_kepler_drift(q.copy(), v.copy(), gm, -0.37)",
        },
        # --- Invalid: non-positive gravitational parameter ---
        {
            "setup": """import numpy as np
q = np.array([1.0, 0.0, 0.0])
v = np.array([0.0, 6.0, 0.0])
def run_model():
    try:
        kepler_drift(q.copy(), v.copy(), 0.0, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_kepler_drift(q.copy(), v.copy(), 0.0, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: zero position vector ---
        {
            "setup": """import numpy as np
q = np.zeros(3)
v = np.array([0.0, 6.0, 0.0])
def run_model():
    try:
        kepler_drift(q.copy(), v.copy(), 4*np.pi**2, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_kepler_drift(q.copy(), v.copy(), 4*np.pi**2, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
