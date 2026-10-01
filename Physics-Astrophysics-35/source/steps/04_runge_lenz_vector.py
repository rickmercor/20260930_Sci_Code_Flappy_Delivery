"""
Compute the Runge-Lenz vector of the relative two-body orbit from its democratic heliocentric coordinates.

For two bodies in democratic heliocentric coordinates, Q is the companion's position relative to the star and P is its barycentric momentum. The centre-of-mass pair decouples, and the relative motion follows H = |P|^2/(2 mu) - G m0 m1/|Q|.

Its Runge-Lenz vector is written R = P x L - c0 Q/|Q| with L = Q x P, where c0 is the constant that makes R conserved under H, the same constant as in Step 1. R points from the barycentre towards periapsis, and its magnitude is proportional to the eccentricity. The direction of R is what defines the orbit's line of apsides throughout this pipeline.

Returns
-------
np.ndarray of shape (3,), float: the Runge-Lenz vector R = P x L - c0 Q/|Q|.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def runge_lenz_vector(Q: np.ndarray, P: np.ndarray, m0: float, m1: float) -> np.ndarray:
    """Runge-Lenz vector of the two-body relative orbit.

    Parameters
    ----------
    Q : np.ndarray
        Shape (3,), companion position relative to the star (au); finite, nonzero.
    P : np.ndarray
        Shape (3,), barycentric momentum of the companion (Msun au / yr); finite.
    m0 : float
        Mass of the star (Msun), a finite number > 0.
    m1 : float
        Mass of the companion (Msun), a finite number > 0.

    Returns
    -------
    R : np.ndarray
        Shape (3,), the Runge-Lenz vector P x L - c0 Q/|Q| with L = Q x P.

    Raises
    ------
    ValueError
        If Q or P does not have shape (3,) or contains a non-finite value; if
        Q is the zero vector; or if m0 or m1 is not a finite number > 0.
    """
    return R  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_runge_lenz_vector(Q: np.ndarray, P: np.ndarray, m0: float, m1: float) -> np.ndarray:
    import numpy as np

    Q = np.asarray(Q, dtype=float)
    P = np.asarray(P, dtype=float)
    if Q.shape != (3,) or P.shape != (3,):
        raise ValueError("Q and P must both have shape (3,)")
    if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(P))):
        raise ValueError("Q and P must be finite")
    for name, val in (("m0", m0), ("m1", m1)):
        if not (isinstance(val, (int, float, np.floating)) and np.isfinite(val) and float(val) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    q = np.sqrt(Q @ Q)
    if q == 0.0:
        raise ValueError("Q must be nonzero")
    G = 4.0 * np.pi ** 2
    m0, m1 = float(m0), float(m1)
    c0 = G * m0 ** 2 * m1 ** 2 / (m0 + m1)
    L = np.cross(Q, P)
    return np.cross(P, L) - c0 * Q / q

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark orbit at pericentre (R along +x) ---
        {
            "setup": """import numpy as np
def digest(x):
    # log-magnitude plus unit direction, so the result is compared at relative precision
    x = np.asarray(x, dtype=float).ravel()
    n = float(np.sqrt(x @ x))
    if not np.isfinite(n) or n == 0.0:
        return (0.0,) * (x.size + 1)
    return tuple([float(np.log(n))] + [float(c) for c in x / n])
m0, m1, a, e = 1.0, 0.07, 0.06, 0.5
M = m0 + m1
mu = m0*m1/M
Q = np.array([a*(1-e), 0.0, 0.0])
P = mu*np.array([0.0, np.sqrt(4*np.pi**2*M*(1+e)/(a*(1-e))), 0.0])
""",
            "call": "digest(runge_lenz_vector(Q.copy(), P.copy(), m0, m1))",
            "gold_call": "digest(_oracle_runge_lenz_vector(Q.copy(), P.copy(), m0, m1))",
        },
        # --- Valid: a generic inclined state with a radial velocity component ---
        {
            "setup": """import numpy as np
def digest(x):
    # log-magnitude plus unit direction, so the result is compared at relative precision
    x = np.asarray(x, dtype=float).ravel()
    n = float(np.sqrt(x @ x))
    if not np.isfinite(n) or n == 0.0:
        return (0.0,) * (x.size + 1)
    return tuple([float(np.log(n))] + [float(c) for c in x / n])
Q = np.array([0.7, -0.4, 0.2])
P = np.array([0.0021, 0.0052, -0.0011])
""",
            "call": "digest(runge_lenz_vector(Q.copy(), P.copy(), 1.1, 0.001))",
            "gold_call": "digest(_oracle_runge_lenz_vector(Q.copy(), P.copy(), 1.1, 0.001))",
        },
        # --- Edge: comparable masses, where the coefficient c0 differs most between plausible normalizations ---
        {
            "setup": """import numpy as np
def digest(x):
    # log-magnitude plus unit direction, so the result is compared at relative precision
    x = np.asarray(x, dtype=float).ravel()
    n = float(np.sqrt(x @ x))
    if not np.isfinite(n) or n == 0.0:
        return (0.0,) * (x.size + 1)
    return tuple([float(np.log(n))] + [float(c) for c in x / n])
Q = np.array([1.3, 0.5, 0.0])
P = np.array([-1.1, 2.4, 0.3])
""",
            "call": "digest(runge_lenz_vector(Q.copy(), P.copy(), 0.8, 0.6))",
            "gold_call": "digest(_oracle_runge_lenz_vector(Q.copy(), P.copy(), 0.8, 0.6))",
        },
        # --- Invalid: zero position vector ---
        {
            "setup": """import numpy as np
Q = np.zeros(3)
P = np.array([0.0, 0.1, 0.0])
def run_model():
    try:
        runge_lenz_vector(Q.copy(), P.copy(), 1.0, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_runge_lenz_vector(Q.copy(), P.copy(), 1.0, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive star mass ---
        {
            "setup": """import numpy as np
Q = np.array([1.0, 0.0, 0.0])
P = np.array([0.0, 0.1, 0.0])
def run_model():
    try:
        runge_lenz_vector(Q.copy(), P.copy(), 0.0, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_runge_lenz_vector(Q.copy(), P.copy(), 0.0, 0.1)
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
