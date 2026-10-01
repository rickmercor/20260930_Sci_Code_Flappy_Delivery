"""
Compute the basic constants of an isolated star-companion two-body system: the mass ratio, the total and reduced masses, the coefficient of the Runge-Lenz vector used by the later steps, and the period of the relative orbit.

All quantities in this pipeline use astronomical units for length, solar masses for mass and years of 365.25 days for time, with the gravitational constant G = 4*pi^2 au^3 Msun^-1 yr^-2 exactly. The system is a star of mass m0 and a companion of mass m1 on a bound relative orbit with semimajor axis a and eccentricity e. In the barycentric frame the relative motion follows the Hamiltonian H = |P|^2/(2 mu) - G m0 m1/|Q|, where Q is the companion's position relative to the star, P is the companion's barycentric momentum and mu is the reduced mass.

Later steps write the Runge-Lenz vector as R = P x L - c0 Q/|Q|, with L = Q x P. The coefficient c0 is the constant that makes R conserved under H; working it out is part of this step. The period is that of the relative orbit under H.

Returns
-------
np.ndarray of shape (5,), float: [m1/m0, total mass M, reduced mass mu, Runge-Lenz coefficient c0, orbital period in years], in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_two_body_constants(m0: float, m1: float, a: float, e: float) -> np.ndarray:
    """Basic constants of the star-companion two-body problem.

    Parameters
    ----------
    m0 : float
        Mass of the star (Msun), a finite number > 0.
    m1 : float
        Mass of the companion (Msun), a finite number > 0.
    a : float
        Semimajor axis of the relative orbit (au), a finite number > 0.
    e : float
        Eccentricity of the relative orbit, a finite number with 0 < e < 1.

    Returns
    -------
    consts : np.ndarray
        Array of shape (5,): [m1/m0, total mass M, reduced mass mu,
        Runge-Lenz coefficient c0, orbital period in years].

    Raises
    ------
    ValueError
        If m0, m1 or a is not a finite number > 0, or if e is not a finite
        number with 0 < e < 1.
    """
    return consts  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_two_body_constants(m0: float, m1: float, a: float, e: float) -> np.ndarray:
    import numpy as np

    for name, val in (("m0", m0), ("m1", m1), ("a", a)):
        if not (isinstance(val, (int, float, np.floating)) and np.isfinite(val) and float(val) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(e, (int, float, np.floating)) and np.isfinite(e) and 0.0 < float(e) < 1.0):
        raise ValueError("e must be a finite number with 0 < e < 1")

    G = 4.0 * np.pi ** 2                      # au^3 / (Msun yr^2)
    m0, m1, a = float(m0), float(m1), float(a)
    M = m0 + m1
    eps = m1 / m0
    mu = m0 * m1 / M
    c0 = G * m0 ** 2 * m1 ** 2 / M            # Runge-Lenz coefficient mu * G * m0 * m1
    period = 2.0 * np.pi * np.sqrt(a ** 3 / (G * M))
    return np.array([eps, M, mu, c0, period], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark system (brown-dwarf companion, a = 0.06 au, e = 0.5) ---
        {
            "setup": """import numpy as np
def digest(out):
    # each output compared on its own, as a sign-preserving log
    out = np.asarray(out, dtype=float).ravel()
    return tuple(float(np.sign(o) * np.log(abs(o) + 1e-300)) for o in out)
args = (1.0, 0.07, 0.06, 0.5)
""",
            "call": "digest(compute_two_body_constants(*args))",
            "gold_call": "digest(_oracle_compute_two_body_constants(*args))",
        },
        # --- Valid: a Jupiter-like companion on a wide, nearly circular orbit ---
        {
            "setup": """import numpy as np
def digest(out):
    # each output compared on its own, as a sign-preserving log
    out = np.asarray(out, dtype=float).ravel()
    return tuple(float(np.sign(o) * np.log(abs(o) + 1e-300)) for o in out)
args = (1.0, 9.547919e-4, 5.2026, 0.0489)
""",
            "call": "digest(compute_two_body_constants(*args))",
            "gold_call": "digest(_oracle_compute_two_body_constants(*args))",
        },
        # --- Boundary: equal masses and a very eccentric orbit, where m1/M and m1/m0 differ most ---
        {
            "setup": """import numpy as np
def digest(out):
    # each output compared on its own, as a sign-preserving log
    out = np.asarray(out, dtype=float).ravel()
    return tuple(float(np.sign(o) * np.log(abs(o) + 1e-300)) for o in out)
args = (0.9, 0.9, 2.5, 0.95)
""",
            "call": "digest(compute_two_body_constants(*args))",
            "gold_call": "digest(_oracle_compute_two_body_constants(*args))",
        },
        # --- Invalid: e = 1 is not a bound orbit ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_two_body_constants(1.0, 0.07, 0.06, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_two_body_constants(1.0, 0.07, 0.06, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: zero companion mass ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_two_body_constants(1.0, 0.0, 0.06, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_two_body_constants(1.0, 0.0, 0.06, 0.5)
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
