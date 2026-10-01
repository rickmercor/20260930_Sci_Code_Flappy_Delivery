"""
Compute the orbit-averaged apsidal precession rate produced by the simulation's general-relativity potential.

All quantities in this pipeline use astronomical units for length, solar masses for mass and years of 365.25 days for time, with the gravitational constant G = 4*pi^2 au^3 Msun^-1 yr^-2 exactly. The speed of light is c = 299792458 m/s, converted to au/yr with 1 au = 1.495978707e11 m.

General relativity enters the simulation as the position-dependent potential V_GR = -3 G^2 m0^2 m1/(c^2 |Q|^2), added to the two-body Hamiltonian H = |P|^2/(2 mu) - G m0 m1/|Q| of Step 4. Its effect on the orbit is a steady prograde rotation of the Runge-Lenz vector. The orbit-averaged rate of that rotation is to be computed to first order in V_GR, for the orbit with semimajor axis a and eccentricity e. The rate must follow from this specific potential and Hamiltonian.

Returns
-------
float, the orbit-averaged apsidal precession rate from V_GR, in rad / yr.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gr_precession_rate(m0: float, m1: float, a: float, e: float) -> float:
    """Orbit-averaged apsidal precession rate from V_GR = -3 G^2 m0^2 m1 / (c^2 |Q|^2) (rad / yr).

    Parameters
    ----------
    m0 : float
        Mass of the star (Msun), a finite number > 0.
    m1 : float
        Mass of the companion (Msun), a finite number > 0.
    a : float
        Semimajor axis of the relative orbit (au), a finite number > 0.
    e : float
        Eccentricity, a finite number with 0 < e < 1.

    Returns
    -------
    rate : float
        The orbit-averaged apsidal precession rate, in rad / yr (positive = prograde).

    Raises
    ------
    ValueError
        If m0, m1 or a is not a finite number > 0, or if e is not a finite
        number with 0 < e < 1.
    """
    return rate  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_gr_precession_rate(m0: float, m1: float, a: float, e: float) -> float:
    import numpy as np

    for name, val in (("m0", m0), ("m1", m1), ("a", a)):
        if not (isinstance(val, (int, float, np.floating)) and np.isfinite(val) and float(val) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(e, (int, float, np.floating)) and np.isfinite(e) and 0.0 < float(e) < 1.0):
        raise ValueError("e must be a finite number with 0 < e < 1")
    G = 4.0 * np.pi ** 2
    c = 299792458.0 * 365.25 * 86400.0 / 1.495978707e11      # speed of light in au / yr
    m0, m1, a, e = float(m0), float(m1), float(a), float(e)
    M = m0 + m1
    return float(3.0 * G ** 1.5 * m0 * np.sqrt(M) / (c ** 2 * a ** 2.5 * (1.0 - e ** 2)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark system ---
        {
            "setup": """import numpy as np
def digest(x):
    # sign-preserving log, so tiny or large rates are compared at relative precision
    x = float(x)
    return float(np.sign(x) * np.log(abs(x) + 1e-300))
""",
            "call": "digest(gr_precession_rate(1.0, 0.07, 0.06, 0.5))",
            "gold_call": "digest(_oracle_gr_precession_rate(1.0, 0.07, 0.06, 0.5))",
        },
        # --- Valid: a Mercury-like orbit around a solar-mass star ---
        {
            "setup": """import numpy as np
def digest(x):
    # sign-preserving log, so tiny or large rates are compared at relative precision
    x = float(x)
    return float(np.sign(x) * np.log(abs(x) + 1e-300))
""",
            "call": "digest(gr_precession_rate(1.0, 1.66e-7, 0.387098, 0.2056))",
            "gold_call": "digest(_oracle_gr_precession_rate(1.0, 1.66e-7, 0.387098, 0.2056))",
        },
        # --- Edge: equal masses, where the given potential and the textbook two-body formula differ most ---
        {
            "setup": """import numpy as np
def digest(x):
    # sign-preserving log, so tiny or large rates are compared at relative precision
    x = float(x)
    return float(np.sign(x) * np.log(abs(x) + 1e-300))
""",
            "call": "digest(gr_precession_rate(1.0, 1.0, 0.1, 0.3))",
            "gold_call": "digest(_oracle_gr_precession_rate(1.0, 1.0, 0.1, 0.3))",
        },
        # --- Invalid: non-positive semimajor axis ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        gr_precession_rate(1.0, 0.07, -0.06, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_gr_precession_rate(1.0, 0.07, -0.06, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: eccentricity out of range ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        gr_precession_rate(1.0, 0.07, 0.06, 1.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_gr_precession_rate(1.0, 0.07, 0.06, 1.2)
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
