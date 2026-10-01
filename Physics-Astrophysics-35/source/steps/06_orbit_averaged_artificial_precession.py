"""
Average the instantaneous apsidal precession rate produced by the error Hamiltonian over one unperturbed Keplerian orbit, giving the secular (artificial) precession rate of the BAB map.

The artificial precession is the rate at which the direction of the Runge-Lenz vector turns about the orbital angular momentum. At a state (Q, P), with R from Step 4 and dR/dt from Step 5, the instantaneous rate is (R x dR/dt) . z_hat / |R|^2, where z_hat = L/|L| is the unit vector along the orbital angular momentum L = Q x P. A positive rate is prograde. The secular rate is the time average of this quantity over one period of the unperturbed relative orbit with the given a and e; to leading order in h, the difference between that orbit and the map's own orbit does not matter.

Build the orbit in the xy-plane with the companion at pericentre on the +x axis, moving in the +y direction at t = 0. Set up the inertial (barycentric) positions and momenta of star and companion and convert them with Step 2. Sample the orbit at n_samples equally spaced times t_k = k T / n_samples (k = 0, ..., n_samples - 1, with T the period from Step 1), advancing the relative motion with Step 3, and return the arithmetic mean of the n_samples instantaneous rates. The integrand is smooth and periodic, so this mean converges very rapidly with n_samples to the exact orbit average.

Returns
-------
float, the orbit-averaged artificial apsidal precession rate in rad / yr (negative = retrograde).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orbit_averaged_artificial_precession(m0: float, m1: float, a: float, e: float, h: float,
                                          n_samples: int = 512) -> float:
    """Orbit-averaged artificial apsidal precession rate of the BAB map (rad / yr).

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
    h : float
        Map step size (yr), a finite number > 0.
    n_samples : int
        Number of equally spaced sample times over one period, an integer >= 16.

    Returns
    -------
    rate : float
        The orbit-averaged precession rate of the Runge-Lenz vector about the
        orbital angular momentum, in rad / yr (positive = prograde).

    Raises
    ------
    ValueError
        If m0, m1 or a is not a finite number > 0; if e is not a finite number
        with 0 < e < 1; if h is not a finite number > 0; or if n_samples is not
        an integer >= 16.
    """
    return rate  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_orbit_averaged_artificial_precession(m0: float, m1: float, a: float, e: float, h: float,
                                                 n_samples: int = 512) -> float:
    import numpy as np

    if not (isinstance(h, (int, float, np.floating)) and np.isfinite(h) and float(h) > 0.0):
        raise ValueError("h must be a finite number > 0")
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or int(n_samples) < 16:
        raise ValueError("n_samples must be an integer >= 16")
    eps, M, mu, c0, period = _oracle_compute_two_body_constants(m0, m1, a, e)
    G = 4.0 * np.pi ** 2
    m0, m1, a, e, h, n = float(m0), float(m1), float(a), float(e), float(h), int(n_samples)

    # relative orbit at pericentre, in the xy-plane, angular momentum along +z
    q_rel = np.array([a * (1.0 - e), 0.0, 0.0])
    v_rel = np.array([0.0, np.sqrt(G * M * (1.0 + e) / (a * (1.0 - e))), 0.0])
    positions = np.array([-(m1 / M) * q_rel, (m0 / M) * q_rel])
    momenta = np.array([-mu * v_rel, mu * v_rel])
    QP = _oracle_democratic_heliocentric_coordinates(np.array([m0, m1]), positions, momenta)
    Q, P = QP[0, 1].copy(), QP[1, 1].copy()

    dt = period / n
    rates = np.empty(n)
    for k in range(n):
        R = _oracle_runge_lenz_vector(Q, P, m0, m1)
        Rdot = _oracle_error_hamiltonian_rl_rate(Q, P, m0, m1, h)
        L = np.cross(Q, P)
        zhat = L / np.sqrt(L @ L)
        rates[k] = (np.cross(R, Rdot) @ zhat) / (R @ R)
        qv = _oracle_kepler_drift(Q, P / mu, G * M, dt)
        Q, P = qv[0], mu * qv[1]
    return float(rates.mean())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark system and step (14 minutes) ---
        {
            "setup": """import numpy as np
def digest(x):
    # sign-preserving log, so tiny or large rates are compared at relative precision
    x = float(x)
    return float(np.sign(x) * np.log(abs(x) + 1e-300))
h = 14/(60*24*365.25)
""",
            "call": "digest(orbit_averaged_artificial_precession(1.0, 0.07, 0.06, 0.5, h, 512))",
            "gold_call": "digest(_oracle_orbit_averaged_artificial_precession(1.0, 0.07, 0.06, 0.5, h, 512))",
        },
        # --- Valid: a Jupiter-like companion on a wide, nearly circular orbit with a 3.65-day step ---
        {
            "setup": """import numpy as np
def digest(x):
    # sign-preserving log, so tiny or large rates are compared at relative precision
    x = float(x)
    return float(np.sign(x) * np.log(abs(x) + 1e-300))
""",
            "call": "digest(orbit_averaged_artificial_precession(1.0, 9.547919e-4, 5.2026, 0.05, 0.01, 256))",
            "gold_call": "digest(_oracle_orbit_averaged_artificial_precession(1.0, 9.547919e-4, 5.2026, 0.05, 0.01, 256))",
        },
        # --- Edge: a high-eccentricity orbit (e = 0.7) ---
        {
            "setup": """import numpy as np
def digest(x):
    # sign-preserving log, so tiny or large rates are compared at relative precision
    x = float(x)
    return float(np.sign(x) * np.log(abs(x) + 1e-300))
""",
            "call": "digest(orbit_averaged_artificial_precession(0.8, 0.02, 0.3, 0.7, 0.001, 512))",
            "gold_call": "digest(_oracle_orbit_averaged_artificial_precession(0.8, 0.02, 0.3, 0.7, 0.001, 512))",
        },
        # --- Boundary: comparable masses (m1/m0 = 0.2), where writing the result in m1/m0 instead of the exact mass factors fails ---
        {
            "setup": """import numpy as np
def digest(x):
    # sign-preserving log, so tiny or large rates are compared at relative precision
    x = float(x)
    return float(np.sign(x) * np.log(abs(x) + 1e-300))
""",
            "call": "digest(orbit_averaged_artificial_precession(1.5, 0.3, 1.0, 0.6, 0.005, 512))",
            "gold_call": "digest(_oracle_orbit_averaged_artificial_precession(1.5, 0.3, 1.0, 0.6, 0.005, 512))",
        },
        # --- Invalid: circular orbit (e = 0, the Runge-Lenz vector vanishes) ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        orbit_averaged_artificial_precession(1.0, 0.07, 0.06, 0.0, 1e-4, 256)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_orbit_averaged_artificial_precession(1.0, 0.07, 0.06, 0.0, 1e-4, 256)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: too few samples ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        orbit_averaged_artificial_precession(1.0, 0.07, 0.06, 0.5, 1e-4, 8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_orbit_averaged_artificial_precession(1.0, 0.07, 0.06, 0.5, 1e-4, 8)
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
