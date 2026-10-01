"""
Transform the inertial positions and momenta of an N-body system, star first, into democratic heliocentric coordinates.

Democratic heliocentric coordinates (Q, P) are a canonical set built from the inertial positions r_i and momenta p_i of N bodies, with i = 0 the star. The zeroth pair describes the system as a whole: Q_0 is the position of the centre of mass and P_0 is the total momentum. For every other body i, Q_i is its position relative to the star and P_i is its momentum measured in the frame that moves with the centre of mass, its barycentric momentum.

Because the transformation is canonical, the N-body Hamiltonian can be rewritten exactly in these variables. The rewritten Hamiltonian splits into the planets' kinetic energy, the star's kinetic energy expressed through the P_i, the star-planet potentials and the planet-planet potentials, and a Wisdom-Holman map in these coordinates integrates those pieces separately. The transformation has to be derived from the definitions above; no formulas are given here.

Returns
-------
np.ndarray of shape (2, N, 3), float: QP[0] = Q (centre-of-mass position, then heliocentric positions), QP[1] = P (total momentum, then barycentric momenta).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def democratic_heliocentric_coordinates(masses: np.ndarray, positions: np.ndarray,
                                        momenta: np.ndarray) -> np.ndarray:
    """Inertial positions and momenta -> democratic heliocentric coordinates.

    Parameters
    ----------
    masses : np.ndarray
        Shape (N,), N >= 2, the masses (Msun), star first; every entry finite and > 0.
    positions : np.ndarray
        Shape (N, 3), inertial positions (au), row i for body i; finite.
    momenta : np.ndarray
        Shape (N, 3), inertial momenta (Msun au / yr), row i for body i; finite.

    Returns
    -------
    QP : np.ndarray
        Shape (2, N, 3): QP[0] holds Q (row 0 the centre-of-mass position,
        row i the position of body i relative to the star) and QP[1] holds P
        (row 0 the total momentum, row i the barycentric momentum of body i).

    Raises
    ------
    ValueError
        If masses is not one-dimensional with at least 2 entries; if positions
        or momenta does not have shape (N, 3); if any input contains a
        non-finite value; or if any mass is <= 0.
    """
    return QP  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_democratic_heliocentric_coordinates(masses: np.ndarray, positions: np.ndarray,
                                                momenta: np.ndarray) -> np.ndarray:
    import numpy as np

    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    momenta = np.asarray(momenta, dtype=float)
    if masses.ndim != 1 or masses.size < 2:
        raise ValueError("masses must be a one-dimensional array with at least 2 entries")
    n = masses.size
    if positions.shape != (n, 3) or momenta.shape != (n, 3):
        raise ValueError("positions and momenta must both have shape (N, 3)")
    if not (np.all(np.isfinite(masses)) and np.all(np.isfinite(positions)) and np.all(np.isfinite(momenta))):
        raise ValueError("all inputs must be finite")
    if np.any(masses <= 0.0):
        raise ValueError("every mass must be > 0")

    M = masses.sum()
    P0 = momenta.sum(axis=0)
    Q = np.empty((n, 3))
    P = np.empty((n, 3))
    Q[0] = (masses[:, None] * positions).sum(axis=0) / M      # centre of mass
    P[0] = P0                                                  # total momentum
    Q[1:] = positions[1:] - positions[0]                       # heliocentric positions
    P[1:] = momenta[1:] - (masses[1:, None] / M) * P0          # barycentric momenta
    return np.stack([Q, P])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: star and companion in the barycentric frame (the configuration the pipeline uses) ---
        {
            "setup": """import numpy as np
m = np.array([1.0, 0.07])
q = np.array([0.03, 0.0, 0.0])
v = np.array([0.0, 32.1, 0.0])
mu = m[0]*m[1]/m.sum()
r = np.array([-(m[1]/m.sum())*q, (m[0]/m.sum())*q])
p = np.array([-mu*v, mu*v])
""",
            "call": "democratic_heliocentric_coordinates(m.copy(), r.copy(), p.copy())",
            "gold_call": "_oracle_democratic_heliocentric_coordinates(m.copy(), r.copy(), p.copy())",
        },
        # --- Valid: three bodies in a frame that is not the barycentric one (nonzero total momentum, offset centre of mass) ---
        {
            "setup": """import numpy as np
m = np.array([1.2, 0.003, 0.0005])
r = np.array([[0.4, -0.1, 0.02], [1.9, 0.7, -0.05], [-3.1, 2.2, 0.3]])
p = np.array([[0.05, 0.3, -0.01], [-0.012, 0.018, 0.001], [-0.0016, -0.0021, 0.0002]])
""",
            "call": "democratic_heliocentric_coordinates(m.copy(), r.copy(), p.copy())",
            "gold_call": "_oracle_democratic_heliocentric_coordinates(m.copy(), r.copy(), p.copy())",
        },
        # --- Edge: five bodies with comparable masses, where m_i/M differs strongly from m_i/m0 ---
        {
            "setup": """import numpy as np
m = np.array([1.0, 0.4, 0.25, 0.1, 0.05])
rng = np.random.default_rng(3)
r = rng.normal(size=(5, 3))
p = rng.normal(size=(5, 3))
""",
            "call": "democratic_heliocentric_coordinates(m.copy(), r.copy(), p.copy())",
            "gold_call": "_oracle_democratic_heliocentric_coordinates(m.copy(), r.copy(), p.copy())",
        },
        # --- Invalid: a non-positive mass ---
        {
            "setup": """import numpy as np
m = np.array([1.0, -0.1])
r = np.zeros((2, 3))
p = np.zeros((2, 3))
def run_model():
    try:
        democratic_heliocentric_coordinates(m.copy(), r.copy(), p.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_democratic_heliocentric_coordinates(m.copy(), r.copy(), p.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: positions with the wrong shape ---
        {
            "setup": """import numpy as np
m = np.array([1.0, 0.1])
r = np.zeros((2, 2))
p = np.zeros((2, 3))
def run_model():
    try:
        democratic_heliocentric_coordinates(m.copy(), r.copy(), p.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_democratic_heliocentric_coordinates(m.copy(), r.copy(), p.copy())
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
