"""
Find the temperature at which the two diagonal entries of the axion mass matrix become equal, defining the level-crossing (resonance) temperature.

As the universe cools, the QCD-coupled flavor field's mass m_a(T) grows from a strongly suppressed value at high temperature toward its zero-temperature value m_{a,0}, while the second flavor field's mass mS stays fixed. A level crossing occurs at the temperature T_x at which the two diagonal entries of the squared-mass matrix from the diagonalization step become equal. This is not, in general, the point where the bare masses m_a(T) and mS are equal. It is also where the splitting between the two eigenvalues is smallest. The condition has a solution only when the value of m_a(T_x)^2 that it requires is one the temperature-dependent mass can actually reach: m_a(T)^2 is bounded above by m_{a,0}^2 and decreases monotonically as temperature increases above T_QCD.

Neither the explicit crossing condition nor a closed-form solution for T_x is given here. Build the condition from the diagonal entries of the mass matrix, and solve it using the temperature-dependent mass relation from the earlier step.

Returns
-------
float, the crossing temperature T_x in GeV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def find_crossing_temperature(m_a0: float, mS: float, Rf: float, T_QCD: float, n: float) -> float:
    """Root-find the level-crossing temperature T_x.

    Parameters
    ----------
    m_a0 : float
        Zero-temperature axion mass (GeV), a finite number > 0.
    mS : float
        Second flavor field's mass (GeV), a finite number > 0.
    Rf : float
        Dimensionless ratio fS/fa, a finite number > 0 and < 1 (Rf < 1 is
        required for a crossing to exist).
    T_QCD : float
        QCD confinement scale (GeV), a finite number > 0.
    n : float
        Exponent of the QCD-coupled field's mass above T_QCD
        (m_a proportional to T^-n, so m_a^2 and the topological
        susceptibility fall off as T^-2n), a finite number > 0.

    Returns
    -------
    T_x : float
        The crossing temperature in GeV, finite and > 0.

    Raises
    ------
    ValueError
        If m_a0, mS, T_QCD, or n is not a finite number > 0; if Rf is not
        a finite number with 0 < Rf < 1; or if no crossing exists for the
        given parameters (the value of m_a(T)^2 that the crossing
        condition requires is at least m_a0^2, so it can never be reached).
    """
    return T_x  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_find_crossing_temperature(m_a0: float, mS: float, Rf: float, T_QCD: float, n: float) -> float:
    import numpy as np
    from scipy.optimize import brentq

    for name, value in (("m_a0", m_a0), ("mS", mS), ("T_QCD", T_QCD), ("n", n)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(Rf, (int, float, np.floating)) and np.isfinite(Rf) and 0.0 < float(Rf) < 1.0):
        raise ValueError("Rf must be a finite number with 0 < Rf < 1")

    m_a0 = float(m_a0)
    mS = float(mS)
    Rf = float(Rf)
    T_QCD = float(T_QCD)
    n = float(n)

    required = mS ** 2 * (1.0 - Rf ** 2)
    if required >= m_a0 ** 2:
        raise ValueError("no crossing exists: mS^2*(1-Rf^2) >= m_a0^2")

    def crossing_eq(T):
        m_a_sq = _oracle_compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n)
        return (m_a_sq + mS ** 2 * Rf ** 2) - mS ** 2

    Tlo, Thi = 1e-8, 1e6
    T_x = brentq(crossing_eq, Tlo, Thi, xtol=1e-14, rtol=1e-13)
    return T_x

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark scenario ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(np.log(abs(x)))
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
T_QCD = 0.100
n = 3.34
""",
            "call": "digest(find_crossing_temperature(m_a0, mS, Rf, T_QCD, n))",
            "gold_call": "digest(_oracle_find_crossing_temperature(m_a0, mS, Rf, T_QCD, n))",
        },
        # --- Valid: a different, larger Rm-like ratio (still within existence) ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(np.log(abs(x)))
m_a0 = 1.0e-13
mS = 3.0e-14
Rf = 0.15
T_QCD = 0.150
n = 4.0
""",
            "call": "digest(find_crossing_temperature(m_a0, mS, Rf, T_QCD, n))",
            "gold_call": "digest(_oracle_find_crossing_temperature(m_a0, mS, Rf, T_QCD, n))",
        },
        # --- Boundary: Rf close to (but less than) 1 ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(np.log(abs(x)))
m_a0 = 1.0e-13
mS = 5.0e-15
Rf = 0.95
T_QCD = 0.100
n = 3.34
""",
            "call": "digest(find_crossing_temperature(m_a0, mS, Rf, T_QCD, n))",
            "gold_call": "digest(_oracle_find_crossing_temperature(m_a0, mS, Rf, T_QCD, n))",
        },
        # --- Consistency: the returned T_x actually solves the crossing condition to
        # high precision ---
        {
            "setup": """import numpy as np
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
T_QCD = 0.100
n = 3.34
def check(fn):
    Tx = fn(m_a0, mS, Rf, T_QCD, n)
    m_a_sq = m_a0**2*min(1.0, (T_QCD/Tx)**(2*n))
    lhs = m_a_sq + mS**2*Rf**2
    rhs = mS**2
    return int(abs(lhs-rhs) < 1e-6*rhs)
""",
            "call": "check(find_crossing_temperature)",
            "gold_call": "check(_oracle_find_crossing_temperature)",
        },
        # --- Consistency: T_x must lie above T_QCD (crossing occurs in the power-law regime
        # for this benchmark family of parameters) ---
        {
            "setup": """import numpy as np
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
T_QCD = 0.100
n = 3.34
def check(fn):
    Tx = fn(m_a0, mS, Rf, T_QCD, n)
    return int(Tx > T_QCD)
""",
            "call": "check(find_crossing_temperature)",
            "gold_call": "check(_oracle_find_crossing_temperature)",
        },
        # --- Invalid: Rf outside (0,1) ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        find_crossing_temperature(1e-13, 1e-14, 1.5, 0.1, 3.34)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_find_crossing_temperature(1e-13, 1e-14, 1.5, 0.1, 3.34)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: no crossing exists for these parameters ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        find_crossing_temperature(1e-14, 1e-13, 0.1, 0.1, 3.34)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_find_crossing_temperature(1e-14, 1e-13, 0.1, 0.1, 3.34)
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
