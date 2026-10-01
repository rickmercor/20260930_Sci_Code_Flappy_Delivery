"""
Compute the squared mass of the QCD-coupled axion flavor field at a given temperature.

The strength of the QCD instanton effects that generate this field's mass is governed by the temperature-dependent topological susceptibility. Well below the QCD confinement scale T_QCD, instanton effects are fully developed and the field's mass sits at its zero-temperature value, m_{a,0}, independent of temperature. Above T_QCD, instanton effects become progressively suppressed as the temperature increases, and the mass itself falls off as a power law in temperature, m_a(T) proportional to T^(-n). Here n is the exponent of the mass, not of the susceptibility: m_a(T)^2, and with it the topological susceptibility, falls off as T^(-2n). The two regimes join continuously at T = T_QCD, where the power-law expression and the constant zero-temperature value must agree.

No closed-form expression combining both regimes into a single formula is given here; it must be constructed from the description above (matching the constant value below T_QCD to the power-law falloff above it, continuously at T_QCD), then evaluated at the requested temperature.

Returns
-------
float, the squared mass m_a(T)^2 in GeV^2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_axion_mass_squared_at_temperature(T: float, m_a0: float, T_QCD: float, n: float) -> float:
    """Temperature-dependent squared mass of the QCD-coupled axion field.

    Parameters
    ----------
    T : float
        Temperature (GeV), a finite number > 0.
    m_a0 : float
        Zero-temperature axion mass (GeV), a finite number > 0.
    T_QCD : float
        QCD confinement scale (GeV), a finite number > 0.
    n : float
        Exponent of the QCD-coupled field's mass above T_QCD
        (m_a proportional to T^-n, so m_a^2 and the topological
        susceptibility fall off as T^-2n), a finite number > 0.

    Returns
    -------
    m_a_sq : float
        The squared mass m_a(T)^2 in GeV^2, finite and > 0.

    Raises
    ------
    ValueError
        If T, m_a0, T_QCD, or n is not a finite number > 0.
    """
    return m_a_sq  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_axion_mass_squared_at_temperature(T: float, m_a0: float, T_QCD: float, n: float) -> float:
    import numpy as np

    for name, value in (("T", T), ("m_a0", m_a0), ("T_QCD", T_QCD), ("n", n)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    T = float(T)
    m_a0 = float(m_a0)
    T_QCD = float(T_QCD)
    n = float(n)

    m_a_sq = m_a0 ** 2 * min(1.0, (T_QCD / T) ** (2.0 * n))
    return m_a_sq

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: temperature well above T_QCD (power-law-suppressed regime) ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(np.log(abs(x)))
T = 0.2
m_a0 = 5.775455048675712e-15
T_QCD = 0.100
n = 3.34
""",
            "call": "digest(compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
            "gold_call": "digest(_oracle_compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
        },
        # --- Valid: temperature below T_QCD (constant regime) ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(np.log(abs(x)))
T = 0.01
m_a0 = 5.775455048675712e-15
T_QCD = 0.100
n = 3.34
""",
            "call": "digest(compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
            "gold_call": "digest(_oracle_compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
        },
        # --- Boundary: T well above T_QCD, different n ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(np.log(abs(x)))
T = 5.0
m_a0 = 1.0e-13
T_QCD = 0.150
n = 4.0
""",
            "call": "digest(compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
            "gold_call": "digest(_oracle_compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
        },
        # --- Consistency: continuity at T = T_QCD exactly ---
        {
            "setup": """import numpy as np
m_a0 = 3.0e-14
T_QCD = 0.100
n = 3.0
def check(fn):
    val_at = fn(T_QCD, m_a0, T_QCD, n)
    return int(abs(val_at - m_a0**2) < 1e-6*m_a0**2)
""",
            "call": "check(compute_axion_mass_squared_at_temperature)",
            "gold_call": "check(_oracle_compute_axion_mass_squared_at_temperature)",
        },
        # --- Boundary: below T_QCD (T = 0.05) the mass equals its zero-temperature value ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(np.log(abs(x)))
T = 0.05
m_a0 = 3.0e-14
T_QCD = 0.100
n = 3.0
""",
            "call": "digest(compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
            "gold_call": "digest(_oracle_compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
        },
        # --- Edge: far below T_QCD (T = 0.001) the mass is still exactly its zero-temperature value,
        # with no residual temperature dependence ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(np.log(abs(x)))
T = 0.001
m_a0 = 3.0e-14
T_QCD = 0.100
n = 3.0
""",
            "call": "digest(compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
            "gold_call": "digest(_oracle_compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
        },
        # --- Valid: above T_QCD at T = 0.2, m_a^2 suppressed by (T_QCD/T)^(2n) = 1/64 ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(np.log(abs(x)))
T = 0.2
m_a0 = 3.0e-14
T_QCD = 0.100
n = 3.0
""",
            "call": "digest(compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
            "gold_call": "digest(_oracle_compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
        },
        # --- Valid: above T_QCD at T = 0.5, suppression of about 6.4e-5 ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(np.log(abs(x)))
T = 0.5
m_a0 = 3.0e-14
T_QCD = 0.100
n = 3.0
""",
            "call": "digest(compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
            "gold_call": "digest(_oracle_compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
        },
        # --- Edge: far above T_QCD at T = 2.0, suppression of about 1.6e-8; with the two previous
        # cases this pins the strictly decreasing power law ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(np.log(abs(x)))
T = 2.0
m_a0 = 3.0e-14
T_QCD = 0.100
n = 3.0
""",
            "call": "digest(compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
            "gold_call": "digest(_oracle_compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n))",
        },
        # --- Invalid: non-positive temperature ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_axion_mass_squared_at_temperature(0.0, 1e-14, 0.1, 3.34)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_axion_mass_squared_at_temperature(0.0, 1e-14, 0.1, 3.34)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive exponent n ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_axion_mass_squared_at_temperature(0.2, 1e-14, 0.1, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_axion_mass_squared_at_temperature(0.2, 1e-14, 0.1, -1.0)
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
