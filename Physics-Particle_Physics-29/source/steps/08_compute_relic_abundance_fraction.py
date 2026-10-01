"""
Orchestrator: compute the present-day dark matter abundance fraction carried by the light mass eigenstate of a two-axion system, from the model's fundamental parameters.

The fraction follows from chaining Steps 1 through 7: compute the zero-temperature mass scales (Step 1), find the level-crossing temperature by root-finding the crossing condition built from the temperature-dependent mass (Steps 2 and 4) and the mass-matrix diagonalization (Step 3), compute the adiabaticity parameter and Landau-Zener probability at that crossing (Step 5), determine the two oscillation temperatures and the initial mass-eigenstate field values (Step 6), and finally assemble the present-day abundance fraction from the pre-crossing WKB densities, the Landau-Zener mixing, and a fresh diagonalization at the present-day temperature (Step 7).

This step is the final orchestrator: called with no arguments, it reproduces the exact benchmark scenario and returns the benchmark abundance fraction.

Returns
-------
float, the present-day abundance fraction f_L carried by the light mass eigenstate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_relic_abundance_fraction(
    fa: float = 1.0e12,
    Rf: float = 0.02,
    Rm: float = 0.10,
    theta_a: float = 0.02,
    theta_s: float = 0.8,
    T_QCD: float = 0.100,
    n: float = 3.34,
    g_star: float = 61.75,
    T0: float = 2.348e-13,
) -> float:
    """Present-day relic abundance fraction of a two-axion Landau-Zener system.

    Parameters
    ----------
    fa : float
        Axion decay constant (GeV), > 0.
    Rf : float
        Dimensionless ratio fS/fa, with 0 < Rf < 1.
    Rm : float
        Dimensionless ratio mS/m_{a,0}, > 0.
    theta_a : float
        Initial misalignment angle of the 'a' flavor field, finite.
    theta_s : float
        Initial misalignment angle of the 'aS' flavor field, finite.
    T_QCD : float
        QCD confinement scale (GeV), > 0.
    n : float
        Exponent of the QCD-coupled field's mass above T_QCD
        (m_a proportional to T^-n; m_a^2 falls off as T^-2n), > 0.
    g_star : float
        Effective relativistic degrees of freedom, > 0.
    T0 : float
        Present-day temperature (GeV), > 0, with T0 < T_QCD.

    Returns
    -------
    f_L : float
        The present-day abundance fraction carried by the light mass
        eigenstate, a finite number in (0, 1).

    Raises
    ------
    ValueError
        If any underlying step raises ValueError on its own inputs, or if
        the top-level parameter constraints described above are violated.
    """
    return f_L  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_relic_abundance_fraction(
    fa: float = 1.0e12,
    Rf: float = 0.02,
    Rm: float = 0.10,
    theta_a: float = 0.02,
    theta_s: float = 0.8,
    T_QCD: float = 0.100,
    n: float = 3.34,
    g_star: float = 61.75,
    T0: float = 2.348e-13,
) -> float:
    import numpy as np

    # Step 1
    m_a0, mS = _oracle_compute_axion_mass_scales(fa, Rm)

    # Step 4 (chains Steps 2-3 internally via find_crossing_temperature's own oracle)
    T_x = _oracle_find_crossing_temperature(m_a0, mS, Rf, T_QCD, n)

    # Step 5
    gamma, P_LZ = _oracle_compute_landau_zener_probability(T_x, m_a0, mS, Rf, T_QCD, n, g_star)

    # Step 6
    T_osc_H, T_osc_L, a_H, a_L = _oracle_compute_oscillation_temperatures_and_initial_fields(
        m_a0, mS, Rf, fa, T_QCD, n, g_star, theta_a, theta_s
    )

    # Step 7
    f_L = _oracle_assemble_relic_abundance_fraction(
        m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n
    )

    return float(f_L)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the exact benchmark scenario, all default arguments ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_relic_abundance_fraction()",
            "gold_call": "_oracle_compute_relic_abundance_fraction()",
        },
        # --- Valid: a larger mixing ratio, Rf = 0.05 (gamma of about 10, a nearly adiabatic
        # crossing with almost no conversion) ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_relic_abundance_fraction(Rf=0.05)",
            "gold_call": "_oracle_compute_relic_abundance_fraction(Rf=0.05)",
        },
        # --- Valid: different initial misalignment angles ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_relic_abundance_fraction(theta_a=0.3, theta_s=1.2)",
            "gold_call": "_oracle_compute_relic_abundance_fraction(theta_a=0.3, theta_s=1.2)",
        },
        # --- Boundary: theta_s = -theta_a / Rf^2 makes a_L formally zero at the initial
        # instant (a tuned but valid configuration) ---
        {
            "setup": """import numpy as np
def check(fn):
    Rf = 0.02
    theta_a = 1.0
    theta_s = theta_a/Rf**2
    fL = fn(Rf=Rf, theta_a=theta_a, theta_s=theta_s)
    return int(0.0 < fL < 1.0)
""",
            "call": "check(compute_relic_abundance_fraction)",
            "gold_call": "check(_oracle_compute_relic_abundance_fraction)",
        },
        # --- Valid: a small mixing ratio, Rf = 0.01 (gamma of about 0.41, P_LZ of about 0.53;
        # f_L of about 0.082) ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_relic_abundance_fraction(Rf=0.01)",
            "gold_call": "_oracle_compute_relic_abundance_fraction(Rf=0.01)",
        },
        # --- Valid: Rf = 0.03 (gamma of about 3.7, P_LZ of about 0.003; f_L of about 0.715) ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_relic_abundance_fraction(Rf=0.03)",
            "gold_call": "_oracle_compute_relic_abundance_fraction(Rf=0.03)",
        },
        # --- Valid: Rf = 0.04 (gamma of about 6.5, P_LZ of about 4e-5; f_L of about 0.591). With cases 1
        # and 2, f_L runs 0.082, 0.502, 0.715, 0.591, 0.461 for Rf = 0.01 to 0.05: it stays in (0, 1)
        # but is not monotonic in Rf, because Rf also sets the initial eigenstate fields ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_relic_abundance_fraction(Rf=0.04)",
            "gold_call": "_oracle_compute_relic_abundance_fraction(Rf=0.04)",
        },
        # --- Invalid: Rf outside (0,1) ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_relic_abundance_fraction(Rf=1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_relic_abundance_fraction(Rf=1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive decay constant ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_relic_abundance_fraction(fa=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_relic_abundance_fraction(fa=0.0)
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
