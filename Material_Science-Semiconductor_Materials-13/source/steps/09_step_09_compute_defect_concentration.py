"""
Return the apparent trap concentration of each level from its capacitance deflection, the shallow doping density of the probed layer and the quiescent capacitance at the reverse bias, under an explicitly supplied multiplicative partial-probing correction.

The capacitance of a junction is set by the width of its space charge region, which in turn is set by the net charge the region contains. Emptying a deep level of density much smaller than the shallow doping perturbs that charge slightly, and expanding the capacitance to first order in the perturbation shows that the fractional change in capacitance is half the ratio of the emitted charge density to the shallow doping density. Inverting that relation gives the trap concentration as twice the doping density times the ratio of the capacitance deflection to the quiescent capacitance, and the factor of two is a consequence of the square-root dependence of capacitance on charge density, not a convention.




The relation assumes that every state of the level is filled during the pulse and empties during the transient, which is why the sign of the deflection is discarded: it records whether majority or minority carriers were captured, not how many states there were. It also assumes the level is probed over the whole region swept between the pulse bias and the reverse bias. In reality the level stays filled in a thin zone near the depletion edge where its energy remains below the Fermi level, so the nominal first-order estimate may require a partial-probing correction. In this step, `correction` is defined as the multiplicative factor in $N_T = 2 N_D |Delta C| correction / C_0$; it is not the geometric fraction of the swept region that contributes. Its physical value depends on the trap depth and on the two bias voltages.




The correction is level dependent because it depends on trap depth as well as the junction-bias geometry. It therefore cannot generally be assumed to cancel between two levels, even when they were measured under the same pulse conditions. The benchmark does not supply the junction quantities needed to evaluate separate corrections, so it explicitly uses correction = 1 for every level and calls the resulting values uncorrected or apparent concentrations. A physically corrected concentration ratio would require one correction factor per level and is outside this task's stated target.

Returns
-------
np.ndarray of shape (n_levels,), float: apparent trap concentration in 1/cm^3.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_defect_concentration(deflection: np.ndarray, doping_density: float,
                                 base_capacitance: float,
                                 correction: float = 1.0) -> np.ndarray:
    """Return the apparent trap concentration from each capacitance deflection.

    Parameters
    ----------
    deflection : np.ndarray
        Capacitance deflection of each level in pF; the sign is ignored.
    doping_density : float
        Shallow donor or acceptor density of the probed layer in 1/cm^3
        (doping_density > 0).
    base_capacitance : float
        Quiescent capacitance at the reverse bias in pF
        (base_capacitance > 0), in the same units as deflection.
    correction : float
        Multiplicative dimensionless partial-probing correction (correction >
        0). The returned concentration is
        2*doping_density*abs(deflection)*correction/base_capacitance. The
        benchmark passes exactly 1.0, so its reported concentrations are
        uncorrected.

    Returns
    -------
    concentration : np.ndarray
        Apparent trap concentration of each level in 1/cm^3 under the supplied
        common multiplicative correction.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return concentration  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_defect_concentration(deflection: np.ndarray, doping_density: float,
                                         base_capacitance: float,
                                         correction: float = 1.0) -> np.ndarray:
    import numpy as np

    for name, value in (("doping_density", doping_density),
                        ("base_capacitance", base_capacitance),
                        ("correction", correction)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float, np.floating, np.integer))
                or not np.isfinite(value) or float(value) <= 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    deflection = np.atleast_1d(np.asarray(deflection, dtype=float))
    if deflection.size == 0 or not np.all(np.isfinite(deflection)):
        raise ValueError("deflection must be non-empty and finite")

    # Expanding the junction capacitance to first order in the emitted charge
    # density gives a factor of two from the square-root dependence of the
    # capacitance on the charge the space charge region contains.
    ratio = np.abs(deflection) / float(base_capacitance)

    return 2.0 * float(doping_density) * ratio * float(correction)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark doping and quiescent capacitance, two levels ---
        {
            "setup": """import numpy as np
deflection = np.array([-0.612, -0.037])
""",
            "call": "compute_defect_concentration(deflection, 2.0e16, 204.5) / 1.0e12 + 1000.0",
            "gold_call": "_oracle_compute_defect_concentration(deflection, 2.0e16, 204.5) / 1.0e12 + 1000.0",
        },
        # --- Valid: the weak level alone ---
        {
            "setup": """import numpy as np
deflection = np.array([-0.612, -0.037])
""",
            "call": "compute_defect_concentration(deflection, 2.0e16, 204.5) / 1.0e12 + 1000.0",
            "gold_call": "_oracle_compute_defect_concentration(deflection, 2.0e16, 204.5) / 1.0e12 + 1000.0",
        },
        # --- Valid: an explicitly common multiplicative factor cancels algebraically ---
        {
            "setup": """import numpy as np
deflection = np.array([-0.612, -0.037])
""",
            "call": "compute_defect_concentration(deflection, 2.0e16, 204.5, 1.37) / 1.0e12 + 1000.0",
            "gold_call": "_oracle_compute_defect_concentration(deflection, 2.0e16, 204.5, 1.37) / 1.0e12 + 1000.0",
        },
        # --- Boundary: a single level given as a bare scalar ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_defect_concentration(0.04, 5.0e15, 120.0) / 1.0e12 + 1000.0",
            "gold_call": "_oracle_compute_defect_concentration(0.04, 5.0e15, 120.0) / 1.0e12 + 1000.0",
        },
        # --- Edge: positive deflections from minority carrier capture ---
        {
            "setup": """import numpy as np
deflection = np.array([0.31, 0.006])
""",
            "call": "compute_defect_concentration(deflection, 1.0e16, 95.0, 0.82) / 1.0e12 + 1000.0",
            "gold_call": "_oracle_compute_defect_concentration(deflection, 1.0e16, 95.0, 0.82) / 1.0e12 + 1000.0",
        },
        # --- Invalid: non-positive doping density ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_defect_concentration(np.array([0.75]), 0.0, 204.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_defect_concentration(np.array([0.75]), 0.0, 204.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive quiescent capacitance ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_defect_concentration(np.array([0.75]), 2.0e16, -204.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_defect_concentration(np.array([0.75]), 2.0e16, -204.5)
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
