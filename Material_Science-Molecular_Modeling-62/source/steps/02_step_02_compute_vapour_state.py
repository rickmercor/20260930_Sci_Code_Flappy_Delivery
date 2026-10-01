"""
Compute the specific gas constant of water vapour, the saturated vapour density from the ideal-gas law and the latent heat of vaporisation from the Watson correlation.

The free-energy accounting of a vapour nucleus is written per unit mass, because the quantity integrated over the nucleus is the vapour density multiplied by a specific Gibbs free-energy increment. Every thermodynamic property entering that increment must therefore be mass-based, and in particular the gas constant appearing in the pressure term is the specific gas constant of water, the universal gas constant divided by the molar mass of water, and not the universal constant itself. Substituting the molar value leaves the pressure term too small by the molar mass, which changes the balance between the two competing terms of the increment and therefore both the location and the height of the barrier. The same specific constant then fixes the vapour density, since at the modest pressures of interest saturated steam is well described as an ideal gas, giving a density of the pressure divided by the product of the specific gas constant and the saturation temperature.


The latent heat is the other mass-based property, and it is not a constant: it falls monotonically from the triple point to zero at the critical point, where the two phases become indistinguishable. The Watson correlation captures that decay with a single power law in the reduced distance to the critical temperature, scaling a reference latent heat by the ratio of the critical temperature minus the target temperature to the critical temperature minus the reference temperature, raised to an exponent close to three-eighths. The correlation is used here rather than a tabulated value because it makes the dependence on the saturation state explicit, and because the exponent, not the reference point, is what controls how quickly the driving force for vaporization collapses as the liquid is pushed towards the critical region. Evaluating it outside the interval between the reference and the critical temperature is meaningless, since the base of the power law then leaves the unit interval or turns negative.

Returns
-------
np.ndarray of shape (3,), float: the specific gas constant in J/(kg K), the saturated vapour density in kg/m^3 and the latent heat of vaporisation in J/kg.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_vapour_state(t_sat: float, p_liquid: float,
                         h_reference: float = 2441.7e3,
                         t_reference: float = 298.15,
                         t_critical: float = 647.096,
                         watson_exponent: float = 0.38) -> np.ndarray:
    """Compute the mass-based vapour properties at the saturation state.

    Parameters
    ----------
    t_sat : float
        Saturation temperature of the liquid in kelvin (0 < t_sat < t_critical).
    p_liquid : float
        Liquid pressure in pascal (> 0).
    h_reference : float
        Reference latent heat of vaporisation in J/kg (> 0).
    t_reference : float
        Temperature in kelvin at which the reference latent heat applies
        (0 < t_reference < t_critical).
    t_critical : float
        Critical temperature of water in kelvin (> 0).
    watson_exponent : float
        Exponent of the Watson correlation (> 0).

    Returns
    -------
    vapour_state : np.ndarray
        Array of shape (3,) holding the specific gas constant of water vapour
        in J/(kg K), the saturated vapour density in kg/m^3 and the latent heat
        of vaporisation in J/kg.
    """
    return vapour_state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_vapour_state(t_sat: float, p_liquid: float,
                                 h_reference: float = 2441.7e3,
                                 t_reference: float = 298.15,
                                 t_critical: float = 647.096,
                                 watson_exponent: float = 0.38) -> np.ndarray:
    universal_gas_constant = 8.314462618
    molar_mass_water = 0.018015268

    for name, value in (("t_sat", t_sat), ("p_liquid", p_liquid),
                        ("h_reference", h_reference), ("t_reference", t_reference),
                        ("t_critical", t_critical),
                        ("watson_exponent", watson_exponent)):
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not float(t_sat) < float(t_critical):
        raise ValueError("t_sat must lie below the critical temperature")
    if not float(t_reference) < float(t_critical):
        raise ValueError("t_reference must lie below the critical temperature")

    # The increment being integrated is specific, so the gas constant must be
    # the mass-based one rather than the universal constant.
    r_specific = universal_gas_constant / molar_mass_water

    # Saturated steam at these pressures is well described as an ideal gas.
    rho_vapour = float(p_liquid) / (r_specific * float(t_sat))

    # Watson correlation for the decay of the latent heat towards the critical point.
    reduced = (float(t_critical) - float(t_sat)) / (float(t_critical) - float(t_reference))
    h_fg = float(h_reference) * reduced ** float(watson_exponent)

    return np.array([r_specific, rho_vapour, h_fg], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark saturation state at one atmosphere (normal scenario) ---
        {
            "setup": """import numpy as np
t_sat = 373.15
p_liquid = 101325.0
""",
            "call": "compute_vapour_state(t_sat, p_liquid)",
            "gold_call": "_oracle_compute_vapour_state(t_sat, p_liquid)",
        },
        # --- Valid: elevated system pressure raises the vapour density ---
        {
            "setup": """import numpy as np
t_sat = 393.35
p_liquid = 2.0 * 101325.0
""",
            "call": "compute_vapour_state(t_sat, p_liquid)",
            "gold_call": "_oracle_compute_vapour_state(t_sat, p_liquid)",
        },
        # --- Boundary: reference temperature itself, where the Watson factor is one ---
        {
            "setup": """import numpy as np
t_sat = 298.15
p_liquid = 3169.0
""",
            "call": "compute_vapour_state(t_sat, p_liquid)",
            "gold_call": "_oracle_compute_vapour_state(t_sat, p_liquid)",
        },
        # --- Edge: near-critical saturation, where the latent heat nearly vanishes ---
        {
            "setup": """import numpy as np
t_sat = 646.0
p_liquid = 2.15e7
""",
            "call": "compute_vapour_state(t_sat, p_liquid)",
            "gold_call": "_oracle_compute_vapour_state(t_sat, p_liquid)",
        },
        # --- Invalid: saturation temperature above the critical point ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_vapour_state(700.0, 101325.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_vapour_state(700.0, 101325.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive pressure ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_vapour_state(373.15, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_vapour_state(373.15, 0.0)
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
