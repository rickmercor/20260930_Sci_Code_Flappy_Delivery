"""
Recover the areal density of a two-dimensional electron gas from a non-contact transport measurement.

An as-grown heterostructure is characterised without any processing by an eddy-current measurement that
returns the sheet resistance together with the room-temperature Hall mobility. For a single carrier
species the sheet conductance is the product of the carrier charge, the areal density and the mobility.

Inputs: sheet_resistance_ohm_per_sq: float > 0, sheet resistance in ohm per square hall_mobility_cm2_per_Vs: float > 0, Hall mobility in cm^2 V^-1 s^-1

 Returns: float, sheet carrier density in m^-2

 Raises: ValueError if either input is not strictly positive.

Returns
-------
float, sheet carrier density in m^-2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sheet_carrier_density(sheet_resistance_ohm_per_sq: float, hall_mobility_cm2_per_Vs: float) -> float:
    """Sheet resistance in ohm/sq and Hall mobility in cm^2/(V s) -> 2DEG sheet carrier density in m^-2.
 
    Raises ValueError if either input is not strictly positive.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sheet_carrier_density(sheet_resistance_ohm_per_sq: float, hall_mobility_cm2_per_Vs: float) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    if not (sheet_resistance_ohm_per_sq > 0.0):
        raise ValueError('sheet resistance must be positive')
    if not (hall_mobility_cm2_per_Vs > 0.0):
        raise ValueError('mobility must be positive')
    return 1.0 / (Q * sheet_resistance_ohm_per_sq * hall_mobility_cm2_per_Vs * 1.0e-4)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\n",
         "call": "round(sheet_carrier_density(690.0, 1500.0) * 1e-16, 9)",
         "gold_call": "round(_oracle_sheet_carrier_density(690.0, 1500.0) * 1e-16, 9)"},
        {"setup": "import numpy as np\n",
         "call": "round(sheet_carrier_density(2000.0, 300.0) * 1e-16, 9)",
         "gold_call": "round(_oracle_sheet_carrier_density(2000.0, 300.0) * 1e-16, 9)"},
        {"setup": "import numpy as np\n",
         "call": "round(sheet_carrier_density(1.0e6, 50.0) * 1e-16, 12)",
         "gold_call": "round(_oracle_sheet_carrier_density(1.0e6, 50.0) * 1e-16, 12)"},
        {"setup": "import numpy as np\ndef run_model():\n    try:\n        sheet_carrier_density(0.0, 1500.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle():\n    try:\n        _oracle_sheet_carrier_density(0.0, 1500.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
