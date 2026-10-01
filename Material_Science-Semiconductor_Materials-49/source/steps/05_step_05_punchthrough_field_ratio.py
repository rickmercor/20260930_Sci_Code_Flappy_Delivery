"""
Field ratio at the punch-through design that makes the constant-mobility, complete-ionisation drift resistance stationary.

A punch-through drift region ends on a heavily doped layer, so the field falls to a finite value at the far edge instead of reaching zero, and the profile is trapezoidal. Writing that terminal field as a fraction of the peak field and demanding that the drift specific on-resistance be stationary with respect to doping at fixed blocking voltage, with the mobility held constant and every donor ionised, pins the fraction. Carrying the stationarity condition through the exact avalanche relation, rather than through a fitted power law, leaves the fraction as a function of the dimensionless field variable alone. The fraction is physically confined to the interval from zero up to but not including one.

Returns
-------
float, the terminal to peak field ratio (no units)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def punchthrough_field_ratio(zeta: float) -> float:
    '''Terminal to peak field ratio that makes the constant-mobility, complete-ionisation drift resistance stationary.

    Parameters
    ----------
    zeta : float
        Dimensionless field variable.

    Returns
    -------
    float
        The field ratio (no units).

    Raises
    ------
    ValueError
        If zeta is not finite or is not strictly positive.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import exp1

def _oracle_punchthrough_field_ratio(zeta: float) -> float:
    for _v in (zeta,):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('zeta must be finite and strictly positive')
    return float((2.0/3.0)*zeta*np.exp(zeta)*exp1(zeta) - 1.0/3.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "punchthrough_field_ratio(10.712565)",
            "gold_call": "_oracle_punchthrough_field_ratio(10.712565)",
        },
        {
            "setup": "import numpy as np",
            "call": "punchthrough_field_ratio(11.993110)",
            "gold_call": "_oracle_punchthrough_field_ratio(11.993110)",
        },
        {
            "setup": "import numpy as np",
            "call": "punchthrough_field_ratio(8.111756)",
            "gold_call": "_oracle_punchthrough_field_ratio(8.111756)",
        },
        {
            "setup": "import numpy as np",
            "call": "punchthrough_field_ratio(4.0)",
            "gold_call": "_oracle_punchthrough_field_ratio(4.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "punchthrough_field_ratio(60.0)",
            "gold_call": "_oracle_punchthrough_field_ratio(60.0)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        punchthrough_field_ratio(0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_punchthrough_field_ratio(0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
