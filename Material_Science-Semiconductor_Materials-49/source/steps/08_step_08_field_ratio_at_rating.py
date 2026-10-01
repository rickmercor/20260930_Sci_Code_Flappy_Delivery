"""
Terminal to peak field ratio forced on the punch-through drift layer by its voltage rating.

Once the doping is chosen the trapezoidal profile has one degree of freedom left, and the blocking requirement removes it: the layer must be thin enough that the field has not decayed to zero at the stop layer, yet thick enough that the area under the profile reaches the rated voltage. Inverting the blocking voltage for the field ratio turns a two parameter design space into a one parameter family, and that family is the constraint along which the drift on-resistance is then minimised. A doping so heavy that the rated voltage is out of reach even at zero terminal field has no punch-through design at all.

Returns
-------
float, the terminal to peak field ratio, dimensionless
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def field_ratio_at_rating(eps_r: float, doping: float, e_crit: float, bv_target: float) -> float:
    '''Terminal-to-peak field ratio that makes the trapezoidal profile block bv_target.

    Parameters
    ----------
    eps_r : float
        Relative permittivity of the semiconductor.
    doping : float
        Drift doping in cm^-3.
    e_crit : float
        Peak field at breakdown in V/cm.
    bv_target : float
        Blocking voltage the design must support in V.

    Returns
    -------
    float
        Field ratio in [0, 1).

    Raises
    ------
    ValueError
        If any of eps_r, doping, e_crit, bv_target is not finite or is not strictly positive.
        A doping that cannot reach bv_target before the field ratio falls to zero is rejected.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_field_ratio_at_rating(eps_r: float, doping: float, e_crit: float, bv_target: float) -> float:
    for _v in (eps_r, doping, e_crit, bv_target):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('eps_r, doping, e_crit, bv_target must be finite and strictly positive')
    f = lambda eta: _oracle_punchthrough_breakdown_voltage(eps_r, doping, e_crit, eta) - bv_target
    if f(0.0) < 0.0:
        raise ValueError('this doping cannot reach bv_target at any punch-through field ratio')
    return float(brentq(f, 0.0, 1.0 - 1.0e-12, xtol=1.0e-15, rtol=8.9e-16, maxiter=300))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "field_ratio_at_rating(9.7, 3.0345717e16, 2.8698979e6, 650.0)",
            "gold_call": "_oracle_field_ratio_at_rating(9.7, 3.0345717e16, 2.8698979e6, 650.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "field_ratio_at_rating(9.7, 2.5e16, 2.9e6, 650.0)",
            "gold_call": "_oracle_field_ratio_at_rating(9.7, 2.5e16, 2.9e6, 650.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "field_ratio_at_rating(9.7, 1.0e16, 3.1e6, 300.0)",
            "gold_call": "_oracle_field_ratio_at_rating(9.7, 1.0e16, 3.1e6, 300.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "field_ratio_at_rating(11.7, 1.0e15, 3.0e5, 100.0)",
            "gold_call": "_oracle_field_ratio_at_rating(11.7, 1.0e15, 3.0e5, 100.0)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        field_ratio_at_rating(9.7, -3.0e16, 2.87e6, 650.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_field_ratio_at_rating(9.7, -3.0e16, 2.87e6, 650.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
