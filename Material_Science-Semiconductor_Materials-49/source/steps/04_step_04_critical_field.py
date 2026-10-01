"""
Convert the dimensionless field variable into the peak field at breakdown.

The dimensionless field variable was defined as the effective field scale divided by the peak field at the blocking junction, so recovering the critical field is a direct inversion. This is the field at the metallurgical junction where the triangular or trapezoidal profile is largest.

Returns
-------
float, the critical electric field in V/cm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def critical_field(b_eff: float, zeta: float) -> float:
    '''Peak electric field at avalanche breakdown.

    Parameters
    ----------
    b_eff : float
        Effective field scale in V/cm.
    zeta : float
        Dimensionless field variable.

    Returns
    -------
    float
        Critical field in V/cm.

    Raises
    ------
    ValueError
        If any of b_eff, zeta is not finite or is not strictly positive.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_critical_field(b_eff: float, zeta: float) -> float:
    for _v in (b_eff, zeta):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('b_eff, zeta must be finite and strictly positive')
    return float(b_eff/zeta)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "critical_field(2.61e7, 10.712565)",
            "gold_call": "_oracle_critical_field(2.61e7, 10.712565)",
        },
        {
            "setup": "import numpy as np",
            "call": "critical_field(1.615e6, 4.043)",
            "gold_call": "_oracle_critical_field(1.615e6, 4.043)",
        },
        {
            "setup": "import numpy as np",
            "call": "critical_field(2.695e7, 12.0)",
            "gold_call": "_oracle_critical_field(2.695e7, 12.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "critical_field(1.895e7, 20.0)",
            "gold_call": "_oracle_critical_field(1.895e7, 20.0)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        critical_field(2.61e7, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_critical_field(2.61e7, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
