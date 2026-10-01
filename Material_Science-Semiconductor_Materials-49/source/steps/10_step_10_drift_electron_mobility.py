"""
Doping dependent electron mobility of the 4H-SiC drift layer at room temperature.

Treating the mobility as a constant makes the conduction loss look better than it is, because a thinner drift region needs heavier doping and heavier doping scatters carriers harder. The room temperature majority electron mobility of 4H-SiC follows a Caughey-Thomas form that falls from a lightly doped plateau of 950 cm^2 V^-1 s^-1 towards a heavily doped floor of 40 cm^2 V^-1 s^-1, with a reference doping of 2e17 cm^-3 and an exponent of 0.61.

Returns
-------
float, the electron mobility in cm^2 V^-1 s^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def drift_electron_mobility(doping: float) -> float:
    '''Room temperature electron mobility of the 4H-SiC drift layer.

    Parameters
    ----------
    doping : float
        Drift doping in cm^-3.

    Returns
    -------
    float
        Electron mobility in cm^2 V^-1 s^-1.

    Raises
    ------
    ValueError
        If doping is not finite or is not strictly positive.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_drift_electron_mobility(doping: float) -> float:
    for _v in (doping,):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('doping must be finite and strictly positive')
    return float(40.0 + 910.0/(1.0 + (doping/2.0e17)**0.61))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "drift_electron_mobility(4.442828e15)",
            "gold_call": "_oracle_drift_electron_mobility(4.442828e15)",
        },
        {
            "setup": "import numpy as np",
            "call": "drift_electron_mobility(1.0e14)",
            "gold_call": "_oracle_drift_electron_mobility(1.0e14)",
        },
        {
            "setup": "import numpy as np",
            "call": "drift_electron_mobility(2.0e17)",
            "gold_call": "_oracle_drift_electron_mobility(2.0e17)",
        },
        {
            "setup": "import numpy as np",
            "call": "drift_electron_mobility(1.0e19)",
            "gold_call": "_oracle_drift_electron_mobility(1.0e19)",
        },
        {
            "setup": "import numpy as np",
            "call": "drift_electron_mobility(1.499584e16)",
            "gold_call": "_oracle_drift_electron_mobility(1.499584e16)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        drift_electron_mobility(-1.0e16)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_drift_electron_mobility(-1.0e16)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
