"""
Blocking voltage supported by the trapezoidal profile.

The blocking voltage is the area under the field profile. For a trapezoid running from the peak field down to the terminal field across the drift thickness this is elementary, and expressing it through the field ratio and the doping keeps every quantity tied to the same design point.

Returns
-------
float, the blocking voltage in V
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def punchthrough_breakdown_voltage(eps_r: float, doping: float, e_crit: float, eta: float) -> float:
    '''Blocking voltage of the punch-through design.

    Parameters
    ----------
    eps_r : float
        Relative permittivity.
    doping : float
        Drift doping in cm^-3.
    e_crit : float
        Critical field in V/cm.
    eta : float
        Terminal to peak field ratio.

    Returns
    -------
    float
        Blocking voltage in V.

    Raises
    ------
    ValueError
        If any of eps_r, doping, e_crit is not finite or is not strictly positive.
        ``eta`` outside ``[0, 1)`` is rejected: a terminal field at or above the
        peak field is not a punch-through profile.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_punchthrough_breakdown_voltage(eps_r: float, doping: float, e_crit: float, eta: float) -> float:
    for _v in (eps_r, doping, e_crit):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('eps_r, doping, e_crit must be finite and strictly positive')
    if not np.isfinite(eta) or not (0.0 <= eta < 1.0):
        raise ValueError('eta must be finite and lie in [0, 1)')
    q = 1.602e-19; eps0 = 8.854e-14
    return float(0.5*(1.0 - eta**2)*(eps_r*eps0/(q*doping))*e_crit**2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "punchthrough_breakdown_voltage(9.7, 4.442828e15, 2.436391e6, 0.280310)",
            "gold_call": "_oracle_punchthrough_breakdown_voltage(9.7, 4.442828e15, 2.436391e6, 0.280310)",
        },
        {
            "setup": "import numpy as np",
            "call": "punchthrough_breakdown_voltage(9.7, 1.499584e16, 2.695323e6, 0.275528)",
            "gold_call": "_oracle_punchthrough_breakdown_voltage(9.7, 1.499584e16, 2.695323e6, 0.275528)",
        },
        {
            "setup": "import numpy as np",
            "call": "punchthrough_breakdown_voltage(11.7, 1.0e15, 2.868e5, 0.0)",
            "gold_call": "_oracle_punchthrough_breakdown_voltage(11.7, 1.0e15, 2.868e5, 0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "punchthrough_breakdown_voltage(5.7, 1.0e15, 3.6906e6, 0.3333333333333333)",
            "gold_call": "_oracle_punchthrough_breakdown_voltage(5.7, 1.0e15, 3.6906e6, 0.3333333333333333)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        punchthrough_breakdown_voltage(9.7, 3.17e16, 2.88e6, -0.2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_punchthrough_breakdown_voltage(9.7, 3.17e16, 2.88e6, -0.2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
