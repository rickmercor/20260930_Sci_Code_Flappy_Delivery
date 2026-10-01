"""
Drift thickness of the trapezoidal punch-through design.

The field falls linearly across the drift region at a rate set by the ionised doping and the permittivity. Going from the peak field down to the terminal field fixes how thick the drift region must be. Complete ionisation of the dopants is assumed.

Returns
-------
float, the drift region thickness in cm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def punchthrough_depletion_width(eps_r: float, doping: float, e_crit: float, eta: float) -> float:
    '''Drift region thickness of the punch-through design.

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
        Drift thickness in cm.

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

def _oracle_punchthrough_depletion_width(eps_r: float, doping: float, e_crit: float, eta: float) -> float:
    for _v in (eps_r, doping, e_crit):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('eps_r, doping, e_crit must be finite and strictly positive')
    if not np.isfinite(eta) or not (0.0 <= eta < 1.0):
        raise ValueError('eta must be finite and lie in [0, 1)')
    q = 1.602e-19; eps0 = 8.854e-14
    return float((1.0 - eta)*(eps_r*eps0/(q*doping))*e_crit)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "punchthrough_depletion_width(9.7, 4.442828e15, 2.436391e6, 0.280310)",
            "gold_call": "_oracle_punchthrough_depletion_width(9.7, 4.442828e15, 2.436391e6, 0.280310)",
        },
        {
            "setup": "import numpy as np",
            "call": "punchthrough_depletion_width(11.7, 1.0e15, 2.868e5, 0.25)",
            "gold_call": "_oracle_punchthrough_depletion_width(11.7, 1.0e15, 2.868e5, 0.25)",
        },
        {
            "setup": "import numpy as np",
            "call": "punchthrough_depletion_width(9.7, 1.0e17, 3.2176e6, 0.0)",
            "gold_call": "_oracle_punchthrough_depletion_width(9.7, 1.0e17, 3.2176e6, 0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "punchthrough_depletion_width(8.9, 1.0e16, 3.1217e6, 0.3333333333333333)",
            "gold_call": "_oracle_punchthrough_depletion_width(8.9, 1.0e16, 3.1217e6, 0.3333333333333333)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        punchthrough_depletion_width(9.7, 3.17e16, 2.88e6, 1.4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_punchthrough_depletion_width(9.7, 3.17e16, 2.88e6, 1.4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
