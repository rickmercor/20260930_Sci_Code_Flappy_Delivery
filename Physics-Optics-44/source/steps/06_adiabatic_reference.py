"""
Return the width a beam would sit at if the response length were ramped linearly from sigma_i to sigma_f over a distance zf and the beam were able to follow that change arbitrarily closely, evaluated at the distance z. Clamp the normalized distance to the closed unit interval. Raise ValueError if zf is not strictly positive or if the ramped response length reaches zero or below.

A beam whose medium is changed slowly enough does not follow the change instantaneously in any detailed sense; what it follows is the width at which it would be in equilibrium for the parameter value reached so far. That instantaneous-equilibrium history is the reference any accelerated protocol is measured against, and the distance over which a ramp stays close to it is what sets how slow slowly has to be.

Returns
-------
float: the equilibrium width at the response length the ramp has reached.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adiabatic_reference(z: float, zf: float, power: float, gamma: float, alpha: float, sigma_i: float, sigma_f: float) -> float:
    """Return the width a beam would sit at if the response length were ramped linearly from sigma_i to sigma_f over a distance zf and the beam were able to follow that change arbitrarily closely, evaluated at the distance z. Clamp the normalized distance to the closed unit interval. Raise ValueError if zf is not strictly positive or if the ramped response length reaches zero or below.

    Returns
    -------
    float: the equilibrium width at the response length the ramp has reached.

    Raises
    ------
    ValueError
        If zf is not strictly positive, or the ramped response length is not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_adiabatic_reference(z: float, zf: float, power: float, gamma: float,
                                alpha: float, sigma_i: float, sigma_f: float) -> float:
    """Equilibrium width at the response length a linear ramp has reached at z."""
    z = float(z); zf = float(zf)
    if zf <= 0.0:
        raise ValueError("zf must be positive")
    s = min(max(z / zf, 0.0), 1.0)
    sigma = sigma_i + (sigma_f - sigma_i) * s
    if sigma <= 0.0:
        raise ValueError("the ramped response length must stay positive")
    return _oracle_equilibrium_width(power, gamma, alpha, sigma)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
            {
                    "setup": "import numpy as np",
                    "call": "adiabatic_reference(1.1, 2.2, 12.0, 0.2, 0.15, 4.0, 0.8)",
                    "gold_call": "_oracle_adiabatic_reference(1.1, 2.2, 12.0, 0.2, 0.15, 4.0, 0.8)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "adiabatic_reference(0.0, 2.2, 12.0, 0.2, 0.15, 4.0, 0.8)",
                    "gold_call": "_oracle_adiabatic_reference(0.0, 2.2, 12.0, 0.2, 0.15, 4.0, 0.8)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "adiabatic_reference(9.9, 2.2, 12.0, 0.2, 0.15, 4.0, 0.8)",
                    "gold_call": "_oracle_adiabatic_reference(9.9, 2.2, 12.0, 0.2, 0.15, 4.0, 0.8)"
            },
            {
                    "setup": "import numpy as np\ndef endpoints_match(fn, eq):\n    a = fn(2.2, 2.2, 12.0, 0.2, 0.15, 4.0, 0.8)\n    b = eq(12.0, 0.2, 0.15, 0.8)\n    return float(abs(a - b) < 1e-9)",
                    "call": "endpoints_match(adiabatic_reference, equilibrium_width)",
                    "gold_call": "endpoints_match(_oracle_adiabatic_reference, _oracle_equilibrium_width)"
            },
            {
                    "setup": "import numpy as np\ndef probe_public():\n    try:\n        adiabatic_reference(1.0, 0.0, 12.0, 0.2, 0.15, 4.0, 0.8)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_gold():\n    try:\n        _oracle_adiabatic_reference(1.0, 0.0, 12.0, 0.2, 0.15, 4.0, 0.8)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_public()",
                    "gold_call": "probe_gold()"
            }
    ]
