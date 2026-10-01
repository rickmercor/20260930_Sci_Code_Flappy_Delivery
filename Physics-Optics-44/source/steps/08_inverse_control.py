"""
Return the value one medium parameter must take so that the reduced description produces exactly the given width and width acceleration, with the other two parameters held at their supplied values. The knob is named by the string 'sigma' for the nonlocal length, 'gamma' for the local Kerr coefficient, or 'alpha2' for the squared confinement strength. The supplied value of the parameter being solved for is ignored in every case. Raise ValueError if the width or the power is not strictly positive, if the knob name is not one of the three, or if no positive nonlocal length reproduces the requested pair.

This is the step that turns a prescribed width history into a physical protocol. Because the reduced relation is algebraic in each parameter separately, it can be solved for one of them in closed form at every propagation distance. The three knobs enter the relation in different ways, so the same width history demands very different control profiles: two of them enter linearly and the third does not.

Returns
-------
float: the required value of the named parameter, the squared strength when the knob is 'alpha2'.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def inverse_control(knob: str, a: float, add: float, power: float, gamma: float, alpha: float, sigma: float) -> float:
    """Return the value one medium parameter must take so that the reduced description produces exactly the given width and width acceleration, with the other two parameters held at their supplied values. The knob is named by the string 'sigma' for the nonlocal length, 'gamma' for the local Kerr coefficient, or 'alpha2' for the squared confinement strength. The supplied value of the parameter being solved for is ignored in every case. Raise ValueError if the width or the power is not strictly positive, if the knob name is not one of the three, or if no positive nonlocal length reproduces the requested pair.

    Returns
    -------
    float: the required value of the named parameter, the squared strength when the knob is 'alpha2'.

    Raises
    ------
    ValueError
        If a or power is not strictly positive, if knob is not 'sigma', 'gamma' or 'alpha2', or if no positive nonlocal length reproduces the requested width and acceleration.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_inverse_control(knob: str, a: float, add: float, power: float,
                            gamma: float, alpha: float, sigma: float) -> float:
    """Invert Eq. (7) for one control knob so that the width follows (a, a'')."""
    a = float(a); add = float(add); power = float(power)
    if a <= 0.0 or power <= 0.0:
        raise ValueError("a and power must be positive")
    if knob == "sigma":
        rhs = (1.0 / a ** 3 - 2.0 * alpha ** 2 * a
               - power * gamma / (np.sqrt(2.0 * np.pi) * a ** 2) - add)
        if rhs <= 0.0:
            raise ValueError("no positive nonlocal length reproduces this trajectory")
        val = (2.0 * power * a / (np.sqrt(np.pi) * rhs)) ** (2.0 / 3.0) - 2.0 * a ** 2
        if val <= 0.0:
            raise ValueError("no positive nonlocal length reproduces this trajectory")
        return float(np.sqrt(val))
    if knob == "gamma":
        rest = (1.0 / a ** 3 - 2.0 * alpha ** 2 * a
                - 2.0 * power * a / (np.sqrt(np.pi) * (2.0 * a ** 2 + sigma ** 2) ** 1.5) - add)
        return float(rest * np.sqrt(2.0 * np.pi) * a ** 2 / power)
    if knob == "alpha2":
        rest = (1.0 / a ** 3 - power * gamma / (np.sqrt(2.0 * np.pi) * a ** 2)
                - 2.0 * power * a / (np.sqrt(np.pi) * (2.0 * a ** 2 + sigma ** 2) ** 1.5) - add)
        return float(rest / (2.0 * a))
    raise ValueError("knob must be 'sigma', 'gamma' or 'alpha2'")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
            {
                    "setup": "import numpy as np",
                    "call": "inverse_control('sigma', 0.7, -0.35, 12.0, 0.2, 0.15, 4.0)",
                    "gold_call": "_oracle_inverse_control('sigma', 0.7, -0.35, 12.0, 0.2, 0.15, 4.0)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "inverse_control('gamma', 0.7, -0.35, 12.0, 0.2, 0.15, 4.0)",
                    "gold_call": "_oracle_inverse_control('gamma', 0.7, -0.35, 12.0, 0.2, 0.15, 4.0)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "inverse_control('alpha2', 0.4610400, 0.0, 12.0, 0.2, 0.15, 4.0)",
                    "gold_call": "_oracle_inverse_control('alpha2', 0.4610400, 0.0, 12.0, 0.2, 0.15, 4.0)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "inverse_control('alpha2', 0.7, -0.35, 12.0, 0.2, 0.15, 4.0)",
                    "gold_call": "_oracle_inverse_control('alpha2', 0.7, -0.35, 12.0, 0.2, 0.15, 4.0)"
            },
            {
                    "setup": "import numpy as np\ndef probe_nonpos_public():\n    try:\n        inverse_control('sigma', -1.0, 0.0, 12.0, 0.2, 0.15, 4.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_nonpos_gold():\n    try:\n        _oracle_inverse_control('sigma', -1.0, 0.0, 12.0, 0.2, 0.15, 4.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_nonpos_public()",
                    "gold_call": "probe_nonpos_gold()"
            },
            {
                    "setup": "import numpy as np\ndef probe_unreach_public():\n    try:\n        inverse_control('sigma', 0.7, 100.0, 12.0, 0.2, 0.15, 4.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_unreach_gold():\n    try:\n        _oracle_inverse_control('sigma', 0.7, 100.0, 12.0, 0.2, 0.15, 4.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_unreach_public()",
                    "gold_call": "probe_unreach_gold()"
            },
            {
                    "setup": "import numpy as np\ndef round_trip(fn, eq):\n    a = eq(12.0, 0.2, 0.15, 4.0)\n    return float(fn('sigma', a, 0.0, 12.0, 0.2, 0.15, 4.0))",
                    "call": "round_trip(inverse_control, equilibrium_width)",
                    "gold_call": "round_trip(_oracle_inverse_control, _oracle_equilibrium_width)"
            },
            {
                    "setup": "import numpy as np\ndef probe_public():\n    try:\n        inverse_control('pitch', 0.7, -0.35, 12.0, 0.2, 0.15, 4.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_gold():\n    try:\n        _oracle_inverse_control('pitch', 0.7, -0.35, 12.0, 0.2, 0.15, 4.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_public()",
                    "gold_call": "probe_gold()"
            }
    ]
