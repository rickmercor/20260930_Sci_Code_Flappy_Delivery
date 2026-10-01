"""
Return the squared frequency of small oscillations of the width about the given width, defined as minus the derivative of the predicted width acceleration with respect to the width, evaluated at that width and at the given medium parameters. Raise ValueError if the width, the power or the response length is not strictly positive.

Writing the width as an equilibrium value plus a small deviation and linearising the predicted acceleration turns it into a harmonic equation for the deviation, driven by the motion of the equilibrium itself. The coefficient of the deviation is this quantity, and it sets the intrinsic distance scale over which the width responds. It is not positive everywhere: each contribution to the acceleration carries its own dependence on the width, and the nonlocal one changes character according to how the response length compares with the beam, so an equilibrium is only stable where the total comes out positive.

Returns
-------
float: the squared oscillation frequency at the given width, which may be negative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def width_curvature(a: float, power: float, gamma: float, alpha: float, sigma: float) -> float:
    """Return the squared frequency of small oscillations of the width about the given width, defined as minus the derivative of the predicted width acceleration with respect to the width, evaluated at that width and at the given medium parameters. Raise ValueError if the width, the power or the response length is not strictly positive.

    Returns
    -------
    float: the squared oscillation frequency at the given width, which may be negative.

    Raises
    ------
    ValueError
        If a, power or sigma is not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_width_curvature(a: float, power: float, gamma: float, alpha: float,
                            sigma: float) -> float:
    """Curvature of the effective potential at width a: minus d(a'')/da."""
    a = float(a); power = float(power); sigma = float(sigma)
    if a <= 0.0 or power <= 0.0:
        raise ValueError("a and power must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return float(3.0 / a ** 4
                 + 2.0 * alpha ** 2
                 - 2.0 * power * gamma / (np.sqrt(2.0 * np.pi) * a ** 3)
                 + 2.0 * power / np.sqrt(np.pi)
                   * (sigma ** 2 - 4.0 * a ** 2) / (2.0 * a ** 2 + sigma ** 2) ** 2.5)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
            {
                    "setup": "import numpy as np",
                    "call": "width_curvature(0.8928089, 12.0, 0.2, 0.15, 4.0)",
                    "gold_call": "_oracle_width_curvature(0.8928089, 12.0, 0.2, 0.15, 4.0)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "width_curvature(0.4610400, 12.0, 0.2, 0.15, 0.8)",
                    "gold_call": "_oracle_width_curvature(0.4610400, 12.0, 0.2, 0.15, 0.8)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "width_curvature(0.7, 20.0, 0.1, 0.1, 0.5)",
                    "gold_call": "_oracle_width_curvature(0.7, 20.0, 0.1, 0.1, 0.5)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "width_curvature(1.5, 12.0, 0.2, 0.15, 2.4)",
                    "gold_call": "_oracle_width_curvature(1.5, 12.0, 0.2, 0.15, 2.4)"
            },
            {
                    "setup": "import numpy as np\ndef derivative_lo(fn, acc):\n    h = 1e-6\n    a, s = 0.5, 0.3\n    num = -(acc(a + h, 12.0, 0.2, 0.15, s) - acc(a - h, 12.0, 0.2, 0.15, s)) / (2.0 * h)\n    return float(abs(fn(a, 12.0, 0.2, 0.15, s) - num) < 1e-5 * abs(num))",
                    "call": "derivative_lo(width_curvature, width_acceleration)",
                    "gold_call": "derivative_lo(_oracle_width_curvature, _oracle_width_acceleration)"
            },
            {
                    "setup": "import numpy as np\ndef derivative_mid(fn, acc):\n    h = 1e-6\n    a, s = 1.1, 2.0\n    num = -(acc(a + h, 12.0, 0.2, 0.15, s) - acc(a - h, 12.0, 0.2, 0.15, s)) / (2.0 * h)\n    return float(abs(fn(a, 12.0, 0.2, 0.15, s) - num) < 1e-5 * abs(num))",
                    "call": "derivative_mid(width_curvature, width_acceleration)",
                    "gold_call": "derivative_mid(_oracle_width_curvature, _oracle_width_acceleration)"
            },
            {
                    "setup": "import numpy as np\ndef derivative_hi(fn, acc):\n    h = 1e-6\n    a, s = 2.0, 5.0\n    num = -(acc(a + h, 12.0, 0.2, 0.15, s) - acc(a - h, 12.0, 0.2, 0.15, s)) / (2.0 * h)\n    return float(abs(fn(a, 12.0, 0.2, 0.15, s) - num) < 1e-5 * abs(num))",
                    "call": "derivative_hi(width_curvature, width_acceleration)",
                    "gold_call": "derivative_hi(_oracle_width_curvature, _oracle_width_acceleration)"
            },
            {
                    "setup": "import numpy as np\ndef probe_public():\n    try:\n        width_curvature(0.0, 12.0, 0.2, 0.15, 4.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_gold():\n    try:\n        _oracle_width_curvature(0.0, 12.0, 0.2, 0.15, 4.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_public()",
                    "gold_call": "probe_gold()"
            }
    ]
