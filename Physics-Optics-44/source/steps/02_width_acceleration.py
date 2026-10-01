"""
Return the second derivative of the beam width with respect to propagation distance that the reduced description of this medium predicts, for a beam of the given width carrying the given power in a medium with the given local Kerr coefficient, confinement strength and nonlocal length. The relation is the one the source derives by reducing the full propagation equation to the beam's collective coordinates. Raise ValueError if the width, the power or the nonlocal length is not strictly positive.

Reducing the field to a single collective coordinate turns the propagation problem into the motion of a particle in an effective potential, whose four contributions are diffraction, the external confinement, the local nonlinearity and the nonlocal one. The nonlocal contribution is the only one that depends on the response length, and it interpolates between the local limit and a strongly averaged one as that length grows past the beam.

Returns
-------
float: the second derivative of the width with respect to propagation distance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def width_acceleration(a: float, power: float, gamma: float, alpha: float, sigma: float) -> float:
    """Return the second derivative of the beam width with respect to propagation distance that the reduced description of this medium predicts, for a beam of the given width carrying the given power in a medium with the given local Kerr coefficient, confinement strength and nonlocal length. The relation is the one the source derives by reducing the full propagation equation to the beam's collective coordinates. Raise ValueError if the width, the power or the nonlocal length is not strictly positive.

    Returns
    -------
    float: the second derivative of the width with respect to propagation distance.

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


def _oracle_width_acceleration(a: float, power: float, gamma: float, alpha: float, sigma: float) -> float:
    a = float(a); power = float(power); gamma = float(gamma)
    alpha = float(alpha); sigma = float(sigma)
    if a <= 0.0 or power <= 0.0 or sigma <= 0.0:
        raise ValueError("a, power and sigma must be positive")
    return float(1.0 / a ** 3
                 - 2.0 * alpha ** 2 * a
                 - power * gamma / (np.sqrt(2.0 * np.pi) * a ** 2)
                 - 2.0 * power * a / (np.sqrt(np.pi) * (2.0 * a ** 2 + sigma ** 2) ** 1.5))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
            {
                    "setup": "import numpy as np",
                    "call": "width_acceleration(1.0, 12.0, 0.2, 0.15, 4.0)",
                    "gold_call": "_oracle_width_acceleration(1.0, 12.0, 0.2, 0.15, 4.0)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "width_acceleration(0.5, 20.0, 0.1, 0.1, 0.5)",
                    "gold_call": "_oracle_width_acceleration(0.5, 20.0, 0.1, 0.1, 0.5)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "width_acceleration(3.0, 1.0, 0.0, 0.0, 1e-4)",
                    "gold_call": "_oracle_width_acceleration(3.0, 1.0, 0.0, 0.0, 1e-4)"
            },
            {
                    "setup": "import numpy as np\ndef free_limit(fn):\n    return float(fn(2.0, 1e-12, 0.0, 0.0, 1.0) * 8.0)",
                    "call": "free_limit(width_acceleration)",
                    "gold_call": "free_limit(_oracle_width_acceleration)"
            },
            {
                    "setup": "import numpy as np\ndef probe_public():\n    try:\n        width_acceleration(0.0, 12.0, 0.2, 0.15, 4.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_gold():\n    try:\n        _oracle_width_acceleration(0.0, 12.0, 0.2, 0.15, 4.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_public()",
                    "gold_call": "probe_gold()"
            }
    ]
