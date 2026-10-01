"""
Return the stability limit on the time step for explicit integration of the discretised bar, from the element size and the material wave speed.

Explicit schemes for elastodynamics are conditionally stable: the step is bounded by the time a wave needs to cross one element.

Returns
-------
float, the critical time step in seconds.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def critical_time_step(n_elements, length, youngs, density):
    """Return the stability limit on the time step for explicit integration of the discretised bar, from the element size and the material wave speed.

    Returns
    -------
    float, the critical time step in seconds.

    Raises
    ------
    ValueError: if n_elements is below 1, or any of length, youngs or density is not positive.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_critical_time_step(n_elements, length, youngs, density):
    if n_elements < 1:
        raise ValueError("n_elements must be at least 1")
    if min(length, youngs, density) <= 0:
        raise ValueError("length, youngs and density must be positive")
    c = np.sqrt(float(youngs) / float(density))
    h = float(length) / int(n_elements)
    return float(h / c)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "critical_time_step(50, 0.254, 211e9, 7847.0)",
         "gold_call": "_oracle_critical_time_step(50, 0.254, 211e9, 7847.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "critical_time_step(1, 1.0, 1.0e11, 1000.0)",
         "gold_call": "_oracle_critical_time_step(1, 1.0, 1.0e11, 1000.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "critical_time_step(5000, 0.254, 211e9, 7847.0)",
         "gold_call": "_oracle_critical_time_step(5000, 0.254, 211e9, 7847.0)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        critical_time_step(3, -1.0, 211e9, 7847.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_critical_time_step(3, -1.0, 211e9, 7847.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
