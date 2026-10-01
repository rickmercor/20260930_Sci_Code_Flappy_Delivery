"""
Return the interface stiffness ceiling used to keep the explicit time step usable, together with the state-variable threshold that corresponds to it. Both come from the source; the quantities the ceiling is built from are what matter.

An interface whose stiffness grows without bound would drive the stable time step to zero, so the law is regularised by capping that stiffness. The cap has to be expressed in quantities the bulk discretisation already fixes.

Returns
-------
ndarray of shape (2,): the stiffness ceiling, then the state-variable threshold.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def cohesive_stiffness_cap(youngs, element_size, alpha, cohesive_strength, critical_opening):
    """Return the interface stiffness ceiling used to keep the explicit time step usable, together with the state-variable threshold that corresponds to it. Both come from the source; the quantities the ceiling is built from are what matter.

    Returns
    -------
    ndarray of shape (2,): the stiffness ceiling, then the state-variable threshold.

    Raises
    ------
    ValueError: if any of youngs, element_size, alpha, cohesive_strength or critical_opening is not positive.
    """
    return np.zeros(1)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cohesive_stiffness_cap(youngs, element_size, alpha, cohesive_strength,
                                   critical_opening):
    if min(float(youngs), float(element_size), float(alpha),
           float(cohesive_strength), float(critical_opening)) <= 0.0:
        raise ValueError("youngs, element_size, alpha, cohesive_strength and "
                         "critical_opening must all be positive")
    # CONVENTION (paper, regularisation): the stiffness cap is tied to the BULK element
    # stiffness through a user parameter alpha, and the damage threshold follows from it.
    k_tilde = float(alpha) * float(youngs) / float(element_size)
    sig = float(cohesive_strength); dc = float(critical_opening)
    d_tilde = sig / (sig + k_tilde * dc)
    return np.array([k_tilde, d_tilde], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "cohesive_stiffness_cap(211e9, 5.08e-3, 4.0, 6.0e8, 6e-4)",
         "gold_call": "_oracle_cohesive_stiffness_cap(211e9, 5.08e-3, 4.0, 6.0e8, 6e-4)"},   # normal
        {"setup": "import numpy as np",
         "call": "cohesive_stiffness_cap(211e9, 5.08e-3, 1.0, 6.0e8, 6e-4)",
         "gold_call": "_oracle_cohesive_stiffness_cap(211e9, 5.08e-3, 1.0, 6.0e8, 6e-4)"},   # boundary
        {"setup": "import numpy as np",
         "call": "cohesive_stiffness_cap(70e9, 1.0e-4, 10.0, 2.0e8, 1e-5)",
         "gold_call": "_oracle_cohesive_stiffness_cap(70e9, 1.0e-4, 10.0, 2.0e8, 1e-5)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        cohesive_stiffness_cap(211e9, 5.08e-3, 0.0, 6.0e8, 6e-4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_cohesive_stiffness_cap(211e9, 5.08e-3, 0.0, 6.0e8, 6e-4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
