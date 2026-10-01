"""
Advance the interface state variable given the current opening measure and its previous value. The source states whether the state may decrease when the interface closes again; follow it.

Interface degradation is a dissipative process, so its state variable is governed by a loading function rather than by the instantaneous opening alone.

Returns
-------
ndarray of shape (n_interfaces,), the updated state variable in [0, 1].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def cohesive_damage_update(effective_opening, critical_opening, damage_previous):
    """Advance the interface state variable given the current opening measure and its previous value. The source states whether the state may decrease when the interface closes again; follow it.

    Returns
    -------
    ndarray of shape (n_interfaces,), the updated state variable in [0, 1].

    Raises
    ------
    ValueError: if critical_opening is not positive, damage_previous lies outside [0, 1], or the two arrays do not share a shape.
    """
    return np.zeros(1)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cohesive_damage_update(effective_opening, critical_opening, damage_previous):
    d_eff = np.asarray(effective_opening, dtype=float)
    d_prev = np.asarray(damage_previous, dtype=float)
    if float(critical_opening) <= 0.0:
        raise ValueError("critical_opening must be positive")
    if np.any(d_prev < 0.0) or np.any(d_prev > 1.0):
        raise ValueError("damage_previous must lie in [0, 1]")
    if d_eff.shape != d_prev.shape:
        raise ValueError("effective_opening and damage_previous must have the same shape")
    # CONVENTION (paper, loading function f = delta/delta_c - d <= 0 with d_dot >= 0):
    # damage is the running MAXIMUM, never decreasing, and saturates at 1.
    return np.clip(np.maximum(d_prev, d_eff / float(critical_opening)), 0.0, 1.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "cohesive_damage_update(np.array([3e-4, 1e-4]), 6e-4, np.array([0.2, 0.4]))",
         "gold_call": "_oracle_cohesive_damage_update(np.array([3e-4, 1e-4]), 6e-4, np.array([0.2, 0.4]))"},   # normal
        {"setup": "import numpy as np",
         "call": "cohesive_damage_update(np.array([0.0]), 6e-4, np.array([0.0]))",
         "gold_call": "_oracle_cohesive_damage_update(np.array([0.0]), 6e-4, np.array([0.0]))"},   # boundary
        {"setup": "import numpy as np",
         "call": "cohesive_damage_update(np.array([9e-4]), 6e-4, np.array([0.1]))",
         "gold_call": "_oracle_cohesive_damage_update(np.array([9e-4]), 6e-4, np.array([0.1]))"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        cohesive_damage_update(np.array([1e-4]), 6e-4, np.array([1.4]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_cohesive_damage_update(np.array([1e-4]), 6e-4, np.array([1.4]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
