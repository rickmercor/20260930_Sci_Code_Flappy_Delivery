"""
Combine the normal and tangential parts of an interface opening into the single scalar measure the interface law is written in terms of. The source fixes how the tangential part is weighted; follow it exactly.

A degradable interface opens in both normal and tangential directions. Mixed-mode interface laws reduce the two to one scalar before evaluating the state.

Returns
-------
ndarray of shape (n_interfaces,), the scalar opening measure per interface.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def cohesive_effective_opening(delta_n, delta_t, beta):
    """Combine the normal and tangential parts of an interface opening into the single scalar measure the interface law is written in terms of. The source fixes how the tangential part is weighted; follow it exactly.

    Returns
    -------
    ndarray of shape (n_interfaces,), the scalar opening measure per interface.

    Raises
    ------
    ValueError: if beta is negative, or delta_t does not have one row per interface.
    """
    return np.zeros(1)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cohesive_effective_opening(delta_n, delta_t, beta):
    dn = np.asarray(delta_n, dtype=float)
    dt = np.atleast_2d(np.asarray(delta_t, dtype=float))
    if float(beta) < 0.0:
        raise ValueError("beta must be non-negative")
    if dt.shape[0] != dn.shape[0]:
        raise ValueError("delta_t must have one row per interface")
    b = float(beta)
    # CONVENTION (paper eq. 17): the mixed-mode effective opening weights the TANGENTIAL
    # part by beta SQUARED under the root. This is NOT the same weighting the traction
    # direction uses; see step 12.
    return np.sqrt(dn * dn + b * b * np.sum(dt * dt, axis=1))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "cohesive_effective_opening(np.array([1e-6, 5e-6]), np.array([[2e-7],[1e-6]]), 1.4)",
         "gold_call": "_oracle_cohesive_effective_opening(np.array([1e-6, 5e-6]), np.array([[2e-7],[1e-6]]), 1.4)"},   # normal
        {"setup": "import numpy as np",
         "call": "cohesive_effective_opening(np.array([0.0]), np.array([[0.0]]), 1.4)",
         "gold_call": "_oracle_cohesive_effective_opening(np.array([0.0]), np.array([[0.0]]), 1.4)"},   # boundary
        {"setup": "import numpy as np",
         "call": "cohesive_effective_opening(np.linspace(0.0, 6e-4, 32), np.full((32,1), 2e-5), 1.4)",
         "gold_call": "_oracle_cohesive_effective_opening(np.linspace(0.0, 6e-4, 32), np.full((32,1), 2e-5), 1.4)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        cohesive_effective_opening(np.array([1e-6]), np.array([[1e-7],[2e-7]]), 1.4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_cohesive_effective_opening(np.array([1e-6]), np.array([[1e-7],[2e-7]]), 1.4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
