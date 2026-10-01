"""
Return the Coulomb friction bound of the source's contact model from the normal contact traction and the coefficient of friction (the paper's Eq. 10). The source's sign convention has the normal contact traction non-positive in compression, and the bound must come out non-negative so that it can bound a magnitude. Recover the exact expression, including its sign, from the paper.

The friction bound is the largest tangential traction a fracture in contact can carry before it slips; it is proportional to how hard the fracture is being pressed together.

Returns
-------
return array like q_n (float64): the Coulomb friction bound b
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def friction_bound(q_n, F):
    """q_n: normal contact traction (<= 0 in compression); F: coefficient of friction.
    Returns a float64 array shaped like q_n with the source's Coulomb friction bound
    (paper Eq. 10), non-negative for a fracture in contact."""
    return np.zeros_like(np.asarray(q_n, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 3: Coulomb friction bound (Eq. 10)."""

import numpy as np


def _oracle_friction_bound(q_n, F):
    q = np.asarray(q_n, dtype=np.float64)
    if q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("q_n must be finite and non-empty")
    Ff = float(F)
    if not np.isfinite(Ff) or Ff < 0.0:
        raise ValueError("F must be finite and non-negative")
    return -Ff * q

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nq_n=_n.array([-4.0e6,-2.0e6,0.0,-1.0e6]);F=1.0', "call": 'friction_bound(q_n, F)', "gold_call": '_oracle_friction_bound(q_n, F)', "tol": 1e-06},
        {"setup": 'import numpy as _n\nq_n=_n.array([-7.5e6,-3.3e6,-0.5e6]);F=0.6', "call": 'friction_bound(q_n, F)', "gold_call": '_oracle_friction_bound(q_n, F)', "tol": 1e-06},
        {"setup": 'import numpy as _n\nq_n=_n.array([0.0,-9.1e6,-2.7e6,-5.5e6]);F=0.85', "call": 'friction_bound(q_n, F)', "gold_call": '_oracle_friction_bound(q_n, F)', "tol": 1e-06},
    ]
