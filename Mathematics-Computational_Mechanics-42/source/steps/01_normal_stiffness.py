"""
Return the source's normal fracture deformation relation g_n(q_n), which maps the normal fracture contact traction to the normal deformation of the fracture (the paper's Eq. 15). Two variants are used: the linear (Schoenberg) relation selected by model='Lin', and the nonlinear Barton-Bandis relation selected by model='BB', which additionally depends on the maximum allowed fracture closure du_max and stiffens as the fracture closes toward it. Recover both exact expressions from the paper; the Barton-Bandis relation is NOT the linear one and its denominator is the source's.

The normal stiffness relation is what converts a normal contact traction into fracture closure. The linear relation has constant stiffness, whereas the nonlinear relation stiffens as the fracture closes and cannot close beyond a maximum value.

Returns
-------
return array like q_n (float64): the normal fracture deformation g_n(q_n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def normal_stiffness(q_n, K_n, du_max, model):
    """q_n: normal fracture contact traction (<= 0 in compression); K_n: normal fracture
    stiffness; du_max: maximum allowed fracture closure; model: 'Lin' or 'BB'.
    Returns a float64 array shaped like q_n with the source's normal deformation relation
    g_n(q_n) (paper Eq. 15). Raises ValueError on an unknown model or a non-positive
    Barton-Bandis denominator."""
    return np.zeros_like(np.asarray(q_n, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 1: normal fracture deformation relation, linear and Barton-Bandis (Eq. 15)."""

import numpy as np


def _oracle_normal_stiffness(q_n, K_n, du_max, model):
    q = np.asarray(q_n, dtype=np.float64)
    if model == "Lin":
        return q / K_n
    if model == "BB":
        den = K_n - q / du_max
        if np.any(den <= 0):
            raise ValueError("Barton-Bandis denominator must stay positive")
        return q / den
    raise ValueError("model must be 'Lin' or 'BB'")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as _n\nq_n=_n.array([-4.0e6,-2.0e6,-1.0e6,0.0]);K_n=2.0e11;du_max=5.0e-5;model='Lin'", "call": 'normal_stiffness(q_n, K_n, du_max, model)', "gold_call": '_oracle_normal_stiffness(q_n, K_n, du_max, model)', "tol": 1e-18},
        {"setup": "import numpy as _n\nq_n=_n.array([-4.0e6,-2.0e6,-1.0e6,0.0]);K_n=2.0e11;du_max=5.0e-5;model='BB'", "call": 'normal_stiffness(q_n, K_n, du_max, model)', "gold_call": '_oracle_normal_stiffness(q_n, K_n, du_max, model)', "tol": 1e-18},
        {"setup": "import numpy as _n\nq_n=_n.array([-7.5e6,-3.3e6,-0.5e6]);K_n=1.5e11;du_max=8.0e-5;model='BB'", "call": 'normal_stiffness(q_n, K_n, du_max, model)', "gold_call": '_oracle_normal_stiffness(q_n, K_n, du_max, model)', "tol": 1e-18},
    ]
