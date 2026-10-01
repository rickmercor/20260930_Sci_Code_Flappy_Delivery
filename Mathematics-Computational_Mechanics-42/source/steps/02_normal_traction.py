"""
Return the normal fracture contact traction produced by a prescribed normal displacement jump, for both families of fracture deformation model. For the spring model (contact=False) the traction is whatever the source's normal relation of Eq. 15 gives when the deformation equals the displacement jump (the paper's Eq. 9), so the fracture can carry traction of either sign. For the contact mechanics model (contact=True) the source instead imposes the unilateral non-penetration conditions of Eq. 12, under which the traction is never tensile and vanishes wherever the fracture is open. Invert the source's relation exactly; the Barton-Bandis inversion is not the linear one.

This is the difference between a fracture treated as a two-sided spring and one treated as a unilateral contact: a spring transmits tension across an opening fracture, whereas a contact model lets the fracture open and carry nothing.

Returns
-------
return array like jump_n (float64): the normal contact traction q_n
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def normal_traction(jump_n, K_n, du_max, model, contact):
    """jump_n: normal displacement jump across the fracture (negative in closure); K_n:
    normal fracture stiffness; du_max: maximum allowed closure; model: 'Lin' or 'BB';
    contact: False for the spring model (paper Eq. 9), True for the contact mechanics
    model with non-penetration (paper Eq. 12). Returns a float64 array shaped like jump_n
    with the normal contact traction q_n. Raises ValueError on an unknown model or a
    closure at or beyond du_max."""
    return np.zeros_like(np.asarray(jump_n, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 2: normal contact traction from the displacement jump (Eqs. 9, 12, 15)."""

import numpy as np


def _oracle_normal_traction(jump_n, K_n, du_max, model, contact):
    jn = np.asarray(jump_n, dtype=np.float64)
    if model == "Lin":
        q = K_n * jn
    elif model == "BB":
        fac = 1.0 + jn / du_max
        if np.any(fac <= 0):
            raise ValueError("closure at or beyond the maximum allowed closure")
        q = K_n * jn / fac
    else:
        raise ValueError("model must be 'Lin' or 'BB'")
    if contact:
        q = np.where(jn >= 0.0, 0.0, q)
    return q

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as _n\njump_n=_n.array([1.8e-5,-1.2e-5,-2.4e-5,0.6e-5,-3.0e-5]);K_n=2.0e11;du_max=5.0e-5;model='Lin';contact=False", "call": 'normal_traction(jump_n, K_n, du_max, model, contact)', "gold_call": '_oracle_normal_traction(jump_n, K_n, du_max, model, contact)', "tol": 1e-06},
        {"setup": "import numpy as _n\njump_n=_n.array([1.8e-5,-1.2e-5,-2.4e-5,0.6e-5,-3.0e-5]);K_n=2.0e11;du_max=5.0e-5;model='BB';contact=True", "call": 'normal_traction(jump_n, K_n, du_max, model, contact)', "gold_call": '_oracle_normal_traction(jump_n, K_n, du_max, model, contact)', "tol": 1e-06},
        {"setup": "import numpy as _n\njump_n=_n.array([-0.9e-5,2.2e-5,-3.6e-5,-1.5e-5]);K_n=1.5e11;du_max=8.0e-5;model='BB';contact=False", "call": 'normal_traction(jump_n, K_n, du_max, model, contact)', "gold_call": '_oracle_normal_traction(jump_n, K_n, du_max, model, contact)', "tol": 1e-06},
    ]
