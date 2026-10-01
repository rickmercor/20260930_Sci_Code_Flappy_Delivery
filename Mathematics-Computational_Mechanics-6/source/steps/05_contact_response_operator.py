"""
Build the operator that maps impulses on the active constraints to the constraint velocities they produce. It must be consistent with how this scheme feeds the impulse back into the configuration, so it is not the operator a purely kinematic argument gives; take its form from the source.

Eliminating the end-of-step acceleration from the balance equation and substituting into the complementarity condition leaves a linear complementarity problem in the impulse alone. This step assembles its matrix.

Returns
-------
ndarray of shape (n_active, n_active); a (0, 0) array when no constraint is active.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def contact_response_operator(stiffness, lumped_mass, contact_rows, active, dt):
    """Build the operator that maps impulses on the active constraints to the constraint velocities they produce. It must be consistent with how this scheme feeds the impulse back into the configuration, so it is not the operator a purely kinematic argument gives; take its form from the source.

    Returns
    -------
    ndarray of shape (n_active, n_active); a (0, 0) array when no constraint is active.

    Raises
    ------
    ValueError: if dt is not positive, or any lumped_mass entry is not positive.
    """
    return np.zeros((1, 1))  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_contact_response_operator(stiffness, lumped_mass, contact_rows, active, dt):
    K = np.asarray(stiffness, dtype=float)
    Md = np.asarray(lumped_mass, dtype=float)
    H = np.atleast_2d(np.asarray(contact_rows, dtype=float))
    act = np.asarray(active, dtype=float) > 0.5
    if dt <= 0:
        raise ValueError("dt must be positive")
    if np.any(Md <= 0):
        raise ValueError("lumped_mass entries must be positive")
    if not act.any():
        return np.zeros((0, 0), dtype=float)
    dt = float(dt)
    HA = H[act.ravel()]
    Minv = 1.0 / Md
    MiHt = Minv[:, None] * HA.T
    # CONVENTION (paper, eq. 46): the MODIFIED Delassus operator. The (dt^2/4) K M^-1
    # correction comes from folding a_{n+1} back through the half-step displacement
    # update. The plain Delassus H M^-1 H^T is the natural wrong answer.
    return HA @ MiHt - (dt * dt / 4.0) * (HA @ (Minv[:, None] * (K @ MiHt)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "contact_response_operator(np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), 0.1)",
         "gold_call": "_oracle_contact_response_operator(np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), 0.1)"},   # normal
        {"setup": "import numpy as np",
         "call": "contact_response_operator(np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([0.0]), 0.1)",
         "gold_call": "_oracle_contact_response_operator(np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([0.0]), 0.1)"},   # boundary
        {"setup": "import numpy as np",
         "call": "contact_response_operator(np.eye(3)*1e6, np.array([2.0,1.0,2.0]), np.array([[1.0,0.0,0.0],[0.0,0.0,1.0]]), np.array([1.0,1.0]), 1e-6)",
         "gold_call": "_oracle_contact_response_operator(np.eye(3)*1e6, np.array([2.0,1.0,2.0]), np.array([[1.0,0.0,0.0],[0.0,0.0,1.0]]), np.array([1.0,1.0]), 1e-6)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        contact_response_operator(np.eye(2), np.array([1.0,-1.0]), np.array([[1.0,0.0]]), np.array([1.0]), 0.1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_contact_response_operator(np.eye(2), np.array([1.0,-1.0]), np.array([[1.0,0.0]]), np.array([1.0]), 0.1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
