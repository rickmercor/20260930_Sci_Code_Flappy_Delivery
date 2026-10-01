"""
Given the response operator from the previous step, solve for the impulses on the active constraints. The right-hand side combines the incoming constraint velocity, the smooth acceleration contribution and the elastic term; how the coefficient of restitution enters it is fixed by the source. Impulses are nonnegative.

The impulse follows from a complementarity condition between constraint velocity and impulse, equivalently a nonnegative quadratic program. Contact can only push.

Returns
-------
ndarray of shape (n_active,) of nonnegative impulses; empty when nothing is active.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def contact_impulse(response_operator, stiffness, lumped_mass, contact_rows, active, smooth_state, v, a, dt, restitution):
    """Given the response operator from the previous step, solve for the impulses on the active constraints. The right-hand side combines the incoming constraint velocity, the smooth acceleration contribution and the elastic term; how the coefficient of restitution enters it is fixed by the source. Impulses are nonnegative.

    Returns
    -------
    ndarray of shape (n_active,) of nonnegative impulses; empty when nothing is active.

    Raises
    ------
    ValueError: if restitution lies outside [0, 1], or response_operator is not square of size n_active.
    """
    return np.zeros(1)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_contact_impulse(response_operator, stiffness, lumped_mass, contact_rows,
                            active, predicted, v, a, dt, restitution):
    K = np.asarray(stiffness, dtype=float)
    Md = np.asarray(lumped_mass, dtype=float)
    H = np.atleast_2d(np.asarray(contact_rows, dtype=float))
    act = np.asarray(active, dtype=float) > 0.5
    if not (0.0 <= float(restitution) <= 1.0):
        raise ValueError("restitution must lie in [0, 1]")
    if not act.any():
        return np.zeros(0, dtype=float)
    v = np.asarray(v, dtype=float); a = np.asarray(a, dtype=float)
    dt = float(dt); e = float(restitution)
    W = np.atleast_2d(np.asarray(response_operator, dtype=float))
    if W.shape[0] != int(act.sum()):
        raise ValueError("response_operator must be square of size n_active")
    HA = H[act.ravel()]
    Minv = 1.0 / Md
    u_pred = np.asarray(predicted, dtype=float)
    # CONVENTION (paper, eq. 47): Newton's impact law puts (1 + e) on v_n, and the
    # elastic term enters as -(dt/2) M^-1 K u_pred.
    b = HA @ ((1.0 + e) * v + 0.5 * dt * a - 0.5 * dt * (Minv * (K @ u_pred)))
    p = np.linalg.solve(W, -b)
    return np.maximum(0.0, p)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "contact_impulse(contact_response_operator(np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), 0.05), np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), smooth_predictor(np.array([-0.01,0.5]), np.array([-2.0,-2.0]), np.zeros(2), 0.05), np.array([-2.0,-2.0]), np.zeros(2), 0.05, 0.5)",
         "gold_call": "_oracle_contact_impulse(contact_response_operator(np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), 0.05), np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), smooth_predictor(np.array([-0.01,0.5]), np.array([-2.0,-2.0]), np.zeros(2), 0.05), np.array([-2.0,-2.0]), np.zeros(2), 0.05, 0.5)"},   # normal
        {"setup": "import numpy as np",
         "call": "contact_impulse(contact_response_operator(np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), 0.05), np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), smooth_predictor(np.array([-0.01,0.5]), np.array([-2.0,-2.0]), np.zeros(2), 0.05), np.array([-2.0,-2.0]), np.zeros(2), 0.05, 0.0)",
         "gold_call": "_oracle_contact_impulse(contact_response_operator(np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), 0.05), np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), smooth_predictor(np.array([-0.01,0.5]), np.array([-2.0,-2.0]), np.zeros(2), 0.05), np.array([-2.0,-2.0]), np.zeros(2), 0.05, 0.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "contact_impulse(contact_response_operator(np.eye(3)*1e5, np.array([2.0,1.0,2.0]), np.array([[1.0,0.0,0.0]]), np.array([1.0]), 1e-6), np.eye(3)*1e5, np.array([2.0,1.0,2.0]), np.array([[1.0,0.0,0.0]]), np.array([1.0]), smooth_predictor(np.array([-1e-6,0.0,1.0]), np.full(3,-5.0), np.zeros(3), 1e-6), np.full(3,-5.0), np.zeros(3), 1e-6, 1.0)",
         "gold_call": "_oracle_contact_impulse(contact_response_operator(np.eye(3)*1e5, np.array([2.0,1.0,2.0]), np.array([[1.0,0.0,0.0]]), np.array([1.0]), 1e-6), np.eye(3)*1e5, np.array([2.0,1.0,2.0]), np.array([[1.0,0.0,0.0]]), np.array([1.0]), smooth_predictor(np.array([-1e-6,0.0,1.0]), np.full(3,-5.0), np.zeros(3), 1e-6), np.full(3,-5.0), np.zeros(3), 1e-6, 1.0)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        contact_impulse(np.eye(1), np.eye(2), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), np.zeros(2), np.zeros(2), np.zeros(2), 0.1, 1.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_contact_impulse(np.eye(1), np.eye(2), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), np.zeros(2), np.zeros(2), np.zeros(2), 0.1, 1.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
