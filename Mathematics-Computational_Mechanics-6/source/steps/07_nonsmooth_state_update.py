"""
Given the smooth state and the impulse, close the step and return the end-of-step configuration, velocity and acceleration. The weight the impulsive velocity carries into the configuration is set by the source and is not the same as its weight into the velocity.

The impulsive part of the motion is added on top of the smooth part. Velocity is the variable the constraint acts on directly; the configuration receives it through the scheme's own update rule.

Returns
-------
ndarray of shape (3, n_dof): row 0 configuration, row 1 velocity, row 2 acceleration.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def nonsmooth_state_update(stiffness, lumped_mass, contact_rows, active, smooth_state, v, a, dt, impulse):
    """Given the smooth state and the impulse, close the step and return the end-of-step configuration, velocity and acceleration. The weight the impulsive velocity carries into the configuration is set by the source and is not the same as its weight into the velocity.

    Returns
    -------
    ndarray of shape (3, n_dof): row 0 configuration, row 1 velocity, row 2 acceleration.

    Raises
    ------
    ValueError: if dt is not positive, any lumped_mass entry is not positive, or smooth_state and v do not share a shape.
    """
    return np.zeros((3, 1))  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_nonsmooth_state_update(stiffness, lumped_mass, contact_rows, active,
                                   predicted, v, a, dt, impulse):
    K = np.asarray(stiffness, dtype=float)
    Md = np.asarray(lumped_mass, dtype=float)
    H = np.atleast_2d(np.asarray(contact_rows, dtype=float))
    act = np.asarray(active, dtype=float) > 0.5
    v = np.asarray(v, dtype=float); a = np.asarray(a, dtype=float)
    p = np.asarray(impulse, dtype=float).ravel()
    if dt <= 0:
        raise ValueError("dt must be positive")
    if np.any(Md <= 0):
        raise ValueError("lumped_mass entries must be positive")
    u_pred = np.asarray(predicted, dtype=float)
    if u_pred.shape != v.shape:
        raise ValueError("predicted and v must have the same shape")
    dt = float(dt)
    Minv = 1.0 / Md
    if act.any() and p.size:
        HA = H[act.ravel()]
        v_hat = (Minv[:, None] * HA.T @ p).ravel()
    else:
        v_hat = np.zeros_like(v)
    # CONVENTION (paper, eq. 40): the impulsive velocity enters the displacement
    # through HALF a step, not a full step.
    u_new = u_pred + 0.5 * dt * v_hat
    a_new = Minv * (-(K @ u_new))
    v_new = v + 0.5 * dt * (a + a_new) + v_hat
    out = np.zeros((3, u_new.shape[0]), dtype=float)
    out[0] = u_new; out[1] = v_new; out[2] = a_new
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "nonsmooth_state_update(np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), smooth_predictor(np.array([-0.01,0.5]), np.array([-2.0,-2.0]), np.zeros(2), 0.05), np.array([-2.0,-2.0]), np.zeros(2), 0.05, np.array([3.0]))",
         "gold_call": "_oracle_nonsmooth_state_update(np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([1.0]), smooth_predictor(np.array([-0.01,0.5]), np.array([-2.0,-2.0]), np.zeros(2), 0.05), np.array([-2.0,-2.0]), np.zeros(2), 0.05, np.array([3.0]))"},   # normal
        {"setup": "import numpy as np",
         "call": "nonsmooth_state_update(np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([0.0]), smooth_predictor(np.array([0.5,0.5]), np.array([-1.0,-1.0]), np.zeros(2), 0.05), np.array([-1.0,-1.0]), np.zeros(2), 0.05, np.zeros(0))",
         "gold_call": "_oracle_nonsmooth_state_update(np.array([[2.0,-1.0],[-1.0,2.0]]), np.array([1.0,1.0]), np.array([[1.0,0.0]]), np.array([0.0]), smooth_predictor(np.array([0.5,0.5]), np.array([-1.0,-1.0]), np.zeros(2), 0.05), np.array([-1.0,-1.0]), np.zeros(2), 0.05, np.zeros(0))"},   # boundary
        {"setup": "import numpy as np",
         "call": "nonsmooth_state_update(np.eye(3)*1e5, np.array([2.0,1.0,2.0]), np.array([[1.0,0.0,0.0]]), np.array([1.0]), smooth_predictor(np.array([-1e-6,0.0,1.0]), np.full(3,-5.0), np.zeros(3), 1e-6), np.full(3,-5.0), np.zeros(3), 1e-6, np.array([12.0]))",
         "gold_call": "_oracle_nonsmooth_state_update(np.eye(3)*1e5, np.array([2.0,1.0,2.0]), np.array([[1.0,0.0,0.0]]), np.array([1.0]), smooth_predictor(np.array([-1e-6,0.0,1.0]), np.full(3,-5.0), np.zeros(3), 1e-6), np.full(3,-5.0), np.zeros(3), 1e-6, np.array([12.0]))"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        nonsmooth_state_update(np.eye(2), np.array([1.0,-1.0]), np.array([[1.0,0.0]]), np.array([1.0]), np.zeros(2), np.zeros(2), np.zeros(2), 0.1, np.array([1.0]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_nonsmooth_state_update(np.eye(2), np.array([1.0,-1.0]), np.array([[1.0,0.0]]), np.array([1.0]), np.zeros(2), np.zeros(2), np.zeros(2), 0.1, np.array([1.0]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
