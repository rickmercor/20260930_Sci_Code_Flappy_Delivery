"""
Decide which unilateral constraints must carry an impulse over the coming step. The source fixes both the configuration this decision is taken on and whether the gap test is inclusive; follow it exactly.

Unilateral contact is a set-valued condition: a constraint carries an impulse only while it is closed. Which configuration the closure test reads is a choice the time-integration scheme makes, and the source states its choice.

Returns
-------
ndarray of shape (n_constraints,), 1.0 where the constraint is active and 0.0 otherwise.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def active_contact_set(u, v, a, dt, contact_rows):
    """Decide which unilateral constraints must carry an impulse over the coming step. The source fixes both the configuration this decision is taken on and whether the gap test is inclusive; follow it exactly.

    Returns
    -------
    ndarray of shape (n_constraints,), 1.0 where the constraint is active and 0.0 otherwise.

    Raises
    ------
    ValueError: if contact_rows has a column count that does not match the number of degrees of freedom.
    """
    return np.zeros(1)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_active_contact_set(u, v, a, dt, contact_rows):
    H = np.atleast_2d(np.asarray(contact_rows, dtype=float))
    if H.shape[1] != np.asarray(u, dtype=float).shape[0]:
        raise ValueError("contact_rows columns must match the number of degrees of freedom")
    # CONVENTION (paper, eq. 39-40): the set is decided on the SMOOTH PREDICTION at
    # t_{n+1}, not on the gap at t_n; and the test is NON-STRICT.
    u_pred = _oracle_smooth_predictor(u, v, a, dt)
    gap = H @ u_pred
    return (gap <= 0.0).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "active_contact_set(np.array([0.5,1.0]), np.array([-3.0,-3.0]), np.zeros(2), 0.1, np.array([[1.0,0.0]]))",
         "gold_call": "_oracle_active_contact_set(np.array([0.5,1.0]), np.array([-3.0,-3.0]), np.zeros(2), 0.1, np.array([[1.0,0.0]]))"},   # normal
        {"setup": "import numpy as np",
         "call": "active_contact_set(np.array([0.0,1.0]), np.zeros(2), np.zeros(2), 0.1, np.array([[1.0,0.0]]))",
         "gold_call": "_oracle_active_contact_set(np.array([0.0,1.0]), np.zeros(2), np.zeros(2), 0.1, np.array([[1.0,0.0]]))"},   # boundary
        {"setup": "import numpy as np",
         "call": "active_contact_set(np.array([1.0,1.0,1.0]), np.array([-1.0,-1.0,-1.0]), np.zeros(3), 0.5, np.array([[1.0,0.0,0.0],[0.0,0.0,1.0]]))",
         "gold_call": "_oracle_active_contact_set(np.array([1.0,1.0,1.0]), np.array([-1.0,-1.0,-1.0]), np.zeros(3), 0.5, np.array([[1.0,0.0,0.0],[0.0,0.0,1.0]]))"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        active_contact_set(np.zeros(3), np.zeros(3), np.zeros(3), 0.1, np.array([[1.0,0.0]]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_active_contact_set(np.zeros(3), np.zeros(3), np.zeros(3), 0.1, np.array([[1.0,0.0]]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
