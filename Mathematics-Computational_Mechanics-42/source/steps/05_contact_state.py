"""
Classify the mechanical state of each point of a fracture in the source's contact model. Row 0 is the fracture opening as the source defines it in Section 2.3.3, that is the part of the normal displacement jump not accounted for by the elastic normal deformation of Eq. 15. Row 1 is the source's slip tendency, the ratio of the tangential traction magnitude to the magnitude of the friction bound; report it as 0.0 wherever the friction bound vanishes, since row 2 already records that the point is open. Row 2 encodes the source's three contact states as 0 for open, 1 for sticking and 2 for sliding, using the source's own criteria: a point with no contact traction is open, a point in contact whose slip tendency is below one is sticking, and a point in contact whose slip tendency has reached one is sliding. Recover the definitions of the opening and the slip tendency from the paper.

The three contact states are what a fracture contact model has to resolve at every point and every time step, and they are what distinguishes a contact model from a spring model, which has no notion of opening or of a friction bound.

Returns
-------
return (3, N) float64: row 0 the fracture opening, row 1 the slip tendency, row 2 the state code
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def contact_state(jump_n, q_n, q_tau, K_n, du_max, model, F):
    """jump_n: (N,) normal displacement jump; q_n: (N,) normal contact traction; q_tau:
    (N,) tangential traction; K_n: normal fracture stiffness; du_max: maximum allowed
    closure; model: 'Lin' or 'BB'; F: coefficient of friction. Returns a float64 array
    (3, N): the fracture opening, the slip tendency (0.0 where the friction bound
    vanishes) and the state code (0 open, 1 sticking, 2 sliding)."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 5: fracture opening, slip tendency and contact state (Section 2.3.3, Eq. 12)."""

import numpy as np


def _gn(q, K_n, du_max, model):
    q = np.asarray(q, dtype=np.float64)
    if model == "Lin":
        return q / K_n
    if model == "BB":
        den = K_n - q / du_max
        if np.any(den <= 0):
            raise ValueError("Barton-Bandis denominator must stay positive")
        return q / den
    raise ValueError("model must be 'Lin' or 'BB'")


def _oracle_contact_state(jump_n, q_n, q_tau, K_n, du_max, model, F):
    jn = np.asarray(jump_n, dtype=np.float64)
    qn = np.asarray(q_n, dtype=np.float64)
    qt = np.asarray(q_tau, dtype=np.float64)
    if jn.shape != qn.shape or jn.shape != qt.shape or jn.size == 0:
        raise ValueError("jump_n, q_n and q_tau must share one non-empty shape")
    if not (np.all(np.isfinite(jn)) and np.all(np.isfinite(qn)) and np.all(np.isfinite(qt))):
        raise ValueError("non-finite input")
    if model not in ("Lin", "BB"):
        raise ValueError("model must be 'Lin' or 'BB'")
    if not np.isfinite(K_n) or K_n <= 0.0 or not np.isfinite(du_max) or du_max <= 0.0:
        raise ValueError("K_n and du_max must be positive and finite")
    if not np.isfinite(F) or F < 0.0:
        raise ValueError("F must be finite and non-negative")
    delta = jn - _gn(qn, K_n, du_max, model)
    b = -F * qn
    s = np.zeros(jn.shape)
    nz = np.abs(b) > 0
    s[nz] = np.abs(qt[nz]) / np.abs(b[nz])
    state = np.ones(jn.shape)
    state[qn == 0.0] = 0.0
    contact = qn != 0.0
    state[contact & (s >= 1.0)] = 2.0
    return np.vstack([delta, s, state])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as _n\njump_n=_n.array([1.8e-5,-1.2e-5,-2.4e-5,-3.0e-5]);q_n=_n.array([0.0,-2.4e6,-4.8e6,-6.0e6]);q_tau=_n.array([0.0,1.0e6,4.8e6,2.0e6]);K_n=2.0e11;du_max=5.0e-5;model='Lin';F=1.0", "call": 'contact_state(jump_n, q_n, q_tau, K_n, du_max, model, F)', "gold_call": '_oracle_contact_state(jump_n, q_n, q_tau, K_n, du_max, model, F)', "tol": 1e-09},
        {"setup": "import numpy as _n\njump_n=_n.array([-0.9e-5,2.2e-5,-3.6e-5,-1.5e-5]);q_n=_n.array([-1.8e6,0.0,-7.2e6,-3.0e6]);q_tau=_n.array([1.8e6,0.0,3.0e6,0.5e6]);K_n=2.0e11;du_max=5.0e-5;model='Lin';F=1.0", "call": 'contact_state(jump_n, q_n, q_tau, K_n, du_max, model, F)', "gold_call": '_oracle_contact_state(jump_n, q_n, q_tau, K_n, du_max, model, F)', "tol": 1e-09},
        {"setup": "import numpy as _n\njump_n=_n.array([-1.0e-5,-2.0e-5,-0.5e-5]);q_n=_n.array([-2.5e6,-5.5e6,-1.1e6]);q_tau=_n.array([1.2e6,3.3e6,1.1e6]);K_n=1.5e11;du_max=8.0e-5;model='BB';F=0.6", "call": 'contact_state(jump_n, q_n, q_tau, K_n, du_max, model, F)', "gold_call": '_oracle_contact_state(jump_n, q_n, q_tau, K_n, du_max, model, F)', "tol": 1e-09},
    ]
