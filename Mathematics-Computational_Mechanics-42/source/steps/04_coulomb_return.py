"""
Apply the source's radial-return map to the tangential fracture traction (the inner projection of the paper's Eq. 13, the Alart-Curnier reformulation of the Coulomb conditions of Eq. 11). A trial tangential traction is formed from the current tangential traction and the tangential velocity jump scaled by the numerical parameter c, and is then scaled back onto the friction cone when it exceeds the bound and left untouched when it does not. The bound is clipped at zero before the comparison so that an open fracture returns no tangential traction. Recover the exact form of the map from the paper. Two-dimensional setting, so the tangential direction has a single component and the norm is the absolute value.

Coulomb friction is a set-valued law: stick while the tangential traction is inside the friction cone, slip on its boundary. The radial-return map turns that inequality statement into a single equation that a Newton solver can handle.

Returns
-------
return array like q_tau (float64): the returned tangential traction
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def coulomb_return(q_tau, jump_vel_tau, b, c):
    """q_tau: current tangential traction; jump_vel_tau: tangential velocity jump; b:
    Coulomb friction bound; c: positive numerical parameter of the return map. Returns a
    float64 array shaped like q_tau with the tangential traction after the source's
    radial-return projection (paper Eq. 13). Raises ValueError if c <= 0."""
    return np.zeros_like(np.asarray(q_tau, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 4: radial-return map for the tangential traction (Eq. 13)."""

import numpy as np


def _oracle_coulomb_return(q_tau, jump_vel_tau, b, c):
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be positive and finite")
    qt = np.asarray(q_tau, dtype=np.float64)
    dv = np.asarray(jump_vel_tau, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    trial = qt + c * dv
    nrm = np.abs(trial)
    cap = np.maximum(bb, 0.0)
    fac = np.ones_like(trial)
    nz = nrm > 0
    fac[nz] = np.minimum(1.0, cap[nz] / nrm[nz])
    return fac * trial

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nq_tau=_n.array([0.0,0.0,0.0,0.0]);jump_vel_tau=_n.array([0.8,-1.1,1.6,-0.4]);b=_n.array([4.0e6,2.0e6,0.0,1.0e6]);c=1.0e6', "call": 'coulomb_return(q_tau, jump_vel_tau, b, c)', "gold_call": '_oracle_coulomb_return(q_tau, jump_vel_tau, b, c)', "tol": 1e-06},
        {"setup": 'import numpy as _n\nq_tau=_n.array([1.0e6,-2.0e6,0.5e6]);jump_vel_tau=_n.array([2.5,0.3,-4.0]);b=_n.array([3.0e6,2.5e6,1.2e6]);c=1.0e6', "call": 'coulomb_return(q_tau, jump_vel_tau, b, c)', "gold_call": '_oracle_coulomb_return(q_tau, jump_vel_tau, b, c)', "tol": 1e-06},
        {"setup": 'import numpy as _n\nq_tau=_n.array([0.0,1.5e6,-0.9e6,0.0]);jump_vel_tau=_n.array([0.0,-3.2,1.1,0.6]);b=_n.array([2.0e6,0.0,5.0e6,-1.0e6]);c=5.0e5', "call": 'coulomb_return(q_tau, jump_vel_tau, b, c)', "gold_call": '_oracle_coulomb_return(q_tau, jump_vel_tau, b, c)', "tol": 1e-06},
    ]
