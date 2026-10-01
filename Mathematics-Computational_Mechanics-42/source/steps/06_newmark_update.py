"""
Advance velocity and acceleration by one step of the time integrator the source uses for the elastic wave equation and for the tangential velocity jump (the paper's Eqs. 20-21), given the newly computed displacement and the previous displacement, velocity and acceleration. The source uses the standard parameter choice stated in the paper. Recover both update formulas exactly; they are the Newmark relations written with the new displacement as the known quantity, not a simple finite difference of the displacement.

An implicit time integrator is what makes the coupled wave-fracture system solvable at each step: the displacement is the unknown solved for, and the velocity and acceleration are recovered from it afterwards.

Returns
-------
return (2, N) float64: row 0 the new velocity, row 1 the new acceleration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def newmark_update(u_new, u_old, v_old, a_old, dt, beta, gamma):
    """u_new, u_old, v_old, a_old: displacement, previous displacement, velocity and
    acceleration; dt: time step; beta, gamma: Newmark parameters. Returns a float64 array
    (2, N) with the updated velocity and acceleration (paper Eqs. 20-21). Raises
    ValueError if dt <= 0 or beta <= 0."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 6: Newmark velocity and acceleration update (Eqs. 20-21)."""

import numpy as np


def _oracle_newmark_update(u_new, u_old, v_old, a_old, dt, beta, gamma):
    if not np.isfinite(dt) or dt <= 0 or beta <= 0:
        raise ValueError("dt and beta must be positive")
    un = np.asarray(u_new, dtype=np.float64)
    uo = np.asarray(u_old, dtype=np.float64)
    vo = np.asarray(v_old, dtype=np.float64)
    ao = np.asarray(a_old, dtype=np.float64)
    v_new = ((1.0 - gamma / beta) * vo
             + dt * (1.0 - gamma / (2.0 * beta)) * ao
             + (gamma / (beta * dt)) * (un - uo))
    a_new = (un - uo - dt * vo - (1.0 - 2.0 * beta) * (dt ** 2 / 2.0) * ao) / (beta * dt ** 2)
    return np.vstack([v_new, a_new])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nu_new=_n.array([2.0e-7,-1.0e-7,0.5e-7]);u_old=_n.array([0.0,0.0,0.0]);v_old=_n.array([1.0,0.5,-0.25]);a_old=_n.array([0.0,1.0e6,-2.0e6]);dt=1.0e-7;beta=0.25;gamma=0.5', "call": 'newmark_update(u_new, u_old, v_old, a_old, dt, beta, gamma)', "gold_call": '_oracle_newmark_update(u_new, u_old, v_old, a_old, dt, beta, gamma)', "tol": 1e-06},
        {"setup": 'import numpy as _n\nu_new=_n.array([1.1e-6,3.0e-7]);u_old=_n.array([2.0e-7,-1.0e-7]);v_old=_n.array([-0.75,2.0]);a_old=_n.array([5.0e5,-3.0e5]);dt=2.5e-8;beta=0.25;gamma=0.5', "call": 'newmark_update(u_new, u_old, v_old, a_old, dt, beta, gamma)', "gold_call": '_oracle_newmark_update(u_new, u_old, v_old, a_old, dt, beta, gamma)', "tol": 1e-06},
        {"setup": 'import numpy as _n\nu_new=_n.array([4.0e-7,-2.2e-7,1.3e-7,0.0]);u_old=_n.array([1.0e-7,1.0e-7,-1.0e-7,0.0]);v_old=_n.array([0.2,-0.4,0.6,0.0]);a_old=_n.array([1.0e6,0.0,-1.0e6,2.0e6]);dt=5.0e-8;beta=0.3;gamma=0.6', "call": 'newmark_update(u_new, u_old, v_old, a_old, dt, beta, gamma)', "gold_call": '_oracle_newmark_update(u_new, u_old, v_old, a_old, dt, beta, gamma)', "tol": 1e-06},
    ]
