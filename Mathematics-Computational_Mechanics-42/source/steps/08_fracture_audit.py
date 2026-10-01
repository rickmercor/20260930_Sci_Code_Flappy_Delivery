"""
Run the full fracture deformation audit over the source's four model variants, in the order S-Lin-Lin, S-Lin-BB, C-Coul-Lin, C-Coul-BB, on the declared fixed fracture configuration with all prescribed jumps scaled by load_scale. For each variant compute the normal contact traction from the prescribed normal displacement jump under that variant's normal relation and model family, then the tangential traction: for the two spring variants the tangential relation of Eq. 16 applies, whereas for the two contact variants the tangential traction follows from the source's radial-return map starting from zero traction with the declared numerical parameter. Then classify the fracture points and advance the tangential jump by one step of the source's time integrator, taking the new tangential jump to be the prescribed tangential jump plus the tangential traction divided by the tangential stiffness. Also advance the normal jump by the same integrator, taking the new normal jump to be the prescribed normal jump plus the normal traction divided by the normal stiffness, and build the absorbing boundary coefficient matrix for the declared material and the outward normal (1, 0); at each fracture point apply that matrix to the two-component velocity (updated normal velocity jump, updated tangential velocity jump) and sum the magnitudes of the resulting boundary tractions. Report the row [sum of normal tractions, sum of tangential traction magnitudes, sum of fracture openings, largest slip tendency, number of open points, number of sticking points, number of sliding points, norm of the updated tangential velocity jump, summed absorbing-boundary traction magnitude, sum of the elastic normal closure g_n(q_n)]. Assemble by calling the earlier sub-problem functions; every earlier step must be reached.

Contrasting the four deformation models on one prescribed state is what exposes their differences: the spring models transmit tension across an opening fracture and can exceed the friction bound, whereas the contact models open instead of pulling and cap the tangential traction. The time integrator is advanced separately in the normal and the tangential direction, and both advances start from the same previous state: the prescribed previous tangential velocity and acceleration vectors also serve as the previous normal velocity and acceleration, so no separate normal history is introduced.

Returns
-------
return (4, 10) float64: one audit row per fracture deformation model
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fracture_audit(load_scale):
    """load_scale: positive multiplier on the declared prescribed displacement jumps.
    Builds the declared fracture configuration and returns a float64 array (4, 10) whose
    rows are the audit of the four model variants S-Lin-Lin, S-Lin-BB, C-Coul-Lin and
    C-Coul-BB, with columns [sum q_n, sum |q_tau|, sum opening, max slip tendency,
    n_open, n_sticking, n_sliding, |v_tau|, summed absorbing-boundary traction,
    sum of elastic normal closure]. Assembled by calling the earlier sub-problem functions.
    Raises ValueError on a non-positive or nonfinite load_scale."""
    return np.zeros((4, 10))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 8 (final orchestrator): fracture deformation model audit."""

import numpy as np

_KN, _KT, _DUMAX, _F = 2.0e11, 2.0e11, 5.0e-5, 1.0
_C = 3.1e6
_RHO, _LAM, _MU = 2600.0, 4.0e9, 4.0e9
_NBND = np.array([1.0, 0.0])
_DT, _BETA, _GAMMA = 1.0e-7, 0.25, 0.5
_JN = np.array([1.8e-5, -1.2e-5, -2.4e-5, 0.6e-5, -3.0e-5, -0.9e-5])
_JT = np.array([0.8e-5, -1.1e-5, 1.6e-5, -0.4e-5, 2.2e-5, 0.5e-5])
_DVT = np.array([0.8, -1.1, 1.6, -0.4, 2.2, 0.5])
_V0 = np.array([0.2, -0.4, 0.6, 0.1, -0.3, 0.5])
_A0 = np.array([1.0e6, 0.0, -1.0e6, 2.0e6, 5.0e5, -5.0e5])
_VARIANTS = (("Lin", False), ("BB", False), ("Lin", True), ("BB", True))


def _oracle_fracture_audit(load_scale):
    if isinstance(load_scale, bool) or not np.isfinite(load_scale) or load_scale <= 0:
        raise ValueError("load_scale must be positive and finite")
    ls = float(load_scale)
    jn = ls * _JN
    jt = ls * _JT
    dvt = ls * _DVT
    D = _oracle_absorbing_matrix(_RHO, _LAM, _MU, _NBND)
    rows = []
    for model, contact in _VARIANTS:
        qn = _oracle_normal_traction(jn, _KN, _DUMAX, model, contact)
        b = _oracle_friction_bound(qn, _F)
        if contact:
            qt = _oracle_coulomb_return(np.zeros_like(jt), dvt, b, _C)
        else:
            qt = _KT * jt
        st = _oracle_contact_state(jn, qn, qt, _KN, _DUMAX, model, _F)
        delta, s, state = st[0], st[1], st[2]
        max_s = float(np.max(s)) if s.size else 0.0
        vt = _oracle_newmark_update(jt + qt / _KT, jt, _V0, _A0, _DT, _BETA, _GAMMA)[0]
        vn = _oracle_newmark_update(jn + qn / _KN, jn, _V0, _A0, _DT, _BETA, _GAMMA)[0]
        tabs = 0.0
        for i in range(vt.size):
            tabs += float(np.linalg.norm(D @ np.array([vn[i], vt[i]])))
        gn = _oracle_normal_stiffness(qn, _KN, _DUMAX, model)
        rows.append([float(np.sum(qn)), float(np.sum(np.abs(qt))), float(np.sum(delta)),
                     max_s, float(np.sum(state == 0.0)), float(np.sum(state == 1.0)),
                     float(np.sum(state == 2.0)), float(np.linalg.norm(vt)),
                     tabs, float(np.sum(gn))])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'load_scale = 1.0', "call": 'fracture_audit(load_scale)', "gold_call": '_oracle_fracture_audit(load_scale)', "tol": 1e-06},
        {"setup": 'load_scale = 0.8', "call": 'fracture_audit(load_scale)', "gold_call": '_oracle_fracture_audit(load_scale)', "tol": 1e-06},
        {"setup": 'load_scale = 1.25', "call": 'fracture_audit(load_scale)', "gold_call": '_oracle_fracture_audit(load_scale)', "tol": 1e-06},
    ]
