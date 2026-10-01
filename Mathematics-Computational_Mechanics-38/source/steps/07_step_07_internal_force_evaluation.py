"""
Performs exactly one internal-force evaluation of the full method for a given displacement field and bond state on the families of step 1, by chaining steps 3 to 6 in the order the source prescribes and then assembling the method's two-term force state with its non-uniform stress contribution and the truncation-corrected bond weights. Returns the internal force density of every point together with the updated history and bond phase-field, which the dynamics of step 8 feed back into the next evaluation. Validates the within-evaluation update ordering and the force-state assembly. Deliberately excluded: time integration.

One internal-force evaluation couples kinematics, constitutive response, damage evolution and force assembly. In the bond-associated correspondence framework the force state of a bond has two contributions - a direct projection of the (degraded) bond stress and a correction that distributes the non-uniform part of the stress through the shape-function derivatives - and the internal force density of a point is the family sum of the difference of the two oriented force states, with the truncation-corrected bond weights of step 1 in every family sum. No-fail bonds never accumulate damage but transmit force normally. The exact assembly and the order in which kinematic quantities, driving force, history, phase-field and degraded stress are updated within the evaluation must be taken from the source.

Returns
-------
dict: 'B' (N,3) internal force density in N/m^3, updated 'calY' (P,) in J/m^3 and 's' (P,) bond phase-field
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def internal_force_evaluation(fam: dict, u: np.ndarray, calY: np.ndarray,
                              s_prev: np.ndarray, E: float, nu: float, Gc: float,
                              s_c: float, c0: float) -> dict:
    """One internal-force evaluation of the method with its damage update.

    Performs exactly one internal-force evaluation of the method for a
    given displacement field and bond state by chaining the earlier steps:
    the kinematic operators of step 3, the bond-associated deformation
    gradients of step 4, the undamaged bond stress and crack driving force
    of step 5 (no-fail bond pairs never accumulate damage: their driving
    force is treated as zero), the history update and the closed-form
    phase-field of step 6, and finally the method's force-state assembly
    (the two-term bond force state with its non-uniform stress
    contribution) with the truncation-corrected bond weights of step 1 in
    every family sum, giving the internal force density of every point.
    The order in which kinematic quantities, driving force, history,
    phase-field and degraded stress are updated within the evaluation -
    in particular which of them use the phase-field carried over from the
    previous evaluation and which use the updated one - must be taken from
    the source. Called once per force evaluation by the dynamics of
    step 8.

    Parameters
    ----------
    fam : dict
        Bond-family dictionary returned by build_families (step 1).
    u : np.ndarray
        (N, 3) displacement of every point in m.
    calY : np.ndarray
        (P,) bond damage-history variable before this evaluation, in J/m^3.
    s_prev : np.ndarray
        (P,) bond phase-field carried over from the previous evaluation.
    E : float
        Young's modulus in Pa (> 0).
    nu : float
        Poisson's ratio in (-1, 0.5).
    Gc : float
        Critical energy release rate in J/m^2 (> 0).
    s_c : float
        Critical phase-field value in (0, 1).
    c0 : float
        Griffith-consistency normalization constant of the kernel (output
        of step 2), in (0, 1).

    Returns
    -------
    dict
        'B' : (N, 3) float array, internal force density of every point in N/m^3;
        'calY' : (P,) float array, updated bond damage-history variable in J/m^3;
        's' : (P,) float array, updated bond phase-field in [0, 1].

    Raises
    ------
    ValueError
        If a material, fracture or normalization parameter is invalid, or
        u, calY, s_prev have inconsistent shapes.
    """
    return {}

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_internal_force_evaluation(fam, u, calY, s_prev, E, nu, Gc, s_c, c0):
    if not (E > 0.0 and -1.0 < nu < 0.5 and Gc > 0.0 and 0.0 < s_c < 1.0 and 0.0 < c0 < 1.0):
        raise ValueError("invalid material, fracture or normalization parameters")
    pk = fam['pk']; pn = fam['pn']; dX = fam['dX']; r = fam['r']
    omb = fam['omega_b']; V = fam['V']; nofail = fam['nofail']
    N = fam['X'].shape[0]
    u = np.asarray(u, dtype=float)
    calY = np.asarray(calY, dtype=float)
    s_prev = np.asarray(s_prev, dtype=float)
    if u.shape != (N, 3) or calY.shape != pk.shape or s_prev.shape != pk.shape:
        raise ValueError("u must be (N, 3); calY and s_prev must hold one value per bond pair")
    # (1) kinematic operators from the phase-field of the PREVIOUS evaluation (step 3)
    ops = _oracle_kinematic_operators(fam, s_prev, s_c)
    # (2) bond-associated deformation gradients (step 4)
    Ft = _oracle_bond_deformation_gradients(fam, ops, u)
    # (3) undamaged bond stress and stress-based crack driving force (step 5)
    resp = _oracle_bond_stress_and_driving_force(Ft, E, nu)
    P0 = resp['P0']
    Y = np.where(nofail, 0.0, resp['Y'])
    # (4) history update and closed-form phase-field (step 6)
    calY_new = np.maximum(calY, Y)
    s = _oracle_bond_phase_field(calY_new, Gc, fam['delta'], c0)
    # (5) degraded stress enters the two-term force state
    Pt = ((1.0 - s)**2)[:, None, None]*P0
    nhat = dX/r[:, None]
    proj = np.eye(3) - nhat[:, :, None]*nhat[:, None, :]
    ZC = (omb*V)[:, None, None]*(Pt @ proj)
    Z = np.zeros((N, 3, 3))
    np.add.at(Z, pk, ZC)
    np.add.at(Z, pn, ZC)
    dphi_kn = ops['dphi_kn']; dphi_nk = ops['dphi_nk']
    t_kn = omb[:, None]*np.einsum('pij,pj->pi', Pt, nhat) + np.einsum('pij,pj->pi', Z[pk], dphi_kn)/V
    t_nk = omb[:, None]*np.einsum('pij,pj->pi', Pt, -nhat) + np.einsum('pij,pj->pi', Z[pn], dphi_nk)/V
    B = np.zeros((N, 3))
    np.add.at(B, pk, (t_kn - t_nk)*V)
    np.add.at(B, pn, (t_nk - t_kn)*V)
    return {'B': B, 'calY': calY_new, 's': s}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\ndef _force_y(build, pf, force, c0, nx, ny, nz, eps, i0, j0, k0):\n    fam = build(nx, ny, nz, 1.0e-3, 2.015e-3)\n    u = np.zeros((len(fam['X']), 3))\n    u[:, 1] = eps*fam['X'][:, 1]\n    calY = np.where(fam['notch'], 1.0e30, 0.0)\n    s = pf(calY, 120.0, 2.015e-3, c0)\n    out = force(fam, u, calY, s, 3.2e10, 0.25, 120.0, 0.95, c0)\n    ijk = fam['ijk']\n    a = int(np.flatnonzero((ijk[:, 0] == i0) & (ijk[:, 1] == j0) & (ijk[:, 2] == k0))[0])\n    return float(out['B'][a, 1])"
    return [
        {"name": "normal_tension_at_notch_tip",
         "setup": setup,
         "call": "_force_y(build_families, bond_phase_field, internal_force_evaluation, pfpd_normalization_constant('cubic'), 16, 10, 2, 1.0e-3, 8, 4, 0)",
         "gold_call": "_force_y(_oracle_build_families, _oracle_bond_phase_field, _oracle_internal_force_evaluation, _oracle_pfpd_normalization_constant('cubic'), 16, 10, 2, 1.0e-3, 8, 4, 0)"},
        {"name": "boundary_smaller_grid_larger_strain",
         "setup": setup,
         "call": "_force_y(build_families, bond_phase_field, internal_force_evaluation, pfpd_normalization_constant('cubic'), 8, 6, 2, 2.0e-3, 4, 3, 0)",
         "gold_call": "_force_y(_oracle_build_families, _oracle_bond_phase_field, _oracle_internal_force_evaluation, _oracle_pfpd_normalization_constant('cubic'), 8, 6, 2, 2.0e-3, 4, 3, 0)"},
        {"name": "edge_axial_compression_no_damage_growth",
         "setup": setup,
         "call": "_force_y(build_families, bond_phase_field, internal_force_evaluation, pfpd_normalization_constant('cubic'), 16, 10, 2, -1.0e-3, 8, 4, 0)",
         "gold_call": "_force_y(_oracle_build_families, _oracle_bond_phase_field, _oracle_internal_force_evaluation, _oracle_pfpd_normalization_constant('cubic'), 16, 10, 2, -1.0e-3, 8, 4, 0)"},
        {"name": "edge_updated_phase_field_count_after_one_evaluation",
         "setup": "import numpy as np\ndef _grown(build, pf, force, c0, eps):\n    fam = build(16, 10, 2, 1.0e-3, 2.015e-3)\n    u = np.zeros((len(fam['X']), 3))\n    u[:, 1] = eps*fam['X'][:, 1]\n    calY = np.where(fam['notch'], 1.0e30, 0.0)\n    s = pf(calY, 120.0, 2.015e-3, c0)\n    out = force(fam, u, calY, s, 3.2e10, 0.25, 120.0, 0.95, c0)\n    return float(np.sum(out['s'][~fam['notch']]))",
         "call": "_grown(build_families, bond_phase_field, internal_force_evaluation, pfpd_normalization_constant('cubic'), 1.0e-3)",
         "gold_call": "_grown(_oracle_build_families, _oracle_bond_phase_field, _oracle_internal_force_evaluation, _oracle_pfpd_normalization_constant('cubic'), 1.0e-3)"},
    ]
