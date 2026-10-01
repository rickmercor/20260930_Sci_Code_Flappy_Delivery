"""
Builds, for a given bond phase-field state on the families of step 1, the kinematically degraded moment matrix (shape tensor) of every point and the shape-function derivatives of both endpoints of every bond pair. Validates the method's separate kinematic degradation function, the role of the critical phase-field value s_c in it, and the way it enters the kinematic operators; the returned operators are the kinematic input of the bond deformation gradients (step 4) and of every internal-force evaluation (step 7). Deliberately excluded: stress, driving force, and evolution.

The moment matrix (shape tensor) of a point is the kernel-weighted second moment of its family bond vectors, and its inverse defines the nonlocal deformation gradient through the shape-function derivatives of the correspondence formulation. In this method damage does not reach the kinematic side through the energetic degradation of the stress: a separate function of the bond phase-field, in which the critical phase-field value s_c plays a specific role, weights the kinematic contribution of every bond so that the moment matrix stays well conditioned while bonds degrade. Which function is used, how s_c enters it, and how it appears in the moment matrix and in the shape-function derivatives must be taken from the source.

Returns
-------
dict: 'h' (P,), degraded moment matrices 'M' (N,3,3) in m^2, shape-function derivatives 'dphi_kn'/'dphi_nk' (P,3) in 1/m
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def kinematic_operators(fam: dict, s: np.ndarray, s_c: float) -> dict:
    """Kinematically degraded moment matrices and shape-function derivatives.

    Given the bond families of step 1 and one bond phase-field value per
    bond pair, applies the method's kinematic degradation - the separate
    function through which damage enters the kinematic side of the
    correspondence formulation, in which the critical phase-field value
    s_c plays the role the source assigns to it - and builds the moment
    matrix (shape tensor) of every point and the shape-function
    derivatives of both endpoints of every bond pair exactly as the source
    prescribes. Which function is used, and how it enters the moment
    matrix and the shape-function derivatives, must be taken from the
    source. These operators are consumed by the bond deformation
    gradients of step 4 and by every internal-force evaluation of step 7.

    Parameters
    ----------
    fam : dict
        Bond-family dictionary returned by build_families (step 1).
    s : np.ndarray
        (P,) array of bond phase-field values in [0, 1], one per bond pair,
        in the pair order of fam['pk'] / fam['pn'].
    s_c : float
        Critical phase-field value in (0, 1).

    Returns
    -------
    dict
        'h' : (P,) float array, kinematic degradation factor of every bond
            pair as the method defines it;
        'M' : (N, 3, 3) float array, degraded moment matrix of every point in m^2;
        'dphi_kn' : (P, 3) float array, shape-function derivative of endpoint
            pn[p] with respect to the family of endpoint pk[p], in 1/m;
        'dphi_nk' : (P, 3) float array, shape-function derivative of endpoint
            pk[p] with respect to the family of endpoint pn[p], in 1/m.

    Raises
    ------
    ValueError
        If s_c is not in (0, 1), or s does not hold one value in [0, 1] per
        bond pair.
    """
    return {}

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_kinematic_operators(fam, s, s_c):
    if not (0.0 < s_c < 1.0):
        raise ValueError("require 0 < s_c < 1")
    pk = fam['pk']; pn = fam['pn']; dX = fam['dX']; om = fam['omega']; V = fam['V']
    N = fam['X'].shape[0]
    s = np.asarray(s, dtype=float)
    if s.shape != pk.shape:
        raise ValueError("s must hold one phase-field value per bond pair")
    if np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("bond phase-field values must lie in [0, 1]")
    h = np.where(s <= s_c, 1.0, ((1.0 - s)/(1.0 - s_c))**2)
    w_h = om * h
    outer = dX[:, :, None]*dX[:, None, :]
    M = np.zeros((N, 3, 3))
    np.add.at(M, pk, (w_h*V)[:, None, None]*outer)
    np.add.at(M, pn, (w_h*V)[:, None, None]*outer)
    Minv = np.linalg.inv(M)
    dphi_kn = (w_h*V)[:, None]*np.einsum('pij,pj->pi', Minv[pk], dX)
    dphi_nk = (w_h*V)[:, None]*np.einsum('pij,pj->pi', Minv[pn], -dX)
    return {'h': h, 'M': M, 'dphi_kn': dphi_kn, 'dphi_nk': dphi_nk}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\ndef _flank_min(build, kin, s_c, s_notch):\n    fam = build(16, 10, 2, 1.0e-3, 2.015e-3)\n    s = np.where(fam['notch'], s_notch, 0.0)\n    M = kin(fam, s, s_c)['M']\n    flank = (fam['ijk'][:, 1] == 4) | (fam['ijk'][:, 1] == 5)\n    return float(np.min(np.linalg.eigvalsh(M[flank])[:, 0]))"
    return [
        {"name": "normal_fully_predamaged_notch",
         "setup": setup,
         "call": "_flank_min(build_families, kinematic_operators, 0.95, 1.0)",
         "gold_call": "_flank_min(_oracle_build_families, _oracle_kinematic_operators, 0.95, 1.0)"},
        {"name": "boundary_notch_below_critical_value",
         "setup": setup,
         "call": "_flank_min(build_families, kinematic_operators, 0.95, 0.9)",
         "gold_call": "_flank_min(_oracle_build_families, _oracle_kinematic_operators, 0.95, 0.9)"},
        {"name": "edge_notch_between_critical_value_and_one",
         "setup": setup,
         "call": "_flank_min(build_families, kinematic_operators, 0.95, 0.975)",
         "gold_call": "_flank_min(_oracle_build_families, _oracle_kinematic_operators, 0.95, 0.975)"},
        {"name": "edge_kinematic_gating_sum_of_degradation_factors",
         "setup": "import numpy as np\ndef _h_sum(build, kin, s_c, s_notch):\n    fam = build(16, 10, 2, 1.0e-3, 2.015e-3)\n    s = np.where(fam['notch'], s_notch, 0.0)\n    return float(np.sum(kin(fam, s, s_c)['h']))",
         "call": "_h_sum(build_families, kinematic_operators, 0.95, 0.975)",
         "gold_call": "_h_sum(_oracle_build_families, _oracle_kinematic_operators, 0.95, 0.975)"},
    ]
