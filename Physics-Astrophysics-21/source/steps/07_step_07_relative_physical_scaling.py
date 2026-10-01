"""
Enforce the source's relative distance from the relativistic cone boundary.

This stage accepts a stored cell after absolute physical scaling and applies
the source's conservative relative certification. The prescribed fraction is
relative to the cell's own admissible anchor. Axis cells use the same physical
scope as the preceding operations.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relative_physical_scaling(stored, radial_cell, fraction=0.1):
    """Return the stored states after relative physical certification.

    Parameters
    ----------
    stored : array_like, shape (3,3,4)
        Finite axis-compatible stored states after absolute certification,
        with an admissible conservative cell mean.
    radial_cell : int
        Radial index a in {0,1} on the same unit-width patch.
    fraction : float
        Finite fraction in [0,1) specifying relative certification strength.

    Returns
    -------
    ndarray, shape (3,3,4)
        Conservatively certified stored state. The input is not modified.
        Invalid input, inadmissible mean, or negative local density/cone
        margin raise ValueError.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_relative_physical_scaling(stored, radial_cell, fraction=0.1):
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('Relative fraction must lie in [0,1)')
    out,chi,anchor,qa = _s6_geometry_anchor(stored,radial_cell)
    local = _oracle_regular_local_state(out,radial_cell)
    radius = np.linalg.norm(local[...,:3],axis=-1)
    if np.any(local[...,0] < 0) or np.any(local[...,3]-radius < 0):
        raise ValueError('Input must satisfy absolute physical constraints')
    delta = fraction*qa/(anchor[3]+np.linalg.norm(anchor[:3]))
    minimum = np.min((1-delta)*local[...,3]-(1+delta)*radius)
    ga = (1-fraction)*qa
    theta = 1.0 if minimum >= 0 else ga/(ga-minimum)
    reference = chi[...,None]*anchor
    return reference+theta*(out-reference)

def _s7_absolute():
    return _oracle_absolute_physical_scaling(_s6_filtered()[0,0],0)

def _s7_near_cone():
    """An a=1 cell whose corner node sits close to the cone, so theta<1 there."""
    base = _oracle_absolute_physical_scaling(_s6_filtered()[1,0],1)
    _,chi,anchor,qa = _s6_geometry_anchor(base,1)
    out = chi[...,None]*anchor
    local = _oracle_regular_local_state(out,1)
    d,mr,mz,_ = local[0,0]
    out[0,0,3] = chi[0,0]*(np.sqrt(d*d+mr*mr+mz*mz)+0.02*qa)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'w = _s7_absolute(); w_gold = _s7_absolute()', 'call':'relative_physical_scaling(w,0)',
         'gold_call':'_oracle_relative_physical_scaling(w_gold,0)'},
        {'setup':'w = _s7_absolute(); w_gold = _s7_absolute()', 'call':'relative_physical_scaling(w,0,fraction=0)',
         'gold_call':'w_gold'},
        {'setup':'w = _s7_absolute(); w_gold = _s7_absolute()', 'call':'relative_physical_scaling(w,0,fraction=0.8)',
         'gold_call':'_oracle_relative_physical_scaling(w_gold,0,fraction=0.8)'},
        {'setup':'w = _s7_near_cone(); w_gold = _s7_near_cone()', 'call':'relative_physical_scaling(w,1)',
         'gold_call':'_oracle_relative_physical_scaling(w_gold,1)'},
        {'setup':'w = _s1_fixture(True)[1,1]; w_gold = _s1_fixture(True)[1,1]', 'call':'relative_physical_scaling(w,1)',
         'gold_call':'w_gold'},
        {'setup':'w = _s7_absolute()\ndef _s7_invalid():\n    try:\n        relative_physical_scaling(w,0,fraction=1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0',
         'call':'_s7_invalid()', 'gold_call':'1'},
    ]
