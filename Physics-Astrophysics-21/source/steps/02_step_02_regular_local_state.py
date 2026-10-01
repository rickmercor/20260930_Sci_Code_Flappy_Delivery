"""
Recover the finite local orthonormal state associated with stored cell data.

A cell K_ab has Q2 nodes at local coordinates (0,1/2,1), reduction weight r,
and stored state W=r U. On a zero-weight face the requested output includes
the regular discrete local state used by the source's stage stabilization.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def regular_local_state(stored, radial_cell):
    """Return the regular local state on one cell, including any axis trace.

    Parameters
    ----------
    stored : array_like, shape (3,3,4)
        Finite stored states in radial-node, axial-node, component order.
        Axis-face stored values must already be zero.
    radial_cell : int
        Radial index a, either 0 or 1, for K_ab in the unit-width patch.

    Returns
    -------
    ndarray, shape (3,3,4)
        Finite local states in component order (D,m_r,m_z,E). The input is not
        modified. Invalid data or a nonzero stored axis face raise ValueError.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_regular_local_state(stored, radial_cell):
    w = np.asarray(stored,dtype=float)
    if w.shape != (3,3,4) or not np.isfinite(w).all() or radial_cell not in (0,1):
        raise ValueError('Invalid cell states or radial index')
    a = int(radial_cell)
    radius = a+np.array([0.,.5,1.])
    out = np.empty_like(w)
    if a == 0:
        if np.any(w[0] != 0):
            raise ValueError('Stored axis values must vanish')
        out[1:] = w[1:]/radius[1:,None,None]
        out[0] = -3*w[0]+4*w[1]-w[2]
    else:
        out[:] = w/radius[:,None,None]
    return out

def _s2_noise(seed):
    return np.random.default_rng(seed).normal(size=(3,3,4))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'w = _oracle_repair_axis(_s1_fixture())[0,0]; w_gold = _oracle_repair_axis(_s1_fixture())[0,0]',
         'call':'regular_local_state(w,0)', 'gold_call':'_oracle_regular_local_state(w_gold,0)'},
        {'setup':'w = _s1_fixture(True)[0,1]',
         'call':'regular_local_state(w,0)',
         'gold_call':'np.broadcast_to(np.array([1.,.1,.2,3.]),(3,3,4)).copy()'},
        {'setup':'w = _s2_noise(82); w_gold = _s2_noise(82)',
         'call':'regular_local_state(w,1)', 'gold_call':'_oracle_regular_local_state(w_gold,1)'},
        {'setup':'w = np.ones((3,3,4))\ndef _s2_invalid():\n    try:\n        regular_local_state(w,0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0',
         'call':'_s2_invalid()', 'gold_call':'1'},
    ]
