"""
Restore an axis-compatible stored representation of a raw tensor nodal stage.

The patch consists of K_ab=[a,a+1]x[b,b+1], a,b in {0,1}, with local nodes
(0,1/2,1), mapped GLL quadrature, and reduction weight chi=r. Components are
(D,m_r,m_z,E). The requested operation is the source's conservative axis repair.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def repair_axis(raw):
    """Return repaired stored nodal values for the four-cell patch.

    Parameters
    ----------
    raw : array_like, shape (2,2,3,3,4)
        Finite raw stored states, indexed by (a,b,radial_node,axial_node,component).

    Returns
    -------
    ndarray, shape (2,2,3,3,4)
        Conservative axis-compatible stored states. The input is not modified.
        Invalid shape or nonfinite entries raise ValueError.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_repair_axis(raw):
    out = np.array(raw, dtype=float, copy=True)
    if out.shape != (2,2,3,3,4) or not np.isfinite(out).all():
        raise ValueError('Expected finite four-cell Q2 stored states')
    out[0,:,1:,:,:] += out[0,:,:1,:,:]/5.0
    out[0,:,0,:,:] = 0.0
    return out

def _s1_fixture(constant=False):
    values = np.empty((2,2,3,3,4))
    x = np.array([0.,.5,1.])
    for a in range(2):
        for b in range(2):
            r,z = np.meshgrid(a+x,b+x,indexing='ij')
            u = np.stack([1+.1*z+.04*a+.01*b,.15*r,
                          .2+.05*z+.03*a,3+.2*z+.07*a+.11*b],axis=-1)
            if constant:
                u[:] = [1.,.1,.2,3.]
            values[a,b] = r[...,None]*u
    if not constant:
        values[0,0,0,1] = [.01,0,.005,.03]
        values[0,0,1,1] = [-.05,.8,.1,.65]
    return values

def _s1_noise(seed):
    return np.random.default_rng(seed).normal(size=(2,2,3,3,4))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'w = _s1_fixture(); w_gold = _s1_fixture()', 'call':'repair_axis(w)',
         'gold_call':'_oracle_repair_axis(w_gold)'},
        {'setup':'w = _s1_fixture(True); w_gold = _s1_fixture(True)', 'call':'repair_axis(w)',
         'gold_call':'w_gold'},
        {'setup':'w = _s1_noise(310); w_gold = _s1_noise(310)',
         'call':'repair_axis(w)', 'gold_call':'_oracle_repair_axis(w_gold)'},
        {'setup':'w = np.zeros((2,2,3,3,4)); w[1,0,2,1,3] = np.nan\ndef _s1_invalid():\n    try:\n        repair_axis(w)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0',
         'call':'_s1_invalid()', 'gold_call':'1'},
    ]
