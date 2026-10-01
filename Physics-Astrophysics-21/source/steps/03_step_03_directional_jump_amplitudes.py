"""
Measure interface irregularity of local Q2 states on a four-cell patch.

The source's directional oscillation indicator uses separate component jump
amplitudes for normal derivative orders zero through two. Physical boundary
faces are excluded. The patch has unit cell widths and mapped GLL quadrature.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def directional_jump_amplitudes(local_states):
    """Return the source's interior-face jump amplitudes.

    Parameters
    ----------
    local_states : array_like, shape (2,2,3,3,4)
        Finite local states at nodes (0,1/2,1), with duplicated DG interfaces.

    Returns
    -------
    ndarray, shape (2,2,3,2,4)
        Entries indexed by (a,b,derivative_order,normal_direction,component).
        Direction 0 is radial and direction 1 is axial. The input is not
        modified. Invalid data raise ValueError.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_directional_jump_amplitudes(local_states):
    u = np.asarray(local_states,dtype=float)
    if u.shape != (2,2,3,3,4) or not np.isfinite(u).all():
        raise ValueError('Expected finite local patch states')
    derivative = np.array([[-3.,4.,-1.],[-1.,0.,1.],[1.,-4.,3.]])
    weights = np.array([1.,4.,1.])/6
    jumps = np.empty((2,2,3,2,4))
    for order in range(3):
        d = np.linalg.matrix_power(derivative,order)
        for a in range(2):
            for b in range(2):
                radial = np.einsum('i,ijk->jk',d[2],u[0,b])-np.einsum('i,ijk->jk',d[0],u[1,b])
                axial = np.einsum('j,ijk->ik',d[2],u[a,0])-np.einsum('j,ijk->ik',d[0],u[a,1])
                jumps[a,b,order,0] = np.sqrt(np.einsum('i,ik->k',weights,radial**2))
                jumps[a,b,order,1] = np.sqrt(np.einsum('i,ik->k',weights,axial**2))
    return jumps

def _s3_fixture():
    w = _oracle_repair_axis(_s1_fixture())
    return np.array([[_oracle_regular_local_state(w[a,b],a) for b in range(2)] for a in range(2)])

def _s3_step():
    u = np.zeros((2,2,3,3,4))
    u[1] = 2
    return u

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'u = _s3_fixture(); u_gold = _s3_fixture()', 'call':'directional_jump_amplitudes(u)',
         'gold_call':'_oracle_directional_jump_amplitudes(u_gold)'},
        {'setup':'u = np.ones((2,2,3,3,4))', 'call':'directional_jump_amplitudes(u)',
         'gold_call':'np.zeros((2,2,3,2,4))'},
        {'setup':'u = _s1_noise(208); u_gold = _s1_noise(208)',
         'call':'directional_jump_amplitudes(u)', 'gold_call':'_oracle_directional_jump_amplitudes(u_gold)'},
        {'setup':'u = _s3_step(); u_gold = _s3_step()',
         'call':'directional_jump_amplitudes(u)', 'gold_call':'_oracle_directional_jump_amplitudes(u_gold)'},
        {'setup':'u = _s3_fixture(); u[0,1,2,0,3] = np.inf\ndef _s3_invalid():\n    try:\n        directional_jump_amplitudes(u)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0',
         'call':'_s3_invalid()', 'gold_call':'1'},
    ]
