"""
Assign source-consistent damping rates to derivative orders on the patch.

Use the source's normalization and numerical noise policy. All four cells form
the global normalization domain. The flat causal speed is one, with Q2
effective resolution on unit widths.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def normalized_damping_rates(local_states, jump_amplitudes):
    """Return order-dependent system damping rates for the four cells.

    Parameters
    ----------
    local_states : array_like, shape (2,2,3,3,4)
        Finite local nodal states with the geometry and ordering of step 3.
    jump_amplitudes : array_like, shape (2,2,3,2,4)
        Nonnegative finite amplitudes in step 3's output ordering.

    Returns
    -------
    ndarray, shape (2,2,3)
        Rates indexed by (a,b,derivative_order), using binary64 precision for
        the source's noise screens. Neither input is modified. Invalid data
        raise ValueError.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

def _oracle_normalized_damping_rates(local_states, jump_amplitudes):
    u = np.asarray(local_states,dtype=float)
    jumps = np.asarray(jump_amplitudes,dtype=float)
    if (u.shape != (2,2,3,3,4) or jumps.shape != (2,2,3,2,4)
            or not np.isfinite(u).all() or not np.isfinite(jumps).all() or np.any(jumps < 0)):
        raise ValueError('Invalid local states or jumps')
    weights = np.array([1.,4.,1.])/6
    mass = weights[:,None]*weights[None,:]
    mean = np.einsum('ij,abijk->k',mass,u)/4
    amp = np.max(abs(u-mean),axis=(0,1,2,3))
    absolute = 1000*np.finfo(float).eps
    component = max(1e-8*np.max(np.linalg.norm(u,axis=-1)),absolute)
    rates = np.zeros((2,2,3))
    for a in range(2):
        for b in range(2):
            for direction in range(2):
                line = u[:,b] if direction == 0 else u[a,:]
                line_mean = np.einsum('ij,cijk->k',mass,line)/2
                denominator = np.maximum(np.max(abs(line-line_mean),axis=(0,1,2)),1e-6*amp)
                retained = (amp > component) & (denominator > absolute)
                for order in range(3):
                    values = np.zeros(4)
                    values[retained] = ((2*order+1)*jumps[a,b,order,direction,retained]
                                        /(2*math.factorial(order)*denominator[retained]))
                    rates[a,b,order] += 5*np.max(values)
    return rates

def _s4_flat_line():
    """One coordinate line nearly constant in m_r while the patch spans 3e3."""
    u = _s3_fixture()
    u[:,0,:,:,1] = 0.3
    u[:,0,0,0,1] = 0.3+3e-7
    u[:,1,:,:,1] = 3.0e3
    return u

def _s4_tall():
    u = _s1_noise(417)
    u[...,2] = 100
    return u

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'u = _s3_fixture(); u_gold = _s3_fixture(); j = _oracle_directional_jump_amplitudes(u_gold); j_gold = _oracle_directional_jump_amplitudes(u_gold)',
         'call':'normalized_damping_rates(u,j)', 'gold_call':'_oracle_normalized_damping_rates(u_gold,j_gold)'},
        {'setup':'u = np.ones((2,2,3,3,4)); j = np.ones((2,2,3,2,4))',
         'call':'normalized_damping_rates(u,j)', 'gold_call':'np.zeros((2,2,3))'},
        {'setup':'u = _s4_tall(); u_gold = _s4_tall(); j = _oracle_directional_jump_amplitudes(u_gold); j_gold = _oracle_directional_jump_amplitudes(u_gold)',
         'call':'normalized_damping_rates(u,j)', 'gold_call':'_oracle_normalized_damping_rates(u_gold,j_gold)'},
        {'setup':'u = _s3_fixture(); u[:,1] *= 1e4; u_gold = _s3_fixture(); u_gold[:,1] *= 1e4; j = _oracle_directional_jump_amplitudes(u_gold); j_gold = _oracle_directional_jump_amplitudes(u_gold)',
         'call':'normalized_damping_rates(u,j)', 'gold_call':'_oracle_normalized_damping_rates(u_gold,j_gold)'},
        {'setup':'u = _s4_flat_line(); u_gold = _s4_flat_line(); j = _oracle_directional_jump_amplitudes(u_gold); j_gold = _oracle_directional_jump_amplitudes(u_gold)',
         'call':'normalized_damping_rates(u,j)', 'gold_call':'_oracle_normalized_damping_rates(u_gold,j_gold)'},
        {'setup':'u = _s3_fixture(); j = _oracle_directional_jump_amplitudes(u); j[1,1,2,0,3] = -1.0\ndef _s4_invalid():\n    try:\n        normalized_damping_rates(u,j)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0',
         'call':'_s4_invalid()', 'gold_call':'1'},
    ]
