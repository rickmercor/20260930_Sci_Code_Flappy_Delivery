"""
Apply local-state oscillation elimination while retaining stored cell means.

The source's filter acts on local Q2 nodal states with reduction weight chi=r.
Its finite discrete prescription must also apply to cells whose reduction
weight vanishes at the axis.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def conservative_local_filter(local_states, rates, dt=0.02, strength=0.02):
    """Return the filtered stored states on the four unit-width cells.

    Parameters
    ----------
    local_states : array_like, shape (2,2,3,3,4)
        Finite regular local states at the patch's Q2 nodes.
    rates : array_like, shape (2,2,3)
        Finite nonnegative order-dependent rates from step 4.
    dt, strength : float
        Finite nonnegative timestep and oscillation-elimination strength.

    Returns
    -------
    ndarray, shape (2,2,3,3,4)
        Filtered stored states W, indexed as in step 1. Neither array input is
        modified. Invalid inputs raise ValueError. Zero dt or strength gives
        the undamped map to stored data.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_conservative_local_filter(local_states, rates, dt=0.02, strength=0.02):
    u = np.asarray(local_states,dtype=float)
    rates = np.asarray(rates,dtype=float)
    if (u.shape != (2,2,3,3,4) or rates.shape != (2,2,3)
            or not np.isfinite(u).all() or not np.isfinite(rates).all() or np.any(rates < 0)
            or not np.isfinite([dt,strength]).all() or min(dt,strength) < 0):
        raise ValueError('Invalid states, rates, timestep or strength')
    x = np.array([0.,.5,1.])
    weights = np.array([1.,4.,1.])/6
    mass = weights[:,None]*weights[None,:]
    xx,yy = np.meshgrid(x,x,indexing='ij')
    basis = np.stack([np.ones_like(xx),2*yy-1,2*xx-1,(2*xx-1)*(2*yy-1)],axis=-1).reshape(9,4)
    alpha = np.exp(-strength*dt*np.cumsum(rates,axis=-1)[...,1:])
    out = np.empty_like(u)
    for a in range(2):
        chi = np.broadcast_to((a+x)[:,None],(3,3))
        wm = (mass*chi).ravel()
        gram = basis.T@(wm[:,None]*basis)
        for b in range(2):
            values = u[a,b].reshape(9,4)
            p0 = np.sum(wm[:,None]*values,axis=0)/np.sum(wm)
            coefficients = np.linalg.solve(gram,basis.T@(wm[:,None]*values))
            p1 = (basis@coefficients).reshape(3,3,4)
            out[a,b] = chi[...,None]*(p0+alpha[a,b,0]*(p1-p0)+alpha[a,b,1]*(u[a,b]-p1))
    return out

def _s5_rates(u):
    return _oracle_normalized_damping_rates(u,_oracle_directional_jump_amplitudes(u))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'u = _s3_fixture(); u_gold = _s3_fixture(); rates = _s5_rates(u_gold); rates_gold = _s5_rates(u_gold)',
         'call':'conservative_local_filter(u,rates)', 'gold_call':'_oracle_conservative_local_filter(u_gold,rates_gold)'},
        {'setup':'u = _s3_fixture(); rates = np.full((2,2,3),10.0)',
         'call':'conservative_local_filter(u,rates,dt=0)',
         'gold_call':'_oracle_repair_axis(_s1_fixture())'},
        {'setup':'u = _s1_noise(181); u_gold = _s1_noise(181); rates = np.full((2,2,3),100.0); rates_gold = np.full((2,2,3),100.0)',
         'call':'conservative_local_filter(u,rates,dt=0.4,strength=0.3)',
         'gold_call':'_oracle_conservative_local_filter(u_gold,rates_gold,dt=0.4,strength=0.3)'},
        {'setup':'u = np.broadcast_to(np.array([1.,.1,.2,3.]),(2,2,3,3,4)).copy(); rates = np.full((2,2,3),1e3)',
         'call':'conservative_local_filter(u,rates)', 'gold_call':'_s1_fixture(True)'},
        {'setup':'u = _s3_fixture(); rates = _s5_rates(u)\ndef _s5_invalid():\n    try:\n        conservative_local_filter(u,rates,strength=-0.01)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0',
         'call':'_s5_invalid()', 'gold_call':'1'},
    ]
