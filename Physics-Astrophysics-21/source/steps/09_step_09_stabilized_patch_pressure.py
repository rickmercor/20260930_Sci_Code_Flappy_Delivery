"""
FINAL ORCHESTRATOR: pressure of the finite prescribed stage on a Q2 patch.

Use the preceding sub-problem functions to evaluate the source's complete
stabilization and pressure recovery for the supplied raw nodal stage. The
selected interface value belongs to K00 at (r,z)=(1,1/2), not to K10.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stabilized_patch_pressure(raw=None, dt=0.02, strength=0.02):
    """Return the K00 pressure at (r,z)=(1,1/2) after the prescribed stage.

    Parameters
    ----------
    raw : array_like, shape (2,2,3,3,4), optional
        Stored patch values ordered (a,b,radial_node,axial_node,component).
        If omitted, the four cells K_ab=[a,a+1]x[b,b+1], a,b in {0,1}, use
        local nodes (0,1/2,1) and raw values
        r*(1+0.1*z+0.04*a+0.01*b, 0.15*r, 0.2+0.05*z+0.03*a,
        3+0.2*z+0.07*a+0.11*b). In K00 overwrite node [0,1] with
        (0.01,0,0.005,0.03) and [1,1] with (-0.05,0.8,0.1,0.65).
    dt, strength : float
        Nonnegative finite timestep and oscillation-elimination strength.
        Both requested absolute floors are 1e-11, relative fraction is 0.1,
        Gamma=5/3, c=1, chi=r; there is no PDE residual or boundary jump.

    Returns
    -------
    float
        Pressure at K00 radial node 2 and axial node 1. The supplied array is
        not modified. Invalid data or any inadmissible conservative cell mean
        raise ValueError. Assemble this result by calling the preceding
        sub-problem functions.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_stabilized_patch_pressure(raw=None, dt=0.02, strength=0.02):
    if not np.isfinite([dt,strength]).all() or min(dt,strength) < 0:
        raise ValueError('Invalid timestep or filter strength')
    if raw is None:
        raw = np.empty((2,2,3,3,4))
        x = np.array([0.,.5,1.])
        for a in range(2):
            for b in range(2):
                r,z = np.meshgrid(a+x,b+x,indexing='ij')
                local = np.stack([1+.1*z+.04*a+.01*b,.15*r,
                                  .2+.05*z+.03*a,3+.2*z+.07*a+.11*b],axis=-1)
                raw[a,b] = r[...,None]*local
        raw[0,0,0,1] = [.01,0,.005,.03]
        raw[0,0,1,1] = [-.05,.8,.1,.65]
    repaired = _oracle_repair_axis(raw)
    weights = np.array([1.,4.,1.])/6
    means = np.einsum('i,j,abijk->abk',weights,weights,repaired)
    if np.any(means[...,0] <= 0) or np.any(means[...,3] <= np.linalg.norm(means[...,:3],axis=-1)):
        raise ValueError('An input cell mean is inadmissible')
    local = np.array([[_oracle_regular_local_state(repaired[a,b],a)
                       for b in range(2)] for a in range(2)])
    jumps = _oracle_directional_jump_amplitudes(local)
    rates = _oracle_normalized_damping_rates(local,jumps)
    filtered = _oracle_conservative_local_filter(local,rates,dt=dt,strength=strength)
    final = np.empty_like(filtered)
    for a in range(2):
        for b in range(2):
            absolute = _oracle_absolute_physical_scaling(filtered[a,b],a)
            final[a,b] = _oracle_relative_physical_scaling(absolute,a)
    target = _oracle_regular_local_state(final[0,0],0)[2,1]
    return _oracle_primitive_pressure(target)

def _s9_perturbed():
    w = _s1_fixture()
    return w+np.random.default_rng(610).normal(scale=1e-5,size=w.shape)

def _s9_inadmissible():
    w = _s1_fixture()
    w[1,1,:,:,3] = 0.1
    return w

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'', 'call':'stabilized_patch_pressure()',
         'gold_call':'1.2212246706514617136238315514'},
        {'setup':'w = _s1_fixture(True); w_gold = _s1_fixture(True)', 'call':'stabilized_patch_pressure(raw=w)',
         'gold_call':'_oracle_primitive_pressure(np.array([1.,.1,.2,3.]))'},
        {'setup':'', 'call':'stabilized_patch_pressure(dt=0)',
         'gold_call':'1.2206959997664078565799395641'},
        {'setup':'w = _s9_perturbed(); w_gold = _s9_perturbed()',
         'call':'stabilized_patch_pressure(raw=w,dt=0.021,strength=0.018)',
         'gold_call':'_oracle_stabilized_patch_pressure(raw=w_gold,dt=0.021,strength=0.018)'},
        {'setup':'w = _s9_inadmissible()\ndef _s9_invalid():\n    try:\n        stabilized_patch_pressure(raw=w)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0',
         'call':'_s9_invalid()', 'gold_call':'1'},
    ]
