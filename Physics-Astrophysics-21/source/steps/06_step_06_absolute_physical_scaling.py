"""
Impose the source's absolute physical constraints on a stored cell state.

The equation of state is relativistic ideal gas in a flat orthonormal frame.
For this operation, physical admissibility is expressed through density D and
cone margin q=E-sqrt(D^2+m_r^2+m_z^2). The cell may contain an axis face. The
requested map is the source's explicit conservative absolute scaling.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def absolute_physical_scaling(stored, radial_cell, density_floor=1e-11, cone_floor=1e-11):
    """Return stored states after the absolute density and cone constraints.

    Parameters
    ----------
    stored : array_like, shape (3,3,4)
        Finite axis-compatible stored nodal states with admissible mean.
    radial_cell : int
        Radial cell index a in {0,1}; nodes are a+(0,1/2,1).
    density_floor, cone_floor : float
        Positive finite requested absolute floors.

    Returns
    -------
    ndarray, shape (3,3,4)
        Conservatively scaled stored states, in (D,m_r,m_z,E) order. The input
        is not modified. Invalid inputs or an inadmissible cell mean raise
        ValueError.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _s6_geometry_anchor(stored, radial_cell):
    w = np.array(stored,dtype=float,copy=True)
    if (w.shape != (3,3,4) or not np.isfinite(w).all()
            or radial_cell not in (0,1)):
        raise ValueError('Invalid stored cell or radial index')
    a = int(radial_cell)
    if a == 0 and np.any(w[0] != 0):
        raise ValueError('Stored axis face must vanish')
    chi = np.broadcast_to((a+np.array([0.,.5,1.]))[:,None],(3,3))
    weights = np.array([1.,4.,1.])/6
    mass = weights[:,None]*weights[None,:]
    anchor = np.einsum('ij,ijk->k',mass,w)/np.sum(mass*chi)
    qa = anchor[3]-np.linalg.norm(anchor[:3])
    if anchor[0] <= 0 or qa <= 0:
        raise ValueError('Cell mean is outside the admissible cone')
    return w,chi,anchor,qa

def _oracle_absolute_physical_scaling(stored, radial_cell, density_floor=1e-11, cone_floor=1e-11):
    if (not np.isfinite([density_floor,cone_floor]).all()
            or min(density_floor,cone_floor) <= 0):
        raise ValueError('Floors must be positive and finite')
    out,chi,anchor,qa = _s6_geometry_anchor(stored,radial_cell)
    reference = chi[...,None]*anchor
    local = _oracle_regular_local_state(out,radial_cell)
    de,qe = min(density_floor,anchor[0]),min(cone_floor,qa)
    minimum = np.min(local[...,0])
    theta_d = 1.0 if minimum >= de else (anchor[0]-de)/(anchor[0]-minimum)
    out[...,0] = reference[...,0]+theta_d*(out[...,0]-reference[...,0])
    local = _oracle_regular_local_state(out,radial_cell)
    minimum = np.min(local[...,3]-np.linalg.norm(local[...,:3],axis=-1))
    theta_q = 1.0 if minimum >= qe else (qa-qe)/(qa-minimum)
    return reference+theta_q*(out-reference)

def _s6_filtered():
    u = _s3_fixture()
    rates = _oracle_normalized_damping_rates(u,_oracle_directional_jump_amplitudes(u))
    return _oracle_conservative_local_filter(u,rates)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'w = _s6_filtered()[0,0]; w_gold = _s6_filtered()[0,0]',
         'call':'absolute_physical_scaling(w,0)', 'gold_call':'_oracle_absolute_physical_scaling(w_gold,0)'},
        {'setup':'w = _s1_fixture(True)[1,0]; w_gold = _s1_fixture(True)[1,0]',
         'call':'absolute_physical_scaling(w,1)', 'gold_call':'w_gold'},
        {'setup':'w = _s1_fixture(True)[0,1]*1e-12; w_gold = _s1_fixture(True)[0,1]*1e-12',
         'call':'absolute_physical_scaling(w,0)*1e12',
         'gold_call':'_oracle_absolute_physical_scaling(w_gold,0)*1e12'},
        {'setup':'w = _s6_filtered()[0,0]; w_gold = _s6_filtered()[0,0]',
         'call':'absolute_physical_scaling(w,0,density_floor=0.6,cone_floor=1.5)',
         'gold_call':'_oracle_absolute_physical_scaling(w_gold,0,density_floor=0.6,cone_floor=1.5)'},
        {'setup':'w = _s6_filtered()[0,0]; w_gold = _s6_filtered()[0,0]',
         'call':'absolute_physical_scaling(w,0,density_floor=0.4,cone_floor=1.0)',
         'gold_call':'_oracle_absolute_physical_scaling(w_gold,0,density_floor=0.4,cone_floor=1.0)'},
        {'setup':'w = -_s1_fixture(True)[1,1]\ndef _s6_invalid():\n    try:\n        absolute_physical_scaling(w,1)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0',
         'call':'_s6_invalid()', 'gold_call':'1'},
    ]
