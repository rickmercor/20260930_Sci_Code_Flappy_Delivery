"""
Prescribed longitudinal magnetic component and its physical gradient.

Prescribed longitudinal magnetic component and its physical gradient.

Use the preceding definition of S_a, P and T, and the supplied continuous
bounds. With u=(T-Tmin)/(Tmax-Tmin), the prescribed component is
cos(0.1+(pi-0.2) (exp(5 u)-1)/(exp(5)-1)).

Returns
-------
[Bx, dBx/dx, dBx/dy, dBx/dz] on the final array axis.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def prescribed_component(x,y,z,phases,bounds):
    """Return [Bx, dBx/dx, dBx/dy, dBx/dz] on the final array axis.

    x,y,z are finite broadcast-compatible real arrays or scalars. phases
    and bounds have the seed_bounds conventions. Output shape is the
    broadcast coordinate shape followed by (4,). Coordinates are in radians
    in the modal arguments.

    Raises
    ------
    ValueError
        If phases or bounds are malformed, if the coordinates are nonfinite
        or cannot broadcast, or if the prescribed values are not finite.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_prescribed_component(x,y,z,phases,bounds):
    p=_check_phases(phases);bounds=_check_bounds(bounds)
    x,y,z=[np.asarray(v,dtype=float) for v in (x,y,z)]
    if not all(np.isfinite(v).all() for v in (x,y,z)):
        raise ValueError('coordinates must be finite')
    try:
        np.broadcast_shapes(x.shape,y.shape,z.shape)
    except ValueError as exc:
        raise ValueError('coordinates must broadcast') from exc
    sx,sy,sz=_wave(x,p[0]),_wave(y,p[1]),_wave(z,p[2])
    u=(sx*sy*sz-bounds[1,0])/np.diff(bounds[1])[0]
    a=.1+(np.pi-.2)*np.expm1(5*u)/np.expm1(5.)
    factor=-np.sin(a)*(np.pi-.2)*5*np.exp(5*u)/np.expm1(5.)/np.diff(bounds[1])[0]
    values=np.broadcast_arrays(np.cos(a),factor*_wave(x,p[0],1)*sy*sz,
                               factor*sx*_wave(y,p[1],1)*sz,factor*sx*sy*_wave(z,p[2],1))
    if not all(np.isfinite(v).all() for v in values):
        raise ValueError('prescribed field is not finite')
    return np.stack(values,axis=-1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\np = np.random.default_rng(17).uniform(0, 6, (3, 3))\n',
      'call': 'prescribed_component(np.array([0.2, 2.1]), 1.3, 0.02, p.copy(), seed_bounds(p.copy()))',
      'gold_call': '_oracle_prescribed_component(np.array([0.2, 2.1]), 1.3, 0.02, p.copy(), '
                   '_oracle_seed_bounds(p.copy()))'},
     {'setup': 'import numpy as np\np = np.zeros((3, 1))\n',
      'call': 'prescribed_component(0.0, 0.0, 0.0, p.copy(), seed_bounds(p.copy()))',
      'gold_call': '_oracle_prescribed_component(0.0, 0.0, 0.0, p.copy(), _oracle_seed_bounds(p.copy()))'},
     {'setup': 'import numpy as np\n'
               'p = np.random.default_rng(17).uniform(0, 6, (3, 3))\n'
               '\n'
               'def case():\n'
               '    _case_bounds = seed_bounds(p.copy())\n'
               '    try:\n'
               '        prescribed_component(np.nan, 0.0, 0.0, p.copy(), _case_bounds)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def gold_case():\n'
               '    _case_bounds = _oracle_seed_bounds(p.copy())\n'
               '    try:\n'
               '        _oracle_prescribed_component(np.nan, 0.0, 0.0, p.copy(), _case_bounds)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'case()',
      'gold_call': 'gold_case()'}]
