"""
Complete magnetic boundary data on z=0.

Complete magnetic boundary data on z=0.

The longitudinal component is prescribed_component. With the same
normalizations u and v=(P-Pmin)/(Pmax-Pmin), the normal component is
sin(0.1+(pi-0.2) (exp(5 u)-1)/(exp(5)-1)) cos(1+(pi-2) v).
The field has unit magnitude and positive y component.

Returns
-------
The complete boundary vector [Bx,By,Bz] on the final axis.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def boundary_field(x,y,phases,bounds):
    """Return the complete boundary vector [Bx,By,Bz] on the final axis.

    x and y are finite broadcast-compatible coordinates. phases and
    bounds follow seed_bounds; output shape is broadcast(x,y).shape+(3,).
    The positive-By branch is required.

    Raises
    ------
    ValueError
        If phases or bounds are malformed, or if the prescribed boundary
        components leave no positive y component at some queried point.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_boundary_field(x,y,phases,bounds):
    p=_check_phases(phases);bounds=_check_bounds(bounds)
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    g=_oracle_prescribed_component(x,y,0.,p,bounds)[...,0]
    v=(_wave(x,p[0])*_wave(y,p[1])-bounds[0,0])/np.diff(bounds[0])[0]
    a2=1-g*g
    normal=np.sqrt(np.maximum(0.,a2))*np.cos(1+(np.pi-2)*v)
    positive2=a2-normal*normal
    if not np.isfinite(positive2).all() or np.any(positive2<=0):
        raise ValueError('boundary is not on the positive branch')
    return np.stack([g,np.sqrt(positive2),normal],axis=-1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\np = np.random.default_rng(27).uniform(-3, 3, (3, 4))\n',
      'call': 'boundary_field(np.array([0.1, 2.3]), 0.8, p.copy(), seed_bounds(p.copy()))',
      'gold_call': '_oracle_boundary_field(np.array([0.1, 2.3]), 0.8, p.copy(), _oracle_seed_bounds(p.copy()))'},
     {'setup': 'import numpy as np\np = np.zeros((3, 1))\n',
      'call': 'boundary_field(0.0, 0.0, p.copy(), seed_bounds(p.copy()))',
      'gold_call': '_oracle_boundary_field(0.0, 0.0, p.copy(), _oracle_seed_bounds(p.copy()))'},
     {'setup': 'import numpy as np\n'
               'p = np.random.default_rng(27).uniform(-3, 3, (3, 4))\n'
               '\n'
               'def case():\n'
               '    try:\n'
               '        boundary_field(0.0, 0.0, p.copy(), np.zeros((2, 2)))\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def gold_case():\n'
               '    try:\n'
               '        _oracle_boundary_field(0.0, 0.0, p.copy(), np.zeros((2, 2)))\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'case()',
      'gold_call': 'gold_case()'}]
