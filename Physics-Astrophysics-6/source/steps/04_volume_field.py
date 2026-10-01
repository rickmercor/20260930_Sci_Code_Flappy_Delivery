"""
Reconstruct the admissible three-dimensional magnetic field.

Reconstruct the magnetic field on a slab already established to admit a

continuously differentiable solution with the prescribed interior longitudinal

component, boundary data, unit magnitude, zero divergence, positive y component,

and x,y periodicity. Smooth admissibility of the requested slab is an input

precondition. The final orchestrator establishes the breakdown height through

characteristic analysis and uses this reconstruction at subcritical heights.

Returns
-------
Field values with shape (len(zs),len(xs),len(ys),3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def volume_field(xs,ys,zs,phases,bounds):
    """Return field values with shape (len(zs),len(xs),len(ys),3).

    xs and ys are nonempty finite one-dimensional physical coordinates in
    [0,2*pi); zs is nonempty, finite, nonnegative and strictly increasing.
    phases and bounds follow seed_bounds and must describe the prescribed
    fields. Return xyz components at the actual Cartesian query points, to
    absolute component accuracy 2e-6.

    The supplied phases and query heights must describe a slab on which the
    continuously differentiable positive-By field exists. The caller must
    establish this admissibility before calling the reconstruction.

    Raises
    ------
    ValueError
        If any input is malformed, nonfinite, out of range or not increasing,
        or the numerical reconstruction cannot be resolved on the supplied
        admissible slab.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

def _oracle_volume_field(xs,ys,zs,phases,bounds):
    phases=_check_phases(phases);bounds=_check_bounds(bounds)
    xs,ys,zs=[np.asarray(v,dtype=float) for v in (xs,ys,zs)]
    if any(v.ndim!=1 or not len(v) or not np.isfinite(v).all() for v in (xs,ys,zs)):
        raise ValueError('query axes must be finite nonempty vectors')
    if np.any(xs<0) or np.any(xs>=2*np.pi) or np.any(ys<0) or np.any(ys>=2*np.pi):
        raise ValueError('periodic coordinates must be in [0,2*pi)')
    if zs[0]<0 or np.any(np.diff(zs)<=0):
        raise ValueError('z queries must be nonnegative and strictly increasing')
    n=2048
    x=xs[:,None];y=2*np.pi*np.arange(n)[None,:]/n
    initial=_oracle_boundary_field(x,y,phases,bounds)[...,2]
    freq=np.fft.fftfreq(n,1/n);freq[n//2]=0
    def derivative(z,flat):
        normal=flat.reshape(len(xs),n)
        data=_oracle_prescribed_component(x,y,z,phases,bounds)
        g,gx,gy,gz=np.moveaxis(data,-1,0)
        p2=1-g*g-normal*normal
        if np.any(p2<=0):
            raise ValueError('positive branch lost')
        positive=np.sqrt(p2)
        normal_y=np.fft.ifft(1j*freq*np.fft.fft(normal,axis=1),axis=1).real
        positive_y=-(g*gy+normal*normal_y)/positive
        return (-gx-positive_y).ravel()
    if zs[-1]==0:
        states=initial.ravel()[None,:]
    else:
        sol=solve_ivp(derivative,(0.,float(zs[-1])),initial.ravel(),method='DOP853',
                      rtol=2e-11,atol=2e-13,t_eval=zs,max_step=max(.0005,float(zs[-1])/200))
        if not sol.success:
            raise ValueError('smooth continuation could not be resolved')
        states=sol.y.T
    output=[]
    for z,flat in zip(zs,states):
        normal=flat.reshape(len(xs),n)
        indices=ys*n/(2*np.pi)
        if np.all(np.abs(indices-np.rint(indices))<1e-10):
            evaluated=normal[:,np.rint(indices).astype(int)%n]
        else:
            spline=CubicSpline(np.linspace(0,2*np.pi,n+1),np.concatenate([normal,normal[:,:1]],axis=1),axis=1,bc_type='periodic')
            evaluated=spline(ys)
        g=_oracle_prescribed_component(x,ys[None,:],float(z),phases,bounds)[...,0]
        p2=1-g*g-evaluated*evaluated
        if np.any(p2<=0):
            raise ValueError('query field leaves the positive branch')
        output.append(np.stack([g,np.sqrt(p2),evaluated],axis=-1))
    return np.array(output)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'p = np.random.default_rng(71).uniform(0, 6, (3, 3))\n'
               'x = np.array([0.2, 1.7])\n'
               'y = np.array([0.1, 1.9, 4.7])\n'
               'z = np.array([0.0, 0.003, 0.009])\n'
               '\n'
               'def _case_encode_volume(value, shape):\n'
               '    result = np.asarray(value)\n'
               '    if result.shape != shape or result.dtype.kind not in "iuf" or not '
               'np.isfinite(result).all():\n'
               '        raise AssertionError("volume_field must return a finite real field with the contracted '
               'shape")\n'
               '    return result * (1e-9 / 2e-6)\n',
      'call': '_case_encode_volume(volume_field(x.copy(), y.copy(), z.copy(), p.copy(), seed_bounds(p.copy())), '
              '(len(z), len(x), len(y), 3))',
      'gold_call': '_case_encode_volume(_oracle_volume_field(x.copy(), y.copy(), z.copy(), p.copy(), '
                   '_oracle_seed_bounds(p.copy())), (len(z), len(x), len(y), 3))'},
     {'setup': 'import numpy as np\n'
               'p = np.array([[5.93, 2.26, 4.93, 3.72, 1.85], [5.8, 5.46, 2.29, 6.11, 1.41], [5.06, 4.28, 2.96, '
               '0.19, 5.62]])\n'
               'x = np.array([2.2143518802])\n'
               'y = np.array([0.4, 1.9, 3.3, 5.0])\n'
               'z = np.array([0.05, 0.15, 0.2369])\n'
               '\n'
               'def _case_encode_volume(value, shape):\n'
               '    result = np.asarray(value)\n'
               '    if result.shape != shape or result.dtype.kind not in "iuf" or not '
               'np.isfinite(result).all():\n'
               '        raise AssertionError("volume_field must return a finite real field with the contracted '
               'shape")\n'
               '    return result * (1e-9 / 2e-6)\n',
      'call': '_case_encode_volume(volume_field(x.copy(), y.copy(), z.copy(), p.copy(), seed_bounds(p.copy())), '
              '(len(z), len(x), len(y), 3))',
      'gold_call': '_case_encode_volume(_oracle_volume_field(x.copy(), y.copy(), z.copy(), p.copy(), '
                   '_oracle_seed_bounds(p.copy())), (len(z), len(x), len(y), 3))'},
     {'setup': 'import numpy as np\n'
               'p = np.zeros((3, 1))\n'
               'x = np.array([0.3])\n'
               'y = np.array([0.2, 2.0])\n'
               'z = np.array([0.0])\n'
               '\n'
               'def _case_encode_volume(value, shape):\n'
               '    result = np.asarray(value)\n'
               '    if result.shape != shape or result.dtype.kind not in "iuf" or not '
               'np.isfinite(result).all():\n'
               '        raise AssertionError("volume_field must return a finite real field with the contracted '
               'shape")\n'
               '    return result * (1e-9 / 2e-6)\n',
      'call': '_case_encode_volume(volume_field(x.copy(), y.copy(), z.copy(), p.copy(), seed_bounds(p.copy())), '
              '(len(z), len(x), len(y), 3))',
      'gold_call': '_case_encode_volume(_oracle_volume_field(x.copy(), y.copy(), z.copy(), p.copy(), '
                   '_oracle_seed_bounds(p.copy())), (len(z), len(x), len(y), 3))'},
     {'setup': 'import numpy as np\n'
               'p = np.random.default_rng(71).uniform(0, 6, (3, 3))\n'
               '\n'
               'def case():\n'
               '    _case_bounds = seed_bounds(p.copy())\n'
               '    try:\n'
               '        volume_field([0.0], [0.0], [-0.1], p.copy(), _case_bounds)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def gold_case():\n'
               '    _case_bounds = _oracle_seed_bounds(p.copy())\n'
               '    try:\n'
               '        _oracle_volume_field([0.0], [0.0], [-0.1], p.copy(), _case_bounds)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'case()',
      'gold_call': 'gold_case()'}]
