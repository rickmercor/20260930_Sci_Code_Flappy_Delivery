"""
Analytic geometry of a smooth ellipsoid at surface sample positions.

The surface is sum_d (x_d/a_d)^2=1. Geometry is supplied analytically in this
task rather than estimated from a background point cloud. The scalar curvature
measure reported in column 3 is the Euclidean norm of the two principal
curvatures at the sample.

Return an (n,4) float array in the input row order.

Columns 0 to 2 hold the outward unit normal at each sample; column 3 holds
the scalar curvature measure described above. points has shape (n,3) with
n>=1 and axes has shape (3,) with every semiaxis strictly positive. All
entries are finite and every sample satisfies the surface equation to
absolute residual 1e-8 or less.

Raises
------
ValueError
    If points or axes has the wrong shape, if n<1, if any entry of points
    or axes is not finite, if any semiaxis is not strictly positive, or if
    any sample exceeds the on-surface residual bound.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def surface_geometry(points: 'np.ndarray | list | tuple', axes: 'np.ndarray | list | tuple') -> 'np.ndarray':
    """Return an (n,4) float array in the input row order.

    Columns 0 to 2 hold the outward unit normal at each sample; column 3 holds
    the scalar curvature measure described above. points has shape (n,3) with
    n>=1 and axes has shape (3,) with every semiaxis strictly positive. All
    entries are finite and every sample satisfies the surface equation to
    absolute residual 1e-8 or less.

    Raises
    ------
    ValueError
        If points or axes has the wrong shape, if n<1, if any entry of points
        or axes is not finite, if any semiaxis is not strictly positive, or if
        any sample exceeds the on-surface residual bound.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_surface_geometry(points: 'np.ndarray | list | tuple', axes: 'np.ndarray | list | tuple') -> 'np.ndarray':
    import numpy as np
    x=np.asarray(points,dtype=float);a=np.asarray(axes,dtype=float)
    if x.ndim!=2 or x.shape[1:]!=(3,) or len(x)<1 or a.shape!=(3,):
        raise ValueError('invalid geometry shapes')
    if not np.isfinite(x).all() or not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError('invalid geometry values')
    if np.max(abs(np.sum((x/a)**2,axis=1)-1))>1e-8:
        raise ValueError('points must lie on ellipsoid')
    g=x/a**2;n=g/np.linalg.norm(g,axis=1)[:,None]
    out=np.empty((len(x),4));out[:,:3]=n
    for i in range(len(x)):
        p=np.eye(3)-np.outer(n[i],n[i])
        shape=p@np.diag(1/a**2)@p/np.linalg.norm(g[i])
        eig=np.linalg.eigvalsh(shape)
        out[i,3]=np.linalg.norm(eig[1:])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nx=np.eye(3);a=np.ones(3)',
      'call': 'surface_geometry(x,a)',
      'gold_call': '_oracle_surface_geometry(x,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nx=np.array([[1.,0.,0.],[0.,.8,0.],[0.,0.,.6]]);a=[1.,.8,.6]',
      'call': 'surface_geometry(x,a)',
      'gold_call': '_oracle_surface_geometry(x,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nx=np.array([[2.,0.,0.]]);a=[2.,2.,2.]',
      'call': 'surface_geometry(x,a)',
      'gold_call': '_oracle_surface_geometry(x,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'u=.7;v=1.1\n'
               'x=np.array([[np.sin(u)*np.cos(v),.8*np.sin(u)*np.sin(v),.6*np.cos(u)]]);a=[1.,.8,.6]',
      'call': 'surface_geometry(x,a)',
      'gold_call': '_oracle_surface_geometry(x,a)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'def _bad_public():\n'
               '    try: surface_geometry([[0.,0.,0.]],[1.,1.,1.]); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2\n'
               'def _bad_gold():\n'
               '    try: _oracle_surface_geometry([[0.,0.,0.]],[1.,1.,1.]); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': '_bad_public()',
      'gold_call': '_bad_gold()',
      'tol': 1e-09}]
