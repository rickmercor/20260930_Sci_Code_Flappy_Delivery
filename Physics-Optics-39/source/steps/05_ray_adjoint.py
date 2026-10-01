"""
Evaluate the source-plane residual operator for the declared non-local reconstruction.

The residual operator uses the declared forward ray geometry and the Euclidean pixel inner product.

Returns
-------
return out  # out : float ndarray, shape (S,ny,nx)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def ray_adjoint(detector: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Evaluate the discrete adjoint of spectral transport.

    Parameters
    ----------
    detector : float ndarray, shape (S,ny,nx)
        Signed detector residuals in incident-flux units.
    shifts : float ndarray, shape (S,2,ny,nx)
        Forward source-to-detector displacements in pixels, ordered [y,x].

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Source-grid spectral residuals in incident-flux units. The adjoint
        uses the Euclidean pixel inner product and the declared forward map.

    Raises
    ------
    ValueError
        If detector and shifts do not have the documented compatible shapes.
    """
    return np.zeros(np.shape(detector), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _epr_stencil(shifts):
    (s, _, ny, nx) = shifts.shape
    (y, x) = np.indices((ny, nx))
    (yy, xx) = (y + shifts[:, 0], x + shifts[:, 1])
    (iy, ix) = (np.floor(yy).astype(int), np.floor(xx).astype(int))
    (fy, fx) = (yy - iy, xx - ix)
    return [(np.arange(s)[:, None, None], (iy + dy) % ny, (ix + dx) % nx, (fy if dy else 1 - fy) * (fx if dx else 1 - fx)) for (dy, dx) in ((0, 0), (0, 1), (1, 0), (1, 1))]

def _oracle_ray_adjoint(detector: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Apply the exact Euclidean transpose of the frozen bilinear ray map.

    Parameters
    ----------
    detector : float ndarray, shape (S,ny,nx)
        Signed detector residuals in incident-flux units.
    shifts : float ndarray, shape (S,2,ny,nx)
        Forward source-to-detector displacements in pixels, ordered [y,x].

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Source-grid residuals in incident-flux units, using the periodic
        tensor-product linear map and the Euclidean discrete inner product.

    Raises
    ------
    ValueError
        If detector and shifts do not have the documented compatible shapes.
    """
    (r, u) = (np.asarray(detector, dtype=float), np.asarray(shifts, dtype=float))
    if r.ndim != 3 or u.shape != (r.shape[0], 2, *r.shape[1:]):
        raise ValueError('Detector and displacement shapes are incompatible')
    out = np.zeros_like(r)
    for (s, y, x, w) in _epr_stencil(u):
        out += w * r[s, y, x]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'identity_adjoint',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u*=0',
      'call': 'ray_adjoint(i,u)',
      'gold_call': '_oracle_ray_adjoint(i,u)'},
     {'name': 'constant_detector_partition_unity',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.ones_like(i)',
      'call': 'ray_adjoint(i,u)',
      'gold_call': '_oracle_ray_adjoint(i,u)'},
     {'name': 'impulse_detector',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i*=0;i[:,3,4]=1;u[:,0]=.4;u[:,1]=.2',
      'call': 'ray_adjoint(i,u)',
      'gold_call': '_oracle_ray_adjoint(i,u)'},
     {'name': 'many_sources_same_residual',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u[:,0]=3-y;u[:,1]=4-x',
      'call': 'ray_adjoint(i,u)',
      'gold_call': '_oracle_ray_adjoint(i,u)'},
     {'name': 'negative_fractional_adjoint',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u[:,0]=-.4;u[:,1]=-.3',
      'call': 'ray_adjoint(i,u)',
      'gold_call': '_oracle_ray_adjoint(i,u)'},
     {'name': 'large_periodic_adjoint',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u[:,0]-=23;u[:,1]+=19',
      'call': 'ray_adjoint(i,u)',
      'gold_call': '_oracle_ray_adjoint(i,u)'},
     {'name': 'signed_nonuniform_dual_field',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.stack((np.sin(X+2*Y),np.cos(2*X-Y)))',
      'call': 'ray_adjoint(i,u)',
      'gold_call': '_oracle_ray_adjoint(i,u)'},
     {'name': 'independent_energy_maps',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u[0]*=0;u[1]*=3',
      'call': 'ray_adjoint(i,u)',
      'gold_call': '_oracle_ray_adjoint(i,u)'}]
