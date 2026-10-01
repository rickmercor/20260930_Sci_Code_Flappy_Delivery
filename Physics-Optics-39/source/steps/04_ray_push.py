"""
Evaluate the detector-plane spectral intensity under the declared finite-displacement transport model.

The non-local model and its grid specialization are defined in the problem statement and source paper.

Returns
-------
return out  # out : float ndarray, shape (S,ny,nx)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def ray_push(source: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Conservatively redistribute source mass along the finite eikonal map.

    Parameters
    ----------
    source : float ndarray, shape (S,ny,nx)
        Signed spectral source masses in incident-flux units.
    shifts : float ndarray, shape (S,2,ny,nx)
        Finite mapped displacements in pixels, ordered [y,x].

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Detector-plane spectral masses under the periodic bilinear
        finite-displacement model. Units and spectral order match source.

    Raises
    ------
    ValueError
        If source and shifts do not have the documented compatible shapes.
    """
    return np.zeros(np.shape(source), dtype=float)

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

def _oracle_ray_push(source: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Conservatively redistribute source mass along the finite eikonal map.

    Parameters
    ----------
    source : float ndarray, shape (S,ny,nx)
        Signed spectral source masses in incident-flux units. Signed values
        permit transporting an additive diffraction-pressure correction.
    shifts : float ndarray, shape (S,2,ny,nx)
        Finite mapped displacements in pixels, ordered [y,x].

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Detector masses, accumulated with tensor-product linear weights at
        floor(y+uy), floor(x+ux) and their upper neighbors, modulo ny,nx.
        Repeated target indices accumulate. Units match source.

    Raises
    ------
    ValueError
        If source and shifts do not have the documented compatible shapes.
    """
    (a, u) = (np.asarray(source, dtype=float), np.asarray(shifts, dtype=float))
    if a.ndim != 3 or u.shape != (a.shape[0], 2, *a.shape[1:]):
        raise ValueError('Source and displacement shapes are incompatible')
    out = np.zeros_like(a)
    for (s, y, x, w) in _epr_stencil(u):
        np.add.at(out, (s, y, x), a * w)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'identity_map',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u*=0',
      'call': 'ray_push(i,u)',
      'gold_call': '_oracle_ray_push(i,u)'},
     {'name': 'integer_translation_wrap',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u[:,0]=2;u[:,1]=-3',
      'call': 'ray_push(i,u)',
      'gold_call': '_oracle_ray_push(i,u)'},
     {'name': 'four_corner_subpixel_split',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i*=0;i[:,3,4]=1;u[:,0]=.25;u[:,1]=.75',
      'call': 'ray_push(i,u)',
      'gold_call': '_oracle_ray_push(i,u)'},
     {'name': 'many_sources_one_target',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u[:,0]=3-y;u[:,1]=4-x',
      'call': 'ray_push(i,u)',
      'gold_call': '_oracle_ray_push(i,u)'},
     {'name': 'negative_fractional_floor',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u[:,0]=-.2;u[:,1]=-.7',
      'call': 'ray_push(i,u)',
      'gold_call': '_oracle_ray_push(i,u)'},
     {'name': 'multi_period_wrap',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u[:,0]+=15;u[:,1]-=28',
      'call': 'ray_push(i,u)',
      'gold_call': '_oracle_ray_push(i,u)'},
     {'name': 'signed_pressure_mass',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.stack((np.cos(X+Y),np.sin(X-Y)))',
      'call': 'ray_push(i,u)',
      'gold_call': '_oracle_ray_push(i,u)'},
     {'name': 'energy_resolved_nonuniform_map',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n',
      'call': 'ray_push(i,u)',
      'gold_call': '_oracle_ray_push(i,u)'}]
