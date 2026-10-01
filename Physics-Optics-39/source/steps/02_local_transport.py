"""
Evaluate the local geometrical detector-intensity model through quadratic propagation order.

Use the source paper's local WKB0 model and the benchmark's spatial derivative convention.

Returns
-------
return out  # out : float ndarray, shape (S,ny,nx)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def local_transport(intensity: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Evaluate the local second-order WKB0 intensity closure.

    Parameters
    ----------
    intensity : float ndarray, shape (S,ny,nx)
        Positive object-plane intensities, normalized to total incident flux.
    shifts : float ndarray, shape (S,2,ny,nx)
        Per-line dimensionless displacements ordered [y,x].

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Local WKB0 detector predictions in incident-flux units, through
        quadratic order in shifts. Every derivative is the centered periodic
        first difference; second derivatives compose that same operator.
        This output is the geometrical contribution; the wave correction is a separate output.

    Raises
    ------
    ValueError
        If intensity and shifts do not have the documented compatible shapes.
    """
    return np.zeros(np.shape(intensity), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _epr_d(a, axis):
    return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / 2.0

def _oracle_local_transport(intensity: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Evaluate the local second-order WKB0 intensity closure.

    Parameters
    ----------
    intensity : float ndarray, shape (S,ny,nx)
        Positive object-plane intensities, normalized to total incident flux.
    shifts : float ndarray, shape (S,2,ny,nx)
        Per-line dimensionless displacements ordered [y,x].

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Local WKB0 detector predictions in incident-flux units, through
        quadratic order in shifts. Every derivative is the centered periodic
        first difference; second derivatives compose that same operator.
        This output excludes the separately evaluated pressure correction.

    Raises
    ------
    ValueError
        If intensity and shifts do not have the documented compatible shapes.
    """
    i = np.asarray(intensity, dtype=float)
    if i.ndim != 3 or np.shape(shifts) != (i.shape[0], 2, *i.shape[1:]):
        raise ValueError('Expected spectral intensities and [y,x] shifts on the same grid')
    (v, u) = (np.asarray(shifts, dtype=float)[:, 0], np.asarray(shifts, dtype=float)[:, 1])
    return i - _epr_d(i * v, -2) - _epr_d(i * u, -1) + 0.5 * _epr_d(_epr_d(i * v * v, -2), -2) + _epr_d(_epr_d(i * v * u, -2), -1) + 0.5 * _epr_d(_epr_d(i * u * u, -1), -1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'zero_distance_identity',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u*=0',
      'call': 'local_transport(i,u)',
      'gold_call': '_oracle_local_transport(i,u)'},
     {'name': 'uniform_translation',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.ones_like(i);u[:]=.4',
      'call': 'local_transport(i,u)',
      'gold_call': '_oracle_local_transport(i,u)'},
     {'name': 'x_only_flux',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u[:,0]=0',
      'call': 'local_transport(i,u)',
      'gold_call': '_oracle_local_transport(i,u)'},
     {'name': 'y_only_flux',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u[:,1]=0',
      'call': 'local_transport(i,u)',
      'gold_call': '_oracle_local_transport(i,u)'},
     {'name': 'mixed_stress_oblique',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n',
      'call': 'local_transport(i,u)',
      'gold_call': '_oracle_local_transport(i,u)'},
     {'name': 'sheared_rays',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u[:,0]=.5*np.cos(X);u[:,1]=.7*np.sin(Y)',
      'call': 'local_transport(i,u)',
      'gold_call': '_oracle_local_transport(i,u)'},
     {'name': 'nyquist_composed_derivative',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'y,x=np.indices((6,8));i=(1+.1*(-1.)**(x+y))[None];u=np.ones((1,2,6,8))*.3',
      'call': 'local_transport(i,u)',
      'gold_call': '_oracle_local_transport(i,u)'},
     {'name': 'multi_pixel_local_stress',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'u*=5',
      'call': 'local_transport(i,u)',
      'gold_call': '_oracle_local_transport(i,u)'}]
