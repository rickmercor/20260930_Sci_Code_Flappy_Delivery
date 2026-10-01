"""
Evaluate the leading wave correction to the source intensity under the declared paraxial convention.

Use the leading WKB1 intensity correction. Its amplitude Laplacian follows the benchmark convention.

Returns
-------
return out  # out : float ndarray, shape (S,ny,nx)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def diffraction_pressure(intensity: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Evaluate the source-plane leading WKB1 diffraction-pressure correction.

    Parameters
    ----------
    intensity : float ndarray, shape (S,ny,nx)
        Strictly positive spectral object intensities in incident-flux units.
    eta : float ndarray, shape (S,)
        Nonnegative dimensionless propagation parameters L/(k_s*p**2).

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Signed leading wave correction in incident-flux units on the source
        grid. Spatial derivatives use the discretization in the main prompt.

    Raises
    ------
    ValueError
        If intensity is not a positive three-dimensional array or eta length differs.
    """
    return np.zeros(np.shape(intensity), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _epr_d(a, axis):
    return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / 2.0

def _epr_lap(a):
    return sum((np.roll(a, -1, axis=j) + np.roll(a, 1, axis=j) - 2 * a for j in (-2, -1)))

def _oracle_diffraction_pressure(intensity: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Evaluate the source-plane leading WKB1 diffraction-pressure correction.

    Parameters
    ----------
    intensity : float ndarray, shape (S,ny,nx)
        Strictly positive spectral object intensities in incident-flux units.
    eta : float ndarray, shape (S,)
        Nonnegative dimensionless propagation parameters L/(k_s*p**2).

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Signed additive O(eta**2) intensity correction. Q is the nearest-
        neighbor five-point periodic Laplacian of sqrt(I), divided by sqrt(I).
        Gradient and divergence use centered periodic first differences.
        The result is in incident-flux units on the source grid.

    Raises
    ------
    ValueError
        If intensity is not a positive three-dimensional array or eta length differs.
    """
    i = np.asarray(intensity, dtype=float)
    if i.ndim != 3 or np.shape(eta) != (i.shape[0],) or np.any(i <= 0):
        raise ValueError('Pressure requires positive spectral intensities and matching eta')
    root = np.sqrt(i)
    q = _epr_lap(root) / root
    div = _epr_d(i * _epr_d(q, -2), -2) + _epr_d(i * _epr_d(q, -1), -1)
    return -0.25 * np.asarray(eta, dtype=float)[:, None, None] ** 2 * div

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'uniform_amplitude',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.ones_like(i)*.4',
      'call': 'diffraction_pressure(i,e)',
      'gold_call': '_oracle_diffraction_pressure(i,e)'},
     {'name': 'zero_propagation',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'e*=0',
      'call': 'diffraction_pressure(i,e)',
      'gold_call': '_oracle_diffraction_pressure(i,e)'},
     {'name': 'x_curvature',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.stack((.4+.2*np.cos(X),.7+.1*np.sin(2*X)))',
      'call': 'diffraction_pressure(i,e)',
      'gold_call': '_oracle_diffraction_pressure(i,e)'},
     {'name': 'y_curvature',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.stack((.4+.2*np.cos(Y),.7+.1*np.sin(2*Y)))',
      'call': 'diffraction_pressure(i,e)',
      'gold_call': '_oracle_diffraction_pressure(i,e)'},
     {'name': 'oblique_curvature',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n',
      'call': 'diffraction_pressure(i,e)',
      'gold_call': '_oracle_diffraction_pressure(i,e)'},
     {'name': 'nyquist_laplacian_gradient_null',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'y,x=np.indices((6,8));i=(1+.2*(-1.)**(x+y))[None];e=np.array([.5])',
      'call': 'diffraction_pressure(i,e)',
      'gold_call': '_oracle_diffraction_pressure(i,e)'},
     {'name': 'steep_positive_amplitude',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.stack((np.exp(.9*np.cos(2*X+Y)),np.exp(-.7*np.sin(X-2*Y))))',
      'call': 'diffraction_pressure(i,e)',
      'gold_call': '_oracle_diffraction_pressure(i,e)'},
     {'name': 'homogeneous_intensity_scaling',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.stack((i[0],7*i[0]));e=np.array([.3,.3])',
      'call': 'diffraction_pressure(i,e)',
      'gold_call': '_oracle_diffraction_pressure(i,e)'}]
