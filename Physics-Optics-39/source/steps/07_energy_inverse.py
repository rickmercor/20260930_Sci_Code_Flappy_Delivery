"""
Evaluate the per-line homogeneous-object inverse on a source residual.

The linearized homogeneous-object inverse uses the continuous-Laplacian FFT convention stated in the problem.

Returns
-------
return out  # out : float ndarray, shape (S,ny,nx)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def energy_inverse(residual: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply the energy-resolved FFT-diagonal homogeneous-object preconditioner.

    Parameters
    ----------
    residual : float ndarray, shape (S,ny,nx)
        Signed source residuals in incident-flux units.
    coefficients : float ndarray, shape (S,)
        Nonnegative a_s=eta_s*db_s/2, measured in pixel squared units.

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Real per-line inverse responses, in the input residual units, using
        the declared continuous-Laplacian and NumPy FFT conventions.

    Raises
    ------
    ValueError
        If residual is not three-dimensional, coefficients have the wrong length,
        or a coefficient is negative.
    """
    return np.zeros(np.shape(residual), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np



def _oracle_energy_inverse(residual: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply the energy-resolved FFT-diagonal homogeneous-object preconditioner.

    Parameters
    ----------
    residual : float ndarray, shape (S,ny,nx)
        Signed source residuals in incident-flux units.
    coefficients : float ndarray, shape (S,)
        Nonnegative a_s=eta_s*db_s/2, measured in pixel squared units.

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Real filtered residuals, same units. The multiplier is the inverse of
        1+a_s*(qx**2+qy**2), qj=2*pi*fftfreq(nj), with NumPy FFT normalization.

    Raises
    ------
    ValueError
        If residual is not three-dimensional, coefficients have the wrong length,
        or a coefficient is negative.
    """
    r = np.asarray(residual, dtype=float)
    if r.ndim != 3 or np.shape(coefficients) != (r.shape[0],) or np.any(np.asarray(coefficients) < 0):
        raise ValueError('Expected a spectral residual and matching nonnegative coefficients')
    (ny, nx) = r.shape[-2:]
    q2 = (2 * np.pi * np.fft.fftfreq(ny))[:, None] ** 2 + (2 * np.pi * np.fft.fftfreq(nx))[None, :] ** 2
    return np.fft.ifft2(np.fft.fft2(r) / (1 + np.asarray(coefficients)[:, None, None] * q2)).real

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'dc_preserved',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.ones_like(i);a=np.array([.4,30.])',
      'call': 'energy_inverse(i,a)',
      'gold_call': '_oracle_energy_inverse(i,a)'},
     {'name': 'contact_filter_identity',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'a=np.zeros(2)',
      'call': 'energy_inverse(i,a)',
      'gold_call': '_oracle_energy_inverse(i,a)'},
     {'name': 'x_eigenmode',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.stack((np.cos(X),np.sin(2*X)));a=np.array([2.,7.])',
      'call': 'energy_inverse(i,a)',
      'gold_call': '_oracle_energy_inverse(i,a)'},
     {'name': 'y_eigenmode',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.stack((np.cos(Y),np.sin(2*Y)));a=np.array([2.,7.])',
      'call': 'energy_inverse(i,a)',
      'gold_call': '_oracle_energy_inverse(i,a)'},
     {'name': 'oblique_eigenmode',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.stack((np.cos(X+Y),np.sin(2*X-Y)));a=np.array([.3,5.])',
      'call': 'energy_inverse(i,a)',
      'gold_call': '_oracle_energy_inverse(i,a)'},
     {'name': 'even_grid_nyquist',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'y,x=np.indices((6,8));i=np.stack(((-1.)**x,(-1.)**(x+y)));a=np.array([.2,3.])',
      'call': 'energy_inverse(i,a)',
      'gold_call': '_oracle_energy_inverse(i,a)'},
     {'name': 'signed_multimode_residual',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i-=.45;a=np.array([.7,4.])',
      'call': 'energy_inverse(i,a)',
      'gold_call': '_oracle_energy_inverse(i,a)'},
     {'name': 'impulse_green_function',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i*=0;i[:,0,0]=1;a=np.array([.02,80.])',
      'call': 'energy_inverse(i,a)',
      'gold_call': '_oracle_energy_inverse(i,a)'}]
