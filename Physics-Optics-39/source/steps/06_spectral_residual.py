"""
Determine the spectral residuals associated with the measured detector image.

Use the benchmark's spectral-consistency convention, expressed in incident-total-flux units.

Returns
-------
return out  # out : float ndarray, shape (S,ny,nx)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def spectral_residual(measured: np.ndarray, predicted: np.ndarray, weight_floor: float) -> np.ndarray:
    """Evaluate the spectral detector residuals.

    Parameters
    ----------
    measured : float ndarray, shape (ny,nx)
        Positive total measured intensity in incident-flux units.
    predicted : float ndarray, shape (S,ny,nx)
        Signed per-line forward predictions in the same units.
    weight_floor : float
        Strictly positive intensity added once to the positive-flux denominator.

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Per-line detector residuals in incident-flux units, with the
        spectral-consistency convention defined in the main prompt.

    Raises
    ------
    ValueError
        If the array shapes are incompatible or weight_floor is not positive.
    """
    return np.zeros(np.shape(predicted), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np



def _oracle_spectral_residual(measured: np.ndarray, predicted: np.ndarray, weight_floor: float) -> np.ndarray:
    """Allocate an incoherent detector mismatch using positive predicted flux.

    Parameters
    ----------
    measured : float ndarray, shape (ny,nx)
        Positive total measured intensity in incident-flux units.
    predicted : float ndarray, shape (S,ny,nx)
        Signed per-line forward predictions in the same units.
    weight_floor : float
        Strictly positive intensity added once to the positive-flux denominator.

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Per-line detector residuals in incident-flux units, using the measured
        minus incoherent predicted total and normalized positive predictions.

    Raises
    ------
    ValueError
        If the array shapes are incompatible or weight_floor is not positive.
    """
    p = np.asarray(predicted, dtype=float)
    if p.ndim != 3 or np.shape(measured) != p.shape[1:] or weight_floor <= 0:
        raise ValueError('Expected matching detector grids and a positive denominator floor')
    positive = np.maximum(p, 0)
    return positive / (positive.sum(axis=0) + weight_floor) * (measured - p.sum(axis=0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'positive_flux_partition',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'm=np.ones((7,9));h=1e-12',
      'call': 'spectral_residual(m,i,h)',
      'gold_call': '_oracle_spectral_residual(m,i,h)'},
     {'name': 'one_negative_line',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i[0]*=-1;m=np.ones((7,9));h=1e-12',
      'call': 'spectral_residual(m,i,h)',
      'gold_call': '_oracle_spectral_residual(m,i,h)'},
     {'name': 'all_negative_lines',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=-np.abs(i);m=np.ones((7,9));h=1e-12',
      'call': 'spectral_residual(m,i,h)',
      'gold_call': '_oracle_spectral_residual(m,i,h)'},
     {'name': 'signed_cancellation_total',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i[1]=-i[0];m=np.ones((7,9));h=1e-12',
      'call': 'spectral_residual(m,i,h)',
      'gold_call': '_oracle_spectral_residual(m,i,h)'},
     {'name': 'exact_detector_match',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'm=i.sum(axis=0);h=1e-12',
      'call': 'spectral_residual(m,i,h)',
      'gold_call': '_oracle_spectral_residual(m,i,h)'},
     {'name': 'denominator_floor_dominates',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i*=1e-14;m=np.ones((7,9));h=1e-12',
      'call': 'spectral_residual(m,i,h)',
      'gold_call': '_oracle_spectral_residual(m,i,h)'},
     {'name': 'monochromatic_residual',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=i[:1];m=np.ones((7,9));h=.01',
      'call': 'spectral_residual(m,i,h)',
      'gold_call': '_oracle_spectral_residual(m,i,h)'},
     {'name': 'pixelwise_active_set_changes',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i-=.4;m=.9+.03*np.cos(X);h=.1',
      'call': 'spectral_residual(m,i,h)',
      'gold_call': '_oracle_spectral_residual(m,i,h)'}]
