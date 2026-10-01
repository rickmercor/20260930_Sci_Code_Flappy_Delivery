"""
Determine the per-line object intensities and transverse ray displacements for the single-material object.

The incident spectrum is spatially uniform. The material and propagation conventions are defined in the main background and the source paper.

Returns
-------
return out  # out : float ndarray, shape (S,3,ny,nx)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def spectral_rays(tau: np.ndarray, fractions: np.ndarray, attenuation: np.ndarray, eta: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Construct single-material spectral intensities and dimensionless ray shifts.

    Parameters
    ----------
    tau : float ndarray, shape (ny,nx)
        Finite reference optical depth (dimensionless), ny,nx >= 3.
    fractions : float ndarray, shape (S,)
        Strictly positive incident line fractions summing to one.
    attenuation : float ndarray, shape (S,)
        Positive absorption coefficients relative to the reference coefficient.
    eta : float ndarray, shape (S,)
        Nonnegative L/(k_s*p**2), in squared-pixel units normalized by p**2.
    db : float ndarray, shape (S,)
        Nonnegative refractive ratio delta_s/beta_s.

    Returns
    -------
    out : float ndarray, shape (S,3,ny,nx)
        Axis 1 is [intensity, y displacement, x displacement]; intensity is
        relative to incident total flux, displacements are in pixels. Spatial
        derivatives are centered periodic first differences on unit pitch.

    Raises
    ------
    ValueError
        If tau is not two-dimensional or the spectral vector lengths differ.
    """
    return np.zeros((len(fractions), 3, *np.shape(tau)), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _epr_d(a, axis):
    return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / 2.0

def _oracle_spectral_rays(tau: np.ndarray, fractions: np.ndarray, attenuation: np.ndarray, eta: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Construct single-material spectral intensities and dimensionless ray shifts.

    Parameters
    ----------
    tau : float ndarray, shape (ny,nx)
        Finite reference optical depth (dimensionless), ny,nx >= 3.
    fractions : float ndarray, shape (S,)
        Strictly positive incident line fractions summing to one.
    attenuation : float ndarray, shape (S,)
        Positive absorption coefficients relative to the reference coefficient.
    eta : float ndarray, shape (S,)
        Nonnegative L/(k_s*p**2), in squared-pixel units normalized by p**2.
    db : float ndarray, shape (S,)
        Nonnegative delta_s/beta_s. Phase is db_s*log(I_s/f_s)/2.

    Returns
    -------
    out : float ndarray, shape (S,3,ny,nx)
        Axis 1 is [intensity, y displacement, x displacement]; intensity is
        relative to incident total flux, displacements are in pixels. Spatial
        derivatives are centered periodic first differences on unit pitch.

    Raises
    ------
    ValueError
        If tau is not two-dimensional or the spectral vector lengths differ.
    """
    t = np.asarray(tau, dtype=float)
    (f, g, e, b) = (np.asarray(v, dtype=float) for v in (fractions, attenuation, eta, db))
    if t.ndim != 2 or not f.ndim == g.ndim == e.ndim == b.ndim == 1 or (not f.shape == g.shape == e.shape == b.shape):
        raise ValueError('Expected a two-dimensional depth and equal-length spectral vectors')
    intensity = f[:, None, None] * np.exp(-g[:, None, None] * t)
    phase = -0.5 * (b * g)[:, None, None] * t
    return np.stack((intensity, e[:, None, None] * _epr_d(phase, -2), e[:, None, None] * _epr_d(phase, -1)), axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'oblique_polychromatic',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n',
      'call': 'spectral_rays(t,f,g,e,b)',
      'gold_call': '_oracle_spectral_rays(t,f,g,e,b)'},
     {'name': 'uniform_slab',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               't=np.full((7,9),.8)',
      'call': 'spectral_rays(t,f,g,e,b)',
      'gold_call': '_oracle_spectral_rays(t,f,g,e,b)'},
     {'name': 'vacuum_normalization',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               't=np.zeros((7,9))',
      'call': 'spectral_rays(t,f,g,e,b)',
      'gold_call': '_oracle_spectral_rays(t,f,g,e,b)'},
     {'name': 'contact_plane',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'e=np.zeros(2)',
      'call': 'spectral_rays(t,f,g,e,b)',
      'gold_call': '_oracle_spectral_rays(t,f,g,e,b)'},
     {'name': 'pure_absorber',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'b=np.zeros(2)',
      'call': 'spectral_rays(t,f,g,e,b)',
      'gold_call': '_oracle_spectral_rays(t,f,g,e,b)'},
     {'name': 'opaque_line_dynamic_range',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'g=np.array([40.,.02])',
      'call': 'spectral_rays(t,f,g,e,b)',
      'gold_call': '_oracle_spectral_rays(t,f,g,e,b)'},
     {'name': 'negative_reconstructed_depth',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               't=t-.8',
      'call': 'spectral_rays(t,f,g,e,b)',
      'gold_call': '_oracle_spectral_rays(t,f,g,e,b)'},
     {'name': 'monochromatic_transposed_grid',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               't=t.T; f=np.ones(1);g=np.array([1.2]);e=np.array([.7]);b=np.array([5.])',
      'call': 'spectral_rays(t,f,g,e,b)',
      'gold_call': '_oracle_spectral_rays(t,f,g,e,b)'}]
