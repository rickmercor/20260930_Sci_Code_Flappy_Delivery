"""
Determine the common reference depth and its spectrally consistent object intensities.

Use the single-material manifold and the projection weights and floor specified in the problem.

Returns
-------
return out  # out : float ndarray, shape (S+1,ny,nx)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def thickness_projection(components: np.ndarray, fractions: np.ndarray, attenuation: np.ndarray, intensity_floor: float) -> np.ndarray:
    """Project updated spectral components to one fraction-weighted optical depth.

    Parameters
    ----------
    components : float ndarray, shape (S,ny,nx)
        Finite updated spectral intensities; signed inputs are allowed.
    fractions : float ndarray, shape (S,)
        Positive incident spectral fractions summing to one.
    attenuation : float ndarray, shape (S,)
        Positive absorption coefficients relative to the reference coefficient.
    intensity_floor : float
        Positive lower bound, less than one, on each transmission I_s/f_s.

    Returns
    -------
    out : float ndarray, shape (S+1,ny,nx)
        Row 0 is the projected dimensionless reference optical depth. Rows
        1..S are the corresponding single-material spectral intensities,
        in incident-flux units. Use the declared incident-fraction weights
        and transmission floor. The reference depth is real-valued.

    Raises
    ------
    ValueError
        If component/vector shapes disagree, a fraction or attenuation is nonpositive,
        or intensity_floor is outside (0,1).
    """
    return np.zeros((len(fractions) + 1, *np.shape(components)[-2:]), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np



def _oracle_thickness_projection(components: np.ndarray, fractions: np.ndarray, attenuation: np.ndarray, intensity_floor: float) -> np.ndarray:
    """Project updated spectral components to one fraction-weighted optical depth.

    Parameters
    ----------
    components : float ndarray, shape (S,ny,nx)
        Finite updated spectral intensities; signed inputs are allowed.
    fractions : float ndarray, shape (S,)
        Positive incident spectral fractions summing to one.
    attenuation : float ndarray, shape (S,)
        Positive absorption coefficients relative to the reference coefficient.
    intensity_floor : float
        Positive lower bound, less than one, on each transmission I_s/f_s.

    Returns
    -------
    out : float ndarray, shape (S+1,ny,nx)
        Row 0 is dimensionless optical depth: the f-weighted mean of
        -log(max(I_s/f_s,intensity_floor))/attenuation_s. Rows 1..S are
        f_s*exp(-attenuation_s*out[0]) in incident-flux units. Transmission
        may exceed one, corresponding to a negative reconstructed depth.

    Raises
    ------
    ValueError
        If component/vector shapes disagree, a fraction or attenuation is nonpositive,
        or intensity_floor is outside (0,1).
    """
    (f, g) = (np.asarray(fractions), np.asarray(attenuation))
    if np.ndim(components) != 3 or f.shape != (np.shape(components)[0],) or g.shape != f.shape or np.any(f <= 0) or np.any(g <= 0) or (not 0 < intensity_floor < 1):
        raise ValueError('Expected matching positive spectral parameters and a transmission floor in (0,1)')
    ts = np.maximum(np.asarray(components) / f[:, None, None], intensity_floor)
    tau = np.sum(-f[:, None, None] * np.log(ts) / g[:, None, None], axis=0) / f.sum()
    return np.concatenate((tau[None], f[:, None, None] * np.exp(-g[:, None, None] * tau)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'consistent_manifold',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=f[:,None,None]*np.exp(-g[:,None,None]*t)',
      'call': 'thickness_projection(i,f,g,1e-12)',
      'gold_call': '_oracle_thickness_projection(i,f,g,1e-12)'},
     {'name': 'monochromatic_logarithm',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=i[:1];f=np.ones(1);g=np.array([2.])',
      'call': 'thickness_projection(i,f,g,1e-12)',
      'gold_call': '_oracle_thickness_projection(i,f,g,1e-12)'},
     {'name': 'negative_update_floor',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i[0]-=.4',
      'call': 'thickness_projection(i,f,g,1e-4)',
      'gold_call': '_oracle_thickness_projection(i,f,g,1e-4)'},
     {'name': 'superunit_transmission',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.stack((f[0]*np.ones((7,9))*2,f[1]*np.ones((7,9))*3))',
      'call': 'thickness_projection(i,f,g,1e-12)',
      'gold_call': '_oracle_thickness_projection(i,f,g,1e-12)'},
     {'name': 'weak_line_weighting',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'f=np.array([.001,.999])',
      'call': 'thickness_projection(i,f,g,1e-12)',
      'gold_call': '_oracle_thickness_projection(i,f,g,1e-12)'},
     {'name': 'disparate_absorption_scales',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'g=np.array([.03,12.])',
      'call': 'thickness_projection(i,f,g,1e-12)',
      'gold_call': '_oracle_thickness_projection(i,f,g,1e-12)'},
     {'name': 'inconsistent_spectral_depths',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=f[:,None,None]*np.exp(-g[:,None,None]*np.stack((t,2*t)))',
      'call': 'thickness_projection(i,f,g,1e-12)',
      'gold_call': '_oracle_thickness_projection(i,f,g,1e-12)'},
     {'name': 'same_total_different_spectrum',
      'setup': 'import numpy as np\n'
               'y,x=np.indices((7,9)); X=2*np.pi*x/9; Y=2*np.pi*y/7\n'
               't=.5+.1*np.cos(X)+.07*np.sin(Y)+.04*np.cos(X+Y)\n'
               'f=np.array([.25,.75]); g=np.array([1.7,.6]); e=np.array([.3,.8]); b=np.array([12.,9.])\n'
               'i=np.stack((.3+.05*np.cos(X)+.03*np.sin(Y), .5+.07*np.cos(X-Y)))\n'
               'u=np.stack((np.stack((.4*np.sin(Y),.6*np.cos(X))), '
               'np.stack((.3*np.cos(X-Y),-.7*np.sin(X)))))\n'
               'i=np.stack((.15+.03*np.cos(X),.65-.03*np.cos(X)))',
      'call': 'thickness_projection(i,f,g,1e-12)',
      'gold_call': '_oracle_thickness_projection(i,f,g,1e-12)'}]
