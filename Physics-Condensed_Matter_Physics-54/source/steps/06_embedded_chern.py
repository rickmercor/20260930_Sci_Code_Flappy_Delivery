"""
Evaluate the open-ribbon local Chern marker from the sampled projector in the cell-periodic Bloch basis. Differentiate the sampled projector in momentum with the source method's prescription for the hybrid marker, combine it with the intra-cell y embedding, use physical open-x coordinates, and retain the primitive-cell-area normalization.

Section II A gives the hybrid local Chern marker. Here physical coordinates use an oblique primitive basis and a cell-periodic Bloch transform. Consequently -i[y,P] becomes partial_k P-i[Y,P]; this embedding correction is derived from the task convention, rather than quoted as the paper’s formula in the same gauge. The area factor is needed for the chosen cell geometry.

Returns
-------
Real NumPy local-Chern-marker array of shape (nx,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def embedded_chern(p: np.ndarray) -> np.ndarray:
    """Evaluate the open-x marker in the cell-periodic Bloch gauge.

    Parameters
    ----------
    p : numpy.ndarray
        Finite complex stack (ny,2*nx,2*nx), ny>=5, nx>=4.
        Samples at k_j=2*pi*j/ny; general matrices allowed.

    Returns
    -------
    numpy.ndarray
        Real marker per cell (nx,), including orbital embedding and area.
        X_(x,A)=x*sqrt(3)/2, X_(x,B)=X_(x,A)+1/sqrt(3);
        Y_(x,alpha)=x/2; Acell=sqrt(3)/2. Differentiate the sampled P in k
        with the source method's prescription for this hybrid marker.
        D=partial_k P-i[Y,P], A=-i[X,P], and return
        Re[(2*pi*i/(ny*Acell))*sum_(j,alpha) diag(P_j[A_j,D_j])].
        X differences are ordinary open-ribbon differences, never wrapped.

    Raises
    ------
    ValueError
        Invalid shape or nonfinite input.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_embedded_chern(p: np.ndarray) -> np.ndarray:
    """Evaluate the open-x marker in the cell-periodic Bloch gauge.

    Parameters
    ----------
    p : numpy.ndarray
        Finite complex stack (ny,2*nx,2*nx), ny>=5, nx>=4.
        Samples at k_j=2*pi*j/ny; general matrices allowed.

    Returns
    -------
    numpy.ndarray
        Real marker per cell (nx,), including orbital embedding and area.
        X_(x,A)=x*sqrt(3)/2, X_(x,B)=X_(x,A)+1/sqrt(3);
        Y_(x,alpha)=x/2; Acell=sqrt(3)/2. Differentiate the sampled P in k
        with the source method's prescription for this hybrid marker.
        D=partial_k P-i[Y,P], A=-i[X,P], and return
        Re[(2*pi*i/(ny*Acell))*sum_(j,alpha) diag(P_j[A_j,D_j])].
        X differences are ordinary open-ribbon differences, never wrapped.

    Raises
    ------
    ValueError
        Invalid shape or nonfinite input.
    """
    p=np.asarray(p,complex)
    if p.ndim!=3 or p.shape[0]<5 or p.shape[1]!=p.shape[2] or p.shape[1]<8 or p.shape[1]%2 or not np.all(np.isfinite(p)):
        raise ValueError('invalid projector samples')
    ny,dim,_=p.shape; nx=dim//2; area=np.sqrt(3)/2
    X=np.repeat(np.arange(nx)*area,2)+np.tile([0,1/np.sqrt(3)],nx)
    Y=np.repeat(np.arange(nx)/2,2)
    modes=np.fft.fftfreq(ny,1/ny)
    if ny%2==0: modes[ny//2]=0
    deriv=np.fft.ifft((1j*modes)[:,None,None]*np.fft.fft(p,axis=0),axis=0)
    A=-1j*(X[:,None]-X[None,:])*p
    D=deriv-1j*(Y[:,None]-Y[None,:])*p
    diag=np.einsum('kij,kji->ki',p,A@D-D@A)
    return (2j*np.pi/area*diag.reshape(ny,nx,2).sum(axis=2).mean(axis=0)).real

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'p_c = fixed_mu_projectors(magnetic_ribbon(2 * np.pi * np.arange(33) / 33, '
               'np.linspace(-0.1, 0.1, 6)), 0.08)\n'
               'p_g = _oracle_fixed_mu_projectors(_oracle_magnetic_ribbon(2 * np.pi * np.arange(33) / '
               '33, np.linspace(-0.1, 0.1, 6)), 0.08)\n',
      'call': 'embedded_chern(deepcopy(p_c))',
      'gold_call': '_oracle_embedded_chern(deepcopy(p_g))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'p_c = fixed_mu_projectors(magnetic_ribbon(2 * np.pi * np.arange(24) / 24, '
               'np.linspace(0.1, -0.1, 7)), 0.02)\n'
               'p_g = _oracle_fixed_mu_projectors(_oracle_magnetic_ribbon(2 * np.pi * np.arange(24) / '
               '24, np.linspace(0.1, -0.1, 7)), 0.02)\n',
      'call': 'embedded_chern(deepcopy(p_c))',
      'gold_call': '_oracle_embedded_chern(deepcopy(p_g))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'rng = np.random.default_rng(18)\n'
               'p = rng.normal(size=(10, 8, 8)) + 1j * rng.normal(size=(10, 8, 8))\n',
      'call': 'embedded_chern(deepcopy(p))',
      'gold_call': '_oracle_embedded_chern(deepcopy(p))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: embedded_chern(deepcopy(np.zeros((4, 8, 8)))))',
      'gold_call': 'raises_value_error(lambda: _oracle_embedded_chern(deepcopy(np.zeros((4, 8, 8)))))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: embedded_chern(deepcopy(np.zeros((5, 7, 7)))))',
      'gold_call': 'raises_value_error(lambda: _oracle_embedded_chern(deepcopy(np.zeros((5, 7, 7)))))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: embedded_chern(deepcopy(np.zeros((5, 8, 9)))))',
      'gold_call': 'raises_value_error(lambda: _oracle_embedded_chern(deepcopy(np.zeros((5, 8, 9)))))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: embedded_chern(deepcopy(np.full((5, 8, 8), np.nan))))',
      'gold_call': 'raises_value_error(lambda: _oracle_embedded_chern(deepcopy(np.full((5, 8, 8), '
                   'np.nan))))'}]
