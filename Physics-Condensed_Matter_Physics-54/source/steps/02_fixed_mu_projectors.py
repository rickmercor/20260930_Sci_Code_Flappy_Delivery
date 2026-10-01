"""
Build each occupied spectral projector with the strict condition E<mu. Support zero, full and momentum-dependent occupied ranks. The result must be independent of eigenvector phases and rotations within occupied degenerate subspaces.

Section II B defines the Streda density response at fixed chemical potential. Edge bands can cross that chemical potential, so imposing half the states at every momentum would change the physical response. A spectral projector retains the occupied subspace without eigenvector-gauge ambiguity.

Returns
-------
Complex NumPy projector stack with the same shape as h.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fixed_mu_projectors(h: np.ndarray, mu: float) -> np.ndarray:
    """Construct zero-temperature occupied projectors at fixed chemical potential.

    Parameters
    ----------
    h : numpy.ndarray
        Finite Hermitian stack (m,d,d), m>=1, even d>=2, Hermiticity
        absolute error <=1e-10. Occupied rank may differ between blocks.
    mu : float
        Finite real chemical potential, held fixed across fluxes.

    Returns
    -------
    numpy.ndarray
        Complex stack matching h, spectral projector onto eigenvalues E<mu.
        E=mu is excluded. Empty and full occupied subspaces are allowed.
        Return projectors, not eigenvectors; never force rank d/2.

    Raises
    ------
    ValueError
        Invalid shape, nonfinite/non-Hermitian h or invalid mu.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_fixed_mu_projectors(h: np.ndarray, mu: float) -> np.ndarray:
    """Construct zero-temperature occupied projectors at fixed chemical potential.

    Parameters
    ----------
    h : numpy.ndarray
        Finite Hermitian stack (m,d,d), m>=1, even d>=2, Hermiticity
        absolute error <=1e-10. Occupied rank may differ between blocks.
    mu : float
        Finite real chemical potential, held fixed across fluxes.

    Returns
    -------
    numpy.ndarray
        Complex stack matching h, spectral projector onto eigenvalues E<mu.
        E=mu is excluded. Empty and full occupied subspaces are allowed.
        Return projectors, not eigenvectors; never force rank d/2.

    Raises
    ------
    ValueError
        Invalid shape, nonfinite/non-Hermitian h or invalid mu.
    """
    h=np.asarray(h,complex)
    if h.ndim!=3 or h.shape[0]<1 or h.shape[1]!=h.shape[2] or h.shape[1]<2 or h.shape[1]%2:
        raise ValueError('invalid Hamiltonian shape')
    if not np.all(np.isfinite(h)) or np.max(np.abs(h-h.swapaxes(-1,-2).conj()))>1e-10:
        raise ValueError('invalid Hermitian matrices')
    if not np.isscalar(mu) or not np.isrealobj(mu) or not np.isfinite(mu):
        raise ValueError('invalid chemical potential')
    e,v=np.linalg.eigh(h)
    return (v*(e<mu)[:,None,:])@v.swapaxes(-1,-2).conj()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'import numpy as np\n'
               '\n'
               'def pack(a):\n'
               '    a = np.asarray(a)\n'
               '    return np.stack((a.real, a.imag), axis=-1)\n'
               'h = np.array([np.diag([-2.0, 0.0, 1.0, 3.0]), np.diag([-3.0, -2.0, -1.0, 0.0]), '
               'np.eye(4), -np.eye(4)])\n',
      'call': 'pack(fixed_mu_projectors(deepcopy(h), deepcopy(0.0)))',
      'gold_call': 'pack(_oracle_fixed_mu_projectors(deepcopy(h), deepcopy(0.0)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'import numpy as np\n'
               '\n'
               'def pack(a):\n'
               '    a = np.asarray(a)\n'
               '    return np.stack((a.real, a.imag), axis=-1)\n'
               'rng = np.random.default_rng(71)\n'
               'a = rng.normal(size=(3, 6, 6)) + 1j * rng.normal(size=(3, 6, 6))\n'
               'h = a + a.swapaxes(-1, -2).conj()\n',
      'call': 'pack(fixed_mu_projectors(deepcopy(h), deepcopy(0.15)))',
      'gold_call': 'pack(_oracle_fixed_mu_projectors(deepcopy(h), deepcopy(0.15)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'import numpy as np\n'
               '\n'
               'def pack(a):\n'
               '    a = np.asarray(a)\n'
               '    return np.stack((a.real, a.imag), axis=-1)\n'
               'h_c = magnetic_ribbon(np.linspace(0, 6, 7), np.linspace(-0.1, 0.1, 6), 0.01)\n'
               'h_g = _oracle_magnetic_ribbon(np.linspace(0, 6, 7), np.linspace(-0.1, 0.1, 6), 0.01)\n',
      'call': 'pack(fixed_mu_projectors(deepcopy(h_c), deepcopy(0.08)))',
      'gold_call': 'pack(_oracle_fixed_mu_projectors(deepcopy(h_g), deepcopy(0.08)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: fixed_mu_projectors(deepcopy(np.zeros((1, 3, 3))), '
              'deepcopy(0.0)))',
      'gold_call': 'raises_value_error(lambda: _oracle_fixed_mu_projectors(deepcopy(np.zeros((1, 3, '
                   '3))), deepcopy(0.0)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: fixed_mu_projectors(deepcopy(np.array([[[0, 1], [0, 0]]])), '
              'deepcopy(0.0)))',
      'gold_call': 'raises_value_error(lambda: _oracle_fixed_mu_projectors(deepcopy(np.array([[[0, 1], '
                   '[0, 0]]])), deepcopy(0.0)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: fixed_mu_projectors(deepcopy(np.zeros((1, 2, 2))), '
              'deepcopy(np.inf)))',
      'gold_call': 'raises_value_error(lambda: _oracle_fixed_mu_projectors(deepcopy(np.zeros((1, 2, '
                   '2))), deepcopy(np.inf)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: fixed_mu_projectors(deepcopy(np.full((1, 2, 2), np.nan)), '
              'deepcopy(0.0)))',
      'gold_call': 'raises_value_error(lambda: _oracle_fixed_mu_projectors(deepcopy(np.full((1, 2, 2), '
                   'np.nan)), deepcopy(0.0)))'}]
