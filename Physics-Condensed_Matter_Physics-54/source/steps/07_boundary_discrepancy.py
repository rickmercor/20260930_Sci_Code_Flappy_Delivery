"""
Form the centered finite-flux density response and compare it with the supplied zero-field Chern profile. Return its spatial mean and the boundary/interior RMS errors, with both ribbon edges in the boundary set. Keep the boundary-minus-interior sign fixed.

Section II B and the Haldane examples in Section III compare the local Chern and Streda markers. Boundaries and finite flux can produce local differences. The two RMS sets and their difference are a new finite-ribbon diagnostic specified by this benchmark; Delta is not itself a topological invariant.

Returns
-------
Real NumPy array of shape (4,), ordered [mean(cS), R_boundary, R_interior, R_boundary-R_interior].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def boundary_discrepancy(c: np.ndarray, nplus: np.ndarray, nminus: np.ndarray, flux: float, edge: int=2) -> np.ndarray:
    """Compare local Chern and symmetric finite-flux Streda markers.

    Parameters
    ----------
    c, nplus, nminus : numpy.ndarray
        Equal finite real vectors (nx,). Densities correspond to +flux/-flux
        at identical chemical potential and geometric gauge convention.
    flux : float
        Finite positive real flux magnitude in flux-quantum units per cell.
    edge : int
        Positive integer, 2*edge<nx; booleans excluded.

    Returns
    -------
    numpy.ndarray
        Real vector [mean(cS), R_edge, R_bulk, R_edge-R_bulk], where
        cS=(nplus-nminus)/(2*flux). R is RMS of cS-c over the first/last
        edge cells or the remaining cells respectively. Count both edges.
        Do not sum or average the two densities before differencing.

    Raises
    ------
    ValueError
        Invalid vectors, flux or edge width.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_boundary_discrepancy(c: np.ndarray, nplus: np.ndarray, nminus: np.ndarray,
                                 flux: float, edge: int = 2) -> np.ndarray:
    """Compare local Chern and symmetric finite-flux Streda markers.

    Parameters
    ----------
    c, nplus, nminus : numpy.ndarray
        Equal finite real vectors (nx,). Densities correspond to +flux/-flux
        at identical chemical potential and geometric gauge convention.
    flux : float
        Finite positive real flux magnitude in flux-quantum units per cell.
    edge : int
        Positive integer, 2*edge<nx; booleans excluded.

    Returns
    -------
    numpy.ndarray
        Real vector [mean(cS), R_edge, R_bulk, R_edge-R_bulk], where
        cS=(nplus-nminus)/(2*flux). R is RMS of cS-c over the first/last
        edge cells or the remaining cells respectively. Count both edges.
        Do not sum or average the two densities before differencing.

    Raises
    ------
    ValueError
        Invalid vectors, flux or edge width.
    """
    c,nplus,nminus=map(np.asarray,(c,nplus,nminus))
    if c.ndim!=1 or c.shape!=nplus.shape or c.shape!=nminus.shape or any(not np.isrealobj(a) or not np.all(np.isfinite(a)) for a in (c,nplus,nminus)):
        raise ValueError('invalid vectors')
    if isinstance(edge,(bool,np.bool_)) or not isinstance(edge,(int,np.integer)) or edge<1 or 2*edge>=len(c):
        raise ValueError('invalid edge width')
    if not np.isscalar(flux) or not np.isrealobj(flux) or not np.isfinite(flux) or flux<=0:
        raise ValueError('invalid flux')
    cs=(nplus-nminus)/(2*flux); d=cs-c
    re=np.sqrt(np.mean(np.r_[d[:edge],d[-edge:]]**2))
    rb=np.sqrt(np.mean(d[edge:-edge]**2))
    return np.array([cs.mean(),re,rb,re-rb])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nfrom copy import deepcopy\n\n',
      'call': 'boundary_discrepancy(deepcopy(np.zeros(6)), deepcopy(np.array([1.0, 2, 3, 4, 5, 6])), '
              'deepcopy(np.zeros(6)), deepcopy(0.5), deepcopy(1))',
      'gold_call': '_oracle_boundary_discrepancy(deepcopy(np.zeros(6)), deepcopy(np.array([1.0, 2, 3, '
                   '4, 5, 6])), deepcopy(np.zeros(6)), deepcopy(0.5), deepcopy(1))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'c = np.array([-0.6, -0.9, -1.0, -0.98, -0.91, -0.7])\n'
               'p = np.array([0.98, 1.01, 1.02, 0.95, 0.94, 0.96])\n'
               'm = p + 0.006 * np.array([0.5, 0.8, 1.01, 0.99, 0.82, 0.51])\n',
      'call': 'boundary_discrepancy(deepcopy(c), deepcopy(p), deepcopy(m), deepcopy(0.003), '
              'deepcopy(2))',
      'gold_call': '_oracle_boundary_discrepancy(deepcopy(c), deepcopy(p), deepcopy(m), '
                   'deepcopy(0.003), deepcopy(2))'},
     {'setup': 'import numpy as np\nfrom copy import deepcopy\n\n',
      'call': 'boundary_discrepancy(deepcopy(np.zeros(7)), deepcopy(np.ones(7)), deepcopy(np.ones(7)), '
              'deepcopy(0.002), deepcopy(3))',
      'gold_call': '_oracle_boundary_discrepancy(deepcopy(np.zeros(7)), deepcopy(np.ones(7)), '
                   'deepcopy(np.ones(7)), deepcopy(0.002), deepcopy(3))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: boundary_discrepancy(deepcopy(np.zeros(6)), '
              'deepcopy(np.zeros(6)), deepcopy(np.zeros(6)), deepcopy(0.0), deepcopy(1)))',
      'gold_call': 'raises_value_error(lambda: _oracle_boundary_discrepancy(deepcopy(np.zeros(6)), '
                   'deepcopy(np.zeros(6)), deepcopy(np.zeros(6)), deepcopy(0.0), deepcopy(1)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: boundary_discrepancy(deepcopy(np.zeros(6)), '
              'deepcopy(np.zeros(6)), deepcopy(np.zeros(6)), deepcopy(0.01), deepcopy(3)))',
      'gold_call': 'raises_value_error(lambda: _oracle_boundary_discrepancy(deepcopy(np.zeros(6)), '
                   'deepcopy(np.zeros(6)), deepcopy(np.zeros(6)), deepcopy(0.01), deepcopy(3)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: boundary_discrepancy(deepcopy(np.zeros(6)), '
              'deepcopy(np.zeros(5)), deepcopy(np.zeros(6)), deepcopy(0.01), deepcopy(1)))',
      'gold_call': 'raises_value_error(lambda: _oracle_boundary_discrepancy(deepcopy(np.zeros(6)), '
                   'deepcopy(np.zeros(5)), deepcopy(np.zeros(6)), deepcopy(0.01), deepcopy(1)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: boundary_discrepancy(deepcopy(np.zeros(6)), '
              'deepcopy(np.zeros(6)), deepcopy(np.zeros(6)), deepcopy(0.01), deepcopy(True)))',
      'gold_call': 'raises_value_error(lambda: _oracle_boundary_discrepancy(deepcopy(np.zeros(6)), '
                   'deepcopy(np.zeros(6)), deepcopy(np.zeros(6)), deepcopy(0.01), deepcopy(True)))'}]
