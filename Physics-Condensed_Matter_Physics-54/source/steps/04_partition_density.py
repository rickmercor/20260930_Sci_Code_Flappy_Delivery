"""
Integrate the local occupied density over every supplied momentum interval using a separate Gauss-Legendre rule. Reconstruct the magnetic Hamiltonian and its fixed-mu projector at the quadrature nodes. Normalize by 2*pi and sum, rather than average, the two orbital densities.

The hybrid position/momentum representation permits the local density to be reconstructed by integration over conserved transverse momentum. Each orbital contributes to the cell density. Partitioned Gauss-Legendre quadrature is a benchmark integration method designed to avoid integrating across known occupation jumps.

Returns
-------
Real NumPy cell-density array of shape (nx,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def partition_density(delta: np.ndarray, flux: float, mu: float, cuts: np.ndarray, order: int) -> np.ndarray:
    """Integrate the occupied cell density separately on each momentum interval.

    Parameters
    ----------
    delta, flux, mu : numpy.ndarray, float, float
        Model inputs accepted by earlier functions; gauge center zero.
    cuts : numpy.ndarray
        Finite real strictly increasing vector of length>=2, endpoints
        0 and 2*pi to absolute tolerance 1e-10, used as supplied.
    order : int
        Gauss-Legendre order >=4 per interval; booleans excluded.

    Returns
    -------
    numpy.ndarray
        Real vector (nx,), integral of sum_alpha P_(x alpha,x alpha)(k)
        over 0..2*pi, divided by 2*pi. Map an order-point Gauss-Legendre
        rule independently to every interval; use E<mu at every node.
        Do not enforce a fixed electron count or sample across a cut.

    Raises
    ------
    ValueError
        Invalid cuts/order or invalid earlier-function inputs.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_partition_density(delta: np.ndarray, flux: float, mu: float,
                               cuts: np.ndarray, order: int) -> np.ndarray:
    """Integrate the occupied cell density separately on each momentum interval.

    Parameters
    ----------
    delta, flux, mu : numpy.ndarray, float, float
        Model inputs accepted by earlier functions; gauge center zero.
    cuts : numpy.ndarray
        Finite real strictly increasing vector of length>=2, endpoints
        0 and 2*pi to absolute tolerance 1e-10, used as supplied.
    order : int
        Gauss-Legendre order >=4 per interval; booleans excluded.

    Returns
    -------
    numpy.ndarray
        Real vector (nx,), integral of sum_alpha P_(x alpha,x alpha)(k)
        over 0..2*pi, divided by 2*pi. Map an order-point Gauss-Legendre
        rule independently to every interval; use E<mu at every node.
        Do not enforce a fixed electron count or sample across a cut.

    Raises
    ------
    ValueError
        Invalid cuts/order or invalid earlier-function inputs.
    """
    cuts=np.asarray(cuts)
    if cuts.ndim!=1 or len(cuts)<2 or not np.isrealobj(cuts) or not np.all(np.isfinite(cuts)) or np.any(np.diff(cuts)<=0) or abs(cuts[0])>1e-10 or abs(cuts[-1]-2*np.pi)>1e-10:
        raise ValueError('invalid partition')
    if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or order<4:
        raise ValueError('invalid quadrature order')
    q,w=np.polynomial.legendre.leggauss(order)
    ks=[]; weights=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        ks.extend((a+b)/2+(b-a)*q/2); weights.extend((b-a)*w/(4*np.pi))
    h=_oracle_magnetic_ribbon(np.array(ks),delta,flux)
    p=_oracle_fixed_mu_projectors(h,mu)
    n=np.diagonal(p,axis1=-2,axis2=-1).real.reshape(len(ks),-1,2).sum(axis=2)
    return np.array(weights)@n

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'd = np.linspace(-0.12, 0.1, 6)\n'
               'cuts_c = fermi_partition(d.copy(), 0.004, 0.08, 128)\n'
               'cuts_g = _oracle_fermi_partition(d, 0.004, 0.08, 128)\n',
      'call': 'partition_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), deepcopy(cuts_c), '
              'deepcopy(8))',
      'gold_call': '_oracle_partition_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), '
                   'deepcopy(cuts_g), deepcopy(8))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'd = np.linspace(-0.12, 0.1, 6)\n'
               'cuts_c = fermi_partition(d.copy(), 0.004, 0.08, 128)\n'
               'cuts_g = _oracle_fermi_partition(d, 0.004, 0.08, 128)\n',
      'call': 'partition_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), deepcopy(cuts_c), '
              'deepcopy(32))',
      'gold_call': '_oracle_partition_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), '
                   'deepcopy(cuts_g), deepcopy(32))'},
     {'setup': 'import numpy as np\nfrom copy import deepcopy\n\n',
      'call': 'partition_density(deepcopy(np.zeros(5)), deepcopy(0.0), deepcopy(20.0), '
              'deepcopy(np.array([0.0, 0.2, 2.0, 2 * np.pi])), deepcopy(8))',
      'gold_call': '_oracle_partition_density(deepcopy(np.zeros(5)), deepcopy(0.0), deepcopy(20.0), '
                   'deepcopy(np.array([0.0, 0.2, 2.0, 2 * np.pi])), deepcopy(8))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'd = np.linspace(-0.12, 0.1, 6)\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: partition_density(deepcopy(d), deepcopy(0.004), '
              'deepcopy(0.08), deepcopy(np.array([0.0, 0.0, 2 * np.pi])), deepcopy(8)))',
      'gold_call': 'raises_value_error(lambda: _oracle_partition_density(deepcopy(d), deepcopy(0.004), '
                   'deepcopy(0.08), deepcopy(np.array([0.0, 0.0, 2 * np.pi])), deepcopy(8)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'd = np.linspace(-0.12, 0.1, 6)\n'
               'cuts_c = fermi_partition(d.copy(), 0.004, 0.08, 128)\n'
               'cuts_g = _oracle_fermi_partition(d, 0.004, 0.08, 128)\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: partition_density(deepcopy(d), deepcopy(0.004), '
              'deepcopy(0.08), deepcopy(cuts_c), deepcopy(3)))',
      'gold_call': 'raises_value_error(lambda: _oracle_partition_density(deepcopy(d), deepcopy(0.004), '
                   'deepcopy(0.08), deepcopy(cuts_g), deepcopy(3)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'd = np.linspace(-0.12, 0.1, 6)\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: partition_density(deepcopy(d), deepcopy(0.004), '
              'deepcopy(0.08), deepcopy(np.array([0.0, 5.0])), deepcopy(8)))',
      'gold_call': 'raises_value_error(lambda: _oracle_partition_density(deepcopy(d), deepcopy(0.004), '
                   'deepcopy(0.08), deepcopy(np.array([0.0, 5.0])), deepcopy(8)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'd = np.linspace(-0.12, 0.1, 6)\n'
               'cuts_c = fermi_partition(d.copy(), 0.004, 0.08, 128)\n'
               'cuts_g = _oracle_fermi_partition(d, 0.004, 0.08, 128)\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: partition_density(deepcopy(d), deepcopy(0.004), '
              'deepcopy(0.08), deepcopy(cuts_c), deepcopy(True)))',
      'gold_call': 'raises_value_error(lambda: _oracle_partition_density(deepcopy(d), deepcopy(0.004), '
                   'deepcopy(0.08), deepcopy(cuts_g), deepcopy(True)))'}]
