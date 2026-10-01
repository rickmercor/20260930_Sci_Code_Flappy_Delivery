"""
Starting from the supplied density estimate, double the per-interval Gauss-Legendre order until the stated maximum-component convergence condition holds. Use step 4 for each refinement. Preserve the partition and compare consecutive summed densities; report nonconvergence instead of silently returning an unconverged estimate.

The finite-flux difference amplifies density errors by 1/(2*f), so density convergence matters directly for the Streda comparison. The order-doubling rule is a reproducible numerical stopping criterion; it is not a rigorous universal error bound or an extrapolation in magnetic field.

Returns
-------
Real NumPy converged cell-density array of shape (nx,); raise ValueError on nonconvergence.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def refine_density(delta: np.ndarray, flux: float, mu: float, cuts: np.ndarray, initial: np.ndarray, order: int=8, tol: float=1e-10, max_order: int=256) -> np.ndarray:
    """Refine partition quadrature with the supplied first density estimate.

    Parameters
    ----------
    delta, flux, mu, cuts : numpy.ndarray, float, float, numpy.ndarray
        Same domains as partition_density. Cuts remain fixed.
    initial : numpy.ndarray
        Finite real vector matching delta, the density at the given order.
        Treat this as an input estimate; do not discard or recompute it.
    order, max_order : int
        Integers, order>=4 and max_order>=2*order; booleans excluded.
    tol : float
        Finite positive real maximum-component convergence threshold.

    Returns
    -------
    numpy.ndarray
        Refined real density (nx,). Double order, recompute the integral,
        and return the new density at the first max(abs(new-old))<=tol.
        Test the candidate at max_order if reached by doubling. Never use
        an order larger than max_order; if none converges raise ValueError.

    Raises
    ------
    ValueError
        Invalid parameters/estimate, nonconvergence, or earlier input errors.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_refine_density(delta: np.ndarray, flux: float, mu: float, cuts: np.ndarray,
                            initial: np.ndarray, order: int = 8,
                            tol: float = 1e-10, max_order: int = 256) -> np.ndarray:
    """Refine partition quadrature with the supplied first density estimate.

    Parameters
    ----------
    delta, flux, mu, cuts : numpy.ndarray, float, float, numpy.ndarray
        Same domains as partition_density. Cuts remain fixed.
    initial : numpy.ndarray
        Finite real vector matching delta, the density at the given order.
        Treat this as an input estimate; do not discard or recompute it.
    order, max_order : int
        Integers, order>=4 and max_order>=2*order; booleans excluded.
    tol : float
        Finite positive real maximum-component convergence threshold.

    Returns
    -------
    numpy.ndarray
        Refined real density (nx,). Double order, recompute the integral,
        and return the new density at the first max(abs(new-old))<=tol.
        Test the candidate at max_order if reached by doubling. Never use
        an order larger than max_order; if none converges raise ValueError.

    Raises
    ------
    ValueError
        Invalid parameters/estimate, nonconvergence, or earlier input errors.
    """
    for n in (order,max_order):
        if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)):
            raise ValueError('orders must be integers')
    if order<4 or max_order<2*order: raise ValueError('invalid order limits')
    if not np.isscalar(tol) or not np.isrealobj(tol) or not np.isfinite(tol) or tol<=0:
        raise ValueError('invalid tolerance')
    prev=np.asarray(initial)
    if prev.shape!=np.asarray(delta).shape or prev.ndim!=1 or not np.isrealobj(prev) or not np.all(np.isfinite(prev)):
        raise ValueError('invalid initial density')
    n=2*order
    while n<=max_order:
        current=_oracle_partition_density(delta,flux,mu,cuts,n)
        if np.max(np.abs(current-prev))<=tol: return current
        prev=current; n*=2
    raise ValueError('density quadrature did not converge')

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'd = np.linspace(-0.12, 0.1, 6)\n'
               'cuts_c = fermi_partition(d.copy(), 0.004, 0.08, 128)\n'
               'cuts_g = _oracle_fermi_partition(d, 0.004, 0.08, 128)\n'
               'initial_c = partition_density(d.copy(), 0.004, 0.08, cuts_c, 8)\n'
               'initial_g = _oracle_partition_density(d, 0.004, 0.08, cuts_g, 8)\n',
      'call': 'refine_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), deepcopy(cuts_c), '
              'deepcopy(initial_c))',
      'gold_call': '_oracle_refine_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), '
                   'deepcopy(cuts_g), deepcopy(initial_g))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'd = np.linspace(-0.12, 0.1, 6)\n'
               'cuts_c = fermi_partition(d.copy(), 0.004, 0.08, 128)\n'
               'cuts_g = _oracle_fermi_partition(d, 0.004, 0.08, 128)\n'
               'initial_c = partition_density(d.copy(), 0.004, 0.08, cuts_c, 8)\n'
               'initial_g = _oracle_partition_density(d, 0.004, 0.08, cuts_g, 8)\n',
      'call': 'refine_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), deepcopy(cuts_c), '
              'deepcopy(initial_c), deepcopy(8), deepcopy(1e-07), deepcopy(128))',
      'gold_call': '_oracle_refine_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), '
                   'deepcopy(cuts_g), deepcopy(initial_g), deepcopy(8), deepcopy(1e-07), '
                   'deepcopy(128))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'd = np.linspace(-0.12, 0.1, 6)\n'
               'cuts_c = fermi_partition(d.copy(), 0.004, 0.08, 128)\n'
               'cuts_g = _oracle_fermi_partition(d, 0.004, 0.08, 128)\n'
               'exact16_c = partition_density(d.copy(), 0.004, 0.08, cuts_c, 16)\n'
               'exact16_g = _oracle_partition_density(d, 0.004, 0.08, cuts_g, 16)\n',
      'call': 'refine_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), deepcopy(cuts_c), '
              'deepcopy(exact16_c), deepcopy(8), deepcopy(1e-12), deepcopy(16))',
      'gold_call': '_oracle_refine_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), '
                   'deepcopy(cuts_g), deepcopy(exact16_g), deepcopy(8), deepcopy(1e-12), '
                   'deepcopy(16))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'd = np.linspace(-0.12, 0.1, 6)\n'
               'cuts_c = fermi_partition(d.copy(), 0.004, 0.08, 128)\n'
               'cuts_g = _oracle_fermi_partition(d, 0.004, 0.08, 128)\n'
               'initial_c = partition_density(d.copy(), 0.004, 0.08, cuts_c, 8)\n'
               'initial_g = _oracle_partition_density(d, 0.004, 0.08, cuts_g, 8)\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: refine_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), '
              'deepcopy(cuts_c), deepcopy(initial_c), deepcopy(8), deepcopy(0.0), deepcopy(256)))',
      'gold_call': 'raises_value_error(lambda: _oracle_refine_density(deepcopy(d), deepcopy(0.004), '
                   'deepcopy(0.08), deepcopy(cuts_g), deepcopy(initial_g), deepcopy(8), deepcopy(0.0), '
                   'deepcopy(256)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'd = np.linspace(-0.12, 0.1, 6)\n'
               'cuts_c = fermi_partition(d.copy(), 0.004, 0.08, 128)\n'
               'cuts_g = _oracle_fermi_partition(d, 0.004, 0.08, 128)\n'
               'initial_c = partition_density(d.copy(), 0.004, 0.08, cuts_c, 8)\n'
               'initial_g = _oracle_partition_density(d, 0.004, 0.08, cuts_g, 8)\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: refine_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), '
              'deepcopy(cuts_c), deepcopy(initial_c), deepcopy(8), deepcopy(1e-10), deepcopy(8)))',
      'gold_call': 'raises_value_error(lambda: _oracle_refine_density(deepcopy(d), deepcopy(0.004), '
                   'deepcopy(0.08), deepcopy(cuts_g), deepcopy(initial_g), deepcopy(8), '
                   'deepcopy(1e-10), deepcopy(8)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'd = np.linspace(-0.12, 0.1, 6)\n'
               'cuts_c = fermi_partition(d.copy(), 0.004, 0.08, 128)\n'
               'cuts_g = _oracle_fermi_partition(d, 0.004, 0.08, 128)\n'
               'exact16_c = partition_density(d.copy(), 0.004, 0.08, cuts_c, 16)\n'
               'exact16_g = _oracle_partition_density(d, 0.004, 0.08, cuts_g, 16)\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: refine_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), '
              'deepcopy(cuts_c), deepcopy(exact16_c + 1.0), deepcopy(8), deepcopy(1e-10), '
              'deepcopy(16)))',
      'gold_call': 'raises_value_error(lambda: _oracle_refine_density(deepcopy(d), deepcopy(0.004), '
                   'deepcopy(0.08), deepcopy(cuts_g), deepcopy(exact16_g + 1.0), deepcopy(8), '
                   'deepcopy(1e-10), deepcopy(16)))'},
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
      'call': 'raises_value_error(lambda: refine_density(deepcopy(d), deepcopy(0.004), deepcopy(0.08), '
              'deepcopy(cuts_c), deepcopy(np.zeros(2))))',
      'gold_call': 'raises_value_error(lambda: _oracle_refine_density(deepcopy(d), deepcopy(0.004), '
                   'deepcopy(0.08), deepcopy(cuts_g), deepcopy(np.zeros(2))))'}]
