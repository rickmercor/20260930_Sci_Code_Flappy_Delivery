"""
Find a deterministic partition at the momentum-dependent Fermi crossings. Follow ordered bands through the prescribed scan, bisect sign-changing brackets, retain exact scan-node hits under the specified tolerance, and merge roots before adding the two endpoints. Use the magnetic ribbon builder from step 1.

The fixed-chemical-potential occupation has jumps when ribbon bands cross the Fermi level. Resolving those crossings separates smooth portions of the density integrand. The scan, ordered-band bisection and root-merging tolerances are benchmark numerical choices, not an algorithm claimed by the source.

Returns
-------
Real one-dimensional NumPy array [0, interior roots..., 2*pi].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fermi_partition(delta: np.ndarray, flux: float, mu: float, scan: int=128) -> np.ndarray:
    """Locate momentum intervals of fixed occupancy by bandwise root bracketing.

    Parameters
    ----------
    delta, flux : numpy.ndarray, float
        Disorder and flux accepted by magnetic_ribbon; gauge center is zero.
    mu : float
        Finite real fixed chemical potential.
    scan : int
        Integer >=8, no booleans; scan+1 equally spaced nodes on [0,2*pi].

    Returns
    -------
    numpy.ndarray
        Sorted real partition including 0 and 2*pi. At scan nodes sort the
        eigenvalues ascending. For each band and adjacent node pair whose
        E-mu values have strictly opposite signs, bisect that ordered band
        until bracket width <=1e-11, then record its midpoint. Also record
        interior scan nodes with abs(E-mu)<=1e-12. Merge interior roots
        separated by <=1e-9, keeping the smaller root; discard roots within
        1e-9 of either endpoint. The task prescribes this detector; it does
        not promise to find tangent roots or multiple roots inside one bin.

    Raises
    ------
    ValueError
        Invalid scan or mu, or invalid magnetic_ribbon parameters.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_fermi_partition(delta: np.ndarray, flux: float, mu: float,
                             scan: int = 128) -> np.ndarray:
    """Locate momentum intervals of fixed occupancy by bandwise root bracketing.

    Parameters
    ----------
    delta, flux : numpy.ndarray, float
        Disorder and flux accepted by magnetic_ribbon; gauge center is zero.
    mu : float
        Finite real fixed chemical potential.
    scan : int
        Integer >=8, no booleans; scan+1 equally spaced nodes on [0,2*pi].

    Returns
    -------
    numpy.ndarray
        Sorted real partition including 0 and 2*pi. At scan nodes sort the
        eigenvalues ascending. For each band and adjacent node pair whose
        E-mu values have strictly opposite signs, bisect that ordered band
        until bracket width <=1e-11, then record its midpoint. Also record
        interior scan nodes with abs(E-mu)<=1e-12. Merge interior roots
        separated by <=1e-9, keeping the smaller root; discard roots within
        1e-9 of either endpoint. The task prescribes this detector; it does
        not promise to find tangent roots or multiple roots inside one bin.

    Raises
    ------
    ValueError
        Invalid scan or mu, or invalid magnetic_ribbon parameters.
    """
    if isinstance(scan,(bool,np.bool_)) or not isinstance(scan,(int,np.integer)) or scan<8:
        raise ValueError('invalid scan count')
    if not np.isscalar(mu) or not np.isrealobj(mu) or not np.isfinite(mu):
        raise ValueError('invalid mu')
    grid=np.linspace(0,2*np.pi,scan+1)
    values=np.linalg.eigvalsh(_oracle_magnetic_ribbon(grid,delta,flux))-mu
    roots=[]
    for b in range(values.shape[1]):
        for i in range(scan):
            if i>0 and abs(values[i,b])<=1e-12: roots.append(grid[i])
            if values[i,b]*values[i+1,b]<0:
                lo,hi=grid[i],grid[i+1]; flo=values[i,b]
                while hi-lo>1e-11:
                    mid=(lo+hi)/2
                    fm=np.linalg.eigvalsh(_oracle_magnetic_ribbon(np.array([mid]),delta,flux))[0,b]-mu
                    if flo*fm<=0: hi=mid
                    else: lo=mid; flo=fm
                roots.append((lo+hi)/2)
    unique=[]
    for r in sorted(roots):
        if 1e-9<r<2*np.pi-1e-9 and (not unique or r-unique[-1]>1e-9): unique.append(r)
    return np.array([0.]+unique+[2*np.pi])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nfrom copy import deepcopy\n\n',
      'call': 'fermi_partition(deepcopy(np.zeros(6)), deepcopy(0.003), deepcopy(0.08), deepcopy(64))',
      'gold_call': '_oracle_fermi_partition(deepcopy(np.zeros(6)), deepcopy(0.003), deepcopy(0.08), '
                   'deepcopy(64))'},
     {'setup': 'import numpy as np\nfrom copy import deepcopy\n\n',
      'call': 'fermi_partition(deepcopy(np.linspace(-0.15, 0.12, 8)), deepcopy(-0.006), '
              'deepcopy(-0.03), deepcopy(96))',
      'gold_call': '_oracle_fermi_partition(deepcopy(np.linspace(-0.15, 0.12, 8)), deepcopy(-0.006), '
                   'deepcopy(-0.03), deepcopy(96))'},
     {'setup': 'import numpy as np\nfrom copy import deepcopy\n\n',
      'call': 'fermi_partition(deepcopy(np.zeros(5)), deepcopy(0.002), deepcopy(20.0), deepcopy(32))',
      'gold_call': '_oracle_fermi_partition(deepcopy(np.zeros(5)), deepcopy(0.002), deepcopy(20.0), '
                   'deepcopy(32))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: fermi_partition(deepcopy(np.zeros(6)), deepcopy(0.003), '
              'deepcopy(0.08), deepcopy(True)))',
      'gold_call': 'raises_value_error(lambda: _oracle_fermi_partition(deepcopy(np.zeros(6)), '
                   'deepcopy(0.003), deepcopy(0.08), deepcopy(True)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: fermi_partition(deepcopy(np.zeros(6)), deepcopy(0.003), '
              'deepcopy(0.08), deepcopy(7)))',
      'gold_call': 'raises_value_error(lambda: _oracle_fermi_partition(deepcopy(np.zeros(6)), '
                   'deepcopy(0.003), deepcopy(0.08), deepcopy(7)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: fermi_partition(deepcopy(np.zeros(6)), deepcopy(0.003), '
              'deepcopy(np.nan)))',
      'gold_call': 'raises_value_error(lambda: _oracle_fermi_partition(deepcopy(np.zeros(6)), '
                   'deepcopy(0.003), deepcopy(np.nan)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               '\n'
               'def raises_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lambda: fermi_partition(deepcopy(np.zeros(3)), deepcopy(0.003), '
              'deepcopy(0.08)))',
      'gold_call': 'raises_value_error(lambda: _oracle_fermi_partition(deepcopy(np.zeros(3)), '
                   'deepcopy(0.003), deepcopy(0.08)))'}]
