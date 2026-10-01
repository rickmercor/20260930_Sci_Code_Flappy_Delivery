"""
Evaluate the final Fock-state expectation with the full Hermitian-string phase.

Majorana-string overlap signs depend on both occupied modes and the phase convention for the full ordered string; treating a high-weight string as an unsigned product of number parities changes physical expectations.

Returns
-------
float, the dimensionless expectation value of the supplied Hermitian expansion in the specified Fock state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contract_fock_expansion(expansion: "np.ndarray", occupation: "np.ndarray") -> float:
    r"""Contract a Hermitian Majorana expansion with a Fock basis state.

    Parameters
    ----------
    expansion : numpy.ndarray
        Real [mask, coefficient] array of shape (K, 2), with masks below 2**(2*N); repeated or unsorted rows and explicit zeros are allowed.
    occupation : numpy.ndarray
        Length-N binary occupation vector, with 1 <= N <= 6 and mode 0 first. All-zero and all-one vectors are supported.

    Returns
    -------
    expectation : float
        Native Python float equal to the dimensionless Fock expectation of the expansion. Empty input gives 0.0.

    Raises
    ------
    ValueError
        If any occupation entry is not zero or one.

    Notes
    -----
    Use the parity-reduced Hermitian gauge from majorana_product_table; its high-weight signs cannot be replaced by a product of pair-string signs without accounting for the basis phase. The convention for the overall phase of the Fock ket does not affect the expectation. Inputs are unchanged; no RNG is used.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from math import fsum
import numpy as np

def _oracle_contract_fock_expansion(expansion: 'np.ndarray', occupation: 'np.ndarray') -> float:
    """Evaluate the specified deterministic reference operation."""
    occupation = np.asarray(occupation)
    if np.any((occupation != 0) & (occupation != 1)):
        raise ValueError('Occupation entries must be binary.')
    n_modes = len(occupation)
    terms = []
    for mask, coefficient in np.asarray(expansion):
        mask = int(mask)
        if any((mask >> 2 * j & 3 in (1, 2) for j in range(n_modes))):
            continue
        pairs = mask.bit_count() // 2
        occupied_pairs = sum((int(occupation[j]) for j in range(n_modes) if mask >> 2 * j & 1))
        sign = (-1) ** (((pairs + 1) // 2 + occupied_pairs) % 2)
        terms.append(float(coefficient) * sign)
    return float(fsum(terms))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary, and edge-case specifications."""
    return [{'name': 'mixed_paired_and_unpaired',
      'setup': 'import numpy as np\n'
               'x=np.array([[0,.17],[3,.2],[15,-.3],[63,.11],[255,-.08],[1023,.09],[4095,.21],[5,.6],[17,-.5]])\n'
               'n=np.array([1,0,0,1,1,0])\n'
               'import numpy as np\n'
               '\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        if not np.array_equal(argument, before):\n'
               '            raise AssertionError("The function mutated an input array.")\n'
               '    return result\n',
      'call': '_preserving_call(contract_fock_expansion, x.copy(), n.copy())',
      'gold_call': '_preserving_call(_oracle_contract_fock_expansion, x.copy(), n.copy())',
      'tol': 1e-10},
     {'name': 'vacuum',
      'setup': 'import numpy as np\n'
               'x=np.array([[0,.17],[3,.2],[15,-.3],[63,.11],[255,-.08],[1023,.09],[4095,.21],[5,.6],[17,-.5]])\n'
               'n=np.array([1,0,0,1,1,0])\n'
               'n=np.zeros(6,int)\n'
               'import numpy as np\n'
               '\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        if not np.array_equal(argument, before):\n'
               '            raise AssertionError("The function mutated an input array.")\n'
               '    return result\n',
      'call': '_preserving_call(contract_fock_expansion, x.copy(), n.copy())',
      'gold_call': '_preserving_call(_oracle_contract_fock_expansion, x.copy(), n.copy())',
      'tol': 1e-10},
     {'name': 'filled_fock_state',
      'setup': 'import numpy as np\n'
               'x=np.array([[0,.17],[3,.2],[15,-.3],[63,.11],[255,-.08],[1023,.09],[4095,.21],[5,.6],[17,-.5]])\n'
               'n=np.array([1,0,0,1,1,0])\n'
               'n=np.ones(6,int)\n'
               'import numpy as np\n'
               '\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        if not np.array_equal(argument, before):\n'
               '            raise AssertionError("The function mutated an input array.")\n'
               '    return result\n',
      'call': '_preserving_call(contract_fock_expansion, x.copy(), n.copy())',
      'gold_call': '_preserving_call(_oracle_contract_fock_expansion, x.copy(), n.copy())',
      'tol': 1e-10},
     {'name': 'empty_expansion',
      'setup': 'import numpy as np\n'
               'x=np.array([[0,.17],[3,.2],[15,-.3],[63,.11],[255,-.08],[1023,.09],[4095,.21],[5,.6],[17,-.5]])\n'
               'n=np.array([1,0,0,1,1,0])\n'
               'x=np.empty((0,2))\n'
               'import numpy as np\n'
               '\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        if not np.array_equal(argument, before):\n'
               '            raise AssertionError("The function mutated an input array.")\n'
               '    return result\n',
      'call': '_preserving_call(contract_fock_expansion, x.copy(), n.copy())',
      'gold_call': '_preserving_call(_oracle_contract_fock_expansion, x.copy(), n.copy())',
      'tol': 1e-10},
     {'name': 'quartic_and_octic_signs',
      'setup': 'import numpy as np\n'
               'x=np.array([[15,.7],[255,.2]])\n'
               'n=np.array([0,1,0,1])\n'
               'import numpy as np\n'
               '\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        if not np.array_equal(argument, before):\n'
               '            raise AssertionError("The function mutated an input array.")\n'
               '    return result\n',
      'call': '_preserving_call(contract_fock_expansion, x.copy(), n.copy())',
      'gold_call': '_preserving_call(_oracle_contract_fock_expansion, x.copy(), n.copy())',
      'tol': 1e-10},
     {'name': 'nonbinary_occupation',
      'setup': 'import numpy as np\n'
               'x=np.array([[0,.17],[3,.2],[15,-.3],[63,.11],[255,-.08],[1023,.09],[4095,.21],[5,.6],[17,-.5]])\n'
               'n=np.array([1,0,0,1,1,0])\n'
               'n[0]=2\n'
               '\n'
               'def _value_error(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_value_error(contract_fock_expansion, x.copy(), n.copy())',
      'gold_call': '_value_error(_oracle_contract_fock_expansion, x.copy(), n.copy())',
      'tol': 1e-10}]
