"""
Evaluate the ordered product phases of binary-indexed Hermitian Majorana strings.

The Hermitian phase convention fixes interference signs throughout fermionic observable propagation. Both commuting and anticommuting products must be represented in the same ordered basis, including strings containing several complete fermion pairs.

Returns
-------
numpy.ndarray of shape (K, 3), the XOR masks and real/imaginary parts of the dimensionless ordered-product phases
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def majorana_product_table(left: "np.ndarray", right: int) -> "np.ndarray":
    r"""Evaluate ordered products of Hermitian Majorana strings.

    Parameters
    ----------
    left : numpy.ndarray
        One-dimensional integer masks, with bit j representing Majorana j; masks are nonnegative and below 2**24. Empty input is allowed.
    right : int
        One nonnegative integer mask below 2**24. Left and right may have even or odd weight.

    Returns
    -------
    products : numpy.ndarray
        Float64 array of shape (len(left), 3), with columns [xor mask, real phase, imaginary phase], in input order. The last two columns encode zeta in mu(left) mu(right) = zeta mu(left XOR right).

    Raises
    ------
    ValueError
        If a supplied mask is negative.

    Notes
    -----
    The ordered basis is mu(v) = i**q(v) times the increasing-index product of its Majoranas, with q(v) = binomial(weight(v), 2) modulo 2. All input arrays are left unchanged. No randomness is used.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_majorana_product_table(left: 'np.ndarray', right: int) -> 'np.ndarray':
    """Evaluate the specified deterministic reference operation."""
    a_values = np.asarray(left)
    b = int(right)
    if b < 0 or np.any(a_values < 0):
        raise ValueError('Masks must be nonnegative.')
    qb = b.bit_count() * (b.bit_count() - 1) // 2 & 1
    out = np.empty((a_values.size, 3), dtype=float)
    for row, value in enumerate(a_values):
        a = int(value)
        x = a ^ b
        qa = a.bit_count() * (a.bit_count() - 1) // 2 & 1
        qx = x.bit_count() * (x.bit_count() - 1) // 2 & 1
        crossings = 0
        remaining = a
        while remaining:
            bit = remaining & -remaining
            crossings += (b & bit - 1).bit_count()
            remaining -= bit
        zeta = (-1) ** crossings * 1j ** ((qa + qb - qx) % 4)
        out[row] = (x, zeta.real, zeta.imag)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary, and edge-case specifications."""
    return [{'name': 'mixed_even_strings',
      'setup': 'import numpy as np\n'
               'a=np.array([0,3,5,15,18,48,63])\n'
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
      'call': '_preserving_call(majorana_product_table, a.copy(), 33)',
      'gold_call': '_preserving_call(_oracle_majorana_product_table, a.copy(), 33)',
      'tol': 1e-10},
     {'name': 'identity_generator',
      'setup': 'import numpy as np\n'
               'a=np.array([0,1,3,15,63])\n'
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
      'call': '_preserving_call(majorana_product_table, a.copy(), 0)',
      'gold_call': '_preserving_call(_oracle_majorana_product_table, a.copy(), 0)',
      'tol': 1e-10},
     {'name': 'odd_parity_products',
      'setup': 'import numpy as np\n'
               'a=np.array([1,2,7,31,255])\n'
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
      'call': '_preserving_call(majorana_product_table, a.copy(), 7)',
      'gold_call': '_preserving_call(_oracle_majorana_product_table, a.copy(), 7)',
      'tol': 1e-10},
     {'name': 'high_weight_gauge',
      'setup': 'import numpy as np\n'
               'a=np.array([15,63,255,1023,4095])\n'
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
      'call': '_preserving_call(majorana_product_table, a.copy(), 693)',
      'gold_call': '_preserving_call(_oracle_majorana_product_table, a.copy(), 693)',
      'tol': 1e-10},
     {'name': 'empty_batch',
      'setup': 'import numpy as np\n'
               'a=np.array([],dtype=int)\n'
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
      'call': '_preserving_call(majorana_product_table, a.copy(), 5)',
      'gold_call': '_preserving_call(_oracle_majorana_product_table, a.copy(), 5)',
      'tol': 1e-10},
     {'name': 'negative_mask',
      'setup': 'import numpy as np\n'
               'a=np.array([0,-1])\n'
               '\n'
               'def _value_error(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_value_error(majorana_product_table, a.copy(), 3)',
      'gold_call': '_value_error(_oracle_majorana_product_table, a.copy(), 3)',
      'tol': 1e-10}]
