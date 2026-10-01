"""
Propagate one symmetric layer with distinct intermediate and endpoint unpaired cutoffs.

A string outside the endpoint subspace may return within the retained commutator order. Temporary excursions therefore must survive inside a layer even though the corresponding strings are removed at its boundary; premature projection changes the finite-step observable.

Returns
-------
numpy.ndarray of shape (K_out, 2), the non-renormalized dimensionless observable expansion at the completed layer boundary
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_trotter_layer(expansion: "np.ndarray", gates: "np.ndarray", n_modes: int, epsilon: float, cap: int) -> "np.ndarray":
    r"""Propagate one second-order layer with a layer-aware unpaired cutoff.

    Parameters
    ----------
    expansion : numpy.ndarray
        Incoming real sparse [mask, coefficient] array of shape (K, 2), with distinct ascending masks and even-weight strings; identity and empty expansions are supported.
    gates : numpy.ndarray
        Chronological [generator mask, angle] array of shape (G, 2) for one symmetric Hubbard layer, as produced by compile_hubbard_layer. Generators are quadratic or paired quartic strings.
    n_modes : int
        Number of fermionic modes, from 1 to 6.
    epsilon : float
        Nonnegative coefficient threshold after every elementary Majorana rotation, with threshold equality retained.
    cap : int
        Nonnegative endpoint cap on the number of unpaired Majoranas. Apply the temporary allowance that preserves leave-and-return paths through second commutator order, then restore this endpoint cap once the layer is complete.

    Returns
    -------
    evolved : numpy.ndarray
        Float64 array of shape (K_out, 2) with nonzero coefficients and ascending unique masks, representing the projected layer output; empty output has shape (0, 2).

    Raises
    ------
    ValueError
        If epsilon or cap is negative.

    Notes
    -----
    Gate rows give state-chronological order, not observable-application order. Use rotate_and_project_expansion for elementary propagation. Do not renormalize retained coefficients. Inputs are unchanged; there is no RNG. The temporary allowance concerns truncation relative to a projected endpoint subspace; it does not assert that fixed-cap dynamics converges to the exact untruncated dynamics.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_propagate_trotter_layer(expansion: 'np.ndarray', gates: 'np.ndarray', n_modes: int, epsilon: float, cap: int) -> 'np.ndarray':
    """Evaluate the specified deterministic reference operation."""
    if epsilon < 0 or cap < 0:
        raise ValueError('Threshold and cap must be nonnegative.')
    result = np.array(expansion, dtype=float, copy=True)
    temporary_cap = min(n_modes, cap + 2)
    for generator, angle in np.asarray(gates)[::-1]:
        result = _oracle_rotate_and_project_expansion(result, int(generator), float(angle), n_modes, epsilon, temporary_cap)
    rows = [row for row in result if sum((int(row[0]) >> 2 * j & 3 in (1, 2) for j in range(n_modes))) <= cap]
    return np.array(rows, dtype=float).reshape(-1, 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary, and edge-case specifications."""
    return [{'name': 'interacting_corner',
      'setup': 'import numpy as np\n'
               '\n'
               'def _dense_expansion(value, n_modes):\n'
               '    value = np.asarray(value, dtype=float)\n'
               '    if value.ndim != 2 or value.shape[1] != 2:\n'
               '        raise ValueError("The result must have two columns.")\n'
               '    masks = value[:, 0].astype(np.int64)\n'
               '    if np.any(value[:, 0] != masks) or np.any(masks < 0) or np.any(masks >= '
               '2**(2*n_modes)):\n'
               '        raise ValueError("The output masks must be valid integers.")\n'
               '    if len(masks) > 1 and np.any(np.diff(masks) <= 0):\n'
               '        raise ValueError("The output masks must be unique and increasing.")\n'
               '    result = np.zeros(2**(2*n_modes))\n'
               '    result[masks] = value[:, 1]\n'
               '    return result\n'
               '\n'
               'x=np.array([[0,.25],[48,-.25],[192,-.25],[240,-.25]])\n'
               'g=np.array([[33.0, -0.105], [18.0, 0.105], [132.0, -0.105], [72.0, 0.105], [513.0, '
               '-0.07665], [258.0, 0.07665], [2052.0, -0.07665], [1032.0, 0.07665], [3.0, 0.3885], [12.0, '
               '0.3885], [15.0, -0.3885], [48.0, 0.3885], [192.0, 0.3885], [240.0, -0.3885], [768.0, '
               '0.3885], [3072.0, 0.3885], [3840.0, -0.3885], [1032.0, 0.07665], [2052.0, -0.07665], '
               '[258.0, 0.07665], [513.0, -0.07665], [72.0, 0.105], [132.0, -0.105], [18.0, 0.105], [33.0, '
               '-0.105]])\n'
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
      'call': '_dense_expansion(_preserving_call(propagate_trotter_layer, x.copy(), g.copy(), 6, 2e-4, 2), '
              '6)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_propagate_trotter_layer, x.copy(), g.copy(), '
                   '6, 2e-4, 2), 6)',
      'tol': 1e-10},
     {'name': 'zero_angle_layer',
      'setup': 'import numpy as np\n'
               '\n'
               'def _dense_expansion(value, n_modes):\n'
               '    value = np.asarray(value, dtype=float)\n'
               '    if value.ndim != 2 or value.shape[1] != 2:\n'
               '        raise ValueError("The result must have two columns.")\n'
               '    masks = value[:, 0].astype(np.int64)\n'
               '    if np.any(value[:, 0] != masks) or np.any(masks < 0) or np.any(masks >= '
               '2**(2*n_modes)):\n'
               '        raise ValueError("The output masks must be valid integers.")\n'
               '    if len(masks) > 1 and np.any(np.diff(masks) <= 0):\n'
               '        raise ValueError("The output masks must be unique and increasing.")\n'
               '    result = np.zeros(2**(2*n_modes))\n'
               '    result[masks] = value[:, 1]\n'
               '    return result\n'
               '\n'
               'x=np.array([[0,.25],[48,-.25],[192,-.25],[240,-.25]])\n'
               'g=np.array([[33.0, -0.105], [18.0, 0.105], [132.0, -0.105], [72.0, 0.105], [513.0, '
               '-0.07665], [258.0, 0.07665], [2052.0, -0.07665], [1032.0, 0.07665], [3.0, 0.3885], [12.0, '
               '0.3885], [15.0, -0.3885], [48.0, 0.3885], [192.0, 0.3885], [240.0, -0.3885], [768.0, '
               '0.3885], [3072.0, 0.3885], [3840.0, -0.3885], [1032.0, 0.07665], [2052.0, -0.07665], '
               '[258.0, 0.07665], [513.0, -0.07665], [72.0, 0.105], [132.0, -0.105], [18.0, 0.105], [33.0, '
               '-0.105]])\n'
               'g[:,1]=0\n'
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
      'call': '_dense_expansion(_preserving_call(propagate_trotter_layer, x.copy(), g.copy(), 6, 2e-4, 2), '
              '6)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_propagate_trotter_layer, x.copy(), g.copy(), '
                   '6, 2e-4, 2), 6)',
      'tol': 1e-10},
     {'name': 'paired_endpoint_excursions',
      'setup': 'import numpy as np\n'
               '\n'
               'def _dense_expansion(value, n_modes):\n'
               '    value = np.asarray(value, dtype=float)\n'
               '    if value.ndim != 2 or value.shape[1] != 2:\n'
               '        raise ValueError("The result must have two columns.")\n'
               '    masks = value[:, 0].astype(np.int64)\n'
               '    if np.any(value[:, 0] != masks) or np.any(masks < 0) or np.any(masks >= '
               '2**(2*n_modes)):\n'
               '        raise ValueError("The output masks must be valid integers.")\n'
               '    if len(masks) > 1 and np.any(np.diff(masks) <= 0):\n'
               '        raise ValueError("The output masks must be unique and increasing.")\n'
               '    result = np.zeros(2**(2*n_modes))\n'
               '    result[masks] = value[:, 1]\n'
               '    return result\n'
               '\n'
               'x=np.array([[0,.25],[48,-.25],[192,-.25],[240,-.25]])\n'
               'g=np.array([[33.0, -0.105], [18.0, 0.105], [132.0, -0.105], [72.0, 0.105], [513.0, '
               '-0.07665], [258.0, 0.07665], [2052.0, -0.07665], [1032.0, 0.07665], [3.0, 0.3885], [12.0, '
               '0.3885], [15.0, -0.3885], [48.0, 0.3885], [192.0, 0.3885], [240.0, -0.3885], [768.0, '
               '0.3885], [3072.0, 0.3885], [3840.0, -0.3885], [1032.0, 0.07665], [2052.0, -0.07665], '
               '[258.0, 0.07665], [513.0, -0.07665], [72.0, 0.105], [132.0, -0.105], [18.0, 0.105], [33.0, '
               '-0.105]])\n'
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
      'call': '_dense_expansion(_preserving_call(propagate_trotter_layer, x.copy(), g.copy(), 6, 2e-4, 0), '
              '6)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_propagate_trotter_layer, x.copy(), g.copy(), '
                   '6, 2e-4, 0), 6)',
      'tol': 1e-10},
     {'name': 'unrestricted_layer',
      'setup': 'import numpy as np\n'
               '\n'
               'def _dense_expansion(value, n_modes):\n'
               '    value = np.asarray(value, dtype=float)\n'
               '    if value.ndim != 2 or value.shape[1] != 2:\n'
               '        raise ValueError("The result must have two columns.")\n'
               '    masks = value[:, 0].astype(np.int64)\n'
               '    if np.any(value[:, 0] != masks) or np.any(masks < 0) or np.any(masks >= '
               '2**(2*n_modes)):\n'
               '        raise ValueError("The output masks must be valid integers.")\n'
               '    if len(masks) > 1 and np.any(np.diff(masks) <= 0):\n'
               '        raise ValueError("The output masks must be unique and increasing.")\n'
               '    result = np.zeros(2**(2*n_modes))\n'
               '    result[masks] = value[:, 1]\n'
               '    return result\n'
               '\n'
               'x=np.array([[0,.25],[48,-.25],[192,-.25],[240,-.25]])\n'
               'g=np.array([[33.0, -0.105], [18.0, 0.105], [132.0, -0.105], [72.0, 0.105], [513.0, '
               '-0.07665], [258.0, 0.07665], [2052.0, -0.07665], [1032.0, 0.07665], [3.0, 0.3885], [12.0, '
               '0.3885], [15.0, -0.3885], [48.0, 0.3885], [192.0, 0.3885], [240.0, -0.3885], [768.0, '
               '0.3885], [3072.0, 0.3885], [3840.0, -0.3885], [1032.0, 0.07665], [2052.0, -0.07665], '
               '[258.0, 0.07665], [513.0, -0.07665], [72.0, 0.105], [132.0, -0.105], [18.0, 0.105], [33.0, '
               '-0.105]])\n'
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
      'call': '_dense_expansion(_preserving_call(propagate_trotter_layer, x.copy(), g.copy(), 6, 0.0, 6), '
              '6)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_propagate_trotter_layer, x.copy(), g.copy(), '
                   '6, 0.0, 6), 6)',
      'tol': 1e-10},
     {'name': 'empty_observable',
      'setup': 'import numpy as np\n'
               '\n'
               'def _dense_expansion(value, n_modes):\n'
               '    value = np.asarray(value, dtype=float)\n'
               '    if value.ndim != 2 or value.shape[1] != 2:\n'
               '        raise ValueError("The result must have two columns.")\n'
               '    masks = value[:, 0].astype(np.int64)\n'
               '    if np.any(value[:, 0] != masks) or np.any(masks < 0) or np.any(masks >= '
               '2**(2*n_modes)):\n'
               '        raise ValueError("The output masks must be valid integers.")\n'
               '    if len(masks) > 1 and np.any(np.diff(masks) <= 0):\n'
               '        raise ValueError("The output masks must be unique and increasing.")\n'
               '    result = np.zeros(2**(2*n_modes))\n'
               '    result[masks] = value[:, 1]\n'
               '    return result\n'
               '\n'
               'x=np.array([[0,.25],[48,-.25],[192,-.25],[240,-.25]])\n'
               'g=np.array([[33.0, -0.105], [18.0, 0.105], [132.0, -0.105], [72.0, 0.105], [513.0, '
               '-0.07665], [258.0, 0.07665], [2052.0, -0.07665], [1032.0, 0.07665], [3.0, 0.3885], [12.0, '
               '0.3885], [15.0, -0.3885], [48.0, 0.3885], [192.0, 0.3885], [240.0, -0.3885], [768.0, '
               '0.3885], [3072.0, 0.3885], [3840.0, -0.3885], [1032.0, 0.07665], [2052.0, -0.07665], '
               '[258.0, 0.07665], [513.0, -0.07665], [72.0, 0.105], [132.0, -0.105], [18.0, 0.105], [33.0, '
               '-0.105]])\n'
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
      'call': '_dense_expansion(_preserving_call(propagate_trotter_layer, x.copy(), g.copy(), 6, 2e-4, 2), '
              '6)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_propagate_trotter_layer, x.copy(), g.copy(), '
                   '6, 2e-4, 2), 6)',
      'tol': 1e-10},
     {'name': 'negative_endpoint_cap',
      'setup': 'import numpy as np\n'
               'x=np.array([[0,.25],[48,-.25],[192,-.25],[240,-.25]])\n'
               'g=np.array([[33.0, -0.105], [18.0, 0.105], [132.0, -0.105], [72.0, 0.105], [513.0, '
               '-0.07665], [258.0, 0.07665], [2052.0, -0.07665], [1032.0, 0.07665], [3.0, 0.3885], [12.0, '
               '0.3885], [15.0, -0.3885], [48.0, 0.3885], [192.0, 0.3885], [240.0, -0.3885], [768.0, '
               '0.3885], [3072.0, 0.3885], [3840.0, -0.3885], [1032.0, 0.07665], [2052.0, -0.07665], '
               '[258.0, 0.07665], [513.0, -0.07665], [72.0, 0.105], [132.0, -0.105], [18.0, 0.105], [33.0, '
               '-0.105]])\n'
               '\n'
               'def _value_error(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_value_error(propagate_trotter_layer, x.copy(), g.copy(), 6, 0.0, -1)',
      'gold_call': '_value_error(_oracle_propagate_trotter_layer, x.copy(), g.copy(), 6, 0.0, -1)',
      'tol': 1e-10},
     {'name': 'negative_threshold',
      'setup': 'import numpy as np\n'
               'x=np.array([[0,.25],[48,-.25],[192,-.25],[240,-.25]])\n'
               'g=np.array([[33.0, -0.105], [18.0, 0.105], [132.0, -0.105], [72.0, 0.105], [513.0, '
               '-0.07665], [258.0, 0.07665], [2052.0, -0.07665], [1032.0, 0.07665], [3.0, 0.3885], [12.0, '
               '0.3885], [15.0, -0.3885], [48.0, 0.3885], [192.0, 0.3885], [240.0, -0.3885], [768.0, '
               '0.3885], [3072.0, 0.3885], [3840.0, -0.3885], [1032.0, 0.07665], [2052.0, -0.07665], '
               '[258.0, 0.07665], [513.0, -0.07665], [72.0, 0.105], [132.0, -0.105], [18.0, 0.105], [33.0, '
               '-0.105]])\n'
               '\n'
               'def _value_error(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_value_error(propagate_trotter_layer, x.copy(), g.copy(), 6, -0.1, 2)',
      'gold_call': '_value_error(_oracle_propagate_trotter_layer, x.copy(), g.copy(), 6, -0.1, 2)',
      'tol': 1e-10}]
