"""
Apply one Majorana rotation with collision-aware coefficient and unpaired-string projection.

A rotation couples anticommuting strings in signed two-dimensional sectors. Summing coincident branches before discarding small coefficients preserves cancellations and constructive interference, while counting unpaired Majoranas retains diagonal many-body strings even when their total weight is large.

Returns
-------
numpy.ndarray of shape (K_out, 2), the sorted sparse dimensionless Hermitian expansion after one Heisenberg rotation and projection
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rotate_and_project_expansion(expansion: "np.ndarray", generator: int, angle: float, n_modes: int, epsilon: float, cap: int) -> "np.ndarray":
    r"""Conjugate an observable by one rotation, then project its expansion.

    Parameters
    ----------
    expansion : numpy.ndarray
        Real array of shape (K, 2), with rows [integer-valued mask, coefficient]; masks lie in [0, 2**(2*n_modes)). Rows may be unsorted or repeated. Empty input is allowed.
    generator : int
        Nonnegative mask of an even-weight Hermitian generator in the same mode space; the identity generator is allowed.
    angle : float
        Real dimensionless angle of exp(-i*angle*mu(generator)/2).
    n_modes : int
        Number of fermionic modes, from 1 to 6; bits 2*j and 2*j+1 form mode j.
    epsilon : float
        Nonnegative absolute coefficient threshold, applied after contributions to each output string have been summed. Equality is retained; exact-zero coefficients are omitted.
    cap : int
        Nonnegative maximum allowed count of singly occupied Majorana pairs. Count a mode once when exactly one of its two Majoranas is present. Values above n_modes have no additional effect.

    Returns
    -------
    projected : numpy.ndarray
        Real float64 array of shape (K_out, 2), with unique masks in increasing order and nonzero coefficients. It represents U-dagger O U after the stated coefficient and unpaired projections; an empty result has shape (0, 2).

    Raises
    ------
    ValueError
        If epsilon or cap is negative.

    Notes
    -----
    All branches are computed from the incoming observable, then equal output masks are merged before either cutoff is evaluated. Inputs are never mutated. Use majorana_product_table for ordered multiplication phases. No random numbers or external operations are used.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from math import cos, fsum, sin
import numpy as np

def _oracle_rotate_and_project_expansion(expansion: 'np.ndarray', generator: int, angle: float, n_modes: int, epsilon: float, cap: int) -> 'np.ndarray':
    """Evaluate the specified deterministic reference operation."""
    expansion = np.asarray(expansion, dtype=float)
    if epsilon < 0 or cap < 0:
        raise ValueError('Threshold and cap must be nonnegative.')
    out = {}
    ct, st = (cos(angle), sin(angle))
    phases = _oracle_majorana_product_table(expansion[:, 0], generator)
    for (mask, coefficient), (target, real_phase, imag_phase) in zip(expansion, phases):
        mask, target = (int(mask), int(target))
        if imag_phase == 0:
            out.setdefault(mask, []).append(float(coefficient))
        else:
            out.setdefault(mask, []).append(float(coefficient) * ct)
            out.setdefault(target, []).append(float(coefficient) * float(imag_phase) * st)
    rows = []
    for mask, contributions in sorted(out.items()):
        coefficient = fsum(contributions)
        unpaired = sum((mask >> 2 * j & 3 in (1, 2) for j in range(n_modes)))
        if coefficient != 0 and abs(coefficient) >= epsilon and (unpaired <= cap):
            rows.append([mask, coefficient])
    return np.array(rows, dtype=float).reshape(-1, 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary, and edge-case specifications."""
    return [{'name': 'mixed_and_repeated',
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
               'x=np.array([[3,.3],[5,.4],[3,-.12],[6,-.7]])\n'
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
      'call': '_dense_expansion(_preserving_call(rotate_and_project_expansion, x.copy(), 5, 0.37, 2, 0.03, '
              '2), 2)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_rotate_and_project_expansion, x.copy(), 5, '
                   '0.37, 2, 0.03, 2), 2)',
      'tol': 1e-10},
     {'name': 'inclusive_coefficient_boundary',
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
               'x=np.array([[0,.12],[3,-.12],[15,.119999],[12,-.120001]])\n'
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
      'call': '_dense_expansion(_preserving_call(rotate_and_project_expansion, x.copy(), 0, 0.0, 2, 0.12, '
              '0), 2)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_rotate_and_project_expansion, x.copy(), 0, '
                   '0.0, 2, 0.12, 0), 2)',
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
      'call': '_dense_expansion(_preserving_call(rotate_and_project_expansion, x.copy(), 33, 0.3, 3, 0.001, '
              '2), 3)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_rotate_and_project_expansion, x.copy(), 33, '
                   '0.3, 3, 0.001, 2), 3)',
      'tol': 1e-10},
     {'name': 'unpaired_not_total_weight',
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
               'x=np.array([[15,.3],[5,.4],[63,-.2]])\n'
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
      'call': '_dense_expansion(_preserving_call(rotate_and_project_expansion, x.copy(), 0, 0.0, 3, 0.0, '
              '0), 3)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_rotate_and_project_expansion, x.copy(), 0, '
                   '0.0, 3, 0.0, 0), 3)',
      'tol': 1e-10},
     {'name': 'merge_before_threshold',
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
               'x=np.array([[3,.06],[3,.06],[15,.2],[15,-.2]])\n'
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
      'call': '_dense_expansion(_preserving_call(rotate_and_project_expansion, x.copy(), 0, 0.0, 2, 0.1, '
              '2), 2)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_rotate_and_project_expansion, x.copy(), 0, '
                   '0.0, 2, 0.1, 2), 2)',
      'tol': 1e-10},
     {'name': 'clifford_rotation',
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
               'x=np.array([[3,.4],[12,-.17]])\n'
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
      'call': '_dense_expansion(_preserving_call(rotate_and_project_expansion, x.copy(), 5, np.pi/2, 2, '
              '1e-12, 2), 2)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_rotate_and_project_expansion, x.copy(), 5, '
                   'np.pi/2, 2, 1e-12, 2), 2)',
      'tol': 1e-10},
     {'name': 'commuting_subspace',
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
               'x=np.array([[0,.2],[3,-.1],[12,.7]])\n'
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
      'call': '_dense_expansion(_preserving_call(rotate_and_project_expansion, x.copy(), 15, -0.6, 2, 0.0, '
              '2), 2)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_rotate_and_project_expansion, x.copy(), 15, '
                   '-0.6, 2, 0.0, 2), 2)',
      'tol': 1e-10},
     {'name': 'negative_threshold',
      'setup': 'import numpy as np\n'
               'x=np.array([[3,.2]])\n'
               '\n'
               'def _value_error(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_value_error(rotate_and_project_expansion, x.copy(), 5, 0.2, 2, -0.1, 2)',
      'gold_call': '_value_error(_oracle_rotate_and_project_expansion, x.copy(), 5, 0.2, 2, -0.1, 2)',
      'tol': 1e-10},
     {'name': 'negative_cap',
      'setup': 'import numpy as np\n'
               'x=np.array([[3,.2]])\n'
               '\n'
               'def _value_error(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_value_error(rotate_and_project_expansion, x.copy(), 5, 0.2, 2, 0.0, -1)',
      'gold_call': '_value_error(_oracle_rotate_and_project_expansion, x.copy(), 5, 0.2, 2, 0.0, -1)',
      'tol': 1e-10}]
