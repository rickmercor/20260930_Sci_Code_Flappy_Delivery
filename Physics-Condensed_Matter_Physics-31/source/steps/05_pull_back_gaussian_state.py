"""
Pull back the evolved observable through an exact Gaussian preparation by weight-sector transformation.

Preparing a correlated Gaussian state can be handled on the observable side before contraction with a Fock state.

Returns
-------
numpy.ndarray of shape (K_full, 2), the dimensionless Gaussian-pulled-back coefficients in every populated Majorana-weight sector
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pull_back_gaussian_state(expansion: "np.ndarray", one_body: "np.ndarray") -> "np.ndarray":
    r"""Pull an observable through a number-conserving Gaussian preparation.

    Parameters
    ----------
    expansion : numpy.ndarray
        Real [mask, coefficient] array of shape (K, 2) in the specified Hermitian gauge. Even or odd weights are supported; empty input is allowed. Masks are below 2**(2*N).
    one_body : numpy.ndarray
        Complex unitary matrix V of shape (N, N), 1 <= N <= 6, defined by W-dagger f_i W = sum_j V[i,j] f_j. Unitarity is interpreted at absolute tolerance 1e-10 with zero relative tolerance.

    Returns
    -------
    pulled : numpy.ndarray
        Float64 [mask, coefficient] array for W-dagger O W. Include every mask in each weight sector present in the input, even when its coefficient vanishes, sorted by mask. Empty input returns shape (0, 2). No coefficient or unpaired cutoff is applied.

    Raises
    ------
    ValueError
        If one_body fails the stated unitarity condition.

    Notes
    -----
    Majoranas are interleaved as gamma_j=f_j-dagger+f_j and gamma-prime_j=i(f_j-dagger-f_j). Inputs remain unchanged; no stochastic sampling or eigenvector gauge choice is used.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from itertools import combinations
from math import fsum
import numpy as np

def _oracle_pull_back_gaussian_state(expansion: 'np.ndarray', one_body: 'np.ndarray') -> 'np.ndarray':
    """Evaluate the specified deterministic reference operation."""
    V = np.asarray(one_body, complex)
    n = len(V)
    d = 2 * n
    if not np.allclose(V.conj().T @ V, np.eye(n), atol=1e-10, rtol=0):
        raise ValueError('The preparation matrix must be unitary.')
    R = np.zeros((d, d))
    R[0::2, 0::2] = V.real
    R[0::2, 1::2] = -V.imag
    R[1::2, 0::2] = V.imag
    R[1::2, 1::2] = V.real
    sectors = {}
    for mask, c in expansion:
        mask = int(mask)
        I = tuple((j for j in range(d) if mask >> j & 1))
        sectors.setdefault(len(I), []).append((I, float(c)))
    result = []
    for w, rows in sorted(sectors.items()):
        if w == 0:
            result.append([0, fsum((c for _, c in rows))])
            continue
        targets = np.array(list(combinations(range(d), w)), dtype=int)
        totals = np.zeros(len(targets))
        for I, c in rows:
            minor = R[np.array(I)[None, :, None], targets[:, None, :]]
            totals += c * np.linalg.det(minor)
        result.extend([[sum((1 << int(j) for j in J)), float(c)] for J, c in zip(targets, totals)])
    return np.array(sorted(result), float).reshape(-1, 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary, and edge-case specifications."""
    return [{'name': 'complex_preparation',
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
               'from scipy.linalg import expm\n'
               'x=np.array([[0,.13],[3,-.27],[15,.31],[6,-.18]])\n'
               'h=np.array([[.2,.7j],[-.7j,-.3]])\n'
               'v=expm(-.37j*h)\n'
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
      'call': '_dense_expansion(_preserving_call(pull_back_gaussian_state, x.copy(), v.copy()), 2)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_pull_back_gaussian_state, x.copy(), '
                   'v.copy()), 2)',
      'tol': 1e-10},
     {'name': 'identity_preparation',
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
               'x=np.array([[3,.2],[15,-.3]])\n'
               'v=np.eye(3)\n'
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
      'call': '_dense_expansion(_preserving_call(pull_back_gaussian_state, x.copy(), v.copy()), 3)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_pull_back_gaussian_state, x.copy(), '
                   'v.copy()), 3)',
      'tol': 1e-10},
     {'name': 'odd_and_even_sectors',
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
               'from scipy.linalg import expm\n'
               'x=np.array([[0,.13],[3,-.27],[15,.31],[6,-.18]])\n'
               'h=np.array([[.2,.7j],[-.7j,-.3]])\n'
               'v=expm(-.37j*h)\n'
               'x=np.array([[1,.4],[3,.2],[7,-.5],[15,.13]])\n'
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
      'call': '_dense_expansion(_preserving_call(pull_back_gaussian_state, x.copy(), v.copy()), 2)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_pull_back_gaussian_state, x.copy(), '
                   'v.copy()), 2)',
      'tol': 1e-10},
     {'name': 'near_identity_without_cutoff',
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
               'from scipy.linalg import expm\n'
               'x=np.array([[0,.13],[3,-.27],[15,.31],[6,-.18]])\n'
               'h=np.array([[.2,.7j],[-.7j,-.3]])\n'
               'v=expm(-.37j*h)\n'
               'v=expm(-1e-7j*h)\n'
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
      'call': '_dense_expansion(_preserving_call(pull_back_gaussian_state, x.copy(), v.copy()), 2)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_pull_back_gaussian_state, x.copy(), '
                   'v.copy()), 2)',
      'tol': 2e-12},
     {'name': 'top_weight_orientation',
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
               'from scipy.linalg import expm\n'
               'h=np.array([[.1,.3j,0],[-.3j,-.2,.4],[0,.4,.6]])\n'
               'v=expm(-.5j*h)\n'
               'x=np.array([[63,-.7]])\n'
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
      'call': '_dense_expansion(_preserving_call(pull_back_gaussian_state, x.copy(), v.copy()), 3)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_pull_back_gaussian_state, x.copy(), '
                   'v.copy()), 3)',
      'tol': 1e-10},
     {'name': 'empty_expansion',
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
               'v=np.eye(2)\n'
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
      'call': '_dense_expansion(_preserving_call(pull_back_gaussian_state, x.copy(), v.copy()), 2)',
      'gold_call': '_dense_expansion(_preserving_call(_oracle_pull_back_gaussian_state, x.copy(), '
                   'v.copy()), 2)',
      'tol': 1e-10},
     {'name': 'nonunitary_input',
      'setup': 'import numpy as np\n'
               'x=np.array([[3,.2]])\n'
               'v=2*np.eye(2)\n'
               '\n'
               'def _value_error(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_value_error(pull_back_gaussian_state, x.copy(), v.copy())',
      'gold_call': '_value_error(_oracle_pull_back_gaussian_state, x.copy(), v.copy())',
      'tol': 1e-10}]
