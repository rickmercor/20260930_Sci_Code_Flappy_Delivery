"""
Calculate the source's forward layer-propagation matrix for the supplied root and fixed dimensionless depth, and its directional derivative induced by tangent.

Use the main problem's forward-wave and differentiation conventions. Root and tangent may be noncommuting complex matrices. depth must be real, finite, and nonnegative.

Returns
-------
Return (X,X_tangent), two complex128 arrays of shape (N,N), containing the forward propagation matrix and its directional derivative at fixed depth. Preserve harmonic ordering and do not mutate the inputs. Raise ValueError for nonreal, nonfinite, or negative depth.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def layer_propagation(
    root: np.ndarray,
    tangent: np.ndarray,
    depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate layer propagation and its Frechet derivative.

    Parameters
    ----------
    root : np.ndarray
        Root matrix G of shape (N, N).
    tangent : np.ndarray
        Directional derivative of G with shape (N, N).
    depth : float
        Fixed finite nonnegative layer depth.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The forward propagation matrix and its tangent, each complex128 (N, N).

    Raises
    ------
    ValueError
        If depth is not real, finite, and nonnegative.
    """
    return None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm_frechet


def _oracle_layer_propagation(
    root: np.ndarray,
    tangent: np.ndarray,
    depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isreal(depth) or not np.isfinite(depth) or depth < 0:
        raise ValueError(
            "depth must be real, finite and nonnegative"
        )

    return expm_frechet(-depth * root, -depth * tangent)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'base = np.array([[0.7, 0.1j], [-0.1j, 0.9]], dtype=np.complex128)\n'
               'direction = np.array([[0.2, 0.04], [0.04, -0.1]], dtype=np.complex128)\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        assert np.array_equal(argument, before), "input arrays must remain unchanged"\n'
               '    return result\n',
      'call': '_preserving_call(layer_propagation, base, direction, 0.0)',
      'gold_call': '_preserving_call(_oracle_layer_propagation, base, direction, 0.0)'},
     {'setup': 'import numpy as np\n'
               'base = np.array([[0.7, 0.1j], [-0.1j, 0.9]], dtype=np.complex128)\n'
               'direction = np.array([[0.2, 0.04], [0.04, -0.1]], dtype=np.complex128)\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        assert np.array_equal(argument, before), "input arrays must remain unchanged"\n'
               '    return result\n',
      'call': '_preserving_call(layer_propagation, base, direction, 0.6)',
      'gold_call': '_preserving_call(_oracle_layer_propagation, base, direction, 0.6)'},
     {'setup': 'import numpy as np\n'
               'base = np.array([[0.7, 0.1j], [-0.1j, 0.9]], dtype=np.complex128)\n'
               'direction = np.array([[0.2, 0.04], [0.04, -0.1]], dtype=np.complex128)\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        assert np.array_equal(argument, before), "input arrays must remain unchanged"\n'
               '    return result\n',
      'call': '_preserving_call(layer_propagation, base, direction, 2.4)',
      'gold_call': '_preserving_call(_oracle_layer_propagation, base, direction, 2.4)'}]
