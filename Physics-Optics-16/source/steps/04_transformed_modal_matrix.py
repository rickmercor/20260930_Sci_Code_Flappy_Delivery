"""
Construct the source's transformed companion modal matrix for the magnetic-field state defined in the main problem, and its directional derivative.

The supplied coupling and root are respectively the TM operator Q and the selected root G; their tangents refer to the same parameter direction. Preserve the task's field, harmonic, and branch conventions and its restrictions on differentiation.

Returns
-------
Return (V,V_tangent), two complex128 arrays of shape (N,N), containing the transformed companion modal matrix and its directional derivative. The fixed magnetic-field basis matrix is not part of the returned tuple. Preserve harmonic ordering and do not mutate the inputs. A singular required solve may raise LinAlgError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transformed_modal(
    coupling: np.ndarray,
    coupling_tangent: np.ndarray,
    root: np.ndarray,
    root_tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the transformed modal matrix and its tangent.

    Parameters
    ----------
    coupling : np.ndarray
        Coupling matrix Q of shape (N, N).
    coupling_tangent : np.ndarray
        Directional derivative of Q with shape (N, N).
    root : np.ndarray
        Nonsingular root matrix G of shape (N, N).
    root_tangent : np.ndarray
        Directional derivative of G with shape (N, N).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The transformed companion modal matrix and its tangent, each complex128 (N, N).

    Raises
    ------
    np.linalg.LinAlgError
        If the root matrix is singular.
    """
    return None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve


def _right_solve(right, left):
    value = solve(left[0].T, right[0].T).T
    tangent = solve(
        left[0].T,
        (right[1] - value @ left[1]).T,
    ).T
    return value, tangent


def _oracle_transformed_modal(
    coupling: np.ndarray,
    coupling_tangent: np.ndarray,
    root: np.ndarray,
    root_tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _right_solve(
        (coupling, coupling_tangent),
        (root, root_tangent),
    )

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
      'call': '_preserving_call(transformed_modal, base, direction, np.array([[2, 0], [1, 4]], '
              'dtype=complex), direction * 0.2)',
      'gold_call': '_preserving_call(_oracle_transformed_modal, base, direction, np.array([[2, 0], [1, '
                   '4]], dtype=complex), direction * 0.2)'},
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
      'call': '_preserving_call(transformed_modal, base, direction, np.eye(2, dtype=complex), '
              'direction * 0.2)',
      'gold_call': '_preserving_call(_oracle_transformed_modal, base, direction, np.eye(2, '
                   'dtype=complex), direction * 0.2)'},
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
      'call': '_preserving_call(transformed_modal, base, direction, base * 1.3, direction * 0.2)',
      'gold_call': '_preserving_call(_oracle_transformed_modal, base, direction, base * 1.3, direction '
                   '* 0.2)'}]
