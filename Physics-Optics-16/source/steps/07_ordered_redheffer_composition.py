"""
Compose the left subsystem followed by the right subsystem using the physical scattering convention of the main problem, and obtain the derivative of the complete cascade.

Both supplied subsystem tangents may be nonzero. The adjoining ports use the same reference basis and harmonic ordering. Retain all four external amplitude-scattering blocks and the task's differentiation restrictions.

Returns
-------
Return the tuple:  (S, S_tangent)  in exactly that order.  Both outputs are NumPy arrays of shape (2, 2, N, N), with complex128 dtype on the task path.  - S: The complete scattering matrix for the left subsystem followed by the right subsystem. - S_tangent: Its directional derivative, including contributions from both supplied input tangents.  Block meanings: - S[0, 0]: Complete left reflection. - S[0, 1]: Complete right-to-left transmission. - S[1, 0]: Complete left-to-right transmission. - S[1, 1]: Complete right reflection.  Preserve the port and harmonic conventions of Step 6. The output blocks need not have the single-layer equalities S[0, 0] = S[1, 1] or S[0, 1] = S[1, 0].  Return amplitude-scattering blocks, not diffraction powers. Do not mutate the input arrays.  A singular feedback solve may raise LinAlgError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def redheffer_compose(
    left: np.ndarray,
    left_tangent: np.ndarray,
    right: np.ndarray,
    right_tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose two ordered scattering matrices and their tangents.

    Parameters
    ----------
    left : np.ndarray
        Left scattering matrix of shape (2, 2, N, N).
    left_tangent : np.ndarray
        Directional derivative of the left matrix.
    right : np.ndarray
        Right scattering matrix of shape (2, 2, N, N).
    right_tangent : np.ndarray
        Directional derivative of the right matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Left-to-right cascade and tangent, each with shape (2, 2, N, N).

    Raises
    ------
    np.linalg.LinAlgError
        If the internal feedback solve is singular.
    """
    return None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve


def _constant(value):
    array = np.asarray(value, dtype=np.complex128)
    return array, np.zeros_like(array)


def _add(left, right):
    return left[0] + right[0], left[1] + right[1]


def _subtract(left, right):
    return left[0] - right[0], left[1] - right[1]


def _multiply(left, right):
    return (
        left[0] @ right[0],
        left[1] @ right[0] + left[0] @ right[1],
    )


def _solve(left, right):
    value = solve(left[0], right[0])
    tangent = solve(left[0], right[1] - left[1] @ value)
    return value, tangent


def _oracle_redheffer_compose(
    left: np.ndarray,
    left_tangent: np.ndarray,
    right: np.ndarray,
    right_tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_blocks = [
        [
            (left[row, column], left_tangent[row, column])
            for column in range(2)
        ]
        for row in range(2)
    ]

    right_blocks = [
        [
            (right[row, column], right_tangent[row, column])
            for column in range(2)
        ]
        for row in range(2)
    ]

    identity = _constant(np.eye(left.shape[-1]))

    feedback = _subtract(
        identity,
        _multiply(left_blocks[1][1], right_blocks[0][0]),
    )

    internal_left = _solve(
        feedback,
        left_blocks[1][0],
    )

    internal_right = _solve(
        feedback,
        _multiply(left_blocks[1][1], right_blocks[0][1]),
    )

    result = [
        [
            _add(
                left_blocks[0][0],
                _multiply(
                    _multiply(
                        left_blocks[0][1],
                        right_blocks[0][0],
                    ),
                    internal_left,
                ),
            ),
            _multiply(
                left_blocks[0][1],
                _add(
                    right_blocks[0][1],
                    _multiply(
                        right_blocks[0][0],
                        internal_right,
                    ),
                ),
            ),
        ],
        [
            _multiply(
                right_blocks[1][0],
                internal_left,
            ),
            _add(
                right_blocks[1][1],
                _multiply(
                    right_blocks[1][0],
                    internal_right,
                ),
            ),
        ],
    ]

    scattering = np.array(
        [[pair[0] for pair in row] for row in result]
    )
    tangent = np.array(
        [[pair[1] for pair in row] for row in result]
    )

    return scattering, tangent

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'identity = np.array([[np.zeros((2,2)), np.eye(2)], [np.eye(2), np.zeros((2,2))]])\n'
               'blocks = np.arange(16, dtype=float).reshape(2,2,2,2) * 0.01\n'
               'zeros = np.zeros_like(blocks)\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        assert np.array_equal(argument, before), "input arrays must remain unchanged"\n'
               '    return result\n',
      'call': '_preserving_call(redheffer_compose, identity, zeros, blocks, blocks * 0.2)',
      'gold_call': '_preserving_call(_oracle_redheffer_compose, identity, zeros, blocks, blocks * '
                   '0.2)'},
     {'setup': 'import numpy as np\n'
               'identity = np.array([[np.zeros((2,2)), np.eye(2)], [np.eye(2), np.zeros((2,2))]])\n'
               'blocks = np.arange(16, dtype=float).reshape(2,2,2,2) * 0.01\n'
               'zeros = np.zeros_like(blocks)\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        assert np.array_equal(argument, before), "input arrays must remain unchanged"\n'
               '    return result\n',
      'call': '_preserving_call(redheffer_compose, blocks, blocks * 0.2, identity, zeros)',
      'gold_call': '_preserving_call(_oracle_redheffer_compose, blocks, blocks * 0.2, identity, '
                   'zeros)'},
     {'setup': 'import numpy as np\n'
               'identity = np.array([[np.zeros((2,2)), np.eye(2)], [np.eye(2), np.zeros((2,2))]])\n'
               'blocks = np.arange(16, dtype=float).reshape(2,2,2,2) * 0.01\n'
               'zeros = np.zeros_like(blocks)\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        assert np.array_equal(argument, before), "input arrays must remain unchanged"\n'
               '    return result\n',
      'call': '_preserving_call(redheffer_compose, blocks, blocks * 0.2, blocks[::-1, ::-1] * 0.7, '
              'blocks * 0.3)',
      'gold_call': '_preserving_call(_oracle_redheffer_compose, blocks, blocks * 0.2, blocks[::-1, '
                   '::-1] * 0.7, blocks * 0.3)'}]
