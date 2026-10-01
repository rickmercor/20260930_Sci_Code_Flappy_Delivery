"""
Construct the source's gap-referenced amplitude-scattering matrix for one homogeneous layer and its directional derivative from the supplied transformed modal and propagation matrices.

The reference is the fixed vacuum medium specified by vacuum_root. Use the incoming/outgoing conventions of the main problem, retain all supplied channels, and preserve matrix-product order.

Returns
-------
Return (S,S_tangent), complex128 arrays of shape (2,2,N,N). The axes are output port, input port, output harmonic, and input harmonic. Port0 is left and port1 is right: S[0,0] is left reflection, S[0,1] right-to-left transmission, S[1,0] left-to-right transmission, and S[1,1] right reflection. S_tangent contains the derivative induced by modal_tangent and propagation_tangent with vacuum_root fixed. Return amplitudes, preserve harmonic order, and do not mutate the inputs. A singular required solve may raise LinAlgError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def layer_scattering(
    modal: np.ndarray,
    modal_tangent: np.ndarray,
    propagation: np.ndarray,
    propagation_tangent: np.ndarray,
    vacuum_root: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct layer scattering and its tangent.

    Parameters
    ----------
    modal : np.ndarray
        Modal matrix V of shape (N, N).
    modal_tangent : np.ndarray
        Directional derivative of V.
    propagation : np.ndarray
        Propagation matrix X of shape (N, N).
    propagation_tangent : np.ndarray
        Directional derivative of X.
    vacuum_root : np.ndarray
        Fixed vacuum-root vector of shape (N,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        S and dS, each complex128 with shape (2, 2, N, N).

    Raises
    ------
    np.linalg.LinAlgError
        If a required matrix solve is singular.
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


def _oracle_layer_scattering(
    modal: np.ndarray,
    modal_tangent: np.ndarray,
    propagation: np.ndarray,
    propagation_tangent: np.ndarray,
    vacuum_root: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    identity = _constant(np.eye(len(vacuum_root)))

    impedance = _solve(
        (modal, modal_tangent),
        _constant(np.diag(vacuum_root)),
    )

    plus = _add(identity, impedance)
    minus = _subtract(identity, impedance)
    propagation_pair = (propagation, propagation_tangent)

    intermediate = _multiply(
        _multiply(propagation_pair, minus),
        _solve(plus, propagation_pair),
    )

    denominator = _subtract(
        plus,
        _multiply(intermediate, minus),
    )

    reflection = _solve(
        denominator,
        _subtract(_multiply(intermediate, plus), minus),
    )

    transmission = _solve(
        denominator,
        _multiply(
            propagation_pair,
            _subtract(
                plus,
                _multiply(minus, _solve(plus, minus)),
            ),
        ),
    )

    scattering = np.array(
        [
            [reflection[0], transmission[0]],
            [transmission[0], reflection[0]],
        ]
    )

    tangent = np.array(
        [
            [reflection[1], transmission[1]],
            [transmission[1], reflection[1]],
        ]
    )

    return scattering, tangent

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'base = np.array([[0.7, 0.1j], [-0.1j, 0.9]], dtype=np.complex128)\n'
               'direction = np.array([[0.2, 0.04], [0.04, -0.1]], dtype=np.complex128)\n'
               'vacuum = np.array([0.5j, 1.2])\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        assert np.array_equal(argument, before), "input arrays must remain unchanged"\n'
               '    return result\n',
      'call': '_preserving_call(layer_scattering, np.diag(vacuum), direction * 0, '
              'np.diag(np.exp(-vacuum * 0.7)), direction * 0, vacuum)',
      'gold_call': '_preserving_call(_oracle_layer_scattering, np.diag(vacuum), direction * 0, '
                   'np.diag(np.exp(-vacuum * 0.7)), direction * 0, vacuum)'},
     {'setup': 'import numpy as np\n'
               'base = np.array([[0.7, 0.1j], [-0.1j, 0.9]], dtype=np.complex128)\n'
               'direction = np.array([[0.2, 0.04], [0.04, -0.1]], dtype=np.complex128)\n'
               'vacuum = np.array([0.5j, 1.2])\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        assert np.array_equal(argument, before), "input arrays must remain unchanged"\n'
               '    return result\n',
      'call': '_preserving_call(layer_scattering, base, direction, np.eye(2), direction * 0, vacuum)',
      'gold_call': '_preserving_call(_oracle_layer_scattering, base, direction, np.eye(2), direction * '
                   '0, vacuum)'},
     {'setup': 'import numpy as np\n'
               'base = np.array([[0.7, 0.1j], [-0.1j, 0.9]], dtype=np.complex128)\n'
               'direction = np.array([[0.2, 0.04], [0.04, -0.1]], dtype=np.complex128)\n'
               'vacuum = np.array([0.5j, 1.2])\n'
               'def _preserving_call(function, *arguments):\n'
               '    snapshots = [(argument, argument.copy()) for argument in arguments if '
               'isinstance(argument, np.ndarray)]\n'
               '    result = function(*arguments)\n'
               '    for argument, before in snapshots:\n'
               '        assert np.array_equal(argument, before), "input arrays must remain unchanged"\n'
               '    return result\n',
      'call': '_preserving_call(layer_scattering, base, direction, base * 0.3, direction * 0.1, '
              'vacuum)',
      'gold_call': '_preserving_call(_oracle_layer_scattering, base, direction, base * 0.3, direction '
                   '* 0.1, vacuum)'}]
