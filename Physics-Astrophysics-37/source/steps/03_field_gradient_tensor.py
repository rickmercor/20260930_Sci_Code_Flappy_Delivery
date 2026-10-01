"""
Spatial gradient tensor of the multimode field.

Spatial gradient tensor of the multimode field.

Because every mode shares one propagation direction, the spatial dependence of the
whole packet enters through a single direction in space, and the gradient tensor
collapses accordingly. Building it explicitly is what lets the later diagnostics be
evaluated at any phase point without finite differences.

Returns
-------
The three by three spatial gradient tensor of the total field.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

import numpy as np


def field_gradient_tensor(
    phases: np.ndarray,
    omega: np.ndarray,
    amplitude: np.ndarray,
    tan_alpha: float = 4.4,
) -> np.ndarray:
    """Return the spatial gradient tensor of the total field at one phase vector.

    Parameters
    ----------
    phases : numpy.ndarray
        Mode phases, one per mode.
    omega : numpy.ndarray
        Mode frequencies, same length as ``phases``.
    amplitude : numpy.ndarray
        Mode amplitudes, same length as ``phases``.
    tan_alpha : float
        Tangent of the common propagation angle. Must be positive.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(3, 3)`` whose entry in row ``j`` and column ``i`` is the
        derivative of field component ``i`` with respect to coordinate ``j``.

    Raises
    ------
    ValueError
        If ``phases``, ``omega`` and ``amplitude`` do not share one shape, if
        ``phases`` is not a non-empty one-dimensional array, or if
        ``tan_alpha`` is not strictly positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_field_gradient_tensor(
    phases: np.ndarray,
    omega: np.ndarray,
    amplitude: np.ndarray,
    tan_alpha: float = 4.4,
) -> np.ndarray:
    phases = np.asarray(phases, dtype=float)
    omega = np.asarray(omega, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    if not (phases.shape == omega.shape == amplitude.shape):
        raise ValueError("phases, omega and amplitude must have the same shape")
    if phases.ndim != 1 or phases.size < 1:
        raise ValueError("phases must be a non-empty one-dimensional array")
    if float(tan_alpha) <= 0.0:
        raise ValueError("tan_alpha must be positive")
    alpha = math.atan(float(tan_alpha))
    direction = np.array([float(tan_alpha), 0.0, 1.0])
    c_sum = float(np.sum(omega * amplitude * np.cos(phases)))
    s_sum = float(np.sum(omega * amplitude * np.sin(phases)))
    response = np.array([-math.cos(alpha) * c_sum, -s_sum, math.sin(alpha) * c_sum])
    return np.outer(direction, response)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the original cases with independent candidate/reference dependencies."""
    return [{'setup': 'def _case_run(_tested_function, mode_spectrum):\n'
               '    import numpy as np\n'
               '    s = mode_spectrum()\n'
               '    p = np.linspace(0.3, 5.1, s.shape[1])\n'
               '    return _tested_function(p.copy(), s[0].copy(), s[1].copy())\n',
      'call': '_case_run(field_gradient_tensor, mode_spectrum)',
      'gold_call': '_case_run(_oracle_field_gradient_tensor, _oracle_mode_spectrum)'},
     {'setup': 'def _case_run(_tested_function, mode_spectrum):\n'
               '    import numpy as np\n'
               '    s = mode_spectrum()\n'
               '    z = np.zeros(s.shape[1])\n'
               '    return _tested_function(z.copy(), s[0].copy(), s[1].copy())\n',
      'call': '_case_run(field_gradient_tensor, mode_spectrum)',
      'gold_call': '_case_run(_oracle_field_gradient_tensor, _oracle_mode_spectrum)'},
     {'setup': 'import numpy as np\n',
      'call': 'field_gradient_tensor(np.array([2.0]), np.array([0.09]), np.array([0.25]), '
              'tan_alpha=0.5)',
      'gold_call': '_oracle_field_gradient_tensor(np.array([2.0]), np.array([0.09]), np.array([0.25]), '
                   'tan_alpha=0.5)'},
     {'setup': 'import numpy as np\n'
               'def _probe_grad(fn):\n'
               '    try:\n'
               '        fn(np.array([0.1]), np.array([0.2]), np.array([0.3]), tan_alpha=-1.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_probe_grad(field_gradient_tensor)',
      'gold_call': '_probe_grad(_oracle_field_gradient_tensor)'}]
