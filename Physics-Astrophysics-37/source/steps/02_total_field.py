"""
Total magnetic field of the multimode wave packet at one phase point.

Total magnetic field of the multimode wave packet at one phase point.

Each mode contributes a transverse fluctuation whose orientation is fixed by the
common propagation angle, and the uniform background adds a unit vector along the
parallel direction. Evaluating the total field at a given vector of mode phases is
the elementary operation on which every later diagnostic rests.

Returns
-------
The three Cartesian components of the total dimensionless magnetic field.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

import numpy as np


def total_field(phases: np.ndarray, amplitude: np.ndarray, tan_alpha: float = 4.4) -> np.ndarray:
    """Return the total dimensionless magnetic field at one phase vector.

    Parameters
    ----------
    phases : numpy.ndarray
        Mode phases, one per mode.
    amplitude : numpy.ndarray
        Mode amplitudes, same length as ``phases``.
    tan_alpha : float
        Tangent of the common propagation angle. Must be positive.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(3,)`` with the Cartesian components of the field,
        normalised to the background field strength.

    Raises
    ------
    ValueError
        If ``phases`` and ``amplitude`` do not have the same shape, if
        ``phases`` is not a non-empty one-dimensional array, or if
        ``tan_alpha`` is not strictly positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_total_field(phases: np.ndarray, amplitude: np.ndarray, tan_alpha: float = 4.4) -> np.ndarray:
    phases = np.asarray(phases, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    if phases.shape != amplitude.shape:
        raise ValueError("phases and amplitude must have the same shape")
    if phases.ndim != 1 or phases.size < 1:
        raise ValueError("phases must be a non-empty one-dimensional array")
    if float(tan_alpha) <= 0.0:
        raise ValueError("tan_alpha must be positive")
    alpha = math.atan(float(tan_alpha))
    a_sum = float(np.sum(amplitude * np.sin(phases)))
    d_sum = float(np.sum(amplitude * np.cos(phases)))
    return np.array([-math.cos(alpha) * a_sum, d_sum, 1.0 + math.sin(alpha) * a_sum])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the original cases with independent candidate/reference dependencies."""
    return [{'setup': 'def _case_run(_tested_function, mode_spectrum):\n'
               '    import numpy as np\n'
               '    s = mode_spectrum()\n'
               '    p = np.linspace(0.3, 5.1, s.shape[1])\n'
               '    return _tested_function(p.copy(), s[1].copy())\n',
      'call': '_case_run(total_field, mode_spectrum)',
      'gold_call': '_case_run(_oracle_total_field, _oracle_mode_spectrum)'},
     {'setup': 'def _case_run(_tested_function, mode_spectrum):\n'
               '    import numpy as np\n'
               '    s = mode_spectrum()\n'
               '    z = np.zeros(s.shape[1])\n'
               '    return _tested_function(z.copy(), s[1].copy())\n',
      'call': '_case_run(total_field, mode_spectrum)',
      'gold_call': '_case_run(_oracle_total_field, _oracle_mode_spectrum)'},
     {'setup': 'import numpy as np\n',
      'call': 'total_field(np.array([1.25]), np.array([0.4]), tan_alpha=12.0)',
      'gold_call': '_oracle_total_field(np.array([1.25]), np.array([0.4]), tan_alpha=12.0)'},
     {'setup': 'import numpy as np\n'
               'def _probe_field(fn):\n'
               '    try:\n'
               '        fn(np.array([0.1, 0.2]), np.array([0.3]))\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_probe_field(total_field)',
      'gold_call': '_probe_field(_oracle_total_field)'}]
