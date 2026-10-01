"""
The dimensionless curvature parameter at one phase point.

The dimensionless curvature parameter at one phase point.

The criterion that decides whether an ion keeps its magnetic moment compares the
field-line curvature radius with the ion gyroradius, and weights that comparison
by how much of the field gradient lies across the background field, which is the
direction the gyromotion samples. Assembling those pieces at a single phase point gives the parameter
whose phase-space minimum governs the onset of chaos.

Returns
-------
The dimensionless curvature parameter evaluated at the given phase vector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def effective_curvature_parameter(
    phases: np.ndarray,
    spectrum: np.ndarray,
    tan_alpha: float = 4.4,
    speed: float = 1.0,
) -> float:
    """Return the dimensionless curvature parameter at one phase vector.

    Parameters
    ----------
    phases : numpy.ndarray
        Mode phases, one per mode.
    spectrum : numpy.ndarray
        Spectrum array of shape ``(4, num_modes)`` as produced by the first step.
    tan_alpha : float
        Tangent of the common propagation angle. Must be positive.
    speed : float
        Dimensionless ion speed entering the gyroradius. Must be positive.

    Returns
    -------
    float
        The parameter value at that phase point.

    Raises
    ------
    ValueError
        If ``spectrum`` does not have shape ``(4, num_modes)``, if ``phases``
        does not hold exactly one entry per mode, or if ``speed`` is not
        strictly positive.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_effective_curvature_parameter(
    phases: np.ndarray,
    spectrum: np.ndarray,
    tan_alpha: float = 4.4,
    speed: float = 1.0,
) -> float:
    phases = np.asarray(phases, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.ndim != 2 or spectrum.shape[0] != 4:
        raise ValueError("spectrum must have shape (4, num_modes)")
    if phases.ndim != 1 or phases.size != spectrum.shape[1]:
        raise ValueError("phases must have one entry per mode")
    if float(speed) <= 0.0:
        raise ValueError("speed must be positive")
    omega, amplitude = spectrum[0], spectrum[1]
    field = _oracle_total_field(phases, amplitude, tan_alpha)
    gradient = _oracle_field_gradient_tensor(phases, omega, amplitude, tan_alpha)
    radius = _oracle_curvature_radius(field, gradient)
    ratio = _oracle_gradient_anisotropy_ratio(gradient)
    gyroradius = float(speed) / float(np.linalg.norm(field))
    return float(radius / gyroradius * ratio)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the original cases with independent candidate/reference dependencies."""
    return [{'setup': 'def _case_run(_tested_function, mode_spectrum):\n'
               '    import numpy as np\n'
               '    s = mode_spectrum()\n'
               '    p = np.linspace(0.3, 5.1, s.shape[1])\n'
               '    return _tested_function(p.copy(), s.copy())\n',
      'call': '_case_run(effective_curvature_parameter, mode_spectrum)',
      'gold_call': '_case_run(_oracle_effective_curvature_parameter, _oracle_mode_spectrum)'},
     {'setup': 'def _case_run(_tested_function, mode_spectrum):\n'
               '    import numpy as np\n'
               '    s = mode_spectrum()\n'
               '    z = np.zeros(s.shape[1])\n'
               '    return _tested_function(z.copy(), s.copy(), speed=0.25)\n',
      'call': '_case_run(effective_curvature_parameter, mode_spectrum)',
      'gold_call': '_case_run(_oracle_effective_curvature_parameter, _oracle_mode_spectrum)'},
     {'setup': 'def _case_run(_tested_function, mode_spectrum):\n'
               '    import numpy as np\n'
               '    s3 = mode_spectrum(num_modes=2, bw2=0.5, omega1=0.4, q=0.5, tan_alpha=8.0)\n'
               '    p3 = np.array([4.7, 2.2])\n'
               '    return _tested_function(p3.copy(), s3.copy(), tan_alpha=8.0, speed=2.0)\n',
      'call': '_case_run(effective_curvature_parameter, mode_spectrum)',
      'gold_call': '_case_run(_oracle_effective_curvature_parameter, _oracle_mode_spectrum)'},
     {'setup': 'def _case_run(_tested_function, mode_spectrum):\n'
               '    import numpy as np\n'
               '    s = mode_spectrum()\n'
               '\n'
               '    def _probe_par(fn):\n'
               '        try:\n'
               '            fn(np.zeros(s.shape[1]), s.copy(), speed=0.0)\n'
               '        except ValueError:\n'
               '            return 1\n'
               '        except Exception:\n'
               '            return 2\n'
               '        return 0\n'
               '    return _probe_par(_tested_function)\n',
      'call': '_case_run(effective_curvature_parameter, mode_spectrum)',
      'gold_call': '_case_run(_oracle_effective_curvature_parameter, _oracle_mode_spectrum)'}]
