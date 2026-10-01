"""
End-to-end benchmark value (final orchestrator step).

End-to-end benchmark value (final orchestrator step).

Chain the spectrum construction and the phase-torus search to produce the single
benchmark number: the smallest value the dimensionless curvature parameter attains
anywhere in the phase space of the specified oblique multimode field, at unit ion
speed.

Returns
-------
The benchmark curvature minimum for the specified configuration.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def chaos_onset_curvature_minimum(
    num_modes: int = 11,
    bw2: float = 0.19,
    omega1: float = 0.11,
    q: float = 1.667,
    tan_alpha: float = 4.4,
    speed: float = 1.0,
    num_starts: int = 4000,
    seed: int = 20260909,
    num_iterations: int = 4000,
) -> float:
    """Return the benchmark phase-space minimum of the curvature parameter.

    Parameters
    ----------
    num_modes : int
        Number of wave modes in the spectrum.
    bw2 : float
        Total dimensionless wave power.
    omega1 : float
        Lowest dimensionless mode frequency.
    q : float
        Exponent of the amplitude power law in frequency.
    tan_alpha : float
        Tangent of the common propagation angle.
    speed : float
        Dimensionless ion speed entering the gyroradius.
    num_starts : int
        Number of independent starting phase vectors for the search.
    seed : int
        Seed of the generator that draws the starting phase vectors.
    num_iterations : int
        Number of descent iterations applied to every start.

    Returns
    -------
    float
        The benchmark value.

    Raises
    ------
    ValueError
        If ``speed`` is not strictly positive, if ``num_starts`` or
        ``num_iterations`` is below one, or if the spectrum parameters are
        invalid (``num_modes`` below one; ``bw2``, ``omega1`` or ``tan_alpha``
        not strictly positive).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_chaos_onset_curvature_minimum(
    num_modes: int = 11,
    bw2: float = 0.19,
    omega1: float = 0.11,
    q: float = 1.667,
    tan_alpha: float = 4.4,
    speed: float = 1.0,
    num_starts: int = 4000,
    seed: int = 20260909,
    num_iterations: int = 4000,
) -> float:
    if float(speed) <= 0.0:
        raise ValueError("speed must be positive")
    if int(num_starts) < 1:
        raise ValueError("num_starts must be at least 1")
    if int(num_iterations) < 1:
        raise ValueError("num_iterations must be at least 1")
    spectrum = _oracle_mode_spectrum(num_modes, bw2, omega1, q, tan_alpha)
    located = _oracle_phase_space_minimum(
        spectrum, tan_alpha, speed, num_starts, seed, num_iterations)
    minimising_phases = np.asarray(located[1:], dtype=float)
    field = _oracle_total_field(minimising_phases, spectrum[1], tan_alpha)
    gradient = _oracle_field_gradient_tensor(
        minimising_phases, spectrum[0], spectrum[1], tan_alpha)
    radius = _oracle_curvature_radius(field, gradient)
    ratio = _oracle_gradient_anisotropy_ratio(gradient)
    if not (radius > 0.0 and ratio >= 1.0):
        raise ValueError("located phase point is not a valid curvature minimum")
    return _oracle_effective_curvature_parameter(
        minimising_phases, spectrum, tan_alpha, speed)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the original cases with independent candidate/reference dependencies."""
    return [{'setup': 'def _case_run(_tested_function, mode_spectrum):\n'
               '    import numpy as np\n'
               '\n'
               '    def _minimum_scale(spectrum, tan_alpha, speed):\n'
               '        cos_alpha = 1.0 / np.sqrt(1.0 + tan_alpha * tan_alpha)\n'
               '        sin_alpha = tan_alpha * cos_alpha\n'
               '        return 0.5 * cos_alpha ** 3 / (speed * sin_alpha * np.sum(spectrum[0] * '
               'spectrum[1]))\n'
               '\n'
               '    def _pack_scalar(value, scale):\n'
               '        v = np.asarray(value, dtype=float)\n'
               '        if v.shape != () or not np.isfinite(v.item()) or v.item() <= 0.0:\n'
               '            return np.zeros(2, dtype=float)\n'
               '        return np.array([v.item() / scale - 1.0, 1.0])\n'
               '    spectrum = mode_spectrum(num_modes=3)\n'
               '    scale = _minimum_scale(spectrum, 4.4, 1.0)\n'
               '    return _pack_scalar(_tested_function(num_modes=3, num_starts=300, '
               'num_iterations=800), scale)\n',
      'call': '_case_run(chaos_onset_curvature_minimum, mode_spectrum)',
      'gold_call': '_case_run(_oracle_chaos_onset_curvature_minimum, _oracle_mode_spectrum)',
      'tol': 1e-07},
     {'setup': 'def _case_run(_tested_function, mode_spectrum):\n'
               '    import numpy as np\n'
               '\n'
               '    def _minimum_scale(spectrum, tan_alpha, speed):\n'
               '        cos_alpha = 1.0 / np.sqrt(1.0 + tan_alpha * tan_alpha)\n'
               '        sin_alpha = tan_alpha * cos_alpha\n'
               '        return 0.5 * cos_alpha ** 3 / (speed * sin_alpha * np.sum(spectrum[0] * '
               'spectrum[1]))\n'
               '\n'
               '    def _pack_scalar(value, scale):\n'
               '        v = np.asarray(value, dtype=float)\n'
               '        if v.shape != () or not np.isfinite(v.item()) or v.item() <= 0.0:\n'
               '            return np.zeros(2, dtype=float)\n'
               '        return np.array([v.item() / scale - 1.0, 1.0])\n'
               '    spectrum = mode_spectrum(num_modes=2, bw2=0.05, omega1=0.3, q=1.0, tan_alpha=1.5)\n'
               '    scale = _minimum_scale(spectrum, 1.5, 1.0)\n'
               '    return _pack_scalar(_tested_function(num_modes=2, bw2=0.05, omega1=0.3, q=1.0, '
               'tan_alpha=1.5, num_starts=200, seed=5, num_iterations=600), scale)\n',
      'call': '_case_run(chaos_onset_curvature_minimum, mode_spectrum)',
      'gold_call': '_case_run(_oracle_chaos_onset_curvature_minimum, _oracle_mode_spectrum)',
      'tol': 1e-07},
     {'setup': 'def _case_run(_tested_function, mode_spectrum):\n'
               '    import numpy as np\n'
               '\n'
               '    def _minimum_scale(spectrum, tan_alpha, speed):\n'
               '        cos_alpha = 1.0 / np.sqrt(1.0 + tan_alpha * tan_alpha)\n'
               '        sin_alpha = tan_alpha * cos_alpha\n'
               '        return 0.5 * cos_alpha ** 3 / (speed * sin_alpha * np.sum(spectrum[0] * '
               'spectrum[1]))\n'
               '\n'
               '    def _pack_scalar(value, scale):\n'
               '        v = np.asarray(value, dtype=float)\n'
               '        if v.shape != () or not np.isfinite(v.item()) or v.item() <= 0.0:\n'
               '            return np.zeros(2, dtype=float)\n'
               '        return np.array([v.item() / scale - 1.0, 1.0])\n'
               '    spectrum = mode_spectrum(num_modes=5, bw2=0.22, omega1=0.06, q=2.0, '
               'tan_alpha=6.0)\n'
               '    scale = _minimum_scale(spectrum, 6.0, 0.8)\n'
               '    return _pack_scalar(_tested_function(num_modes=5, bw2=0.22, omega1=0.06, q=2.0, '
               'tan_alpha=6.0, speed=0.8, num_starts=250, seed=31, num_iterations=3000), scale)\n',
      'call': '_case_run(chaos_onset_curvature_minimum, mode_spectrum)',
      'gold_call': '_case_run(_oracle_chaos_onset_curvature_minimum, _oracle_mode_spectrum)',
      'tol': 1e-07},
     {'setup': 'import numpy as np\n'
               'def _probe_end(fn):\n'
               '    try:\n'
               '        fn(num_modes=3, bw2=-0.1)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_probe_end(chaos_onset_curvature_minimum)',
      'gold_call': '_probe_end(_oracle_chaos_onset_curvature_minimum)'}]
