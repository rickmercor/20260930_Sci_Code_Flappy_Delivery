"""
Global minimum of the curvature parameter over the phase torus.

Global minimum of the curvature parameter over the phase torus.

The parameter of the previous step depends on every mode phase, so the quantity
that decides chaos onset is its smallest value anywhere on the torus of phases.
That minimum cannot be written down in closed form once several modes are
present, so it is located numerically, and the search has to be thorough enough
that repeating it from independent starting points returns the same value.

The minimising phase vector is not unique: replacing every phase psi by pi - psi,
for example, leaves the parameter unchanged. Any phase vector that attains the
global minimum is a valid output, provided the returned value is the parameter
evaluated at the returned phases.

Returns
-------
The smallest value the curvature parameter attains, followed by the mode phases
that attain it, each reduced to the interval [0, 2 pi).

Returns
-------
The smallest value the curvature parameter attains, followed by the mode phases that attain it, each reduced to the interval [0, 2 pi).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

import numpy as np


def phase_space_minimum(
    spectrum: np.ndarray,
    tan_alpha: float = 4.4,
    speed: float = 1.0,
    num_starts: int = 4000,
    seed: int = 20260909,
    num_iterations: int = 4000,
) -> np.ndarray:
    """Locate the global minimum of the curvature parameter over all mode phases.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectrum array of shape ``(4, num_modes)`` as produced by the first step.
    tan_alpha : float
        Tangent of the common propagation angle. Must be positive.
    speed : float
        Dimensionless ion speed entering the gyroradius. Must be positive.
    num_starts : int
        Number of independent starting phase vectors. Must be at least 1.
    seed : int
        Seed of the generator that draws the starting phase vectors.
    num_iterations : int
        Number of descent iterations applied to every start. Must be at least 1.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(num_modes + 1,)`` holding the minimum value found
        followed by the mode phases that attain it, each reduced to the
        interval ``[0, 2*pi)``. The value equals the curvature parameter
        evaluated at the returned phases and is converged with respect to the
        number of starts. Any minimising phase vector is acceptable, since the
        minimiser is not unique.

    Raises
    ------
    ValueError
        If ``spectrum`` does not have shape ``(4, num_modes)``, if ``tan_alpha``
        or ``speed`` is not strictly positive, or if ``num_starts`` or
        ``num_iterations`` is below one.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _descent_objective_parts(phases, omega, amplitude, sin_alpha, cos_alpha):
    sin_phase, cos_phase = np.sin(phases), np.cos(phases)
    a_sum = (amplitude * sin_phase).sum(-1)
    d_sum = (amplitude * cos_phase).sum(-1)
    c_sum = (omega * amplitude * cos_phase).sum(-1)
    s_sum = (omega * amplitude * sin_phase).sum(-1)
    strength = 1.0 + a_sum * a_sum + d_sum * d_sum + 2.0 * sin_alpha * a_sum
    bend = c_sum * c_sum + s_sum * s_sum
    return sin_phase, cos_phase, a_sum, d_sum, c_sum, s_sum, strength, bend


def _descent_value(phases, omega, amplitude, sin_alpha, cos_alpha, speed):
    _, _, _, _, _, _, strength, bend = _descent_objective_parts(
        phases, omega, amplitude, sin_alpha, cos_alpha)
    return strength ** 1.5 / (speed * np.sqrt(bend) * sin_alpha)


def _descent_log_gradient(phases, omega, amplitude, sin_alpha, cos_alpha):
    sin_phase, cos_phase, a_sum, d_sum, c_sum, s_sum, strength, bend = \
        _descent_objective_parts(phases, omega, amplitude, sin_alpha, cos_alpha)
    d_a = amplitude * cos_phase
    d_d = -amplitude * sin_phase
    d_c = -omega * amplitude * sin_phase
    d_s = omega * amplitude * cos_phase
    d_strength = (2.0 * a_sum + 2.0 * sin_alpha)[..., None] * d_a + (2.0 * d_sum)[..., None] * d_d
    d_bend = 2.0 * (c_sum[..., None] * d_c + s_sum[..., None] * d_s)
    return (1.5 * d_strength / strength[..., None]
            - 0.5 * d_bend / bend[..., None])


def _oracle_phase_space_minimum(
    spectrum: np.ndarray,
    tan_alpha: float = 4.4,
    speed: float = 1.0,
    num_starts: int = 4000,
    seed: int = 20260909,
    num_iterations: int = 4000,
) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.ndim != 2 or spectrum.shape[0] != 4:
        raise ValueError("spectrum must have shape (4, num_modes)")
    if float(tan_alpha) <= 0.0:
        raise ValueError("tan_alpha must be positive")
    if float(speed) <= 0.0:
        raise ValueError("speed must be positive")
    if int(num_starts) < 1:
        raise ValueError("num_starts must be at least 1")
    if int(num_iterations) < 1:
        raise ValueError("num_iterations must be at least 1")
    omega, amplitude = spectrum[0], spectrum[1]
    alpha = math.atan(float(tan_alpha))
    sin_alpha, cos_alpha = math.sin(alpha), math.cos(alpha)
    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(0.0, 2.0 * math.pi, size=(int(num_starts), spectrum.shape[1]))
    step = np.full(int(num_starts), 0.2)
    current = np.log(_descent_value(phases, omega, amplitude, sin_alpha, cos_alpha, float(speed)))
    for _ in range(int(num_iterations)):
        trial = phases - step[:, None] * _descent_log_gradient(
            phases, omega, amplitude, sin_alpha, cos_alpha)
        candidate = np.log(_descent_value(trial, omega, amplitude, sin_alpha, cos_alpha, float(speed)))
        better = candidate < current
        phases[better] = trial[better]
        current[better] = candidate[better]
        step = np.clip(np.where(better, step * 1.1, step * 0.5), 1e-12, 1.0)
    best = phases[int(np.argmin(current))] % (2.0 * math.pi)
    value = _oracle_effective_curvature_parameter(best, spectrum, float(tan_alpha), float(speed))
    return np.concatenate(([value], best))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the original cases with independent candidate/reference dependencies."""
    return [{'setup': 'def _case_run(_tested_function, effective_curvature_parameter, mode_spectrum):\n'
               '    import numpy as np\n'
               '\n'
               '    def _pack_minimum(result, spectrum, tan_alpha=4.4, speed=1.0):\n'
               '        r = np.asarray(result, dtype=float)\n'
               '        if r.shape != (spectrum.shape[1] + 1,) or not np.all(np.isfinite(r)) or r[0] '
               '<= 0.0:\n'
               '            return np.zeros(4, dtype=float)\n'
               '        phases = r[1:]\n'
               '        in_range = float(np.all((phases >= 0.0) & (phases < 2.0 * np.pi)))\n'
               '        again = effective_curvature_parameter(phases.copy(), spectrum.copy(), '
               'tan_alpha, speed)\n'
               '        consistent = float(abs(again - r[0]) <= 1e-09 * abs(again))\n'
               '        cos_alpha = 1.0 / np.sqrt(1.0 + tan_alpha * tan_alpha)\n'
               '        sin_alpha = tan_alpha * cos_alpha\n'
               '        scale = 0.5 * cos_alpha ** 3 / (speed * sin_alpha * np.sum(spectrum[0] * '
               'spectrum[1]))\n'
               '        return np.array([r[0] / scale - 1.0, 1.0, in_range, consistent])\n'
               '    s = mode_spectrum(num_modes=3, bw2=0.19, omega1=0.11, tan_alpha=4.4)\n'
               '    return _pack_minimum(_tested_function(s.copy(), num_starts=300, '
               'num_iterations=800), s)\n',
      'call': '_case_run(phase_space_minimum, effective_curvature_parameter, mode_spectrum)',
      'gold_call': '_case_run(_oracle_phase_space_minimum, _oracle_effective_curvature_parameter, '
                   '_oracle_mode_spectrum)',
      'tol': 1e-07},
     {'setup': 'def _case_run(_tested_function, effective_curvature_parameter, mode_spectrum):\n'
               '    import numpy as np\n'
               '\n'
               '    def _pack_minimum(result, spectrum, tan_alpha=4.4, speed=1.0):\n'
               '        r = np.asarray(result, dtype=float)\n'
               '        if r.shape != (spectrum.shape[1] + 1,) or not np.all(np.isfinite(r)) or r[0] '
               '<= 0.0:\n'
               '            return np.zeros(4, dtype=float)\n'
               '        phases = r[1:]\n'
               '        in_range = float(np.all((phases >= 0.0) & (phases < 2.0 * np.pi)))\n'
               '        again = effective_curvature_parameter(phases.copy(), spectrum.copy(), '
               'tan_alpha, speed)\n'
               '        consistent = float(abs(again - r[0]) <= 1e-09 * abs(again))\n'
               '        cos_alpha = 1.0 / np.sqrt(1.0 + tan_alpha * tan_alpha)\n'
               '        sin_alpha = tan_alpha * cos_alpha\n'
               '        scale = 0.5 * cos_alpha ** 3 / (speed * sin_alpha * np.sum(spectrum[0] * '
               'spectrum[1]))\n'
               '        return np.array([r[0] / scale - 1.0, 1.0, in_range, consistent])\n'
               '    s1 = mode_spectrum(num_modes=1, bw2=0.19, omega1=0.11, tan_alpha=4.4)\n'
               '    return _pack_minimum(_tested_function(s1.copy(), num_starts=64, '
               'num_iterations=600), s1)\n',
      'call': '_case_run(phase_space_minimum, effective_curvature_parameter, mode_spectrum)',
      'gold_call': '_case_run(_oracle_phase_space_minimum, _oracle_effective_curvature_parameter, '
                   '_oracle_mode_spectrum)',
      'tol': 1e-07},
     {'setup': 'def _case_run(_tested_function, effective_curvature_parameter, mode_spectrum):\n'
               '    import numpy as np\n'
               '\n'
               '    def _pack_minimum(result, spectrum, tan_alpha=4.4, speed=1.0):\n'
               '        r = np.asarray(result, dtype=float)\n'
               '        if r.shape != (spectrum.shape[1] + 1,) or not np.all(np.isfinite(r)) or r[0] '
               '<= 0.0:\n'
               '            return np.zeros(4, dtype=float)\n'
               '        phases = r[1:]\n'
               '        in_range = float(np.all((phases >= 0.0) & (phases < 2.0 * np.pi)))\n'
               '        again = effective_curvature_parameter(phases.copy(), spectrum.copy(), '
               'tan_alpha, speed)\n'
               '        consistent = float(abs(again - r[0]) <= 1e-09 * abs(again))\n'
               '        cos_alpha = 1.0 / np.sqrt(1.0 + tan_alpha * tan_alpha)\n'
               '        sin_alpha = tan_alpha * cos_alpha\n'
               '        scale = 0.5 * cos_alpha ** 3 / (speed * sin_alpha * np.sum(spectrum[0] * '
               'spectrum[1]))\n'
               '        return np.array([r[0] / scale - 1.0, 1.0, in_range, consistent])\n'
               '    s4 = mode_spectrum(num_modes=4, bw2=0.35, omega1=0.09, q=2.2, tan_alpha=7.0)\n'
               '    return _pack_minimum(_tested_function(s4.copy(), tan_alpha=7.0, speed=1.5, '
               'num_starts=200, seed=77, num_iterations=3000), s4, tan_alpha=7.0, speed=1.5)\n',
      'call': '_case_run(phase_space_minimum, effective_curvature_parameter, mode_spectrum)',
      'gold_call': '_case_run(_oracle_phase_space_minimum, _oracle_effective_curvature_parameter, '
                   '_oracle_mode_spectrum)',
      'tol': 1e-07},
     {'setup': 'def _case_run(_tested_function, mode_spectrum):\n'
               '    import numpy as np\n'
               '    s = mode_spectrum()\n'
               '\n'
               '    def _probe_min(fn):\n'
               '        try:\n'
               '            fn(s.copy(), num_starts=0)\n'
               '        except ValueError:\n'
               '            return 1\n'
               '        except Exception:\n'
               '            return 2\n'
               '        return 0\n'
               '    return _probe_min(_tested_function)\n',
      'call': '_case_run(phase_space_minimum, mode_spectrum)',
      'gold_call': '_case_run(_oracle_phase_space_minimum, _oracle_mode_spectrum)'}]
