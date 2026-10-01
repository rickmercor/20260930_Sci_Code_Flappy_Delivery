#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math

import numpy as np


def mode_spectrum(
    num_modes: int = 11,
    bw2: float = 0.19,
    omega1: float = 0.11,
    q: float = 1.667,
    tan_alpha: float = 4.4,
    band_width: float = 0.08,
) -> np.ndarray:
    if int(num_modes) < 1:
        raise ValueError("num_modes must be at least 1")
    if float(bw2) <= 0.0:
        raise ValueError("bw2 must be positive")
    if float(omega1) <= 0.0:
        raise ValueError("omega1 must be positive")
    if float(tan_alpha) <= 0.0:
        raise ValueError("tan_alpha must be positive")
    if float(band_width) <= 0.0:
        raise ValueError("band_width must be positive")
    omega = np.linspace(float(omega1), float(omega1) + float(band_width), int(num_modes))
    weights = (omega / float(omega1)) ** (-float(q))
    amplitude = np.sqrt(float(bw2) * weights / weights.sum())
    k_par = omega.copy()
    k_perp = k_par * float(tan_alpha)
    return np.vstack((omega, amplitude, k_perp, k_par))

import math

import numpy as np


def total_field(phases: np.ndarray, amplitude: np.ndarray, tan_alpha: float = 4.4) -> np.ndarray:
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

import math

import numpy as np


def field_gradient_tensor(
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

import numpy as np


def curvature_radius(field: np.ndarray, gradient: np.ndarray) -> float:
    field = np.asarray(field, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if field.shape != (3,):
        raise ValueError("field must have shape (3,)")
    if gradient.shape != (3, 3):
        raise ValueError("gradient must have shape (3, 3)")
    magnitude_squared = float(field @ field)
    if magnitude_squared <= 0.0:
        raise ValueError("field magnitude must be positive")
    directional = field @ gradient
    bend = float(np.linalg.norm(directional))
    if bend <= 0.0:
        raise ValueError("field line is locally straight, curvature radius undefined")
    return magnitude_squared / bend

import numpy as np


def gradient_anisotropy_ratio(gradient: np.ndarray) -> float:
    gradient = np.asarray(gradient, dtype=float)
    if gradient.shape != (3, 3):
        raise ValueError("gradient must have shape (3, 3)")
    unit = np.array([0.0, 0.0, 1.0])
    full = float(np.linalg.norm(gradient))
    if full <= 0.0:
        raise ValueError("gradient vanishes, ratio undefined")
    across = float(np.linalg.norm(gradient - np.outer(unit, unit @ gradient)))
    if across <= 0.0:
        raise ValueError("cross-field gradient vanishes, ratio undefined")
    return full / across

import numpy as np


def effective_curvature_parameter(
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
    field = total_field(phases, amplitude, tan_alpha)
    gradient = field_gradient_tensor(phases, omega, amplitude, tan_alpha)
    radius = curvature_radius(field, gradient)
    ratio = gradient_anisotropy_ratio(gradient)
    gyroradius = float(speed) / float(np.linalg.norm(field))
    return float(radius / gyroradius * ratio)

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


def phase_space_minimum(
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
    value = effective_curvature_parameter(best, spectrum, float(tan_alpha), float(speed))
    return np.concatenate(([value], best))

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
    if float(speed) <= 0.0:
        raise ValueError("speed must be positive")
    if int(num_starts) < 1:
        raise ValueError("num_starts must be at least 1")
    if int(num_iterations) < 1:
        raise ValueError("num_iterations must be at least 1")
    spectrum = mode_spectrum(num_modes, bw2, omega1, q, tan_alpha)
    located = phase_space_minimum(
        spectrum, tan_alpha, speed, num_starts, seed, num_iterations)
    minimising_phases = np.asarray(located[1:], dtype=float)
    field = total_field(minimising_phases, spectrum[1], tan_alpha)
    gradient = field_gradient_tensor(
        minimising_phases, spectrum[0], spectrum[1], tan_alpha)
    radius = curvature_radius(field, gradient)
    ratio = gradient_anisotropy_ratio(gradient)
    if not (radius > 0.0 and ratio >= 1.0):
        raise ValueError("located phase point is not a valid curvature minimum")
    return effective_curvature_parameter(
        minimising_phases, spectrum, tan_alpha, speed)
SCICODE_GOLD_EOF
