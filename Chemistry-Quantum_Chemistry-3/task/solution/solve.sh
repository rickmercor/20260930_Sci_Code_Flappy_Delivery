#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def build_powder_grid(n_beta: int, n_alpha: int, n_gamma: int) -> "np.ndarray":
    for name, value in (("n_beta", n_beta), ("n_alpha", n_alpha), ("n_gamma", n_gamma)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if int(value) < 1:
            raise ValueError(name + " must be >= 1")

    n_beta = int(n_beta)
    n_alpha = int(n_alpha)
    n_gamma = int(n_gamma)

    nodes, weights = np.polynomial.legendre.leggauss(n_beta)
    beta = np.arccos(nodes)
    # The Gauss-Legendre weights on [-1, 1] sum to 2; halving normalizes the
    # polar integral, and the azimuthal grids contribute equal weights.
    beta_weight = weights / 2.0
    alpha = 2.0 * np.pi * np.arange(n_alpha) / n_alpha
    gamma = 2.0 * np.pi * np.arange(n_gamma) / n_gamma

    grid = np.empty((n_beta * n_alpha * n_gamma, 4), dtype=float)
    row = 0
    for k in range(n_beta):
        for i in range(n_alpha):
            for l in range(n_gamma):
                grid[row] = (alpha[i], beta[k], gamma[l],
                             beta_weight[k] / (n_alpha * n_gamma))
                row += 1
    return grid

import numpy as np
from math import factorial


def _reduced_wigner_d2(beta: float) -> "np.ndarray":
    """Return the real reduced Wigner matrix d^(2)(beta) indexed from m = 2 down to m = -2."""
    rank = 2
    cos_half = np.cos(beta / 2.0)
    sin_half = np.sin(beta / 2.0)
    matrix = np.zeros((5, 5), dtype=float)
    orders = list(range(2, -3, -1))
    for row, m_out in enumerate(orders):
        for col, m_in in enumerate(orders):
            total = 0.0
            for k in range(2 * rank + 1):
                exponents = (rank + m_in - k, k, m_out - m_in + k, rank - m_out - k)
                if min(exponents) < 0:
                    continue
                numerator = ((-1.0) ** (m_out - m_in + k)) * np.sqrt(
                    factorial(rank + m_out) * factorial(rank - m_out)
                    * factorial(rank + m_in) * factorial(rank - m_in))
                denominator = 1.0
                for value in exponents:
                    denominator *= factorial(value)
                total += (numerator / denominator
                          * cos_half ** (2 * rank + m_in - m_out - 2 * k)
                          * sin_half ** (m_out - m_in + 2 * k))
            matrix[row, col] = total
    return matrix


def _wigner_index(order: int) -> int:
    """Map a second-rank order m to its row or column index in the reduced Wigner matrix."""
    return 2 - order


def _wigner_d2_element(matrix: "np.ndarray", m_out: int, m_in: int) -> float:
    """Return the reduced Wigner element d^(2)_{m_out, m_in} from a precomputed matrix."""
    return float(matrix[_wigner_index(m_out), _wigner_index(m_in)])


def compute_shielding_fourier_components(omega_iso: float, omega_aniso: float, eta: float,
                                                 alpha_pr: float, beta_pr: float, gamma_pr: float,
                                                 beta_rl: float) -> "np.ndarray":
    values = (omega_iso, omega_aniso, eta, alpha_pr, beta_pr, gamma_pr, beta_rl)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("all shielding parameters must be finite")
    eta = float(eta)
    if eta < 0.0 or eta > 1.0:
        raise ValueError("eta must satisfy 0 <= eta <= 1")

    d_pr = _reduced_wigner_d2(float(beta_pr))
    d_rl = _reduced_wigner_d2(float(beta_rl))

    def _wigner_rotation(m_out, m_in):
        return (np.exp(-1j * m_out * float(alpha_pr))
                * _wigner_d2_element(d_pr, m_out, m_in)
                * np.exp(-1j * m_in * float(gamma_pr)))

    components = np.zeros(5, dtype=complex)
    for index, m in enumerate(range(-2, 3)):
        principal = (_wigner_rotation(0, -m)
                     - (eta / np.sqrt(6.0)) * (_wigner_rotation(-2, -m) + _wigner_rotation(2, -m)))
        components[index] = (float(omega_aniso) * principal
                             * _wigner_d2_element(d_rl, -m, 0))
        if m == 0:
            components[index] += float(omega_iso)
    return components

import numpy as np


def compute_offset_trajectory(components: "np.ndarray", omega_r: float,
                                      times: "np.ndarray") -> "np.ndarray":
    components = np.asarray(components, dtype=complex)
    times = np.asarray(times, dtype=float)
    if components.ndim < 1 or components.shape[-1] != 5:
        raise ValueError("components must have a trailing axis of length 5")
    if times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")
    if not np.isfinite(float(omega_r)):
        raise ValueError("omega_r must be finite")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite")

    orders = np.arange(-2, 3)
    phases = np.exp(1j * float(omega_r) * np.outer(times, orders))
    return np.real(np.tensordot(components, phases, axes=([-1], [1])))

import numpy as np


def build_interval_quaternion(omega_rf: "np.ndarray", phi_rf: "np.ndarray",
                                      delta_omega: "np.ndarray", dt: float) -> "np.ndarray":
    dt = float(dt)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    try:
        omega_rf, phi_rf, delta_omega = np.broadcast_arrays(
            np.asarray(omega_rf, dtype=float),
            np.asarray(phi_rf, dtype=float),
            np.asarray(delta_omega, dtype=float))
    except ValueError:
        raise ValueError("omega_rf, phi_rf and delta_omega must be broadcast compatible")
    if not (np.all(np.isfinite(omega_rf)) and np.all(np.isfinite(phi_rf))
            and np.all(np.isfinite(delta_omega))):
        raise ValueError("omega_rf, phi_rf and delta_omega must be finite")

    magnitude = np.sqrt(omega_rf ** 2 + delta_omega ** 2)
    half_angle = magnitude * dt / 2.0
    safe = np.where(magnitude == 0.0, 1.0, magnitude)
    sine = np.sin(half_angle)
    transverse = sine * omega_rf / safe

    quaternion = np.stack([np.cos(half_angle),
                           -transverse * np.cos(phi_rf),
                           -transverse * np.sin(phi_rf),
                           -sine * delta_omega / safe], axis=-1)
    return quaternion

import numpy as np


def _quaternion_multiply(left: "np.ndarray", right: "np.ndarray") -> "np.ndarray":
    """Return the scalar-first quaternion product left * right, broadcasting over leading axes."""
    a2, b2, c2, d2 = (left[..., 0], left[..., 1], left[..., 2], left[..., 3])
    a1, b1, c1, d1 = (right[..., 0], right[..., 1], right[..., 2], right[..., 3])
    return np.stack([a2 * a1 - b2 * b1 - c2 * c1 - d2 * d1,
                     a2 * b1 + b2 * a1 + c2 * d1 - d2 * c1,
                     a2 * c1 - b2 * d1 + c2 * a1 + d2 * b1,
                     a2 * d1 + b2 * c1 - c2 * b1 + d2 * a1], axis=-1)


def compose_quaternion_sequence(quaternions: "np.ndarray") -> "np.ndarray":
    quaternions = np.asarray(quaternions, dtype=float)
    if quaternions.ndim < 2:
        raise ValueError("quaternions must have at least two axes")
    if quaternions.shape[-1] != 4:
        raise ValueError("quaternions must have a trailing axis of length 4")
    if quaternions.shape[-2] < 1:
        raise ValueError("quaternions must hold at least one interval")
    if not np.all(np.isfinite(quaternions)):
        raise ValueError("quaternions must be finite")

    # Copy so that a single-interval sequence does not alias the caller's array.
    q_total = quaternions[..., 0, :].copy()
    for index in range(1, quaternions.shape[-2]):
        q_total = _quaternion_multiply(quaternions[..., index, :], q_total)
    return q_total

import numpy as np


def extract_effective_field(q_total: "np.ndarray", tau_m: float) -> "np.ndarray":
    tau_m = float(tau_m)
    if not np.isfinite(tau_m) or tau_m <= 0.0:
        raise ValueError("tau_m must be positive and finite")
    q_total = np.asarray(q_total, dtype=float)
    if q_total.ndim < 1 or q_total.shape[-1] != 4:
        raise ValueError("q_total must have a trailing axis of length 4")
    if not np.all(np.isfinite(q_total)):
        raise ValueError("q_total must be finite")

    scalar = q_total[..., 0]
    vector = q_total[..., 1:]
    norm = np.sqrt(np.sum(vector ** 2, axis=-1))
    magnitude = 2.0 * np.arctan2(norm, scalar) / tau_m
    safe = np.where(norm == 0.0, 1.0, norm)
    axis = -vector / safe[..., None]
    return magnitude[..., None] * axis

import numpy as np


def compute_fidelity_and_gradient(amplitudes: "np.ndarray", phases: "np.ndarray", dt: float,
                                          offsets: "np.ndarray", crystallites: "np.ndarray",
                                          omega_aniso: float, eta: float, beta_rl: float,
                                          omega_r: float, q_target: "np.ndarray") -> "np.ndarray":
    amplitudes = np.asarray(amplitudes, dtype=float)
    phases = np.asarray(phases, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    crystallites = np.asarray(crystallites, dtype=float)
    q_target = np.asarray(q_target, dtype=float)

    if amplitudes.ndim != 1 or phases.ndim != 1 or amplitudes.shape != phases.shape:
        raise ValueError("amplitudes and phases must be one-dimensional arrays of equal length")
    if amplitudes.size < 1:
        raise ValueError("amplitudes must hold at least one interval")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if offsets.ndim != 1 or offsets.size < 1:
        raise ValueError("offsets must be a non-empty one-dimensional array")
    if crystallites.ndim != 2 or crystallites.shape[1] != 4 or crystallites.shape[0] < 1:
        raise ValueError("crystallites must have shape (n_crystallites, 4)")
    if q_target.shape != (4,):
        raise ValueError("q_target must have shape (4,)")
    if not (0.0 <= float(eta) <= 1.0):
        raise ValueError("eta must satisfy 0 <= eta <= 1")

    n_steps = amplitudes.size
    n_offsets = offsets.size
    times = np.arange(1, n_steps + 1) * dt

    components = np.array([
        compute_shielding_fourier_components(0.0, omega_aniso, eta,
                                                     row[0], row[1], row[2], beta_rl)
        for row in crystallites])
    base = compute_offset_trajectory(components, omega_r, times)
    # One row per crystallite and offset pair, each holding the interval offsets.
    delta = (base[:, None, :] + offsets[None, :, None]).reshape(-1, n_steps)
    weights = np.repeat(crystallites[:, 3], n_offsets) / n_offsets

    quaternions = build_interval_quaternion(amplitudes[None, :], phases[None, :],
                                                    delta, dt)

    identity = np.zeros((delta.shape[0], 4))
    identity[:, 0] = 1.0
    prefix = [identity]
    for index in range(n_steps):
        prefix.append(_quaternion_multiply(quaternions[:, index, :], prefix[-1]))
    suffix = [None] * (n_steps + 1)
    suffix[n_steps] = identity
    for index in range(n_steps - 1, -1, -1):
        suffix[index] = _quaternion_multiply(suffix[index + 1], quaternions[:, index, :])

    fidelity = float(weights @ (prefix[n_steps] @ q_target))

    magnitude = np.sqrt(amplitudes[None, :] ** 2 + delta ** 2)
    safe = np.where(magnitude == 0.0, 1.0, magnitude)
    half_angle = magnitude * dt / 2.0
    sine = np.sin(half_angle)
    cosine = np.cos(half_angle)
    cos_phi = np.cos(phases)[None, :]
    sin_phi = np.sin(phases)[None, :]

    transverse = (delta ** 2 / safe ** 3 * sine
                  + amplitudes[None, :] ** 2 * dt / (2.0 * safe ** 2) * cosine)
    longitudinal = delta * (amplitudes[None, :] / safe ** 3 * sine
                            - amplitudes[None, :] * dt / (2.0 * safe ** 2) * cosine)
    d_amplitude = np.stack([-dt / 2.0 * amplitudes[None, :] / safe * sine,
                            -transverse * cos_phi,
                            -transverse * sin_phi,
                            longitudinal], axis=-1)
    # Limit of a vanishing rotation, where the axis direction is set by the phase alone.
    degenerate = (magnitude == 0.0)
    if np.any(degenerate):
        limit = np.stack([np.zeros_like(sine), -dt / 2.0 * np.broadcast_to(cos_phi, sine.shape),
                          -dt / 2.0 * np.broadcast_to(sin_phi, sine.shape),
                          np.zeros_like(sine)], axis=-1)
        d_amplitude = np.where(degenerate[..., None], limit, d_amplitude)

    ratio = sine * amplitudes[None, :] / safe
    d_phase = np.stack([np.zeros_like(ratio), ratio * sin_phi,
                        -ratio * cos_phi, np.zeros_like(ratio)], axis=-1)

    gradient_amplitude = np.empty(n_steps)
    gradient_phase = np.empty(n_steps)
    for index in range(n_steps):
        left = suffix[index + 1]
        chain_a = _quaternion_multiply(left, _quaternion_multiply(d_amplitude[:, index, :],
                                                                  prefix[index]))
        chain_p = _quaternion_multiply(left, _quaternion_multiply(d_phase[:, index, :],
                                                                  prefix[index]))
        gradient_amplitude[index] = weights @ (chain_a @ q_target)
        gradient_phase[index] = weights @ (chain_p @ q_target)

    return np.concatenate([[fidelity], gradient_amplitude, gradient_phase])

import numpy as np


def run_pipeline(amplitudes_hz: "np.ndarray", phases_deg: "np.ndarray", dt: float,
                         omega_r: float, omega_aniso: float, eta: float, offsets: "np.ndarray",
                         grid_shape: tuple, flip_angle: float, n_iter: int, amp_step: float,
                         phase_step: float, amp_max: float) -> float:
    if isinstance(n_iter, bool) or not isinstance(n_iter, (int, np.integer)):
        raise ValueError("n_iter must be an integer")
    if int(n_iter) < 0:
        raise ValueError("n_iter must be non-negative")
    amp_max = float(amp_max)
    if not np.isfinite(amp_max) or amp_max <= 0.0:
        raise ValueError("amp_max must be positive and finite")
    amp_step = float(amp_step)
    phase_step = float(phase_step)
    if not (np.isfinite(amp_step) and np.isfinite(phase_step)) or amp_step < 0.0 or phase_step < 0.0:
        raise ValueError("amp_step and phase_step must be finite and non-negative")
    try:
        shape = tuple(grid_shape)
    except TypeError:
        raise ValueError("grid_shape must hold three integers")
    if len(shape) != 3:
        raise ValueError("grid_shape must hold three integers")

    beta_rl = float(np.arccos(1.0 / np.sqrt(3.0)))
    offsets = np.asarray(offsets, dtype=float)
    amplitudes = 2.0 * np.pi * np.asarray(amplitudes_hz, dtype=float)
    phases = np.deg2rad(np.asarray(phases_deg, dtype=float))
    n_steps = amplitudes.size
    tau_m = float(dt) * n_steps

    half = 0.5 * float(flip_angle)
    q_target = np.array([np.cos(half), -np.sin(half), 0.0, 0.0])

    crystallites = build_powder_grid(int(shape[0]), int(shape[1]), int(shape[2]))

    for _ in range(int(n_iter)):
        result = compute_fidelity_and_gradient(amplitudes, phases, dt, offsets,
                                                       crystallites, omega_aniso, eta,
                                                       beta_rl, omega_r, q_target)
        gradient_amplitude = result[1:n_steps + 1]
        gradient_phase = result[n_steps + 1:]
        scale_amplitude = np.max(np.abs(gradient_amplitude))
        scale_phase = np.max(np.abs(gradient_phase))
        if scale_amplitude > 0.0:
            amplitudes = np.clip(amplitudes + amp_step * gradient_amplitude / scale_amplitude,
                                 0.0, amp_max)
        if scale_phase > 0.0:
            phases = phases + phase_step * gradient_phase / scale_phase

    times = np.arange(1, n_steps + 1) * float(dt)
    components = np.array([
        compute_shielding_fourier_components(0.0, omega_aniso, eta,
                                                     row[0], row[1], row[2], beta_rl)
        for row in crystallites])
    base = compute_offset_trajectory(components, omega_r, times)
    delta = (base[:, None, :] + offsets[None, :, None]).reshape(-1, n_steps)
    weights = np.repeat(crystallites[:, 3], offsets.size) / offsets.size

    quaternions = build_interval_quaternion(amplitudes[None, :], phases[None, :],
                                                    delta, dt)
    q_total = compose_quaternion_sequence(quaternions)
    fields = extract_effective_field(q_total, tau_m)
    return float(weights @ fields[:, 0] / (2.0 * np.pi))
SCICODE_GOLD_EOF
