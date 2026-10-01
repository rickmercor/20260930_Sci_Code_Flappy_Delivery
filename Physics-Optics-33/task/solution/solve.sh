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


def compute_peak_intensity(pulse_energy_j: float, waist_radius_m: float, duration_fwhm_s: float) -> float:
    if not np.isfinite(pulse_energy_j) or pulse_energy_j <= 0.0:
        raise ValueError("pulse_energy_j must be a positive finite number")
    if not np.isfinite(waist_radius_m) or waist_radius_m <= 0.0:
        raise ValueError("waist_radius_m must be a positive finite number")
    if not np.isfinite(duration_fwhm_s) or duration_fwhm_s <= 0.0:
        raise ValueError("duration_fwhm_s must be a positive finite number")

    peak_fluence_jm2 = 2.0 * float(pulse_energy_j) / (np.pi * float(waist_radius_m) ** 2)
    peak_fluence_jcm2 = peak_fluence_jm2 * 1.0e-4
    temporal_factor = (2.0 / float(duration_fwhm_s)) * np.sqrt(np.log(2.0) / np.pi)
    return float(peak_fluence_jcm2 * temporal_factor)

import numpy as np


def compute_readout_geometry(pump_wavelength_m: float, probe_wavelength_m: float, medium_index: float, pixel_pitch_m: float, magnification: float, corrected_pixel_count: float) -> "np.ndarray":
    values = {
        "pump_wavelength_m": pump_wavelength_m,
        "probe_wavelength_m": probe_wavelength_m,
        "medium_index": medium_index,
        "pixel_pitch_m": pixel_pitch_m,
        "magnification": magnification,
        "corrected_pixel_count": corrected_pixel_count,
    }
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("%s must be a positive finite number" % name)

    fringe_period_m = 0.5 * float(pump_wavelength_m)
    sin_bragg = float(probe_wavelength_m) / (2.0 * float(medium_index) * fringe_period_m)
    if sin_bragg >= 1.0:
        raise ValueError("probe wavelength is too long for a first Bragg order to exist")

    bragg_angle_rad = float(np.arcsin(sin_bragg))
    axial_angle_rad = 0.5 * np.pi - bragg_angle_rad
    grating_length_m = float(pixel_pitch_m) * float(corrected_pixel_count) / (
        float(magnification) * np.sin(axial_angle_rad)
    )
    return np.array(
        [fringe_period_m, bragg_angle_rad, axial_angle_rad, grating_length_m], dtype=float
    )

import numpy as np


def compute_interference_intensity(times_s: "np.ndarray", fringe_phases_rad: "np.ndarray", axial_position_m: float, peak_intensity_wcm2: float, duration_fwhm_s: float) -> "np.ndarray":
    times = np.asarray(times_s, dtype=float)
    phases = np.asarray(fringe_phases_rad, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("times_s must be a non-empty one-dimensional array")
    if phases.ndim != 1 or phases.size < 1:
        raise ValueError("fringe_phases_rad must be a non-empty one-dimensional array")
    if not np.isfinite(peak_intensity_wcm2) or peak_intensity_wcm2 < 0.0:
        raise ValueError("peak_intensity_wcm2 must be a non-negative finite number")
    if not np.isfinite(duration_fwhm_s) or duration_fwhm_s <= 0.0:
        raise ValueError("duration_fwhm_s must be a positive finite number")
    if not np.isfinite(axial_position_m):
        raise ValueError("axial_position_m must be finite")

    # Counter-propagation maps the axial coordinate onto a pump-pump delay.
    speed_of_light_ms = 2.99792458e8
    delay_s = 2.0 * float(axial_position_m) / speed_of_light_ms
    decay = 4.0 * np.log(2.0) / float(duration_fwhm_s) ** 2

    forward = float(peak_intensity_wcm2) * np.exp(-decay * (times + 0.5 * delay_s) ** 2)
    backward = float(peak_intensity_wcm2) * np.exp(-decay * (times - 0.5 * delay_s) ** 2)
    cross = 2.0 * np.sqrt(forward * backward)

    return (forward + backward)[None, :] + cross[None, :] * np.cos(phases)[:, None]

import numpy as np


def compute_ionization_rate(intensity_wcm2: "np.ndarray", ionization_potential_ev: float, pump_wavelength_m: float) -> "np.ndarray":
    intensity = np.asarray(intensity_wcm2, dtype=float)
    if not np.all(np.isfinite(intensity)):
        raise ValueError("intensity_wcm2 entries must be finite")
    if not np.isfinite(ionization_potential_ev) or ionization_potential_ev <= 0.0:
        raise ValueError("ionization_potential_ev must be a positive finite number")
    if not np.isfinite(pump_wavelength_m) or pump_wavelength_m <= 0.0:
        raise ValueError("pump_wavelength_m must be a positive finite number")

    speed_of_light_ms = 2.99792458e8
    atomic_intensity_wcm2 = 3.5094452e16
    atomic_time_s = 2.4188843265e-17
    hartree_ev = 27.211386245988
    residual_charge = 1.0

    ip_au = float(ionization_potential_ev) / hartree_ev
    kappa = np.sqrt(2.0 * ip_au)
    n_star = residual_charge / kappa
    omega_au = (2.0 * np.pi * speed_of_light_ms / float(pump_wavelength_m)) * atomic_time_s

    positive = intensity > 0.0
    safe = np.where(positive, intensity, 1.0)
    field_au = np.sqrt(safe / atomic_intensity_wcm2)

    gamma = omega_au * kappa / field_au
    keldysh_factor = (3.0 / (2.0 * gamma)) * (
        (1.0 + 1.0 / (2.0 * gamma ** 2)) * np.arcsinh(gamma)
        - np.sqrt(1.0 + gamma ** 2) / (2.0 * gamma)
    )

    log_rate_au = (2.0 * n_star - 1.0) * np.log(2.0 * kappa ** 3 / field_au) - (
        2.0 * kappa ** 3 / (3.0 * field_au)
    ) * keldysh_factor

    with np.errstate(over="ignore", under="ignore"):
        rate_au = np.where(positive, np.exp(log_rate_au), 0.0)
    return rate_au / atomic_time_s

import numpy as np


def compute_electron_density(axial_position_m: float, duration_fwhm_s: float, peak_intensity_wcm2: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, n_fringe_phase: int, n_time: int, time_window_factor: float) -> "np.ndarray":
    if not np.isfinite(neutral_density_m3) or neutral_density_m3 < 0.0:
        raise ValueError("neutral_density_m3 must be a non-negative finite number")
    if int(n_fringe_phase) != n_fringe_phase or int(n_fringe_phase) < 1:
        raise ValueError("n_fringe_phase must be an integer of at least one")
    if int(n_time) != n_time or int(n_time) < 2:
        raise ValueError("n_time must be an integer of at least two")
    if not np.isfinite(time_window_factor) or time_window_factor <= 0.0:
        raise ValueError("time_window_factor must be a positive finite number")
    if not np.isfinite(duration_fwhm_s) or duration_fwhm_s <= 0.0:
        raise ValueError("duration_fwhm_s must be a positive finite number")
    if not np.isfinite(axial_position_m):
        raise ValueError("axial_position_m must be finite")

    speed_of_light_ms = 2.99792458e8
    phases = 2.0 * np.pi * np.arange(int(n_fringe_phase), dtype=float) / float(n_fringe_phase)
    half_window_s = float(time_window_factor) * float(duration_fwhm_s) + abs(
        float(axial_position_m)
    ) / speed_of_light_ms
    times = np.linspace(-half_window_s, half_window_s, int(n_time))

    intensity = compute_interference_intensity(
        times, phases, axial_position_m, peak_intensity_wcm2, duration_fwhm_s
    )
    rate = compute_ionization_rate(
        intensity, ionization_potential_ev, pump_wavelength_m
    )
    widths = np.diff(times)
    exposure = np.sum(0.5 * (rate[:, 1:] + rate[:, :-1]) * widths, axis=1)
    return float(neutral_density_m3) * (1.0 - np.exp(-exposure))

import numpy as np


def compute_first_harmonic(axial_positions_m: "np.ndarray", duration_fwhm_s: float, peak_intensity_wcm2: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, n_fringe_phase: int, n_time: int, time_window_factor: float) -> "np.ndarray":
    positions = np.asarray(axial_positions_m, dtype=float)
    if positions.ndim != 1 or positions.size < 1:
        raise ValueError("axial_positions_m must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(positions)):
        raise ValueError("axial_positions_m entries must be finite")

    if int(n_fringe_phase) != n_fringe_phase or int(n_fringe_phase) < 1:
        raise ValueError("n_fringe_phase must be an integer of at least one")
    phases = 2.0 * np.pi * np.arange(int(n_fringe_phase), dtype=float) / float(n_fringe_phase)
    weights = np.cos(phases)

    amplitudes = np.empty(positions.size, dtype=float)
    for index, position in enumerate(positions):
        density = compute_electron_density(
            float(position),
            duration_fwhm_s,
            peak_intensity_wcm2,
            neutral_density_m3,
            ionization_potential_ev,
            pump_wavelength_m,
            n_fringe_phase,
            n_time,
            time_window_factor,
        )
        amplitudes[index] = 2.0 * np.sum(density * weights) / float(n_fringe_phase)
    return amplitudes

import numpy as np


def compute_grating_length(duration_fwhm_s: float, pulse_energy_j: float, waist_radius_m: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, z_max_m: float, n_axial: int, n_fringe_phase: int, n_time: int, time_window_factor: float) -> float:
    if not np.isfinite(z_max_m) or z_max_m <= 0.0:
        raise ValueError("z_max_m must be a positive finite number")
    if int(n_axial) != n_axial or int(n_axial) < 2:
        raise ValueError("n_axial must be an integer of at least two")

    peak_intensity = compute_peak_intensity(
        pulse_energy_j, waist_radius_m, duration_fwhm_s
    )
    positions = np.linspace(0.0, float(z_max_m), int(n_axial))
    amplitudes = compute_first_harmonic(
        positions,
        duration_fwhm_s,
        peak_intensity,
        neutral_density_m3,
        ionization_potential_ev,
        pump_wavelength_m,
        n_fringe_phase,
        n_time,
        time_window_factor,
    )
    envelope = amplitudes ** 2
    half_level = 0.5 * envelope[0]

    below = np.nonzero(envelope < half_level)[0]
    if below.size == 0:
        raise ValueError("axial envelope does not reach half maximum within z_max_m")
    index = int(below[0])

    z_lo, z_hi = positions[index - 1], positions[index]
    e_lo, e_hi = envelope[index - 1], envelope[index]
    z_half = z_lo + (half_level - e_lo) * (z_hi - z_lo) / (e_hi - e_lo)
    return float(2.0 * z_half)

import numpy as np


def compute_calibration_curve(durations_s: "np.ndarray", pulse_energy_j: float, waist_radius_m: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, z_max_m: float, n_axial: int, n_fringe_phase: int, n_time: int, time_window_factor: float) -> "np.ndarray":
    durations = np.asarray(durations_s, dtype=float)
    if durations.ndim != 1 or durations.size < 1:
        raise ValueError("durations_s must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("durations_s entries must be positive and finite")

    lengths = np.empty(durations.size, dtype=float)
    for index, duration in enumerate(durations):
        lengths[index] = compute_grating_length(
            float(duration),
            pulse_energy_j,
            waist_radius_m,
            neutral_density_m3,
            ionization_potential_ev,
            pump_wavelength_m,
            z_max_m,
            n_axial,
            n_fringe_phase,
            n_time,
            time_window_factor,
        )
    return lengths

import numpy as np


def retrieve_pulse_duration(pump_wavelength_m: float, probe_wavelength_m: float, medium_index: float, pixel_pitch_m: float, magnification: float, corrected_pixel_count: float, calibration_durations_s: "np.ndarray", pulse_energy_j: float, waist_radius_m: float, neutral_density_m3: float, ionization_potential_ev: float, z_max_m: float, n_axial: int, n_fringe_phase: int, n_time: int, time_window_factor: float) -> float:
    durations = np.asarray(calibration_durations_s, dtype=float)
    if durations.ndim != 1 or durations.size < 2:
        raise ValueError("calibration_durations_s must hold at least two durations")
    if np.any(np.diff(durations) <= 0.0):
        raise ValueError("calibration_durations_s must be strictly increasing")

    geometry = compute_readout_geometry(
        pump_wavelength_m,
        probe_wavelength_m,
        medium_index,
        pixel_pitch_m,
        magnification,
        corrected_pixel_count,
    )
    measured_length_m = float(geometry[3])

    lengths = compute_calibration_curve(
        durations,
        pulse_energy_j,
        waist_radius_m,
        neutral_density_m3,
        ionization_potential_ev,
        pump_wavelength_m,
        z_max_m,
        n_axial,
        n_fringe_phase,
        n_time,
        time_window_factor,
    )
    if np.any(np.diff(lengths) <= 0.0):
        raise ValueError("calibrated lengths must increase strictly with duration")
    if measured_length_m < lengths[0] or measured_length_m > lengths[-1]:
        raise ValueError("measured grating length lies outside the calibrated range")

    retrieved_duration_s = float(np.interp(measured_length_m, lengths, durations))
    return retrieved_duration_s * 1.0e15
SCICODE_GOLD_EOF
