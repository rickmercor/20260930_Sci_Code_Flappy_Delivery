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

def adaptive_sampling_plan(desired_hz, length_m, reflectivity_a, reflectivity_b, c_m_s=299792458.0):
    """Paper Algorithms 1--3 with explicitly frozen integer conventions."""
    import math
    import numpy as np
    desired_hz = float(desired_hz)
    length_m = float(length_m)
    ra = float(reflectivity_a)
    rb = float(reflectivity_b)
    c_m_s = float(c_m_s)
    if not np.all(np.isfinite(np.asarray(
            [desired_hz, length_m, ra, rb, c_m_s], dtype=float))):
        raise ValueError("inputs must be finite")
    if desired_hz <= 0 or length_m <= 0 or c_m_s <= 0:
        raise ValueError("frequencies, length, and c must be positive")
    if not (0 < ra < 1 and 0 < rb < 1):
        raise ValueError("intensity reflectivities must lie in (0,1)")
    f_2t = c_m_s / (2.0 * length_m)
    q = math.sqrt(ra * rb)
    n_eff = 1.0 / abs(math.log(q))
    n_max = int(math.ceil(5.0 * n_eff))
    eta = f_2t / desired_hz
    n_roundtrips = 1
    n_subhistories = 1
    partial = 0.0
    if desired_hz > f_2t:
        n_subhistories = max(1, int(math.floor(1.0 / eta + 0.5)))
        f_calc = f_2t * n_subhistories
    elif desired_hz < f_2t / n_max:
        n_roundtrips = n_max
        f_calc = desired_hz
        partial = 1.0
    else:
        k0 = int(math.floor(eta))
        boundary = 2.0 * k0 * (k0 + 1.0) / (2.0 * k0 + 1.0)
        n_roundtrips = k0 if eta < boundary else k0 + 1
        f_calc = f_2t / n_roundtrips
    accuracy = 1.0 - abs(f_calc - desired_hz) / desired_hz
    return np.array([f_calc, 1.0 / f_calc, float(n_roundtrips),
                     float(n_subhistories), partial, accuracy], dtype=float)

def cavity_critical_velocity(wavelength_m, length_m, finesse, c_m_s=299792458.0):
    """Critical speed from Eq. (4) of the attached paper."""
    import math
    wavelength_m = float(wavelength_m)
    length_m = float(length_m)
    finesse = float(finesse)
    c_m_s = float(c_m_s)
    if not all(math.isfinite(value) for value in
               (wavelength_m, length_m, finesse, c_m_s)):
        raise ValueError("all physical inputs must be finite")
    if wavelength_m <= 0 or length_m <= 0 or finesse <= 0 or c_m_s <= 0:
        raise ValueError("all physical inputs must be positive")
    return float(wavelength_m * math.pi * c_m_s / (4.0 * length_m * finesse**2))

def moving_cavity_drive(n_roundtrips, length_m, wavelength_m, critical_velocity_m_s,
                                velocity_multiplier, displacement_amplitude_m,
                                displacement_hz, amplitude_depth, amplitude_hz,
                                phase_depth, phase_hz, phase_offset=0.35,
                                c_m_s=299792458.0):
    """Create the centered round-trip grid, relative cavity length, and complex input."""
    import math
    import numpy as np
    scalar_inputs = np.asarray([
        n_roundtrips, length_m, wavelength_m, critical_velocity_m_s,
        velocity_multiplier, displacement_amplitude_m, displacement_hz,
        amplitude_depth, amplitude_hz, phase_depth, phase_hz,
        phase_offset, c_m_s,
    ], dtype=float)
    if not np.all(np.isfinite(scalar_inputs)):
        raise ValueError("all drive inputs must be finite")
    n = int(n_roundtrips)
    if n != n_roundtrips or n < 5 or n % 2 != 1:
        raise ValueError("n_roundtrips must be an odd integer at least 5")
    if (length_m <= 0 or wavelength_m <= 0 or critical_velocity_m_s <= 0
            or displacement_amplitude_m < 0 or c_m_s <= 0):
        raise ValueError("physical scale parameters must be positive")
    if not (0 <= amplitude_depth < 1):
        raise ValueError("amplitude_depth must be in [0,1)")
    f_2t = c_m_s / (2.0 * length_m)
    if max(abs(displacement_hz), abs(amplitude_hz), abs(phase_hz)) >= 0.5 * f_2t:
        raise ValueError("drive frequencies must be below half the round-trip rate")
    dt = 1.0 / f_2t
    time_s = (np.arange(n, dtype=float) - (n - 1) / 2.0) * dt
    displacement_m = (velocity_multiplier * critical_velocity_m_s * time_s
                      + displacement_amplitude_m
                      * np.sin(2.0 * math.pi * displacement_hz * time_s + phase_offset))
    amplitude = 1.0 + amplitude_depth * np.cos(2.0 * math.pi * amplitude_hz * time_s)
    phase = phase_depth * np.sin(2.0 * math.pi * phase_hz * time_s)
    input_field = amplitude * np.exp(1j * phase)
    return np.column_stack((time_s, displacement_m, input_field.real, input_field.imag))

def propagate_cavity_field(drive, wavelength_m, reflectivity_a, reflectivity_b):
    """Round-trip recurrence E_j=t_a E_in,j+q exp(-4 pi i delta_d/lambda) E_{j-1}."""
    import math
    import numpy as np
    drive = np.asarray(drive, dtype=float)
    if drive.ndim != 2 or drive.shape[1] != 4 or drive.shape[0] < 1:
        raise ValueError("drive must have shape (n,4)")
    if not np.all(np.isfinite(drive)):
        raise ValueError("drive must be finite")
    optical_inputs = np.asarray(
        [wavelength_m, reflectivity_a, reflectivity_b], dtype=float
    )
    if not np.all(np.isfinite(optical_inputs)):
        raise ValueError("optical inputs must be finite")
    if wavelength_m <= 0 or not (0 < reflectivity_a < 1 and 0 < reflectivity_b < 1):
        raise ValueError("invalid wavelength or reflectivity")
    input_field = drive[:, 2] + 1j * drive[:, 3]
    t_a = math.sqrt(1.0 - float(reflectivity_a))
    q = math.sqrt(float(reflectivity_a) * float(reflectivity_b))
    phase = np.exp(-4j * math.pi * drive[:, 1] / float(wavelength_m))
    field = np.empty(drive.shape[0], dtype=complex)
    previous = 0.0j
    for j in range(drive.shape[0]):
        previous = t_a * input_field[j] + q * phase[j] * previous
        field[j] = previous
    return field

def cavity_observables(field, drive, demodulation_phase):
    """Return columns [intracavity power, PDH] using Eq. (3)."""
    import numpy as np
    field = np.asarray(field, dtype=complex)
    drive = np.asarray(drive, dtype=float)
    if field.ndim != 1 or drive.ndim != 2 or drive.shape != (field.size, 4):
        raise ValueError("field and drive shapes disagree")
    if (not np.all(np.isfinite(field)) or not np.all(np.isfinite(drive))
            or not np.isfinite(float(demodulation_phase))):
        raise ValueError("inputs must be finite")
    input_field = drive[:, 2] + 1j * drive[:, 3]
    power = np.abs(field)**2
    pdh = -np.imag(np.exp(1j * float(demodulation_phase)) * np.conj(input_field) * field)
    return np.column_stack((power, pdh)).astype(float)

def sampling_diagnostics(time_s, observables, stride, spectral_hz):
    """Time-domain errors plus two-tone Hann-windowed PDH spectral distortion."""
    import numpy as np
    time_s = np.asarray(time_s, dtype=float)
    observables = np.asarray(observables, dtype=float)
    try:
        stride_value = float(stride)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("stride must be a finite positive integer") from exc
    if not np.isfinite(stride_value):
        raise ValueError("stride must be a finite positive integer")
    stride_i = int(stride_value)
    if stride_i != stride_value or stride_i < 1:
        raise ValueError("stride must be a positive integer")
    if time_s.ndim != 1 or observables.shape != (time_s.size, 2) or time_s.size < 5:
        raise ValueError("observables must have shape (len(time_s),2)")
    if (not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0)
            or not np.all(np.isfinite(observables))):
        raise ValueError("time must increase and inputs must be finite")
    spectral_hz = np.asarray(spectral_hz, dtype=float)
    if (spectral_hz.shape != (2,) or np.any(spectral_hz <= 0)
            or not np.all(np.isfinite(spectral_hz))):
        raise ValueError("spectral_hz must contain two finite positive frequencies")
    if np.max(spectral_hz) >= 0.5 / np.min(np.diff(time_s)):
        raise ValueError("spectral frequencies exceed the physical-grid Nyquist rate")
    sample_index = np.arange(0, time_s.size, stride_i, dtype=int)
    if sample_index.size < 4:
        raise ValueError("at least four samples are required")
    last = int(sample_index[-1])
    tref = time_s[:last + 1]
    pref = observables[:last + 1, 0]
    dref = observables[:last + 1, 1]
    ts = time_s[sample_index]
    ps = observables[sample_index, 0]
    ds = observables[sample_index, 1]
    peak_ref = float(np.max(pref))
    if peak_ref <= 0:
        raise ValueError("reference peak power must be positive")
    peak_error = abs(float(np.max(ps)) - peak_ref) / peak_ref

    def crossings(t, y):
        idx = np.where(((y[:-1] <= 0) & (y[1:] > 0)) |
                       ((y[:-1] >= 0) & (y[1:] < 0)))[0]
        values = []
        for i in idx:
            denominator = y[i + 1] - y[i]
            if denominator != 0:
                values.append(t[i] - y[i] * (t[i + 1] - t[i]) / denominator)
        return np.asarray(values, dtype=float)

    reference_crossings = crossings(tref, dref)
    sampled_crossings = crossings(ts, ds)
    if reference_crossings.size == 0 or sampled_crossings.size == 0:
        raise ValueError("PDH trace must contain a zero crossing")
    peak_time = tref[int(np.argmax(pref))]
    z_ref = reference_crossings[int(np.argmin(np.abs(reference_crossings - peak_time)))]
    z_sample = sampled_crossings[int(np.argmin(np.abs(sampled_crossings - z_ref)))]
    zero_time_error = abs(float(z_sample - z_ref))
    interpolated = np.interp(tref, ts, ds)
    denominator = float(np.sqrt(np.mean(dref**2)))
    if denominator == 0:
        raise ValueError("reference PDH RMS must be nonzero")
    pdh_nrmse = float(np.sqrt(np.mean((interpolated - dref)**2)) / denominator)

    def coefficients(t, y):
        u = (t - t[0]) / (t[-1] - t[0])
        window = np.sin(np.pi * u)**2
        window_sum = float(np.sum(window))
        centered = y - float(np.sum(window * y) / window_sum)
        return np.asarray([
            np.sum(window * centered * np.exp(-2j * np.pi * f * t)) / window_sum
            for f in spectral_hz
        ], dtype=complex)

    reference_coefficients = coefficients(tref, dref)
    sampled_coefficients = coefficients(ts, ds)
    spectral_denominator = float(np.sum(np.abs(reference_coefficients)**2))
    if spectral_denominator <= 0:
        raise ValueError("reference two-tone PDH energy must be positive")
    spectral_distortion = float(np.sqrt(
        np.sum(np.abs(sampled_coefficients - reference_coefficients)**2)
        / spectral_denominator
    ))
    return np.array([peak_error, zero_time_error, pdh_nrmse,
                     spectral_distortion], dtype=float)

def ringdown_tail_inference(time_s, field, stride, velocity_m_s,
                                    length_m, wavelength_m, reflectivity_a,
                                    reflectivity_b, finesse,
                                    tail_start_storage=4.0,
                                    tail_end_storage=24.0,
                                    c_m_s=299792458.0):
    """Tail-only complex LS incident-power bias using paper Eqs. (5)--(6)."""
    import math
    import numpy as np
    time_s = np.asarray(time_s, dtype=float)
    field = np.asarray(field, dtype=complex)
    stride_i = int(stride)
    if (time_s.ndim != 1 or field.shape != time_s.shape or time_s.size < 5
            or np.any(np.diff(time_s) <= 0) or not np.all(np.isfinite(time_s))
            or not np.all(np.isfinite(field))):
        raise ValueError("time and field must be finite aligned vectors")
    if stride_i != stride or stride_i < 1:
        raise ValueError("stride must be a positive integer")
    if (length_m <= 0 or wavelength_m <= 0 or finesse <= 0 or c_m_s <= 0
            or not (0 < reflectivity_a < 1 and 0 < reflectivity_b < 1)):
        raise ValueError("invalid optical parameters")
    if not (0 < tail_start_storage < tail_end_storage):
        raise ValueError("tail bounds must be positive and ordered")
    v = float(velocity_m_s)
    v_critical = wavelength_m * math.pi * c_m_s / (4.0 * length_m * finesse**2)
    if not np.isfinite(v) or abs(v) < v_critical * (1.0 - 1e-12):
        raise ValueError("continuum tail fit requires |v| >= v_cr")
    T = float(length_m) / float(c_m_s)
    tau = 2.0 * T / abs(math.log(math.sqrt(float(reflectivity_a) * float(reflectivity_b))))
    k = 2.0 * math.pi / float(wavelength_m)
    q = math.sqrt(float(reflectivity_a) * float(reflectivity_b))
    t_a = math.sqrt(1.0 - float(reflectivity_a))

    def unit_input_kernel(t):
        transient_prefactor = np.sqrt(1j * math.pi / (2.0 * k * v * T))
        transient_prefactor *= np.exp(1j * T / (2.0 * k * v * tau**2))
        transient = transient_prefactor * np.exp(
            -t / tau - 1j * k * v * t**2 / (2.0 * T)
        )
        adiabatic = 1.0 / (1.0 - q * np.exp(-2j * k * v * t))
        return t_a * (transient + adiabatic)

    full_index = np.where((time_s >= tail_start_storage * tau)
                          & (time_s <= tail_end_storage * tau))[0]
    retained_index = np.arange(0, time_s.size, stride_i, dtype=int)
    retained_index = retained_index[
        (time_s[retained_index] >= tail_start_storage * tau)
        & (time_s[retained_index] <= tail_end_storage * tau)
    ]
    if full_index.size < 16 or retained_index.size < 3:
        raise ValueError("insufficient samples in the continuum-valid tail")
    g_full = unit_input_kernel(time_s[full_index])
    g_retained = unit_input_kernel(time_s[retained_index])
    alpha_full = np.vdot(g_full, field[full_index]) / np.vdot(g_full, g_full)
    alpha_retained = (np.vdot(g_retained, field[retained_index])
                      / np.vdot(g_retained, g_retained))
    power_full = float(abs(alpha_full)**2)
    power_retained = float(abs(alpha_retained)**2)
    if power_full <= 0 or not np.isfinite(power_full + power_retained):
        raise ValueError("tail fit produced invalid incident power")
    relative_power_bias = abs(power_retained - power_full) / power_full
    return np.array([power_full, power_retained, relative_power_bias,
                     float(retained_index.size)], dtype=float)

def transformed_sigma_points(reflectivity_a, reflectivity_b, finesse,
                                     transformed_covariance):
    """Seven positive-weight sigma points in [logit(Ra),logit(Rb),log(F)]."""
    import math
    import numpy as np
    ra = float(reflectivity_a)
    rb = float(reflectivity_b)
    finesse = float(finesse)
    covariance = np.asarray(transformed_covariance, dtype=float)
    if not (0.0 < ra < 1.0 and 0.0 < rb < 1.0 and finesse > 0.0):
        raise ValueError("invalid nominal optical parameters")
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError("transformed_covariance must be a finite 3-by-3 matrix")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-14):
        raise ValueError("transformed_covariance must be symmetric")
    try:
        root = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("transformed_covariance must be positive definite") from exc
    mean = np.array([
        math.log(ra / (1.0 - ra)),
        math.log(rb / (1.0 - rb)),
        math.log(finesse),
    ], dtype=float)
    transformed = [mean]
    scale = math.sqrt(6.0)
    for k in range(3):
        transformed.extend((mean + scale * root[:, k], mean - scale * root[:, k]))

    def inverse_map(x):
        ra_i = 1.0 / (1.0 + math.exp(-float(x[0])))
        rb_i = 1.0 / (1.0 + math.exp(-float(x[1])))
        return [ra_i, rb_i, math.exp(float(x[2]))]

    weights = [0.5] + [1.0 / 12.0] * 6
    return np.asarray([inverse_map(x) + [weights[i]]
                       for i, x in enumerate(transformed)], dtype=float)

def uncertainty_risk_table(candidate_desired_hz, sigma_sampling_plans,
                                   sigma_metrics, sigma_weights, risk_quantile,
                                   peak_tolerance, zero_tolerance_s, rms_tolerance,
                                   spectral_tolerance, tail_power_tolerance,
                                   memory_limit_roundtrips):
    """Propagate five separate metric distributions into a risk table."""
    import numpy as np
    candidates = np.asarray(candidate_desired_hz, dtype=float)
    plans = np.asarray(sigma_sampling_plans, dtype=float)
    metrics = np.asarray(sigma_metrics, dtype=float)
    weights = np.asarray(sigma_weights, dtype=float)
    if (candidates.ndim != 1 or candidates.size == 0
            or not np.all(np.isfinite(candidates)) or np.any(candidates <= 0)):
        raise ValueError("candidate frequencies must be a finite positive vector")
    if plans.ndim != 3 or plans.shape[1:] != (candidates.size, 6):
        raise ValueError("sigma_sampling_plans must have shape (H,C,6)")
    if metrics.shape != plans.shape[:2] + (5,) or weights.shape != (plans.shape[0],):
        raise ValueError("metric, weight, and plan shapes disagree")
    if (not np.all(np.isfinite(plans)) or not np.all(np.isfinite(metrics))
            or not np.all(np.isfinite(weights)) or np.any(metrics < 0)
            or np.any(weights <= 0)):
        raise ValueError("plans, metrics, and positive weights must be finite")
    if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("sigma weights must sum to one")
    tolerances = np.asarray([peak_tolerance, zero_tolerance_s, rms_tolerance,
                             spectral_tolerance, tail_power_tolerance], dtype=float)
    try:
        risk_value = float(risk_quantile)
        memory_value = float(memory_limit_roundtrips)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("risk, tolerance, or memory input is invalid") from exc
    if (not np.isfinite(risk_value) or risk_value < 0
            or not np.isfinite(memory_value) or memory_value < 1
            or memory_value != int(memory_value)
            or np.any(tolerances <= 0) or not np.all(np.isfinite(tolerances))):
        raise ValueError("risk, tolerance, or memory input is invalid")
    memory_value = int(memory_value)
    mean_metric = np.einsum("h,hcm->cm", weights, metrics)
    metric_std = np.sqrt(np.einsum(
        "h,hcm->cm", weights, (metrics - mean_metric[None, :, :])**2
    ))
    upper_metric = mean_metric + risk_value * metric_std
    normalized_upper = upper_metric / tolerances[None, :]
    composite = np.max(normalized_upper, axis=1)
    max_stride = np.max(plans[:, :, 2], axis=0)
    any_partial = np.max(plans[:, :, 4], axis=0)
    rows = []
    for i, desired in enumerate(candidates):
        feasible = float(any_partial[i] == 0.0
                         and max_stride[i] <= memory_value
                         and composite[i] <= 1.0)
        rows.append([desired, plans[0, i, 0], max_stride[i],
                     normalized_upper[i, 0], normalized_upper[i, 1],
                     normalized_upper[i, 2], normalized_upper[i, 3],
                     normalized_upper[i, 4], mean_metric[i, 4],
                     metric_std[i, 4], upper_metric[i, 4], composite[i], feasible])
    return np.asarray(rows, dtype=float)

def adaptive_cavity_sampling_benchmark(candidate_desired_hz, velocity_multipliers,
                                                n_roundtrips, length_m, wavelength_m,
                                                reflectivity_a, reflectivity_b, finesse,
                                                demodulation_phase, displacement_fraction,
                                                displacement_hz, amplitude_depth, amplitude_hz,
                                                phase_depth, phase_hz, phase_offset,
                                                peak_tolerance, zero_tolerance_s, rms_tolerance,
                                                spectral_hz, spectral_tolerance,
                                                tail_power_tolerance, memory_limit_roundtrips,
                                                transformed_covariance, risk_quantile,
                                                c_m_s=299792458.0):
    """Return selected risk-loaded tail-fit incident-power bias in ppm."""
    import numpy as np
    candidates = np.asarray(candidate_desired_hz, dtype=float)
    multipliers = np.asarray(velocity_multipliers, dtype=float)
    if (candidates.ndim != 1 or candidates.size == 0
            or not np.all(np.isfinite(candidates)) or np.any(candidates <= 0)):
        raise ValueError("candidate frequencies must be a finite positive vector")
    if multipliers.ndim != 1 or multipliers.size == 0 or not np.all(np.isfinite(multipliers)):
        raise ValueError("velocity multipliers must be finite")
    nonadiabatic = np.where(np.abs(multipliers) >= 1.0)[0]
    if nonadiabatic.size == 0:
        raise ValueError("at least one |velocity multiplier| >= 1 is required")
    sigma = transformed_sigma_points(
        reflectivity_a, reflectivity_b, finesse, transformed_covariance
    )
    plans = np.empty((sigma.shape[0], candidates.size, 6), dtype=float)
    metrics = np.empty((sigma.shape[0], candidates.size, 5), dtype=float)
    for h, (ra_h, rb_h, finesse_h, _) in enumerate(sigma):
        v_critical = cavity_critical_velocity(wavelength_m, length_m, finesse_h, c_m_s)
        displacement_amplitude_m = displacement_fraction * wavelength_m / finesse_h
        observable_traces = []
        tail_traces = {}
        for j, multiplier in enumerate(multipliers):
            drive = moving_cavity_drive(
                n_roundtrips, length_m, wavelength_m, v_critical, float(multiplier),
                displacement_amplitude_m, displacement_hz, amplitude_depth, amplitude_hz,
                phase_depth, phase_hz, phase_offset, c_m_s
            )
            field = propagate_cavity_field(drive, wavelength_m, ra_h, rb_h)
            observable_traces.append((drive[:, 0], cavity_observables(
                field, drive, demodulation_phase
            )))
            if j in nonadiabatic:
                calibration_drive = moving_cavity_drive(
                    n_roundtrips, length_m, wavelength_m, v_critical, float(multiplier),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, phase_offset, c_m_s
                )
                calibration_field = propagate_cavity_field(
                    calibration_drive, wavelength_m, ra_h, rb_h
                )
                tail_traces[j] = (calibration_drive[:, 0], calibration_field)
        plans[h] = np.asarray([
            adaptive_sampling_plan(float(desired), length_m, ra_h, rb_h, c_m_s)
            for desired in candidates
        ], dtype=float)
        for i in range(candidates.size):
            stride = int(plans[h, i, 2])
            ordinary = np.asarray([
                sampling_diagnostics(t, obs, stride, spectral_hz)
                for t, obs in observable_traces
            ], dtype=float)
            metrics[h, i, :4] = np.max(ordinary, axis=0)
            tail_biases = []
            for j in nonadiabatic:
                t, field = tail_traces[int(j)]
                fit = ringdown_tail_inference(
                    t, field, stride, float(multipliers[j]) * v_critical,
                    length_m, wavelength_m, ra_h, rb_h, finesse_h,
                    4.0, 24.0, c_m_s
                )
                tail_biases.append(fit[2])
            metrics[h, i, 4] = max(tail_biases)
    risk_table = uncertainty_risk_table(
        candidates, plans, metrics, sigma[:, 3], risk_quantile,
        peak_tolerance, zero_tolerance_s, rms_tolerance, spectral_tolerance,
        tail_power_tolerance, memory_limit_roundtrips
    )
    feasible = np.where(risk_table[:, 12] == 1.0)[0]
    if feasible.size == 0:
        raise ValueError("no candidate satisfies every constraint")
    selected = min(feasible.tolist(), key=lambda i: (risk_table[i, 1], i))
    return float(round(risk_table[selected, 10] * 1.0e6, 6))
SCICODE_GOLD_EOF
