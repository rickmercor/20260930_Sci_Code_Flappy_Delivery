"""
Compose the discrete moving-cavity diagnostics with the nonadiabatic tail inverse problem and uncertainty certification.

Call every preceding public step. At each mapped physical sigma node, evaluate all supplied velocities and take componentwise maxima of the first four diagnostics. For the fifth diagnostic, fit only velocities whose multipliers satisfy |multiplier|>=1 and take their maximum relative tail-power bias. For every candidate and diagnostic, take these nodewise velocity maxima before forming the weighted population mean and standard deviation across sigma nodes; add risk_quantile times the deviation in physical units, then normalize once by the matching tolerance. Feasibility requires no partial sampling plan at any node, maximum stride across nodes no greater than memory_limit_roundtrips, and all five normalized bounds independently no greater than one. Select the feasible candidate with smallest unrounded centre-node effective rate; break exact ties by original candidate order. The calibration trace is unit-input, purely linear, and unmodulated.

Returns
-------
Return one Python float: selected fifth tail-power-bias upper bound times 1e6 in ppm, rounded to six decimal places.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    """Return the selected risk-loaded tail-power bias as one float in ppm.\n\n    Nodewise velocity maxima precede cross-node population moments. The tail uses\n    only |velocity_multiplier| >= 1. Every normalized bound and the maximum retained\n    stride constraint is tested independently. Invalid inputs or no feasible candidate\n    raise ValueError.\n    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_adaptive_cavity_sampling_benchmark(candidate_desired_hz, velocity_multipliers,
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
    sigma = _oracle_transformed_sigma_points(
        reflectivity_a, reflectivity_b, finesse, transformed_covariance
    )
    plans = np.empty((sigma.shape[0], candidates.size, 6), dtype=float)
    metrics = np.empty((sigma.shape[0], candidates.size, 5), dtype=float)
    for h, (ra_h, rb_h, finesse_h, _) in enumerate(sigma):
        v_critical = _oracle_cavity_critical_velocity(wavelength_m, length_m, finesse_h, c_m_s)
        displacement_amplitude_m = displacement_fraction * wavelength_m / finesse_h
        observable_traces = []
        tail_traces = {}
        for j, multiplier in enumerate(multipliers):
            drive = _oracle_moving_cavity_drive(
                n_roundtrips, length_m, wavelength_m, v_critical, float(multiplier),
                displacement_amplitude_m, displacement_hz, amplitude_depth, amplitude_hz,
                phase_depth, phase_hz, phase_offset, c_m_s
            )
            field = _oracle_propagate_cavity_field(drive, wavelength_m, ra_h, rb_h)
            observable_traces.append((drive[:, 0], _oracle_cavity_observables(
                field, drive, demodulation_phase
            )))
            if j in nonadiabatic:
                calibration_drive = _oracle_moving_cavity_drive(
                    n_roundtrips, length_m, wavelength_m, v_critical, float(multiplier),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, phase_offset, c_m_s
                )
                calibration_field = _oracle_propagate_cavity_field(
                    calibration_drive, wavelength_m, ra_h, rb_h
                )
                tail_traces[j] = (calibration_drive[:, 0], calibration_field)
        plans[h] = np.asarray([
            _oracle_adaptive_sampling_plan(float(desired), length_m, ra_h, rb_h, c_m_s)
            for desired in candidates
        ], dtype=float)
        for i in range(candidates.size):
            stride = int(plans[h, i, 2])
            ordinary = np.asarray([
                _oracle_sampling_diagnostics(t, obs, stride, spectral_hz)
                for t, obs in observable_traces
            ], dtype=float)
            metrics[h, i, :4] = np.max(ordinary, axis=0)
            tail_biases = []
            for j in nonadiabatic:
                t, field = tail_traces[int(j)]
                fit = _oracle_ringdown_tail_inference(
                    t, field, stride, float(multipliers[j]) * v_critical,
                    length_m, wavelength_m, ra_h, rb_h, finesse_h,
                    4.0, 24.0, c_m_s
                )
                tail_biases.append(fit[2])
            metrics[h, i, 4] = max(tail_biases)
    risk_table = _oracle_uncertainty_risk_table(
        candidates, plans, metrics, sigma[:, 3], risk_quantile,
        peak_tolerance, zero_tolerance_s, rms_tolerance, spectral_tolerance,
        tail_power_tolerance, memory_limit_roundtrips
    )
    feasible = np.where(risk_table[:, 12] == 1.0)[0]
    if feasible.size == 0:
        raise ValueError("no candidate satisfies every constraint")
    selected = min(feasible.tolist(), key=lambda i: (risk_table[i, 1], i))
    return float(round(risk_table[selected, 10] * 1.0e6, 6))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":'import numpy as np\ncandidate_desired_hz=np.array([400000.0, 650000.0, 900000.0],dtype=float); velocity_multipliers=np.array([-1.0, 0.75],dtype=float)\nn_roundtrips=12001; length_m=2.5; wavelength_m=1.064e-06; reflectivity_a=0.98; reflectivity_b=0.999; finesse=350.0\ndemodulation_phase=.2; displacement_fraction=.1; displacement_hz=12000.; amplitude_depth=.06; amplitude_hz=7000.; phase_depth=.1; phase_hz=17000.; phase_offset=.3\npeak_tolerance=1.; zero_tolerance_s=8e-6; rms_tolerance=2.; spectral_hz=np.array([12000.,17000.]); spectral_tolerance=1.; tail_power_tolerance=2.\nmemory_limit_roundtrips=800; transformed_covariance=np.array([[0.006, 0.002, -0.0004], [0.002, 0.015, 0.0005], [-0.0004, 0.0005, 0.0015]],dtype=float); risk_quantile=.5\n',"call":'adaptive_cavity_sampling_benchmark(candidate_desired_hz,velocity_multipliers,n_roundtrips,length_m,wavelength_m,reflectivity_a,reflectivity_b,finesse,demodulation_phase,displacement_fraction,displacement_hz,amplitude_depth,amplitude_hz,phase_depth,phase_hz,phase_offset,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_hz,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips,transformed_covariance,risk_quantile)',"gold_call":'_oracle_adaptive_cavity_sampling_benchmark(candidate_desired_hz,velocity_multipliers,n_roundtrips,length_m,wavelength_m,reflectivity_a,reflectivity_b,finesse,demodulation_phase,displacement_fraction,displacement_hz,amplitude_depth,amplitude_hz,phase_depth,phase_hz,phase_offset,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_hz,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips,transformed_covariance,risk_quantile)'},
        {"setup":'import numpy as np\ncandidate_desired_hz=np.array([350000.0, 700000.0, 1000000.0],dtype=float); velocity_multipliers=np.array([-1.1, -0.6, 0.9],dtype=float)\nn_roundtrips=14001; length_m=3.0; wavelength_m=1.55e-06; reflectivity_a=0.982; reflectivity_b=0.9992; finesse=380.0\ndemodulation_phase=.2; displacement_fraction=.1; displacement_hz=12000.; amplitude_depth=.06; amplitude_hz=7000.; phase_depth=.1; phase_hz=17000.; phase_offset=.3\npeak_tolerance=1.; zero_tolerance_s=8e-6; rms_tolerance=2.; spectral_hz=np.array([12000.,17000.]); spectral_tolerance=1.; tail_power_tolerance=2.\nmemory_limit_roundtrips=800; transformed_covariance=np.array([[0.006, 0.002, -0.0004], [0.002, 0.015, 0.0005], [-0.0004, 0.0005, 0.0015]],dtype=float); risk_quantile=.5\n',"call":'adaptive_cavity_sampling_benchmark(candidate_desired_hz,velocity_multipliers,n_roundtrips,length_m,wavelength_m,reflectivity_a,reflectivity_b,finesse,demodulation_phase,displacement_fraction,displacement_hz,amplitude_depth,amplitude_hz,phase_depth,phase_hz,phase_offset,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_hz,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips,transformed_covariance,risk_quantile)',"gold_call":'_oracle_adaptive_cavity_sampling_benchmark(candidate_desired_hz,velocity_multipliers,n_roundtrips,length_m,wavelength_m,reflectivity_a,reflectivity_b,finesse,demodulation_phase,displacement_fraction,displacement_hz,amplitude_depth,amplitude_hz,phase_depth,phase_hz,phase_offset,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_hz,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips,transformed_covariance,risk_quantile)'},
        {"setup":'import numpy as np\ncandidate_desired_hz=np.array([420000.0, 760000.0, 1100000.0],dtype=float); velocity_multipliers=np.array([-0.9, 0.55, 1.05],dtype=float)\nn_roundtrips=16001; length_m=2.2; wavelength_m=7.8e-07; reflectivity_a=0.975; reflectivity_b=0.9988; finesse=300.0\ndemodulation_phase=.2; displacement_fraction=.1; displacement_hz=12000.; amplitude_depth=.06; amplitude_hz=7000.; phase_depth=.1; phase_hz=17000.; phase_offset=.3\npeak_tolerance=1.; zero_tolerance_s=8e-6; rms_tolerance=2.; spectral_hz=np.array([12000.,17000.]); spectral_tolerance=1.; tail_power_tolerance=2.\nmemory_limit_roundtrips=800; transformed_covariance=np.array([[0.006, 0.002, -0.0004], [0.002, 0.015, 0.0005], [-0.0004, 0.0005, 0.0015]],dtype=float); risk_quantile=.5\n',"call":'adaptive_cavity_sampling_benchmark(candidate_desired_hz,velocity_multipliers,n_roundtrips,length_m,wavelength_m,reflectivity_a,reflectivity_b,finesse,demodulation_phase,displacement_fraction,displacement_hz,amplitude_depth,amplitude_hz,phase_depth,phase_hz,phase_offset,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_hz,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips,transformed_covariance,risk_quantile)',"gold_call":'_oracle_adaptive_cavity_sampling_benchmark(candidate_desired_hz,velocity_multipliers,n_roundtrips,length_m,wavelength_m,reflectivity_a,reflectivity_b,finesse,demodulation_phase,displacement_fraction,displacement_hz,amplitude_depth,amplitude_hz,phase_depth,phase_hz,phase_offset,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_hz,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips,transformed_covariance,risk_quantile)'},
        {"setup":'import numpy as np\ncandidate_desired_hz=np.array([400000.0, 650000.0, 900000.0],dtype=float); velocity_multipliers=np.array([-1.0, 0.75],dtype=float)\nn_roundtrips=12001; length_m=2.5; wavelength_m=1.064e-06; reflectivity_a=0.98; reflectivity_b=0.999; finesse=350.0\ndemodulation_phase=.2; displacement_fraction=.1; displacement_hz=12000.; amplitude_depth=.06; amplitude_hz=7000.; phase_depth=.1; phase_hz=17000.; phase_offset=.3\npeak_tolerance=1.; zero_tolerance_s=8e-6; rms_tolerance=2.; spectral_hz=np.array([12000.,17000.]); spectral_tolerance=.01; tail_power_tolerance=2.\nmemory_limit_roundtrips=120; transformed_covariance=np.array([[0.006, 0.002, -0.0004], [0.002, 0.015, 0.0005], [-0.0004, 0.0005, 0.0015]],dtype=float); risk_quantile=.5\n',"call":'adaptive_cavity_sampling_benchmark(candidate_desired_hz,velocity_multipliers,n_roundtrips,length_m,wavelength_m,reflectivity_a,reflectivity_b,finesse,demodulation_phase,displacement_fraction,displacement_hz,amplitude_depth,amplitude_hz,phase_depth,phase_hz,phase_offset,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_hz,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips,transformed_covariance,risk_quantile)',"gold_call":'_oracle_adaptive_cavity_sampling_benchmark(candidate_desired_hz,velocity_multipliers,n_roundtrips,length_m,wavelength_m,reflectivity_a,reflectivity_b,finesse,demodulation_phase,displacement_fraction,displacement_hz,amplitude_depth,amplitude_hz,phase_depth,phase_hz,phase_offset,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_hz,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips,transformed_covariance,risk_quantile)'},
    ]
