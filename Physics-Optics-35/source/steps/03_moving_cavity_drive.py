"""
Construct the centered round-trip time grid, relative optical-path drive, and amplitude/phase-modulated complex input field from the critical speed supplied by the preceding step.

Column order is [time_s, delta_d_m, Re(E_in), Im(E_in)]. The nominal resonant length is removed modulo 2*pi, so delta_d alone enters the phase. The orchestrator supplies v_cr and the physical sinusoidal displacement amplitude in metres. The displacement term is displacement_amplitude_m*sin(2*pi*displacement_hz*t + phase_offset), with no mirror retardation.

Returns
-------
Return one real NumPy array of shape (n_roundtrips,4) with the fixed column order.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def moving_cavity_drive(n_roundtrips, length_m, wavelength_m, critical_velocity_m_s,
                        velocity_multiplier, displacement_amplitude_m,
                        displacement_hz, amplitude_depth, amplitude_hz,
                        phase_depth, phase_hz, phase_offset=0.35,
                        c_m_s=299792458.0):
    """Return the deterministic n-by-4 drive array."""
    return np.empty((0, 4), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_moving_cavity_drive(n_roundtrips, length_m, wavelength_m, critical_velocity_m_s,
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"", "call":"moving_cavity_drive(101,2.75,1064e-9,5.7e-4,1.1,4.8e-10,18000.,0.12,11000.,0.18,23000.)", "gold_call":"_oracle_moving_cavity_drive(101,2.75,1064e-9,5.7e-4,1.1,4.8e-10,18000.,0.12,11000.,0.18,23000.)"},
        {"setup":"", "call":"moving_cavity_drive(103,2.75,1064e-9,5.7e-4,-0.8,4.8e-10,18000.,0.12,11000.,0.18,23000.)", "gold_call":"_oracle_moving_cavity_drive(103,2.75,1064e-9,5.7e-4,-0.8,4.8e-10,18000.,0.12,11000.,0.18,23000.)"},
        {"setup":"", "call":"moving_cavity_drive(51,1.5,1550e-9,2e-4,0.,1e-10,5000.,0.2,3000.,0.3,4000.)", "gold_call":"_oracle_moving_cavity_drive(51,1.5,1550e-9,2e-4,0.,1e-10,5000.,0.2,3000.,0.3,4000.)"},
        {"setup":"", "call":"moving_cavity_drive(75,3.,780e-9,3e-4,0.65,0.,7000.,0.,4000.,0.,9000.,0.)", "gold_call":"_oracle_moving_cavity_drive(75,3.,780e-9,3e-4,0.65,0.,7000.,0.,4000.,0.,9000.,0.)"},
        {"setup":"", "call":"moving_cavity_drive(77,2.,1064e-9,4e-4,-1.25,2e-10,0.,0.1,6000.,0.2,8000.,-0.2)", "gold_call":"_oracle_moving_cavity_drive(77,2.,1064e-9,4e-4,-1.25,2e-10,0.,0.1,6000.,0.2,8000.,-0.2)"},
        {"setup":"", "call":"moving_cavity_drive(99,4.,1550e-9,1e-4,1.,3e-10,9000.,0.15,0.,0.25,0.,0.7)", "gold_call":"_oracle_moving_cavity_drive(99,4.,1550e-9,1e-4,1.,3e-10,9000.,0.15,0.,0.25,0.,0.7)"},
        {"setup":"", "call":"moving_cavity_drive(55,2.2,532e-9,6e-4,-0.4,8e-11,15000.,0.05,12000.,0.1,17000.,1.1)", "gold_call":"_oracle_moving_cavity_drive(55,2.2,532e-9,6e-4,-0.4,8e-11,15000.,0.05,12000.,0.1,17000.,1.1)"},
        {"setup":"", "call":"moving_cavity_drive(101,2.75,1064e-9,5.7e-4,1.1,4.8e-10,18000.,0.12,11000.,0.18,23000.)[:,1]/1064e-9", "gold_call":"_oracle_moving_cavity_drive(101,2.75,1064e-9,5.7e-4,1.1,4.8e-10,18000.,0.12,11000.,0.18,23000.)[:,1]/1064e-9"},
        {"setup":"", "call":"moving_cavity_drive(103,2.75,1064e-9,5.7e-4,-0.8,4.8e-10,18000.,0.12,11000.,0.18,23000.)[:,1]/1064e-9", "gold_call":"_oracle_moving_cavity_drive(103,2.75,1064e-9,5.7e-4,-0.8,4.8e-10,18000.,0.12,11000.,0.18,23000.)[:,1]/1064e-9"},
        {"setup":"", "call":"moving_cavity_drive(55,2.2,532e-9,6e-4,-0.4,8e-11,15000.,0.05,12000.,0.1,17000.,1.1)[:,1]/532e-9", "gold_call":"_oracle_moving_cavity_drive(55,2.2,532e-9,6e-4,-0.4,8e-11,15000.,0.05,12000.,0.1,17000.,1.1)[:,1]/532e-9"}
    ]
