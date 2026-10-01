"""
Compute the pulse duration recorded by the diagnostic, from the imaged extent of the first diffraction order and the numerical calibration of the plasma structure.

This is the end-to-end evaluation the measurement performs. It combines the readout geometry, which turns the corrected pixel count of the recorded axial envelope into a length inside the plasma, with the calibration of grating length against trial pump duration built for the gas, pump energy, focal spot and numerical settings of the shot, and returns the duration consistent with the recorded image.



The trial durations must be supplied in seconds in increasing order, and the calibration is inverted at the measured length by linear interpolation of duration against length. The returned duration is expressed in femtoseconds; every other argument carries the same meaning and units as in the earlier evaluations.

Returns
-------
Return retrieved_duration_fs, a single float: the retrieved pulse duration expressed in femtoseconds.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def retrieve_pulse_duration(pump_wavelength_m: float, probe_wavelength_m: float, medium_index: float, pixel_pitch_m: float, magnification: float, corrected_pixel_count: float, calibration_durations_s: "np.ndarray", pulse_energy_j: float, waist_radius_m: float, neutral_density_m3: float, ionization_potential_ev: float, z_max_m: float, n_axial: int, n_fringe_phase: int, n_time: int, time_window_factor: float) -> float:
    '''Return the retrieved pulse duration in femtoseconds.

    Parameters
    ----------
    pump_wavelength_m : float
        Central wavelength of both pumps, in metres. Must be positive.
    probe_wavelength_m : float
        Central wavelength of the readout probe, in metres. Must be positive and
        small enough for the Bragg order to exist.
    medium_index : float
        Refractive index of the gas at the probe wavelength. Must be positive.
    pixel_pitch_m : float
        Physical pixel pitch of the camera, in metres. Must be positive.
    magnification : float
        Object-to-image magnification of the imaging system. Must be positive.
    corrected_pixel_count : float
        Corrected full width at half maximum of the recorded axial envelope, in
        pixels. Must be positive.
    calibration_durations_s : np.ndarray
        One-dimensional array of at least two trial durations in seconds, strictly
        increasing and all positive.
    pulse_energy_j : float
        Energy of each individual pump, in joules. Must be positive.
    waist_radius_m : float
        1/e^2 intensity radius of the common focal spot, in metres. Must be positive.
    neutral_density_m3 : float
        Neutral number density of the gas, in inverse cubic metres. Must be
        non-negative.
    ionization_potential_ev : float
        Ionization potential of the gas, in electronvolts. Must be positive.
    z_max_m : float
        Largest sampled axial position, in metres. Must be positive.
    n_axial : int
        Number of uniformly spaced axial positions from zero to z_max_m inclusive.
        Must be at least two.
    n_fringe_phase : int
        Number of uniformly spaced fringe phases covering one full period. Must be
        at least one.
    n_time : int
        Number of uniformly spaced time samples. Must be at least two.
    time_window_factor : float
        Multiple of the pulse duration by which the time interval extends beyond the
        outermost envelope centre. Must be positive.

    Returns
    -------
    retrieved_duration_fs : float
        Retrieved pulse duration, in femtoseconds.

    Raises
    ------
    ValueError
        If fewer than two trial durations are supplied, if the calibrated lengths are
        not strictly increasing with duration, if the measured length falls outside
        the calibrated range, or if any quantity forwarded to the earlier evaluations
        is invalid.
    '''
    return retrieved_duration_fs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_retrieve_pulse_duration(pump_wavelength_m: float, probe_wavelength_m: float, medium_index: float, pixel_pitch_m: float, magnification: float, corrected_pixel_count: float, calibration_durations_s: "np.ndarray", pulse_energy_j: float, waist_radius_m: float, neutral_density_m3: float, ionization_potential_ev: float, z_max_m: float, n_axial: int, n_fringe_phase: int, n_time: int, time_window_factor: float) -> float:
    durations = np.asarray(calibration_durations_s, dtype=float)
    if durations.ndim != 1 or durations.size < 2:
        raise ValueError("calibration_durations_s must hold at least two durations")
    if np.any(np.diff(durations) <= 0.0):
        raise ValueError("calibration_durations_s must be strictly increasing")

    geometry = _oracle_compute_readout_geometry(
        pump_wavelength_m,
        probe_wavelength_m,
        medium_index,
        pixel_pitch_m,
        magnification,
        corrected_pixel_count,
    )
    measured_length_m = float(geometry[3])

    lengths = _oracle_compute_calibration_curve(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
LAM_c1 = 800.0e-9
LPR_c1 = 400.0e-9
NMED_c1 = 1.000
PIX_c1 = 3.45e-6
MAG_c1 = 7.00
Q_c1 = 24.0
TS_c1 = np.array([25.0e-15, 55.0e-15, 85.0e-15, 115.0e-15])
E_c1 = 1.20e-3
W0_c1 = 40.0e-6
N0_c1 = 2.45e25
IPOT_c1 = 15.7596
ZMAX_c1 = 3.0e-5
NZ_c1 = 21
NPHI_c1 = 12
NT_c1 = 61
WF_c1 = 4.0
""",
            "call": 'retrieve_pulse_duration(LAM_c1, LPR_c1, NMED_c1, PIX_c1, MAG_c1, Q_c1, TS_c1.copy(), E_c1, W0_c1, N0_c1, IPOT_c1, ZMAX_c1, NZ_c1, NPHI_c1, NT_c1, WF_c1)',
            "gold_call": '_oracle_retrieve_pulse_duration(LAM_c1, LPR_c1, NMED_c1, PIX_c1, MAG_c1, Q_c1, TS_c1.copy(), E_c1, W0_c1, N0_c1, IPOT_c1, ZMAX_c1, NZ_c1, NPHI_c1, NT_c1, WF_c1)',
        },
        {
            "setup": """import numpy as np
LAM_c2 = 800.0e-9
LPR_c2 = 400.0e-9
NMED_c2 = 1.000
PIX_c2 = 3.45e-6
MAG_c2 = 7.00
Q_c2 = 18.0
TS_c2 = np.array([25.0e-15, 55.0e-15, 85.0e-15, 115.0e-15])
E_c2 = 1.20e-3
W0_c2 = 40.0e-6
N0_c2 = 2.45e25
IPOT_c2 = 15.7596
ZMAX_c2 = 3.0e-5
NZ_c2 = 21
NPHI_c2 = 12
NT_c2 = 61
WF_c2 = 4.0
""",
            "call": 'retrieve_pulse_duration(LAM_c2, LPR_c2, NMED_c2, PIX_c2, MAG_c2, Q_c2, TS_c2.copy(), E_c2, W0_c2, N0_c2, IPOT_c2, ZMAX_c2, NZ_c2, NPHI_c2, NT_c2, WF_c2)',
            "gold_call": '_oracle_retrieve_pulse_duration(LAM_c2, LPR_c2, NMED_c2, PIX_c2, MAG_c2, Q_c2, TS_c2.copy(), E_c2, W0_c2, N0_c2, IPOT_c2, ZMAX_c2, NZ_c2, NPHI_c2, NT_c2, WF_c2)',
        },
        {
            "setup": """import numpy as np
LAM_c3 = 800.0e-9
LPR_c3 = 400.0e-9
NMED_c3 = 1.000
PIX_c3 = 3.45e-6
MAG_c3 = 7.00
Q_c3 = 30.0
TS_c3 = np.array([25.0e-15, 55.0e-15, 85.0e-15, 115.0e-15])
E_c3 = 1.20e-3
W0_c3 = 40.0e-6
N0_c3 = 2.45e25
IPOT_c3 = 15.7596
ZMAX_c3 = 3.0e-5
NZ_c3 = 21
NPHI_c3 = 12
NT_c3 = 61
WF_c3 = 4.0
""",
            "call": 'retrieve_pulse_duration(LAM_c3, LPR_c3, NMED_c3, PIX_c3, MAG_c3, Q_c3, TS_c3.copy(), E_c3, W0_c3, N0_c3, IPOT_c3, ZMAX_c3, NZ_c3, NPHI_c3, NT_c3, WF_c3)',
            "gold_call": '_oracle_retrieve_pulse_duration(LAM_c3, LPR_c3, NMED_c3, PIX_c3, MAG_c3, Q_c3, TS_c3.copy(), E_c3, W0_c3, N0_c3, IPOT_c3, ZMAX_c3, NZ_c3, NPHI_c3, NT_c3, WF_c3)',
        },
        {
            "setup": """import numpy as np
LAM_c4 = 800.0e-9
LPR_c4 = 266.0e-9
NMED_c4 = 1.000
PIX_c4 = 3.45e-6
MAG_c4 = 7.00
Q_c4 = 24.0
TS_c4 = np.array([25.0e-15, 55.0e-15, 85.0e-15, 115.0e-15])
E_c4 = 1.20e-3
W0_c4 = 40.0e-6
N0_c4 = 2.45e25
IPOT_c4 = 15.7596
ZMAX_c4 = 3.0e-5
NZ_c4 = 21
NPHI_c4 = 12
NT_c4 = 61
WF_c4 = 4.0
""",
            "call": 'retrieve_pulse_duration(LAM_c4, LPR_c4, NMED_c4, PIX_c4, MAG_c4, Q_c4, TS_c4.copy(), E_c4, W0_c4, N0_c4, IPOT_c4, ZMAX_c4, NZ_c4, NPHI_c4, NT_c4, WF_c4)',
            "gold_call": '_oracle_retrieve_pulse_duration(LAM_c4, LPR_c4, NMED_c4, PIX_c4, MAG_c4, Q_c4, TS_c4.copy(), E_c4, W0_c4, N0_c4, IPOT_c4, ZMAX_c4, NZ_c4, NPHI_c4, NT_c4, WF_c4)',
        },
        {
            "setup": """import numpy as np
LAM_c5 = 800.0e-9
LPR_c5 = 400.0e-9
NMED_c5 = 1.000
PIX_c5 = 6.5e-6
MAG_c5 = 12.0
Q_c5 = 15.0
TS_c5 = np.array([25.0e-15, 55.0e-15, 85.0e-15, 115.0e-15])
E_c5 = 1.20e-3
W0_c5 = 40.0e-6
N0_c5 = 2.45e25
IPOT_c5 = 15.7596
ZMAX_c5 = 3.0e-5
NZ_c5 = 21
NPHI_c5 = 12
NT_c5 = 61
WF_c5 = 4.0
""",
            "call": 'retrieve_pulse_duration(LAM_c5, LPR_c5, NMED_c5, PIX_c5, MAG_c5, Q_c5, TS_c5.copy(), E_c5, W0_c5, N0_c5, IPOT_c5, ZMAX_c5, NZ_c5, NPHI_c5, NT_c5, WF_c5)',
            "gold_call": '_oracle_retrieve_pulse_duration(LAM_c5, LPR_c5, NMED_c5, PIX_c5, MAG_c5, Q_c5, TS_c5.copy(), E_c5, W0_c5, N0_c5, IPOT_c5, ZMAX_c5, NZ_c5, NPHI_c5, NT_c5, WF_c5)',
        },
        {
            "setup": """import numpy as np
LAM_c6 = 800.0e-9
LPR_c6 = 400.0e-9
NMED_c6 = 1.000
PIX_c6 = 3.45e-6
MAG_c6 = 7.00
Q_c6 = 24.0
TS_c6 = np.array([25.0e-15, 45.0e-15, 65.0e-15, 85.0e-15, 105.0e-15])
E_c6 = 6.0e-4
W0_c6 = 40.0e-6
N0_c6 = 2.45e25
IPOT_c6 = 15.7596
ZMAX_c6 = 3.0e-5
NZ_c6 = 21
NPHI_c6 = 12
NT_c6 = 61
WF_c6 = 4.0
""",
            "call": 'retrieve_pulse_duration(LAM_c6, LPR_c6, NMED_c6, PIX_c6, MAG_c6, Q_c6, TS_c6.copy(), E_c6, W0_c6, N0_c6, IPOT_c6, ZMAX_c6, NZ_c6, NPHI_c6, NT_c6, WF_c6)',
            "gold_call": '_oracle_retrieve_pulse_duration(LAM_c6, LPR_c6, NMED_c6, PIX_c6, MAG_c6, Q_c6, TS_c6.copy(), E_c6, W0_c6, N0_c6, IPOT_c6, ZMAX_c6, NZ_c6, NPHI_c6, NT_c6, WF_c6)',
        },
        {
            "setup": """import numpy as np
LAM_c7 = 800.0e-9
LPR_c7 = 400.0e-9
NMED_c7 = 1.000
PIX_c7 = 3.45e-6
MAG_c7 = 7.00
Q_c7 = 24.0
TS_c7 = np.array([60.0e-15, 115.0e-15])
E_c7 = 1.20e-3
W0_c7 = 40.0e-6
N0_c7 = 2.45e25
IPOT_c7 = 15.7596
ZMAX_c7 = 3.0e-5
NZ_c7 = 21
NPHI_c7 = 12
NT_c7 = 61
WF_c7 = 4.0
""",
            "call": 'retrieve_pulse_duration(LAM_c7, LPR_c7, NMED_c7, PIX_c7, MAG_c7, Q_c7, TS_c7.copy(), E_c7, W0_c7, N0_c7, IPOT_c7, ZMAX_c7, NZ_c7, NPHI_c7, NT_c7, WF_c7)',
            "gold_call": '_oracle_retrieve_pulse_duration(LAM_c7, LPR_c7, NMED_c7, PIX_c7, MAG_c7, Q_c7, TS_c7.copy(), E_c7, W0_c7, N0_c7, IPOT_c7, ZMAX_c7, NZ_c7, NPHI_c7, NT_c7, WF_c7)',
        },
        {
            "setup": """import numpy as np
LAM_c8 = 800.0e-9
LPR_c8 = 400.0e-9
NMED_c8 = 1.000
PIX_c8 = 3.45e-6
MAG_c8 = 7.00
Q_c8 = 200.0
TS_c8 = np.array([25.0e-15, 55.0e-15, 85.0e-15, 115.0e-15])
E_c8 = 1.20e-3
W0_c8 = 40.0e-6
N0_c8 = 2.45e25
IPOT_c8 = 15.7596
ZMAX_c8 = 3.0e-5
NZ_c8 = 21
NPHI_c8 = 12
NT_c8 = 61
WF_c8 = 4.0

def run_c8(fn):
    try:
        fn(LAM_c8, LPR_c8, NMED_c8, PIX_c8, MAG_c8, Q_c8, TS_c8.copy(), E_c8, W0_c8, N0_c8, IPOT_c8, ZMAX_c8, NZ_c8, NPHI_c8, NT_c8, WF_c8)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c8(retrieve_pulse_duration)',
            "gold_call": 'run_c8(_oracle_retrieve_pulse_duration)',
        },
        {
            "setup": """import numpy as np
LAM_c9 = 800.0e-9
LPR_c9 = 400.0e-9
NMED_c9 = 1.000
PIX_c9 = 3.45e-6
MAG_c9 = 7.00
Q_c9 = 24.0
TS_c9 = np.array([85.0e-15, 25.0e-15, 115.0e-15])
E_c9 = 1.20e-3
W0_c9 = 40.0e-6
N0_c9 = 2.45e25
IPOT_c9 = 15.7596
ZMAX_c9 = 3.0e-5
NZ_c9 = 21
NPHI_c9 = 12
NT_c9 = 61
WF_c9 = 4.0

def run_c9(fn):
    try:
        fn(LAM_c9, LPR_c9, NMED_c9, PIX_c9, MAG_c9, Q_c9, TS_c9.copy(), E_c9, W0_c9, N0_c9, IPOT_c9, ZMAX_c9, NZ_c9, NPHI_c9, NT_c9, WF_c9)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c9(retrieve_pulse_duration)',
            "gold_call": 'run_c9(_oracle_retrieve_pulse_duration)',
        },
        {
            "setup": """import numpy as np
LAM_c10 = 800.0e-9
LPR_c10 = 400.0e-9
NMED_c10 = 1.000
PIX_c10 = 3.45e-6
MAG_c10 = 7.00
Q_c10 = 24.0
TS_c10 = np.array([88.6e-15])
E_c10 = 1.20e-3
W0_c10 = 40.0e-6
N0_c10 = 2.45e25
IPOT_c10 = 15.7596
ZMAX_c10 = 3.0e-5
NZ_c10 = 21
NPHI_c10 = 12
NT_c10 = 61
WF_c10 = 4.0

def run_c10(fn):
    try:
        fn(LAM_c10, LPR_c10, NMED_c10, PIX_c10, MAG_c10, Q_c10, TS_c10.copy(), E_c10, W0_c10, N0_c10, IPOT_c10, ZMAX_c10, NZ_c10, NPHI_c10, NT_c10, WF_c10)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c10(retrieve_pulse_duration)',
            "gold_call": 'run_c10(_oracle_retrieve_pulse_duration)',
        },
    ]
