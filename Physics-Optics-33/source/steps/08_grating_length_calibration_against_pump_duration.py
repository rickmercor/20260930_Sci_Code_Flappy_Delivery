"""
Compute the calibration of effective grating length against pump duration for one pump energy and focusing geometry.

Because no closed-form relation exists between the grating length and the pulse duration, the diagnostic is calibrated numerically: the length that would be written is evaluated for a grid of trial durations under the physical conditions of the measurement, and the measured length is later inverted against the result. The calibration must therefore be produced with the same gas, pump energy, focal spot and numerical settings that describe the shot being analysed.



The returned array follows the order of the supplied trial durations, which are in seconds, and holds lengths in metres. The remaining arguments carry the same meaning and units as in the single-duration evaluation.

Returns
-------
Return grating_lengths_m, a NumPy array of dtype float64 and shape (durations_s.size,) holding the effective grating length in metres for each supplied trial duration, in the supplied order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_calibration_curve(durations_s: "np.ndarray", pulse_energy_j: float, waist_radius_m: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, z_max_m: float, n_axial: int, n_fringe_phase: int, n_time: int, time_window_factor: float) -> "np.ndarray":
    '''Return the grating length for each trial pump duration.

    Parameters
    ----------
    durations_s : np.ndarray
        One-dimensional, non-empty array of trial durations in seconds. All entries
        must be positive.
    pulse_energy_j : float
        Energy of each individual pump, in joules. Must be positive.
    waist_radius_m : float
        1/e^2 intensity radius of the common focal spot, in metres. Must be positive.
    neutral_density_m3 : float
        Neutral number density of the gas, in inverse cubic metres. Must be
        non-negative.
    ionization_potential_ev : float
        Ionization potential of the gas, in electronvolts. Must be positive.
    pump_wavelength_m : float
        Central wavelength of both pumps, in metres. Must be positive.
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
    grating_lengths_m : np.ndarray
        Array of shape (durations_s.size,) holding the grating length in metres for
        each trial duration.

    Raises
    ------
    ValueError
        If the array of durations is empty, not one-dimensional or contains a
        non-positive entry, or if any quantity forwarded to the single-duration
        evaluation is invalid.
    '''
    return grating_lengths_m

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_calibration_curve(durations_s: "np.ndarray", pulse_energy_j: float, waist_radius_m: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, z_max_m: float, n_axial: int, n_fringe_phase: int, n_time: int, time_window_factor: float) -> "np.ndarray":
    durations = np.asarray(durations_s, dtype=float)
    if durations.ndim != 1 or durations.size < 1:
        raise ValueError("durations_s must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("durations_s entries must be positive and finite")

    lengths = np.empty(durations.size, dtype=float)
    for index, duration in enumerate(durations):
        lengths[index] = _oracle_compute_grating_length(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
TS_c1 = np.array([25.0e-15, 55.0e-15, 85.0e-15, 115.0e-15])
E_c1 = 1.20e-3
W0_c1 = 40.0e-6
N0_c1 = 2.45e25
IPOT_c1 = 15.7596
LAM_c1 = 800.0e-9
ZMAX_c1 = 3.0e-5
NZ_c1 = 31
NPHI_c1 = 16
NT_c1 = 81
WF_c1 = 4.0
""",
            "call": 'compute_calibration_curve(TS_c1.copy(), E_c1, W0_c1, N0_c1, IPOT_c1, LAM_c1, ZMAX_c1, NZ_c1, NPHI_c1, NT_c1, WF_c1)',
            "gold_call": '_oracle_compute_calibration_curve(TS_c1.copy(), E_c1, W0_c1, N0_c1, IPOT_c1, LAM_c1, ZMAX_c1, NZ_c1, NPHI_c1, NT_c1, WF_c1)',
        },
        {
            "setup": """import numpy as np
TS_c2 = np.array([80.0e-15, 85.0e-15, 90.0e-15, 95.0e-15])
E_c2 = 1.20e-3
W0_c2 = 40.0e-6
N0_c2 = 2.45e25
IPOT_c2 = 15.7596
LAM_c2 = 800.0e-9
ZMAX_c2 = 3.0e-5
NZ_c2 = 31
NPHI_c2 = 16
NT_c2 = 81
WF_c2 = 4.0
""",
            "call": 'compute_calibration_curve(TS_c2.copy(), E_c2, W0_c2, N0_c2, IPOT_c2, LAM_c2, ZMAX_c2, NZ_c2, NPHI_c2, NT_c2, WF_c2)',
            "gold_call": '_oracle_compute_calibration_curve(TS_c2.copy(), E_c2, W0_c2, N0_c2, IPOT_c2, LAM_c2, ZMAX_c2, NZ_c2, NPHI_c2, NT_c2, WF_c2)',
        },
        {
            "setup": """import numpy as np
TS_c3 = np.array([88.6e-15])
E_c3 = 1.20e-3
W0_c3 = 40.0e-6
N0_c3 = 2.45e25
IPOT_c3 = 15.7596
LAM_c3 = 800.0e-9
ZMAX_c3 = 3.0e-5
NZ_c3 = 31
NPHI_c3 = 16
NT_c3 = 81
WF_c3 = 4.0
""",
            "call": 'compute_calibration_curve(TS_c3.copy(), E_c3, W0_c3, N0_c3, IPOT_c3, LAM_c3, ZMAX_c3, NZ_c3, NPHI_c3, NT_c3, WF_c3)',
            "gold_call": '_oracle_compute_calibration_curve(TS_c3.copy(), E_c3, W0_c3, N0_c3, IPOT_c3, LAM_c3, ZMAX_c3, NZ_c3, NPHI_c3, NT_c3, WF_c3)',
        },
        {
            "setup": """import numpy as np
TS_c4 = np.array([100.0e-15, 40.0e-15, 70.0e-15])
E_c4 = 1.20e-3
W0_c4 = 40.0e-6
N0_c4 = 2.45e25
IPOT_c4 = 15.7596
LAM_c4 = 800.0e-9
ZMAX_c4 = 3.0e-5
NZ_c4 = 31
NPHI_c4 = 16
NT_c4 = 81
WF_c4 = 4.0
""",
            "call": 'compute_calibration_curve(TS_c4.copy(), E_c4, W0_c4, N0_c4, IPOT_c4, LAM_c4, ZMAX_c4, NZ_c4, NPHI_c4, NT_c4, WF_c4)',
            "gold_call": '_oracle_compute_calibration_curve(TS_c4.copy(), E_c4, W0_c4, N0_c4, IPOT_c4, LAM_c4, ZMAX_c4, NZ_c4, NPHI_c4, NT_c4, WF_c4)',
        },
        {
            "setup": """import numpy as np
TS_c5 = np.array([30.0e-15, 60.0e-15, 90.0e-15])
E_c5 = 2.0e-5
W0_c5 = 40.0e-6
N0_c5 = 2.45e25
IPOT_c5 = 15.7596
LAM_c5 = 800.0e-9
ZMAX_c5 = 3.0e-5
NZ_c5 = 31
NPHI_c5 = 16
NT_c5 = 81
WF_c5 = 4.0
""",
            "call": 'compute_calibration_curve(TS_c5.copy(), E_c5, W0_c5, N0_c5, IPOT_c5, LAM_c5, ZMAX_c5, NZ_c5, NPHI_c5, NT_c5, WF_c5)',
            "gold_call": '_oracle_compute_calibration_curve(TS_c5.copy(), E_c5, W0_c5, N0_c5, IPOT_c5, LAM_c5, ZMAX_c5, NZ_c5, NPHI_c5, NT_c5, WF_c5)',
        },
        {
            "setup": """import numpy as np
TS_c6 = np.array([40.0e-15, 80.0e-15])
E_c6 = 1.20e-3
W0_c6 = 40.0e-6
N0_c6 = 2.45e25
IPOT_c6 = 24.5874
LAM_c6 = 800.0e-9
ZMAX_c6 = 3.0e-5
NZ_c6 = 31
NPHI_c6 = 16
NT_c6 = 81
WF_c6 = 4.0
""",
            "call": 'compute_calibration_curve(TS_c6.copy(), E_c6, W0_c6, N0_c6, IPOT_c6, LAM_c6, ZMAX_c6, NZ_c6, NPHI_c6, NT_c6, WF_c6)',
            "gold_call": '_oracle_compute_calibration_curve(TS_c6.copy(), E_c6, W0_c6, N0_c6, IPOT_c6, LAM_c6, ZMAX_c6, NZ_c6, NPHI_c6, NT_c6, WF_c6)',
        },
        {
            "setup": """import numpy as np
TS_c7 = np.array([60.0e-15, 100.0e-15])
E_c7 = 1.20e-3
W0_c7 = 40.0e-6
N0_c7 = 2.45e25
IPOT_c7 = 15.7596
LAM_c7 = 800.0e-9
ZMAX_c7 = 3.0e-5
NZ_c7 = 61
NPHI_c7 = 32
NT_c7 = 161
WF_c7 = 4.0
""",
            "call": 'compute_calibration_curve(TS_c7.copy(), E_c7, W0_c7, N0_c7, IPOT_c7, LAM_c7, ZMAX_c7, NZ_c7, NPHI_c7, NT_c7, WF_c7)',
            "gold_call": '_oracle_compute_calibration_curve(TS_c7.copy(), E_c7, W0_c7, N0_c7, IPOT_c7, LAM_c7, ZMAX_c7, NZ_c7, NPHI_c7, NT_c7, WF_c7)',
        },
        {
            "setup": """import numpy as np
TS_c8 = np.zeros(0)
E_c8 = 1.20e-3
W0_c8 = 40.0e-6
N0_c8 = 2.45e25
IPOT_c8 = 15.7596
LAM_c8 = 800.0e-9
ZMAX_c8 = 3.0e-5
NZ_c8 = 31
NPHI_c8 = 16
NT_c8 = 81
WF_c8 = 4.0

def run_c8(fn):
    try:
        fn(TS_c8.copy(), E_c8, W0_c8, N0_c8, IPOT_c8, LAM_c8, ZMAX_c8, NZ_c8, NPHI_c8, NT_c8, WF_c8)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c8(compute_calibration_curve)',
            "gold_call": 'run_c8(_oracle_compute_calibration_curve)',
        },
        {
            "setup": """import numpy as np
TS_c9 = np.array([50.0e-15, 0.0])
E_c9 = 1.20e-3
W0_c9 = 40.0e-6
N0_c9 = 2.45e25
IPOT_c9 = 15.7596
LAM_c9 = 800.0e-9
ZMAX_c9 = 3.0e-5
NZ_c9 = 31
NPHI_c9 = 16
NT_c9 = 81
WF_c9 = 4.0

def run_c9(fn):
    try:
        fn(TS_c9.copy(), E_c9, W0_c9, N0_c9, IPOT_c9, LAM_c9, ZMAX_c9, NZ_c9, NPHI_c9, NT_c9, WF_c9)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c9(compute_calibration_curve)',
            "gold_call": 'run_c9(_oracle_compute_calibration_curve)',
        },
        {
            "setup": """import numpy as np
TS_c10 = np.full((2, 2), 50.0e-15)
E_c10 = 1.20e-3
W0_c10 = 40.0e-6
N0_c10 = 2.45e25
IPOT_c10 = 15.7596
LAM_c10 = 800.0e-9
ZMAX_c10 = 3.0e-5
NZ_c10 = 31
NPHI_c10 = 16
NT_c10 = 81
WF_c10 = 4.0

def run_c10(fn):
    try:
        fn(TS_c10.copy(), E_c10, W0_c10, N0_c10, IPOT_c10, LAM_c10, ZMAX_c10, NZ_c10, NPHI_c10, NT_c10, WF_c10)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c10(compute_calibration_curve)',
            "gold_call": 'run_c10(_oracle_compute_calibration_curve)',
        },
    ]
