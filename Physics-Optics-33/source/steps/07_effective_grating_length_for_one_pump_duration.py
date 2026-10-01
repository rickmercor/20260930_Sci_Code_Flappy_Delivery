"""
Compute the effective grating length written by a pump pulse of the supplied duration at fixed pump energy and focusing geometry.

The pump energy and the focal spot are properties of the beamline and stay fixed while the compressor is detuned, so changing the duration changes the peak intensity of each pump as well as the interval over which the two pumps overlap. The length returned here is the quantity the camera measures: the full width at half maximum of the axial envelope of the first-order diffraction signal.



The axial envelope is symmetric about the point of exact temporal coincidence, so it is sampled only at non-negative axial positions, on the supplied number of uniformly spaced positions running from zero to the supplied maximum inclusive. The length is twice the position at which the envelope first falls to half its value at the origin, located by linear interpolation between the two bracketing samples. The returned length is in metres, the duration and the time-window factor refer to the pump pulse, the energy is in joules, the waist radius and the maximum axial position are in metres, the neutral density is in inverse cubic metres, the ionization potential is in electronvolts and the pump wavelength is in metres.

Returns
-------
Return grating_length_m, a single float: the effective grating length in metres for the supplied pump duration.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_grating_length(duration_fwhm_s: float, pulse_energy_j: float, waist_radius_m: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, z_max_m: float, n_axial: int, n_fringe_phase: int, n_time: int, time_window_factor: float) -> float:
    '''Return the effective grating length for one pump duration.

    Parameters
    ----------
    duration_fwhm_s : float
        Full width at half maximum of each pump's Gaussian temporal intensity
        envelope, in seconds. Must be positive.
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
    grating_length_m : float
        Effective grating length, in metres.

    Raises
    ------
    ValueError
        If z_max_m is not positive, if n_axial is below two, if the axial envelope
        does not fall to half its value at the origin anywhere within the sampled
        range, or if any quantity forwarded to the earlier evaluations is invalid.
    '''
    return grating_length_m

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_grating_length(duration_fwhm_s: float, pulse_energy_j: float, waist_radius_m: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, z_max_m: float, n_axial: int, n_fringe_phase: int, n_time: int, time_window_factor: float) -> float:
    if not np.isfinite(z_max_m) or z_max_m <= 0.0:
        raise ValueError("z_max_m must be a positive finite number")
    if int(n_axial) != n_axial or int(n_axial) < 2:
        raise ValueError("n_axial must be an integer of at least two")

    peak_intensity = _oracle_compute_peak_intensity(
        pulse_energy_j, waist_radius_m, duration_fwhm_s
    )
    positions = np.linspace(0.0, float(z_max_m), int(n_axial))
    amplitudes = _oracle_compute_first_harmonic(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
TAU_c1 = 88.6e-15
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
            "call": 'compute_grating_length(TAU_c1, E_c1, W0_c1, N0_c1, IPOT_c1, LAM_c1, ZMAX_c1, NZ_c1, NPHI_c1, NT_c1, WF_c1)',
            "gold_call": '_oracle_compute_grating_length(TAU_c1, E_c1, W0_c1, N0_c1, IPOT_c1, LAM_c1, ZMAX_c1, NZ_c1, NPHI_c1, NT_c1, WF_c1)',
        },
        {
            "setup": """import numpy as np
TAU_c2 = 25.0e-15
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
            "call": 'compute_grating_length(TAU_c2, E_c2, W0_c2, N0_c2, IPOT_c2, LAM_c2, ZMAX_c2, NZ_c2, NPHI_c2, NT_c2, WF_c2)',
            "gold_call": '_oracle_compute_grating_length(TAU_c2, E_c2, W0_c2, N0_c2, IPOT_c2, LAM_c2, ZMAX_c2, NZ_c2, NPHI_c2, NT_c2, WF_c2)',
        },
        {
            "setup": """import numpy as np
TAU_c3 = 115.0e-15
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
            "call": 'compute_grating_length(TAU_c3, E_c3, W0_c3, N0_c3, IPOT_c3, LAM_c3, ZMAX_c3, NZ_c3, NPHI_c3, NT_c3, WF_c3)',
            "gold_call": '_oracle_compute_grating_length(TAU_c3, E_c3, W0_c3, N0_c3, IPOT_c3, LAM_c3, ZMAX_c3, NZ_c3, NPHI_c3, NT_c3, WF_c3)',
        },
        {
            "setup": """import numpy as np
TAU_c4 = 70.0e-15
E_c4 = 6.0e-4
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
            "call": 'compute_grating_length(TAU_c4, E_c4, W0_c4, N0_c4, IPOT_c4, LAM_c4, ZMAX_c4, NZ_c4, NPHI_c4, NT_c4, WF_c4)',
            "gold_call": '_oracle_compute_grating_length(TAU_c4, E_c4, W0_c4, N0_c4, IPOT_c4, LAM_c4, ZMAX_c4, NZ_c4, NPHI_c4, NT_c4, WF_c4)',
        },
        {
            "setup": """import numpy as np
TAU_c5 = 40.0e-15
E_c5 = 1.20e-3
W0_c5 = 40.0e-6
N0_c5 = 2.45e25
IPOT_c5 = 15.7596
LAM_c5 = 800.0e-9
ZMAX_c5 = 6.0e-6
NZ_c5 = 2
NPHI_c5 = 16
NT_c5 = 81
WF_c5 = 4.0
""",
            "call": 'compute_grating_length(TAU_c5, E_c5, W0_c5, N0_c5, IPOT_c5, LAM_c5, ZMAX_c5, NZ_c5, NPHI_c5, NT_c5, WF_c5)',
            "gold_call": '_oracle_compute_grating_length(TAU_c5, E_c5, W0_c5, N0_c5, IPOT_c5, LAM_c5, ZMAX_c5, NZ_c5, NPHI_c5, NT_c5, WF_c5)',
        },
        {
            "setup": """import numpy as np
TAU_c6 = 60.0e-15
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
            "call": 'compute_grating_length(TAU_c6, E_c6, W0_c6, N0_c6, IPOT_c6, LAM_c6, ZMAX_c6, NZ_c6, NPHI_c6, NT_c6, WF_c6)',
            "gold_call": '_oracle_compute_grating_length(TAU_c6, E_c6, W0_c6, N0_c6, IPOT_c6, LAM_c6, ZMAX_c6, NZ_c6, NPHI_c6, NT_c6, WF_c6)',
        },
        {
            "setup": """import numpy as np
TAU_c7 = 50.0e-15
E_c7 = 1.20e-3
W0_c7 = 25.0e-6
N0_c7 = 2.45e25
IPOT_c7 = 15.7596
LAM_c7 = 400.0e-9
ZMAX_c7 = 3.0e-5
NZ_c7 = 31
NPHI_c7 = 16
NT_c7 = 81
WF_c7 = 4.0
""",
            "call": 'compute_grating_length(TAU_c7, E_c7, W0_c7, N0_c7, IPOT_c7, LAM_c7, ZMAX_c7, NZ_c7, NPHI_c7, NT_c7, WF_c7)',
            "gold_call": '_oracle_compute_grating_length(TAU_c7, E_c7, W0_c7, N0_c7, IPOT_c7, LAM_c7, ZMAX_c7, NZ_c7, NPHI_c7, NT_c7, WF_c7)',
        },
        {
            "setup": """import numpy as np
TAU_c8 = 115.0e-15
E_c8 = 1.20e-3
W0_c8 = 40.0e-6
N0_c8 = 2.45e25
IPOT_c8 = 15.7596
LAM_c8 = 800.0e-9
ZMAX_c8 = 2.0e-7
NZ_c8 = 5
NPHI_c8 = 16
NT_c8 = 81
WF_c8 = 4.0

def run_c8(fn):
    try:
        fn(TAU_c8, E_c8, W0_c8, N0_c8, IPOT_c8, LAM_c8, ZMAX_c8, NZ_c8, NPHI_c8, NT_c8, WF_c8)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c8(compute_grating_length)',
            "gold_call": 'run_c8(_oracle_compute_grating_length)',
        },
        {
            "setup": """import numpy as np
TAU_c9 = 88.6e-15
E_c9 = 1.20e-3
W0_c9 = 40.0e-6
N0_c9 = 0.0
IPOT_c9 = 15.7596
LAM_c9 = 800.0e-9
ZMAX_c9 = 3.0e-5
NZ_c9 = 31
NPHI_c9 = 16
NT_c9 = 81
WF_c9 = 4.0

def run_c9(fn):
    try:
        fn(TAU_c9, E_c9, W0_c9, N0_c9, IPOT_c9, LAM_c9, ZMAX_c9, NZ_c9, NPHI_c9, NT_c9, WF_c9)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c9(compute_grating_length)',
            "gold_call": 'run_c9(_oracle_compute_grating_length)',
        },
        {
            "setup": """import numpy as np
TAU_c10 = 88.6e-15
E_c10 = 1.20e-3
W0_c10 = 40.0e-6
N0_c10 = 2.45e25
IPOT_c10 = 15.7596
LAM_c10 = 800.0e-9
ZMAX_c10 = 3.0e-5
NZ_c10 = 1
NPHI_c10 = 16
NT_c10 = 81
WF_c10 = 4.0

def run_c10(fn):
    try:
        fn(TAU_c10, E_c10, W0_c10, N0_c10, IPOT_c10, LAM_c10, ZMAX_c10, NZ_c10, NPHI_c10, NT_c10, WF_c10)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c10(compute_grating_length)',
            "gold_call": 'run_c10(_oracle_compute_grating_length)',
        },
    ]
