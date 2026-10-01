"""
Compute the free-electron density left behind across one fringe of the pattern at a single axial position, after both pumps have passed.

Ionization is irreversible on the timescale of the pumps and depletes the neutral population, which is uniform at the supplied density before the pulses arrive; on the picosecond delay at which the structure is read out, recombination, diffusion and collisional ionization have not yet acted, so the density computed here is the density the probe sees.



The fringe is sampled at the supplied number of uniformly spaced phases covering one full period and starting at zero, and the returned array holds the density at those phases in that order. The time integration uses the supplied number of uniformly spaced samples on an interval that is symmetric about the midpoint of the two pump envelopes and extends beyond the outermost envelope centre by the supplied window factor times the pulse duration, evaluated with the composite trapezoidal rule. Densities are in inverse cubic metres, the axial position is in metres, the duration is in seconds, the peak intensity of each individual pump is in watts per square centimetre, the ionization potential is in electronvolts and the pump wavelength is in metres.

Returns
-------
Return electron_density_m3, a NumPy array of dtype float64 and shape (n_fringe_phase,) holding the free-electron density in inverse cubic metres at the uniformly spaced fringe phases, in increasing phase order starting at zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_electron_density(axial_position_m: float, duration_fwhm_s: float, peak_intensity_wcm2: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, n_fringe_phase: int, n_time: int, time_window_factor: float) -> "np.ndarray":
    '''Return the fringe-resolved free-electron density at one axial position.

    Parameters
    ----------
    axial_position_m : float
        Axial coordinate at which the density is evaluated, in metres.
    duration_fwhm_s : float
        Full width at half maximum of each pump's Gaussian temporal intensity
        envelope, in seconds. Must be positive.
    peak_intensity_wcm2 : float
        On-axis peak intensity of each individual pump, in watts per square
        centimetre. Must be non-negative.
    neutral_density_m3 : float
        Neutral number density of the gas before the pulses arrive, in inverse cubic
        metres. Must be non-negative.
    ionization_potential_ev : float
        Ionization potential of the gas, in electronvolts. Must be positive.
    pump_wavelength_m : float
        Central wavelength of both pumps, in metres. Must be positive.
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
    electron_density_m3 : np.ndarray
        Array of shape (n_fringe_phase,) holding the free-electron density in
        inverse cubic metres.

    Raises
    ------
    ValueError
        If the neutral density is negative, if the number of fringe phases is below
        one, if the number of time samples is below two, if the window factor is not
        positive, or if any quantity forwarded to the intensity or rate evaluation is
        invalid.
    '''
    return electron_density_m3

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_electron_density(axial_position_m: float, duration_fwhm_s: float, peak_intensity_wcm2: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, n_fringe_phase: int, n_time: int, time_window_factor: float) -> "np.ndarray":
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

    intensity = _oracle_compute_interference_intensity(
        times, phases, axial_position_m, peak_intensity_wcm2, duration_fwhm_s
    )
    rate = _oracle_compute_ionization_rate(
        intensity, ionization_potential_ev, pump_wavelength_m
    )
    widths = np.diff(times)
    exposure = np.sum(0.5 * (rate[:, 1:] + rate[:, :-1]) * widths, axis=1)
    return float(neutral_density_m3) * (1.0 - np.exp(-exposure))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
Z_c1 = 0.0
TAU_c1 = 88.6e-15
IP0_c1 = 4.4915e14
N0_c1 = 2.45e25
IPOT_c1 = 15.7596
LAM_c1 = 800.0e-9
NPHI_c1 = 16
NT_c1 = 101
WF_c1 = 4.0
""",
            "call": 'compute_electron_density(Z_c1, TAU_c1, IP0_c1, N0_c1, IPOT_c1, LAM_c1, NPHI_c1, NT_c1, WF_c1)',
            "gold_call": '_oracle_compute_electron_density(Z_c1, TAU_c1, IP0_c1, N0_c1, IPOT_c1, LAM_c1, NPHI_c1, NT_c1, WF_c1)',
        },
        {
            "setup": """import numpy as np
Z_c2 = 6.8e-6
TAU_c2 = 88.6e-15
IP0_c2 = 4.4915e14
N0_c2 = 2.45e25
IPOT_c2 = 15.7596
LAM_c2 = 800.0e-9
NPHI_c2 = 16
NT_c2 = 101
WF_c2 = 4.0
""",
            "call": 'compute_electron_density(Z_c2, TAU_c2, IP0_c2, N0_c2, IPOT_c2, LAM_c2, NPHI_c2, NT_c2, WF_c2)',
            "gold_call": '_oracle_compute_electron_density(Z_c2, TAU_c2, IP0_c2, N0_c2, IPOT_c2, LAM_c2, NPHI_c2, NT_c2, WF_c2)',
        },
        {
            "setup": """import numpy as np
Z_c3 = 2.5e-5
TAU_c3 = 88.6e-15
IP0_c3 = 4.4915e14
N0_c3 = 2.45e25
IPOT_c3 = 15.7596
LAM_c3 = 800.0e-9
NPHI_c3 = 16
NT_c3 = 101
WF_c3 = 4.0
""",
            "call": 'compute_electron_density(Z_c3, TAU_c3, IP0_c3, N0_c3, IPOT_c3, LAM_c3, NPHI_c3, NT_c3, WF_c3)',
            "gold_call": '_oracle_compute_electron_density(Z_c3, TAU_c3, IP0_c3, N0_c3, IPOT_c3, LAM_c3, NPHI_c3, NT_c3, WF_c3)',
        },
        {
            "setup": """import numpy as np
Z_c4 = -6.8e-6
TAU_c4 = 88.6e-15
IP0_c4 = 4.4915e14
N0_c4 = 2.45e25
IPOT_c4 = 15.7596
LAM_c4 = 800.0e-9
NPHI_c4 = 16
NT_c4 = 101
WF_c4 = 4.0
""",
            "call": 'compute_electron_density(Z_c4, TAU_c4, IP0_c4, N0_c4, IPOT_c4, LAM_c4, NPHI_c4, NT_c4, WF_c4)',
            "gold_call": '_oracle_compute_electron_density(Z_c4, TAU_c4, IP0_c4, N0_c4, IPOT_c4, LAM_c4, NPHI_c4, NT_c4, WF_c4)',
        },
        {
            "setup": """import numpy as np
Z_c5 = 3.0e-6
TAU_c5 = 88.6e-15
IP0_c5 = 4.4915e14
N0_c5 = 0.0
IPOT_c5 = 15.7596
LAM_c5 = 800.0e-9
NPHI_c5 = 16
NT_c5 = 101
WF_c5 = 4.0
""",
            "call": 'compute_electron_density(Z_c5, TAU_c5, IP0_c5, N0_c5, IPOT_c5, LAM_c5, NPHI_c5, NT_c5, WF_c5)',
            "gold_call": '_oracle_compute_electron_density(Z_c5, TAU_c5, IP0_c5, N0_c5, IPOT_c5, LAM_c5, NPHI_c5, NT_c5, WF_c5)',
        },
        {
            "setup": """import numpy as np
Z_c6 = 4.0e-6
TAU_c6 = 60.0e-15
IP0_c6 = 6.6e14
N0_c6 = 2.45e25
IPOT_c6 = 15.7596
LAM_c6 = 800.0e-9
NPHI_c6 = 2
NT_c6 = 81
WF_c6 = 4.0
""",
            "call": 'compute_electron_density(Z_c6, TAU_c6, IP0_c6, N0_c6, IPOT_c6, LAM_c6, NPHI_c6, NT_c6, WF_c6)',
            "gold_call": '_oracle_compute_electron_density(Z_c6, TAU_c6, IP0_c6, N0_c6, IPOT_c6, LAM_c6, NPHI_c6, NT_c6, WF_c6)',
        },
        {
            "setup": """import numpy as np
Z_c7 = 2.0e-6
TAU_c7 = 40.0e-15
IP0_c7 = 2.0e14
N0_c7 = 2.45e25
IPOT_c7 = 24.5874
LAM_c7 = 800.0e-9
NPHI_c7 = 24
NT_c7 = 121
WF_c7 = 5.0
""",
            # Compare ionized fraction in ppm; 1e-9 ppm = 1e-15 of neutral density.
            "tol": 1e-9,
            "call": '1.0e6 * compute_electron_density(Z_c7, TAU_c7, IP0_c7, N0_c7, IPOT_c7, LAM_c7, NPHI_c7, NT_c7, WF_c7) / N0_c7',
            "gold_call": '1.0e6 * _oracle_compute_electron_density(Z_c7, TAU_c7, IP0_c7, N0_c7, IPOT_c7, LAM_c7, NPHI_c7, NT_c7, WF_c7) / N0_c7',
        },
        {
            "setup": """import numpy as np
Z_c8 = 0.0
TAU_c8 = 88.6e-15
IP0_c8 = 4.4915e14
N0_c8 = 2.45e25
IPOT_c8 = 15.7596
LAM_c8 = 800.0e-9
NPHI_c8 = 16
NT_c8 = 1
WF_c8 = 4.0

def run_c8(fn):
    try:
        fn(Z_c8, TAU_c8, IP0_c8, N0_c8, IPOT_c8, LAM_c8, NPHI_c8, NT_c8, WF_c8)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c8(compute_electron_density)',
            "gold_call": 'run_c8(_oracle_compute_electron_density)',
        },
        {
            "setup": """import numpy as np
Z_c9 = 0.0
TAU_c9 = 88.6e-15
IP0_c9 = 4.4915e14
N0_c9 = 2.45e25
IPOT_c9 = 15.7596
LAM_c9 = 800.0e-9
NPHI_c9 = 0
NT_c9 = 101
WF_c9 = 4.0

def run_c9(fn):
    try:
        fn(Z_c9, TAU_c9, IP0_c9, N0_c9, IPOT_c9, LAM_c9, NPHI_c9, NT_c9, WF_c9)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c9(compute_electron_density)',
            "gold_call": 'run_c9(_oracle_compute_electron_density)',
        },
        {
            "setup": """import numpy as np
Z_c10 = 0.0
TAU_c10 = 88.6e-15
IP0_c10 = 4.4915e14
N0_c10 = 2.45e25
IPOT_c10 = 15.7596
LAM_c10 = 800.0e-9
NPHI_c10 = 16
NT_c10 = 101
WF_c10 = 0.0

def run_c10(fn):
    try:
        fn(Z_c10, TAU_c10, IP0_c10, N0_c10, IPOT_c10, LAM_c10, NPHI_c10, NT_c10, WF_c10)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c10(compute_electron_density)',
            "gold_call": 'run_c10(_oracle_compute_electron_density)',
        },
    ]
