"""
Compute the amplitude of the first spatial harmonic of the written electron density at each supplied axial position.

The density left behind by the pumps is periodic across the fringes of the ionizing pattern, and it is the periodic part of that distribution, rather than the mean level, that the Bragg readout responds to. This step reduces the fringe-resolved density at each axial position to a single amplitude.



The returned amplitude is normalized so that it is the coefficient multiplying the cosine of the fringe phase in the expansion of the fringe-resolved density, evaluated on the same uniform phase grid used to sample the fringe. The returned array follows the order of the supplied axial positions, which are in metres, and holds amplitudes in inverse cubic metres. The duration is in seconds, the peak intensity of each individual pump in watts per square centimetre, the neutral density in inverse cubic metres, the ionization potential in electronvolts and the pump wavelength in metres.

Returns
-------
Return harmonic_amplitude_m3, a NumPy array of dtype float64 and shape (axial_positions_m.size,) holding the first-harmonic electron-density amplitude in inverse cubic metres, one entry per supplied axial position and in the supplied order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_first_harmonic(axial_positions_m: "np.ndarray", duration_fwhm_s: float, peak_intensity_wcm2: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, n_fringe_phase: int, n_time: int, time_window_factor: float) -> "np.ndarray":
    '''Return the first-harmonic density amplitude at each axial position.

    Parameters
    ----------
    axial_positions_m : np.ndarray
        One-dimensional, non-empty array of axial coordinates in metres.
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
    harmonic_amplitude_m3 : np.ndarray
        Array of shape (axial_positions_m.size,) holding the first-harmonic density
        amplitude in inverse cubic metres.

    Raises
    ------
    ValueError
        If the array of axial positions is empty or not one-dimensional, or if any
        quantity forwarded to the fringe-resolved density evaluation is invalid.
    '''
    return harmonic_amplitude_m3

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_first_harmonic(axial_positions_m: "np.ndarray", duration_fwhm_s: float, peak_intensity_wcm2: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, n_fringe_phase: int, n_time: int, time_window_factor: float) -> "np.ndarray":
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
        density = _oracle_compute_electron_density(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
ZS_c1 = np.linspace(0.0, 1.2e-5, 7)
TAU_c1 = 88.6e-15
IP0_c1 = 4.4915e14
N0_c1 = 2.45e25
IPOT_c1 = 15.7596
LAM_c1 = 800.0e-9
NPHI_c1 = 16
NT_c1 = 101
WF_c1 = 4.0
""",
            "call": 'compute_first_harmonic(ZS_c1.copy(), TAU_c1, IP0_c1, N0_c1, IPOT_c1, LAM_c1, NPHI_c1, NT_c1, WF_c1)',
            "gold_call": '_oracle_compute_first_harmonic(ZS_c1.copy(), TAU_c1, IP0_c1, N0_c1, IPOT_c1, LAM_c1, NPHI_c1, NT_c1, WF_c1)',
        },
        {
            "setup": """import numpy as np
ZS_c2 = np.linspace(-9.0e-6, 9.0e-6, 7)
TAU_c2 = 88.6e-15
IP0_c2 = 4.4915e14
N0_c2 = 2.45e25
IPOT_c2 = 15.7596
LAM_c2 = 800.0e-9
NPHI_c2 = 16
NT_c2 = 101
WF_c2 = 4.0
""",
            "call": 'compute_first_harmonic(ZS_c2.copy(), TAU_c2, IP0_c2, N0_c2, IPOT_c2, LAM_c2, NPHI_c2, NT_c2, WF_c2)',
            "gold_call": '_oracle_compute_first_harmonic(ZS_c2.copy(), TAU_c2, IP0_c2, N0_c2, IPOT_c2, LAM_c2, NPHI_c2, NT_c2, WF_c2)',
        },
        {
            "setup": """import numpy as np
ZS_c3 = np.array([0.0])
TAU_c3 = 88.6e-15
IP0_c3 = 4.4915e14
N0_c3 = 2.45e25
IPOT_c3 = 15.7596
LAM_c3 = 800.0e-9
NPHI_c3 = 16
NT_c3 = 101
WF_c3 = 4.0
""",
            "call": 'compute_first_harmonic(ZS_c3.copy(), TAU_c3, IP0_c3, N0_c3, IPOT_c3, LAM_c3, NPHI_c3, NT_c3, WF_c3)',
            "gold_call": '_oracle_compute_first_harmonic(ZS_c3.copy(), TAU_c3, IP0_c3, N0_c3, IPOT_c3, LAM_c3, NPHI_c3, NT_c3, WF_c3)',
        },
        {
            "setup": """import numpy as np
ZS_c4 = np.array([2.0e-5, 2.5e-5, 3.0e-5])
TAU_c4 = 88.6e-15
IP0_c4 = 4.4915e14
N0_c4 = 2.45e25
IPOT_c4 = 15.7596
LAM_c4 = 800.0e-9
NPHI_c4 = 16
NT_c4 = 101
WF_c4 = 4.0
""",
            "call": 'compute_first_harmonic(ZS_c4.copy(), TAU_c4, IP0_c4, N0_c4, IPOT_c4, LAM_c4, NPHI_c4, NT_c4, WF_c4)',
            "gold_call": '_oracle_compute_first_harmonic(ZS_c4.copy(), TAU_c4, IP0_c4, N0_c4, IPOT_c4, LAM_c4, NPHI_c4, NT_c4, WF_c4)',
        },
        {
            "setup": """import numpy as np
ZS_c5 = np.linspace(0.0, 8.0e-6, 5)
TAU_c5 = 88.6e-15
IP0_c5 = 4.4915e14
N0_c5 = 0.0
IPOT_c5 = 15.7596
LAM_c5 = 800.0e-9
NPHI_c5 = 16
NT_c5 = 101
WF_c5 = 4.0
""",
            "call": 'compute_first_harmonic(ZS_c5.copy(), TAU_c5, IP0_c5, N0_c5, IPOT_c5, LAM_c5, NPHI_c5, NT_c5, WF_c5)',
            "gold_call": '_oracle_compute_first_harmonic(ZS_c5.copy(), TAU_c5, IP0_c5, N0_c5, IPOT_c5, LAM_c5, NPHI_c5, NT_c5, WF_c5)',
        },
        {
            "setup": """import numpy as np
ZS_c6 = np.linspace(0.0, 4.0e-6, 5)
TAU_c6 = 25.0e-15
IP0_c6 = 1.5915e15
N0_c6 = 2.45e25
IPOT_c6 = 15.7596
LAM_c6 = 800.0e-9
NPHI_c6 = 20
NT_c6 = 121
WF_c6 = 4.0
""",
            "call": 'compute_first_harmonic(ZS_c6.copy(), TAU_c6, IP0_c6, N0_c6, IPOT_c6, LAM_c6, NPHI_c6, NT_c6, WF_c6)',
            "gold_call": '_oracle_compute_first_harmonic(ZS_c6.copy(), TAU_c6, IP0_c6, N0_c6, IPOT_c6, LAM_c6, NPHI_c6, NT_c6, WF_c6)',
        },
        {
            "setup": """import numpy as np
ZS_c7 = np.linspace(0.0, 6.0e-6, 5)
TAU_c7 = 60.0e-15
IP0_c7 = 3.0e14
N0_c7 = 2.45e25
IPOT_c7 = 24.5874
LAM_c7 = 800.0e-9
NPHI_c7 = 12
NT_c7 = 81
WF_c7 = 5.0
""",
            "call": 'compute_first_harmonic(ZS_c7.copy(), TAU_c7, IP0_c7, N0_c7, IPOT_c7, LAM_c7, NPHI_c7, NT_c7, WF_c7)',
            "gold_call": '_oracle_compute_first_harmonic(ZS_c7.copy(), TAU_c7, IP0_c7, N0_c7, IPOT_c7, LAM_c7, NPHI_c7, NT_c7, WF_c7)',
        },
        {
            "setup": """import numpy as np
ZS_c8 = np.zeros(0)
TAU_c8 = 88.6e-15
IP0_c8 = 4.4915e14
N0_c8 = 2.45e25
IPOT_c8 = 15.7596
LAM_c8 = 800.0e-9
NPHI_c8 = 16
NT_c8 = 101
WF_c8 = 4.0

def run_c8(fn):
    try:
        fn(ZS_c8.copy(), TAU_c8, IP0_c8, N0_c8, IPOT_c8, LAM_c8, NPHI_c8, NT_c8, WF_c8)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c8(compute_first_harmonic)',
            "gold_call": 'run_c8(_oracle_compute_first_harmonic)',
        },
        {
            "setup": """import numpy as np
ZS_c9 = np.zeros((2, 2))
TAU_c9 = 88.6e-15
IP0_c9 = 4.4915e14
N0_c9 = 2.45e25
IPOT_c9 = 15.7596
LAM_c9 = 800.0e-9
NPHI_c9 = 16
NT_c9 = 101
WF_c9 = 4.0

def run_c9(fn):
    try:
        fn(ZS_c9.copy(), TAU_c9, IP0_c9, N0_c9, IPOT_c9, LAM_c9, NPHI_c9, NT_c9, WF_c9)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c9(compute_first_harmonic)',
            "gold_call": 'run_c9(_oracle_compute_first_harmonic)',
        },
        {
            "setup": """import numpy as np
ZS_c10 = np.array([0.0])
TAU_c10 = 88.6e-15
IP0_c10 = 4.4915e14
N0_c10 = 2.45e25
IPOT_c10 = 15.7596
LAM_c10 = 800.0e-9
NPHI_c10 = 16
NT_c10 = 1
WF_c10 = 4.0

def run_c10(fn):
    try:
        fn(ZS_c10.copy(), TAU_c10, IP0_c10, N0_c10, IPOT_c10, LAM_c10, NPHI_c10, NT_c10, WF_c10)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c10(compute_first_harmonic)',
            "gold_call": 'run_c10(_oracle_compute_first_harmonic)',
        },
    ]
