"""
Compute the instantaneous intensity seen by the gas at one axial position, as a function of time and of position within the fringe pattern.

The two pumps counter-propagate along a common axis, carry the same central wavelength and the same peak intensity, and are brought into exact temporal coincidence at the axial origin; the axial coordinate supplied here is measured from that origin along the common axis, positive in the direction of the first pump. Position within one fringe of the resulting pattern is parameterized by a phase in radians, zero at a fringe maximum of the interference term, and the supplied time samples have their origin midway between the centres of the two pump envelopes at that axial position. Each pump has a Gaussian temporal intensity envelope of the supplied full width at half maximum.



The returned array has shape (number of fringe phases, number of time samples): the first axis indexes the supplied phases in the order given and the second indexes the supplied time samples in the order given. Intensities are in watts per square centimetre, times in seconds and the axial position in metres.

Returns
-------
Return intensity_wcm2, a NumPy array of dtype float64 and shape (fringe_phases_rad.size, times_s.size). Entry [j, i] is the instantaneous intensity in watts per square centimetre at the j-th supplied fringe phase and the i-th supplied time sample, with both axes preserving the supplied order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_interference_intensity(times_s: "np.ndarray", fringe_phases_rad: "np.ndarray", axial_position_m: float, peak_intensity_wcm2: float, duration_fwhm_s: float) -> "np.ndarray":
    '''Return the instantaneous intensity on a fringe-phase by time grid.

    Parameters
    ----------
    times_s : np.ndarray
        One-dimensional, non-empty array of time samples in seconds.
    fringe_phases_rad : np.ndarray
        One-dimensional, non-empty array of fringe phases in radians.
    axial_position_m : float
        Axial coordinate at which the intensity is evaluated, in metres.
    peak_intensity_wcm2 : float
        On-axis peak intensity of each individual pump, in watts per square
        centimetre. Must be non-negative.
    duration_fwhm_s : float
        Full width at half maximum of each pump's Gaussian temporal intensity
        envelope, in seconds. Must be positive.

    Returns
    -------
    intensity_wcm2 : np.ndarray
        Array of shape (fringe_phases_rad.size, times_s.size) holding the
        instantaneous intensity in watts per square centimetre.

    Raises
    ------
    ValueError
        If either input array is empty or not one-dimensional, if the peak intensity
        is negative, or if the duration is not positive.
    '''
    return intensity_wcm2

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_interference_intensity(times_s: "np.ndarray", fringe_phases_rad: "np.ndarray", axial_position_m: float, peak_intensity_wcm2: float, duration_fwhm_s: float) -> "np.ndarray":
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
T_c1 = np.linspace(-3.0e-13, 3.0e-13, 41)
PH_c1 = 2.0 * np.pi * np.arange(8) / 8.0
Z_c1 = 0.0
IP_c1 = 4.4915e14
TAU_c1 = 88.6e-15
""",
            "call": 'compute_interference_intensity(T_c1.copy(), PH_c1.copy(), Z_c1, IP_c1, TAU_c1)',
            "gold_call": '_oracle_compute_interference_intensity(T_c1.copy(), PH_c1.copy(), Z_c1, IP_c1, TAU_c1)',
        },
        {
            "setup": """import numpy as np
T_c2 = np.linspace(-3.0e-13, 3.0e-13, 41)
PH_c2 = 2.0 * np.pi * np.arange(8) / 8.0
Z_c2 = 6.8e-6
IP_c2 = 4.4915e14
TAU_c2 = 88.6e-15
""",
            "call": 'compute_interference_intensity(T_c2.copy(), PH_c2.copy(), Z_c2, IP_c2, TAU_c2)',
            "gold_call": '_oracle_compute_interference_intensity(T_c2.copy(), PH_c2.copy(), Z_c2, IP_c2, TAU_c2)',
        },
        {
            "setup": """import numpy as np
T_c3 = np.linspace(-3.0e-13, 3.0e-13, 41)
PH_c3 = 2.0 * np.pi * np.arange(8) / 8.0
Z_c3 = 3.0e-5
IP_c3 = 4.4915e14
TAU_c3 = 88.6e-15
""",
            "call": 'compute_interference_intensity(T_c3.copy(), PH_c3.copy(), Z_c3, IP_c3, TAU_c3)',
            "gold_call": '_oracle_compute_interference_intensity(T_c3.copy(), PH_c3.copy(), Z_c3, IP_c3, TAU_c3)',
        },
        {
            "setup": """import numpy as np
T_c4 = np.linspace(-3.0e-13, 3.0e-13, 41)
PH_c4 = 2.0 * np.pi * np.arange(8) / 8.0
Z_c4 = -6.8e-6
IP_c4 = 4.4915e14
TAU_c4 = 88.6e-15
""",
            "call": 'compute_interference_intensity(T_c4.copy(), PH_c4.copy(), Z_c4, IP_c4, TAU_c4)',
            "gold_call": '_oracle_compute_interference_intensity(T_c4.copy(), PH_c4.copy(), Z_c4, IP_c4, TAU_c4)',
        },
        {
            "setup": """import numpy as np
T_c5 = np.linspace(-3.0e-13, 3.0e-13, 41)
PH_c5 = 2.0 * np.pi * np.arange(8) / 8.0
Z_c5 = 6.8e-6
IP_c5 = 0.0
TAU_c5 = 88.6e-15
""",
            "call": 'compute_interference_intensity(T_c5.copy(), PH_c5.copy(), Z_c5, IP_c5, TAU_c5)',
            "gold_call": '_oracle_compute_interference_intensity(T_c5.copy(), PH_c5.copy(), Z_c5, IP_c5, TAU_c5)',
        },
        {
            "setup": """import numpy as np
T_c6 = np.array([0.0])
PH_c6 = np.array([2.0 * np.pi / 3.0])
Z_c6 = 4.0e-6
IP_c6 = 1.0e15
TAU_c6 = 40.0e-15
""",
            "call": 'compute_interference_intensity(T_c6.copy(), PH_c6.copy(), Z_c6, IP_c6, TAU_c6)',
            "gold_call": '_oracle_compute_interference_intensity(T_c6.copy(), PH_c6.copy(), Z_c6, IP_c6, TAU_c6)',
        },
        {
            "setup": """import numpy as np
T_c7 = np.linspace(-3.0e-13, 3.0e-13, 41)
PH_c7 = 2.0 * np.pi * np.arange(8) / 8.0
Z_c7 = 3.0e-5
IP_c7 = 1.0e13
TAU_c7 = 1.0e-12
""",
            "call": 'compute_interference_intensity(T_c7.copy(), PH_c7.copy(), Z_c7, IP_c7, TAU_c7)',
            "gold_call": '_oracle_compute_interference_intensity(T_c7.copy(), PH_c7.copy(), Z_c7, IP_c7, TAU_c7)',
        },
        {
            "setup": """import numpy as np
T_c8 = np.linspace(-3.0e-13, 3.0e-13, 41)
PH_c8 = 2.0 * np.pi * np.arange(8) / 8.0
Z_c8 = 0.0
IP_c8 = -1.0
TAU_c8 = 88.6e-15

def run_c8(fn):
    try:
        fn(T_c8.copy(), PH_c8.copy(), Z_c8, IP_c8, TAU_c8)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c8(compute_interference_intensity)',
            "gold_call": 'run_c8(_oracle_compute_interference_intensity)',
        },
        {
            "setup": """import numpy as np
T_c9 = np.linspace(-3.0e-13, 3.0e-13, 41)
PH_c9 = 2.0 * np.pi * np.arange(8) / 8.0
Z_c9 = 0.0
IP_c9 = 4.4915e14
TAU_c9 = 0.0

def run_c9(fn):
    try:
        fn(T_c9.copy(), PH_c9.copy(), Z_c9, IP_c9, TAU_c9)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c9(compute_interference_intensity)',
            "gold_call": 'run_c9(_oracle_compute_interference_intensity)',
        },
        {
            "setup": """import numpy as np
T_c10 = np.zeros(0)
PH_c10 = np.zeros(4)
Z_c10 = 0.0
IP_c10 = 4.4915e14
TAU_c10 = 88.6e-15

def run_c10(fn):
    try:
        fn(T_c10.copy(), PH_c10.copy(), Z_c10, IP_c10, TAU_c10)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c10(compute_interference_intensity)',
            "gold_call": 'run_c10(_oracle_compute_interference_intensity)',
        },
    ]
