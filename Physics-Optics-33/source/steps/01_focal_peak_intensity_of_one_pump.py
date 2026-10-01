"""
Compute the on-axis peak intensity of one focused pump pulse from its energy, focal spot size and temporal width.

Both pump beams in this measurement are identical and are described by a Gaussian transverse profile and a Gaussian temporal envelope. The focal spot is specified by its 1/e^2 intensity radius and the temporal envelope by its full width at half maximum. The pump energy and the focusing geometry are held fixed while the temporal width is varied, so this quantity changes whenever the duration changes. The returned intensity is expressed in watts per square centimetre, the unit in which the medium's ionization response is evaluated, while the energy, radius and duration are supplied in joules, metres and seconds.

Returns
-------
Return peak_intensity_wcm2, a single float: the on-axis peak intensity of one pump in watts per square centimetre.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_peak_intensity(pulse_energy_j: float, waist_radius_m: float, duration_fwhm_s: float) -> float:
    '''Return the on-axis peak intensity of a focused Gaussian pump pulse.

    Parameters
    ----------
    pulse_energy_j : float
        Energy carried by the pulse, in joules. Must be positive.
    waist_radius_m : float
        1/e^2 intensity radius of the focal spot, in metres. Must be positive.
    duration_fwhm_s : float
        Full width at half maximum of the Gaussian temporal intensity envelope,
        in seconds. Must be positive.

    Returns
    -------
    peak_intensity_wcm2 : float
        On-axis peak intensity, in watts per square centimetre.

    Raises
    ------
    ValueError
        If any of the energy, the radius or the duration is not positive.
    '''
    return peak_intensity_wcm2

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_peak_intensity(pulse_energy_j: float, waist_radius_m: float, duration_fwhm_s: float) -> float:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
E_c1 = 0.0012
W_c1 = 4e-05
T_c1 = 8.86e-14
""",
            "call": 'compute_peak_intensity(E_c1, W_c1, T_c1)',
            "gold_call": '_oracle_compute_peak_intensity(E_c1, W_c1, T_c1)',
        },
        {
            "setup": """import numpy as np
E_c2 = 0.0012
W_c2 = 4e-05
T_c2 = 2.5e-14
""",
            "call": 'compute_peak_intensity(E_c2, W_c2, T_c2)',
            "gold_call": '_oracle_compute_peak_intensity(E_c2, W_c2, T_c2)',
        },
        {
            "setup": """import numpy as np
E_c3 = 0.0012
W_c3 = 4e-05
T_c3 = 1.15e-13
""",
            "call": 'compute_peak_intensity(E_c3, W_c3, T_c3)',
            "gold_call": '_oracle_compute_peak_intensity(E_c3, W_c3, T_c3)',
        },
        {
            "setup": """import numpy as np
E_c4 = 2e-05
W_c4 = 4e-05
T_c4 = 6e-14
""",
            "call": 'compute_peak_intensity(E_c4, W_c4, T_c4)',
            "gold_call": '_oracle_compute_peak_intensity(E_c4, W_c4, T_c4)',
        },
        {
            "setup": """import numpy as np
E_c5 = 0.0012
W_c5 = 5e-06
T_c5 = 3e-14
""",
            "call": 'compute_peak_intensity(E_c5, W_c5, T_c5)',
            "gold_call": '_oracle_compute_peak_intensity(E_c5, W_c5, T_c5)',
        },
        {
            "setup": """import numpy as np
E_c6 = 0.0012
W_c6 = 0.0002
T_c6 = 5e-13
""",
            "call": 'compute_peak_intensity(E_c6, W_c6, T_c6)',
            "gold_call": '_oracle_compute_peak_intensity(E_c6, W_c6, T_c6)',
        },
        {
            "setup": """import numpy as np
E_c7 = 1e-12
W_c7 = 4e-05
T_c7 = 8.86e-14
""",
            "call": 'compute_peak_intensity(E_c7, W_c7, T_c7)',
            "gold_call": '_oracle_compute_peak_intensity(E_c7, W_c7, T_c7)',
        },
        {
            "setup": """import numpy as np
E_c8 = 0.0
W_c8 = 4e-05
T_c8 = 8.86e-14

def run_c8(fn):
    try:
        fn(E_c8, W_c8, T_c8)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c8(compute_peak_intensity)',
            "gold_call": 'run_c8(_oracle_compute_peak_intensity)',
        },
        {
            "setup": """import numpy as np
E_c9 = 0.0012
W_c9 = -4e-05
T_c9 = 8.86e-14

def run_c9(fn):
    try:
        fn(E_c9, W_c9, T_c9)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c9(compute_peak_intensity)',
            "gold_call": 'run_c9(_oracle_compute_peak_intensity)',
        },
        {
            "setup": """import numpy as np
E_c10 = 0.0012
W_c10 = 4e-05
T_c10 = 0.0

def run_c10(fn):
    try:
        fn(E_c10, W_c10, T_c10)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c10(compute_peak_intensity)',
            "gold_call": 'run_c10(_oracle_compute_peak_intensity)',
        },
    ]
