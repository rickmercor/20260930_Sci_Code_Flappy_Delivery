"""
Compute the fringe period of the ionizing standing wave, the probe Bragg angle, the angle between the grating axis and the diffracted order, and the object-space grating length implied by the recorded pixel count.

The two pumps are identical in wavelength and counter-propagate along a common axis, so the ionizing pattern they write is a standing wave whose planes lie perpendicular to that axis. The probe is a separate, weak, narrowband beam at its own wavelength, travelling in a gas of the stated refractive index, and it is aligned so that its first diffracted order is the Bragg order of that structure. The camera views the diffracted order and its pixel grid therefore samples a projection of the axial extent of the plasma, magnified by the imaging system; the supplied count is the corrected full width at half maximum of that recorded envelope, in pixels, after the point-spread function has been removed.



The four returned quantities are, in order, the fringe period in metres, the Bragg angle in radians, the angle in radians between the grating axis and the propagation direction of the first diffracted order, and the grating length in metres in the object space of the plasma. Both angles are returned in the first quadrant.

Returns
-------
Return geometry, a NumPy array of shape (4,) and dtype float64 holding, in this order: 1. the fringe period of the ionizing standing wave in metres; 2. the probe Bragg angle in radians, measured between the probe and the grating planes and returned in the first quadrant; 3. the angle in radians between the grating axis and the propagation direction of the first diffracted order, also in the first quadrant; 4. the grating length in metres in the object space of the plasma.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_readout_geometry(pump_wavelength_m: float, probe_wavelength_m: float, medium_index: float, pixel_pitch_m: float, magnification: float, corrected_pixel_count: float) -> "np.ndarray":
    '''Return the readout geometry and the object-space grating length.

    Parameters
    ----------
    pump_wavelength_m : float
        Central wavelength of both counter-propagating pumps, in metres. Must be
        positive.
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

    Returns
    -------
    geometry : np.ndarray
        Array of shape (4,) holding the fringe period in metres, the Bragg angle in
        radians, the angle between the grating axis and the first diffracted order
        in radians, and the grating length in metres.

    Raises
    ------
    ValueError
        If any supplied quantity is not positive, or if the probe wavelength is too
        long for the first Bragg order of the structure to exist.
    '''
    return geometry

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_readout_geometry(pump_wavelength_m: float, probe_wavelength_m: float, medium_index: float, pixel_pitch_m: float, magnification: float, corrected_pixel_count: float) -> "np.ndarray":
    values = {
        "pump_wavelength_m": pump_wavelength_m,
        "probe_wavelength_m": probe_wavelength_m,
        "medium_index": medium_index,
        "pixel_pitch_m": pixel_pitch_m,
        "magnification": magnification,
        "corrected_pixel_count": corrected_pixel_count,
    }
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("%s must be a positive finite number" % name)

    fringe_period_m = 0.5 * float(pump_wavelength_m)
    sin_bragg = float(probe_wavelength_m) / (2.0 * float(medium_index) * fringe_period_m)
    if sin_bragg >= 1.0:
        raise ValueError("probe wavelength is too long for a first Bragg order to exist")

    bragg_angle_rad = float(np.arcsin(sin_bragg))
    axial_angle_rad = 0.5 * np.pi - bragg_angle_rad
    grating_length_m = float(pixel_pitch_m) * float(corrected_pixel_count) / (
        float(magnification) * np.sin(axial_angle_rad)
    )
    return np.array(
        [fringe_period_m, bragg_angle_rad, axial_angle_rad, grating_length_m], dtype=float
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
LP_c1 = 8e-07
LPR_c1 = 4e-07
N_c1 = 1.0
P_c1 = 3.45e-06
M_c1 = 7.0
Q_c1 = 24.0
""",
            "call": 'compute_readout_geometry(LP_c1, LPR_c1, N_c1, P_c1, M_c1, Q_c1)',
            "gold_call": '_oracle_compute_readout_geometry(LP_c1, LPR_c1, N_c1, P_c1, M_c1, Q_c1)',
        },
        {
            "setup": """import numpy as np
LP_c2 = 8e-07
LPR_c2 = 2.66e-07
N_c2 = 1.0
P_c2 = 3.45e-06
M_c2 = 7.0
Q_c2 = 24.0
""",
            "call": 'compute_readout_geometry(LP_c2, LPR_c2, N_c2, P_c2, M_c2, Q_c2)',
            "gold_call": '_oracle_compute_readout_geometry(LP_c2, LPR_c2, N_c2, P_c2, M_c2, Q_c2)',
        },
        {
            "setup": """import numpy as np
LP_c3 = 8e-07
LPR_c3 = 4e-07
N_c3 = 1.00028
P_c3 = 3.45e-06
M_c3 = 7.0
Q_c3 = 24.0
""",
            "call": 'compute_readout_geometry(LP_c3, LPR_c3, N_c3, P_c3, M_c3, Q_c3)',
            "gold_call": '_oracle_compute_readout_geometry(LP_c3, LPR_c3, N_c3, P_c3, M_c3, Q_c3)',
        },
        {
            "setup": """import numpy as np
LP_c4 = 8e-07
LPR_c4 = 4e-07
N_c4 = 1.0
P_c4 = 6.5e-06
M_c4 = 12.0
Q_c4 = 15.0
""",
            "call": 'compute_readout_geometry(LP_c4, LPR_c4, N_c4, P_c4, M_c4, Q_c4)',
            "gold_call": '_oracle_compute_readout_geometry(LP_c4, LPR_c4, N_c4, P_c4, M_c4, Q_c4)',
        },
        {
            "setup": """import numpy as np
LP_c5 = 8e-07
LPR_c5 = 4e-07
N_c5 = 1.0
P_c5 = 3.45e-06
M_c5 = 7.0
Q_c5 = 1.0
""",
            "call": 'compute_readout_geometry(LP_c5, LPR_c5, N_c5, P_c5, M_c5, Q_c5)',
            "gold_call": '_oracle_compute_readout_geometry(LP_c5, LPR_c5, N_c5, P_c5, M_c5, Q_c5)',
        },
        {
            "setup": """import numpy as np
LP_c6 = 1.03e-06
LPR_c6 = 5.15e-07
N_c6 = 1.0
P_c6 = 3.45e-06
M_c6 = 7.0
Q_c6 = 24.0
""",
            "call": 'compute_readout_geometry(LP_c6, LPR_c6, N_c6, P_c6, M_c6, Q_c6)',
            "gold_call": '_oracle_compute_readout_geometry(LP_c6, LPR_c6, N_c6, P_c6, M_c6, Q_c6)',
        },
        {
            "setup": """import numpy as np
LP_c7 = 8e-07
LPR_c7 = 7.95e-07
N_c7 = 1.0
P_c7 = 3.45e-06
M_c7 = 7.0
Q_c7 = 24.0
""",
            "call": 'compute_readout_geometry(LP_c7, LPR_c7, N_c7, P_c7, M_c7, Q_c7)',
            "gold_call": '_oracle_compute_readout_geometry(LP_c7, LPR_c7, N_c7, P_c7, M_c7, Q_c7)',
        },
        {
            "setup": """import numpy as np
LP_c8 = 8e-07
LPR_c8 = 8e-07
N_c8 = 1.0
P_c8 = 3.45e-06
M_c8 = 7.0
Q_c8 = 24.0

def run_c8(fn):
    try:
        fn(LP_c8, LPR_c8, N_c8, P_c8, M_c8, Q_c8)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c8(compute_readout_geometry)',
            "gold_call": 'run_c8(_oracle_compute_readout_geometry)',
        },
        {
            "setup": """import numpy as np
LP_c9 = 8e-07
LPR_c9 = 4e-07
N_c9 = 1.0
P_c9 = 3.45e-06
M_c9 = 7.0
Q_c9 = 0.0

def run_c9(fn):
    try:
        fn(LP_c9, LPR_c9, N_c9, P_c9, M_c9, Q_c9)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c9(compute_readout_geometry)',
            "gold_call": 'run_c9(_oracle_compute_readout_geometry)',
        },
        {
            "setup": """import numpy as np
LP_c10 = 8e-07
LPR_c10 = 4e-07
N_c10 = 1.0
P_c10 = 3.45e-06
M_c10 = -7.0
Q_c10 = 24.0

def run_c10(fn):
    try:
        fn(LP_c10, LPR_c10, N_c10, P_c10, M_c10, Q_c10)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c10(compute_readout_geometry)',
            "gold_call": 'run_c10(_oracle_compute_readout_geometry)',
        },
    ]
