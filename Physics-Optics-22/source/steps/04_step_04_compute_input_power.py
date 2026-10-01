"""
Compute the total radial power of the unit-amplitude entrance field.

The unit-amplitude entrance field retains radial rings.



$$U(r,0)=J_0(k\sin\theta\,r)e^{-r^2/w^2}.$$



Power is integrated over the full transverse plane with the cylindrical area measure $2\pi r\,dr$. The returned value has units of square metres under this unit-amplitude convention and is independent of the thermal lens. At zero cone angle the entrance field is a Gaussian envelope.

Returns
-------
A positive float gives the total entrance power in square metres under unit field amplitude.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_input_power(wave_number: float, waist: float, cone_angle: float) -> float:
    r"""Return the entrance power in the unit-amplitude field convention.

    Parameters
    ----------
    wave_number : float
        Positive finite internal $k$, in $\mathrm{m^{-1}}$.
    waist : float
        Positive finite amplitude-envelope radius $w$, in $\mathrm{m}$.
    cone_angle : float
        Finite internal angle $0\leq\theta<\pi/2$, in radians.

    Returns
    -------
    power : float
        Positive total $2\pi\int_0^\infty |U(r,0)|^2r\,dr$, in
        $\mathrm{m^2}$ under unit field amplitude.

    Raises
    ------
    ValueError
        If the wave number or waist is nonpositive or nonfinite, or the
        angle is nonfinite or outside its stated interval.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import i0e


def _beam_parameters(wave_number: float, waist: float, cone_angle: float) -> tuple:
    wave_number = _finite_scalar(wave_number, "wave_number", True)
    waist = _finite_scalar(waist, "waist", True)
    cone_angle = _finite_scalar(cone_angle, "cone_angle")
    if cone_angle >= np.pi / 2.0:
        raise ValueError("cone_angle must be less than pi/2")
    return wave_number, waist, wave_number * np.sin(cone_angle)


def _oracle_compute_input_power(
    wave_number: float, waist: float, cone_angle: float
) -> float:
    _, waist, transverse = _beam_parameters(wave_number, waist, cone_angle)
    argument = transverse**2 * waist**2 / 4.0
    return float(np.pi * waist**2 * i0e(argument) / 2.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return power cases scaled to resolve the physical normalization."""
    return [
        {
            "setup": """import numpy as np
""",
            "call": "1e9 * compute_input_power(2*np.pi*1.82/1.064e-6, 7e-5, .003)",
            "gold_call": "1e9 * _oracle_compute_input_power(2*np.pi*1.82/1.064e-6, 7e-5, .003)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
""",
            "call": "1e9 * compute_input_power(1e7, 1e-4, 0.0)",
            "gold_call": "1e9 * _oracle_compute_input_power(1e7, 1e-4, 0.0)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
""",
            "call": "1e9 * compute_input_power(1e7, 3.3e-4, .03)",
            "gold_call": "1e9 * _oracle_compute_input_power(1e7, 3.3e-4, .03)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np

def _raises(function):
    try:
        function(1e7, 0.0, .003)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_raises(compute_input_power)",
            "gold_call": "_raises(_oracle_compute_input_power)",
        },
    ]
