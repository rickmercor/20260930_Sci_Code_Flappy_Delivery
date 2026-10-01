"""
Compute the dimensionless first moment of the nucleus volume against the linear wall-normal temperature profile.

A nucleus growing in a wall boundary layer does not sit in a uniform superheat. The temperature falls linearly from its wall value over the height of the nucleus, so the free-energy accounting needs not the plain volume of the nucleus but its volume weighted by the local temperature. Because the profile is linear and anchored at the wall, that weighted volume is fixed by the plain volume minus the first moment of the volume about the wall, divided by the apex height over which the gradient acts. The result is again a pure shape factor multiplying the wall temperature, the cylinder depth and the squared radius, so the shape dependence of the model is carried by exactly two numbers: the volume factor of the previous step and the moment factor computed here.




Evaluating the first moment over the circular segment introduces a term that the plain area integral does not contain, namely two-thirds of the cube of the sine of the contact angle, and dividing by the apex height introduces the factor one plus the cosine of the contact angle in the denominator. Combining them gives a moment factor equal to three times the volume factor, less twice the cube of the sine of the angle, all divided by three times one plus the cosine of the angle. This is where the contact angle enters the model a second time and with a different weighting from the first, and it is the reason the barrier cannot be written as the classical bulk result multiplied by a single geometric shape factor. A shallow nucleus on a poorly wetted surface keeps most of its volume in the hottest liquid near the wall and so has a moment factor close to its volume factor, whereas a tall nucleus reaches into liquid that has already cooled and loses proportionally more of its superheat.

Returns
-------
float: the temperature-weighted volume of the segment divided by the wall temperature, the cylinder depth and the squared radius, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_thermal_volume_moment(theta: float, volume_factor: float) -> float:
    """Compute the dimensionless temperature-weighted volume factor.

    Parameters
    ----------
    theta : float
        Liquid-side contact angle in radians, strictly between 0 and pi.
    volume_factor : float
        Dimensionless volume factor of the same segment, as returned by the
        previous step (> 0).

    Returns
    -------
    moment_factor : float
        Dimensionless factor such that the integral of the temperature over the
        nucleus equals the wall-adjacent liquid temperature multiplied by the
        cylinder depth, the squared radius and this factor.
    """
    return moment_factor  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_thermal_volume_moment(theta: float, volume_factor: float) -> float:
    if not (isinstance(theta, (int, float, np.floating)) and not isinstance(theta, bool)
            and np.isfinite(theta)):
        raise ValueError("theta must be a finite number")
    if not (isinstance(volume_factor, (int, float, np.floating))
            and not isinstance(volume_factor, bool)
            and np.isfinite(volume_factor) and float(volume_factor) > 0.0):
        raise ValueError("volume_factor must be a finite number > 0")
    theta = float(theta)
    if not (0.0 < theta < np.pi):
        raise ValueError("theta must lie strictly between 0 and pi radians")

    apex_factor = 1.0 + np.cos(theta)
    if apex_factor <= 0.0:
        raise ValueError("the apex height factor must be positive")

    # Plain volume less the first moment about the wall, divided by the apex height.
    numerator = 3.0 * float(volume_factor) - 2.0 * np.sin(theta) ** 3
    return float(numerator / (3.0 * apex_factor))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: hydrophilic equilibrium contact angle (normal scenario) ---
        {
            "setup": """import numpy as np
theta = np.deg2rad(62.5)
volume_factor = np.pi - theta + 0.5 * np.sin(2.0 * theta)
""",
            "call": "compute_thermal_volume_moment(theta, volume_factor)",
            "gold_call": "_oracle_compute_thermal_volume_moment(theta, volume_factor)",
        },
        # --- Valid: hydrophobic equilibrium contact angle ---
        {
            "setup": """import numpy as np
theta = np.deg2rad(118.6)
volume_factor = np.pi - theta + 0.5 * np.sin(2.0 * theta)
""",
            "call": "compute_thermal_volume_moment(theta, volume_factor)",
            "gold_call": "_oracle_compute_thermal_volume_moment(theta, volume_factor)",
        },
        # --- Boundary: right angle, where the apex height equals the radius ---
        {
            "setup": """import numpy as np
theta = 0.5 * np.pi
volume_factor = np.pi
""",
            "call": "compute_thermal_volume_moment(theta, volume_factor)",
            "gold_call": "_oracle_compute_thermal_volume_moment(theta, volume_factor)",
        },
        # --- Edge: strongly dewetted nucleus flattened against the wall ---
        {
            "setup": """import numpy as np
theta = 3.0
volume_factor = np.pi - theta + 0.5 * np.sin(2.0 * theta)
""",
            "call": "compute_thermal_volume_moment(theta, volume_factor)",
            "gold_call": "_oracle_compute_thermal_volume_moment(theta, volume_factor)",
        },
        # --- Invalid: contact angle outside the admissible interval ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_thermal_volume_moment(np.pi, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_thermal_volume_moment(np.pi, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive volume factor ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_thermal_volume_moment(1.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_thermal_volume_moment(1.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
