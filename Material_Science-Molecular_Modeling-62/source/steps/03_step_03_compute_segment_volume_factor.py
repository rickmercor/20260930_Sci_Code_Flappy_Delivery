"""
Compute the dimensionless factor that turns the squared nucleus radius and the cylinder depth into the volume of the truncated cylindrical nucleus.

A bubble nucleating in a liquid film only a few nanometres thick and periodic across the flow-normal direction is not a spherical cap. The tractable idealisation used for such films is a cylinder of fixed depth whose cross-section is the circular segment left when a circle of radius r is cut by the substrate plane. The cut is placed so that the half width of the segment at the wall is r sin(theta) and the apex sits at height r(1 + cos(theta)), which is the same as saying that the centre of the circle lies a distance r cos(theta) above the wall. With that placement the liquid-side contact angle at the three-phase line is exactly theta, so a small theta describes a nucleus that bulges well above its own centre, the shape a bubble takes on a strongly wetted surface, while a theta above ninety degrees describes a shallow lens hugging the wall.




The volume of the nucleus is the cylinder depth multiplied by the area of that segment, and the area follows from integrating the chord width over the wall-normal coordinate from the wall to the apex. Carrying out the integral in the shifted variable measured from the circle centre gives an area of r squared multiplied by pi minus theta plus one half the sine of twice theta, so the whole shape dependence of the volume collapses into a single dimensionless factor. Two limits check it: at theta equal to ninety degrees the factor is pi, the area of a half disc, and at theta approaching pi the factor tends to zero because the segment is being squeezed out of existence. This factor multiplies the term that prices the latent heat of the vaporised mass, and it is the only place in the model where the size of the nucleus enters as a plain volume rather than as a volume weighted by the temperature it sits in.

Returns
-------
float: the dimensionless cross-sectional area of the circular segment divided by the squared nucleus radius, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_segment_volume_factor(theta: float) -> float:
    """Compute the dimensionless volume factor of the truncated cylindrical nucleus.

    Parameters
    ----------
    theta : float
        Liquid-side contact angle in radians, strictly between 0 and pi.

    Returns
    -------
    volume_factor : float
        Dimensionless factor such that the nucleus volume equals the cylinder
        depth multiplied by the squared radius multiplied by this factor.
    """
    return volume_factor  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_segment_volume_factor(theta: float) -> float:
    if not (isinstance(theta, (int, float, np.floating)) and not isinstance(theta, bool)
            and np.isfinite(theta)):
        raise ValueError("theta must be a finite number")
    theta = float(theta)
    if not (0.0 < theta < np.pi):
        raise ValueError("theta must lie strictly between 0 and pi radians")

    # Area of the circular segment above the wall, divided by the squared radius.
    return float(np.pi - theta + 0.5 * np.sin(2.0 * theta))

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
""",
            "call": "compute_segment_volume_factor(theta)",
            "gold_call": "_oracle_compute_segment_volume_factor(theta)",
        },
        # --- Valid: hydrophobic equilibrium contact angle ---
        {
            "setup": """import numpy as np
theta = np.deg2rad(118.6)
""",
            "call": "compute_segment_volume_factor(theta)",
            "gold_call": "_oracle_compute_segment_volume_factor(theta)",
        },
        # --- Boundary: right angle, where the segment is exactly a half disc ---
        {
            "setup": """import numpy as np
theta = 0.5 * np.pi
""",
            "call": "compute_segment_volume_factor(theta)",
            "gold_call": "_oracle_compute_segment_volume_factor(theta)",
        },
        # --- Edge: nearly complete dewetting, where the segment almost vanishes ---
        {
            "setup": """import numpy as np
theta = np.pi - 1.0e-6
""",
            "call": "compute_segment_volume_factor(theta)",
            "gold_call": "_oracle_compute_segment_volume_factor(theta)",
        },
        # --- Invalid: angle at the closed end of the admissible interval ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_segment_volume_factor(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_segment_volume_factor(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: angle supplied in degrees rather than radians ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_segment_volume_factor(62.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_segment_volume_factor(62.5)
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
