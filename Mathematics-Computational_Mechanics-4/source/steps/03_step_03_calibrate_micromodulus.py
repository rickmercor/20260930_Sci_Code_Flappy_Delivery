"""
Calibrate the bond micro-modulus constant against the classical plane-strain strain energy density.

The bond micro-potential is quadratic in the bond stretch, pi_ij = 0.5 * c * omega(||X_ij||) * s_ij ** 2 * ||X_ij||, where omega is the influence function and c is a constant that depends on the horizon. The constant is fixed by requiring that, for a uniform horizon and in the bulk of the body, the total internal energy under a homogeneous isotropic extension equal the classical plane-strain value for a material whose Poisson's ratio takes the value imposed by the bond-based formulation. That total is the double sum over every point and every member of that point's family, in which each bond therefore appears twice, and the calibration accounts for both appearances.

The influence function is the power law omega(r) = r ** (-influence_exponent), so the calibration reduces to a single weighted moment of the influence function over the neighbourhood, evaluated in polar coordinates. The moment converges only for influence_exponent < 3.

The function is evaluated for an array of horizons and returns an array of the same shape, giving the constant for each. The returned value is the constant appearing in the micro-potential itself and is used unchanged wherever that micro-potential or its derivative is required.

Raises ValueError if: horizons is empty; horizons contains non-finite values; any horizon is not strictly positive; youngs_modulus is not finite and strictly positive; influence_exponent is not finite; influence_exponent is greater than or equal to 3.

A peridynamic constitutive model is useful only if it reproduces classical elasticity in the limit of small, smooth deformation. The bond micro-modulus is not a free parameter but the quantity that enforces this correspondence. The standard calibration subjects the body to a homogeneous isotropic extension, for which every bond in the neighbourhood carries the same stretch, computes the resulting nonlocal strain energy density at a point far from the boundary, and equates it to the strain energy density that classical linear elasticity predicts for the same deformation.



In two dimensions this comparison is carried out in polar coordinates, and because the micro-potential contains one factor of bond length while the area element contributes another, the neighbourhood integral collapses to a single weighted second moment of the influence function. The calibrated constant is therefore inversely proportional to that moment, and consequently depends on the horizon: refining the discretisation shrinks the horizon, shrinks the moment, and raises the micro-modulus, in such a way that the macroscopic stiffness is preserved. When the influence function is a power law the moment has a closed form, and it diverges once the singularity at the origin becomes too strong for the two-dimensional measure to integrate.

How the internal energy is written determines what the constant means. A formulation that sums over unordered pairs counts each bond once; one that sums over every point and every member of its family counts each bond twice, and the constant calibrated against the second is half the constant calibrated against the first. Both appear in the literature and the two are not interchangeable, so the energy expression the calibration targets must be stated rather than assumed.

Bond-based peridynamics constrains Poisson's ratio, because a central-force pair interaction cannot represent independent bulk and shear response. The admissible value differs between plane strain and plane stress, so the two states carry different calibration constants and are not interchangeable. Influence functions that weight short bonds more heavily than long ones are used to concentrate the interaction near the point and improve the recovery of local behaviour; the classical prototype microelastic brittle model is the special case in which the influence function is constant.

Returns
-------
np.ndarray, the calibrated plane-strain micro-potential constant for each input horizon, same shape as horizons
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def calibrate_micromodulus(youngs_modulus: float, horizons: np.ndarray,
                           influence_exponent: float) -> np.ndarray:
    '''Calibrate the plane-strain bond micro-modulus for each given horizon.

    Parameters
    ----------
    youngs_modulus : float
        Young's modulus E of the material.
    horizons : np.ndarray
        Array of horizon values, of any shape.
    influence_exponent : float
        Exponent a in the influence function omega(r) = r ** (-a). Must be < 3.

    Returns
    -------
    micromodulus : np.ndarray
        Array of the same shape as horizons, giving the constant appearing in
        the micro-potential for each horizon.
    '''
    micromodulus = np.zeros_like(np.asarray(horizons, dtype=float))
    return micromodulus  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_calibrate_micromodulus(youngs_modulus: float, horizons: np.ndarray,
                                   influence_exponent: float) -> np.ndarray:
    """Reference implementation."""
    d = np.asarray(horizons, dtype=float)
    E = float(youngs_modulus)
    a = float(influence_exponent)
    if d.size < 1:
        raise ValueError("horizons must contain at least one entry")
    if not np.all(np.isfinite(d)):
        raise ValueError("horizons must be finite")
    if np.any(d <= 0.0):
        raise ValueError("horizons must be > 0")
    if not (np.isfinite(E) and E > 0.0):
        raise ValueError("youngs_modulus must be finite and > 0")
    if not np.isfinite(a):
        raise ValueError("influence_exponent must be finite")
    if a >= 3.0:
        raise ValueError("influence_exponent must be < 3 for the weighted moment to converge")

    moment = d ** (3.0 - a) / (3.0 - a)
    return 8.0 * E / (5.0 * np.pi * moment)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the two production horizons with the singular influence function ---
        {
            "setup": """import numpy as np
youngs_modulus = 72e9
horizons = np.array([0.00603, 0.003015], dtype=float)
influence_exponent = 2.0
""",
            "call": "calibrate_micromodulus(youngs_modulus, horizons, influence_exponent)",
            "gold_call": "_oracle_calibrate_micromodulus(youngs_modulus, horizons, influence_exponent)",
        },
        # --- Boundary: constant influence function, the classical prototype model ---
        {
            "setup": """import numpy as np
youngs_modulus = 72e9
horizons = np.array([0.00603], dtype=float)
influence_exponent = 0.0
""",
            "call": "calibrate_micromodulus(youngs_modulus, horizons, influence_exponent)",
            "gold_call": "_oracle_calibrate_micromodulus(youngs_modulus, horizons, influence_exponent)",
        },
        # --- Edge: reciprocal influence function, horizons spanning two orders ---
        {
            "setup": """import numpy as np
youngs_modulus = 190e9
horizons = np.array([0.003015, 0.001, 0.05], dtype=float)
influence_exponent = 1.0
""",
            "call": "calibrate_micromodulus(youngs_modulus, horizons, influence_exponent)",
            "gold_call": "_oracle_calibrate_micromodulus(youngs_modulus, horizons, influence_exponent)",
        },
        # --- Edge: non-integer exponent close to the divergence threshold ---
        {
            "setup": """import numpy as np
youngs_modulus = 1.0
horizons = np.array([2.0, 0.5], dtype=float)
influence_exponent = 2.5
""",
            "call": "calibrate_micromodulus(youngs_modulus, horizons, influence_exponent)",
            "gold_call": "_oracle_calibrate_micromodulus(youngs_modulus, horizons, influence_exponent)",
        },
        # --- Invalid: exponent at the divergence threshold ---
        {
            "setup": """import numpy as np
horizons = np.array([0.00603], dtype=float)
def run_model():
    try:
        calibrate_micromodulus(72e9, horizons, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_calibrate_micromodulus(72e9, horizons, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive horizon ---
        {
            "setup": """import numpy as np
horizons = np.array([0.00603, 0.0], dtype=float)
def run_model():
    try:
        calibrate_micromodulus(72e9, horizons, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_calibrate_micromodulus(72e9, horizons, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive Young's modulus ---
        {
            "setup": """import numpy as np
horizons = np.array([0.00603], dtype=float)
def run_model():
    try:
        calibrate_micromodulus(-72e9, horizons, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_calibrate_micromodulus(-72e9, horizons, 2.0)
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
