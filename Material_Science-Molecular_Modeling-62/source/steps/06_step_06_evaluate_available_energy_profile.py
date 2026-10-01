"""
Evaluate the available-energy increment of the nucleus over a grid of nucleus radii.

The shape of the available-energy increment against nucleus radius is what turns a thermodynamic identity into a nucleation criterion. The increment is the squared radius multiplied by a bracket holding the balance coefficient plus the Laplace coefficient multiplied by the logarithm of one plus the capillary length divided by the radius. At small radius, the logarithm diverges, but only as the logarithm of the reciprocal radius, so the squared radius still wins, and the increment starts from zero at the origin. The Laplace term is positive and dominates while the nucleus is small, so the increment climbs: a nucleus that appears by fluctuation is being pushed back towards extinction, because shrinking lowers the available energy. The balance term, which weighs vaporization cost against superheat gain, eventually takes over, and when it is negative the increment turns and falls, so beyond a certain radius growth lowers the available energy and proceeds without further help. The turning point is therefore the whole content of the model: its height is the activation energy that a thermal fluctuation must supply, and its abscissa is the critical radius.


Evaluating the increment on a grid rather than only at its extremum is what makes the qualitative character of the state visible before any extremum is extracted, and it is the only cheap check on a root that has to be found numerically. If the balance coefficient is negative the profile rises, turns once and falls, and the turning point is a genuine maximum. If it is non-negative, the profile is increasing over the whole positive axis, so no barrier exists and the liquid at that state cannot nucleate within this model at any size. Reading the profile also guards against a common misreading in which a stationary point is located by differentiating and then reported without checking that it is a maximum rather than a minimum, which converts a nucleation threshold into its opposite.

Returns
-------
np.ndarray with the same shape as radii, float: the available-energy increment of the nucleus in joules at each radius.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def evaluate_available_energy_profile(coefficients: np.ndarray,
                                      radii: np.ndarray) -> np.ndarray:
    """Evaluate the available-energy increment over a grid of nucleus radii.

    Parameters
    ----------
    coefficients : np.ndarray
        Array of shape (3,) holding the balance coefficient in J/m^2, the
        Laplace coefficient in J/m^2 and the capillary length in metres.
    radii : np.ndarray
        One-dimensional array of strictly positive nucleus radii in metres.

    Returns
    -------
    profile : np.ndarray
        Array with the same shape as radii holding the available-energy
        increment in joules.
    """
    return profile  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_available_energy_profile(coefficients: np.ndarray,
                                              radii: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coefficients, dtype=float).ravel()
    if coeffs.size != 3 or not np.all(np.isfinite(coeffs)):
        raise ValueError("coefficients must hold three finite entries")

    grid = np.asarray(radii, dtype=float)
    if grid.ndim != 1 or grid.size < 1:
        raise ValueError("radii must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError("radii must be finite and strictly positive")

    balance, laplace, capillary = coeffs
    if capillary <= 0.0:
        raise ValueError("the capillary length must be strictly positive")

    return grid ** 2 * (balance + laplace * np.log1p(capillary / grid))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: governing state over a grid spanning the peak ---
        {
            "setup": """import numpy as np
coefficients = np.array([-9.1325677596e-04, 6.9227891610e-04, 1.3362941030e-06])
radii = np.linspace(1.0e-9, 1.0e-6, 11)
""",
            "call": "evaluate_available_energy_profile(coefficients, radii)",
            "gold_call": "_oracle_evaluate_available_energy_profile(coefficients, radii)",
        },
        # --- Valid: logarithmic grid resolving the small-radius rise ---
        {
            "setup": """import numpy as np
coefficients = np.array([-2.6549062633e-04, 1.9787305158e-04, 1.3362941030e-06])
radii = np.logspace(-10, -6, 9)
""",
            "call": "evaluate_available_energy_profile(coefficients, radii)",
            "gold_call": "_oracle_evaluate_available_energy_profile(coefficients, radii)",
        },
        # --- Boundary: single radius, so the profile reduces to one value ---
        {
            "setup": """import numpy as np
coefficients = np.array([-9.1325677596e-04, 6.9227891610e-04, 1.3362941030e-06])
radii = np.array([2.877383718731e-07])
""",
            "call": "evaluate_available_energy_profile(coefficients, radii)",
            "gold_call": "_oracle_evaluate_available_energy_profile(coefficients, radii)",
        },
        # --- Edge: positive balance coefficient, so the profile never turns over ---
        {
            "setup": """import numpy as np
coefficients = np.array([2.5e-04, 1.0e-04, 1.3362941030e-06])
radii = np.linspace(1.0e-9, 5.0e-7, 7)
""",
            "call": "evaluate_available_energy_profile(coefficients, radii)",
            "gold_call": "_oracle_evaluate_available_energy_profile(coefficients, radii)",
        },
        # --- Invalid: a non-positive radius on the grid ---
        {
            "setup": """import numpy as np
coefficients = np.array([-1.0e-03, 1.0e-04, 1.3362941030e-06])
radii = np.array([0.0, 1.0e-7])
def run_model():
    try:
        evaluate_available_energy_profile(coefficients, radii)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_available_energy_profile(coefficients, radii)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: coefficient array of the wrong length ---
        {
            "setup": """import numpy as np
radii = np.array([1.0e-7])
def run_model():
    try:
        evaluate_available_energy_profile(np.array([-1.0e-03, 1.0e-04]), radii)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_available_energy_profile(np.array([-1.0e-03, 1.0e-04]), radii)
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
