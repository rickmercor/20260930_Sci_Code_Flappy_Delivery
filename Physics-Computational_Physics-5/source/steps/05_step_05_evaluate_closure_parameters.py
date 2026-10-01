"""
Convert the coefficients of a three-pole Pade approximant of the kinetic response into the three real coefficients of the corresponding heat-flux closure.

Combining the linearised moment equations with a rational approximant of the kinetic response leaves the heat flux as a fixed combination of the velocity, the pressure measured against the electrostatic potential, and the temperature. The three coefficients of that combination are algebraic functions of the approximant alone, and they are real whenever the approximant respects the parity symmetry of a Maxwellian.

Returns
-------
np.ndarray of shape (3,), float: the velocity, potential-corrected pressure and temperature coefficients of the heat-flux closure.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_closure_parameters(pade_coefficients: np.ndarray) -> np.ndarray:
    """Convert Pade coefficients into heat-flux closure parameters.

    Parameters
    ----------
    pade_coefficients : np.ndarray
        Complex array of shape (3,) holding the numerator coefficient, the
        linear denominator coefficient and the quadratic denominator
        coefficient of the approximant, in the convention of sub-problem 03.

    Returns
    -------
    closure_parameters : np.ndarray
        Array of shape (3,) of native floats holding, in order, the
        coefficient multiplying the normalised velocity, the coefficient
        multiplying the normalised pressure measured against the normalised
        electrostatic potential, and the coefficient multiplying the normalised
        temperature, in the heat-flux closure.

    Raises
    ------
    ValueError
        If ``pade_coefficients`` is not a finite array of shape (3,), if its
        numerator coefficient vanishes, or if any resulting closure parameter
        has an imaginary part exceeding 1e-6 times the larger of one and its
        own magnitude.
    """
    return closure_parameters  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_closure_parameters(pade_coefficients: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coefficients = np.asarray(pade_coefficients, dtype=complex)
    if coefficients.shape != (3,):
        raise ValueError("pade_coefficients must be an array of shape (3,)")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("pade_coefficients must contain only finite entries")

    numerator, linear, quadratic = (complex(value) for value in coefficients)
    if numerator == 0.0:
        raise ValueError("the numerator coefficient must not vanish")

    # Eliminating the heat flux between the linearised pressure equation and the
    # approximated dispersion relation leaves these three combinations.
    velocity = (linear - 3.0 * numerator) / numerator
    potential = (1.0 + quadratic / 2.0) / (1j * numerator)
    temperature = 1.0 / (1j * numerator)

    parameters = np.array([velocity, potential, temperature], dtype=complex)
    tolerance = 1.0e-6 * np.maximum(1.0, np.abs(parameters))
    if np.any(np.abs(parameters.imag) > tolerance):
        raise ValueError("the closure parameters must be real to within 1e-6")
    return parameters.real.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the Hammett-Perkins coefficients, for which the velocity and
        #     potential coefficients both collapse to zero (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
root_pi = np.sqrt(np.pi)
pade_coefficients = np.array([-1j * root_pi / 2.0, -3.0j * root_pi / 2.0, -2.0 + 0.0j])
""",
            "call": "sig(evaluate_closure_parameters(pade_coefficients), 1.0)",
            "gold_call": "sig(_oracle_evaluate_closure_parameters(pade_coefficients), 1.0)",
        },
        # --- Valid: a member whose quadratic coefficient is not minus two, so the
        #     potential coefficient survives ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
root_pi = np.sqrt(np.pi)
pade_coefficients = np.array([-1j * root_pi * (np.pi - 3.0) / (4.0 - np.pi),
                              -1j * root_pi / (4.0 - np.pi),
                              -(3.0 * np.pi - 8.0) / (4.0 - np.pi) + 0.0j])
""",
            "call": "sig(evaluate_closure_parameters(pade_coefficients), 1.0)",
            "gold_call": "sig(_oracle_evaluate_closure_parameters(pade_coefficients), 1.0)",
        },
        # --- Valid: a root-matched pair of coefficients from the benchmark wave
        #     number, which are much smaller in magnitude ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
pade_coefficients = np.array([-1.2806320000j, -5.5761380000j, -2.0 + 0.0j])
""",
            "call": "sig(evaluate_closure_parameters(pade_coefficients), 1.0)",
            "gold_call": "sig(_oracle_evaluate_closure_parameters(pade_coefficients), 1.0)",
        },
        # --- Boundary: a very large numerator coefficient, which drives both the
        #     potential and the temperature coefficient toward zero ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
pade_coefficients = np.array([-208.6979220000j, -689.0308880000j, -2.0 + 0.0j])
""",
            "call": "sig(evaluate_closure_parameters(pade_coefficients), 1.0)",
            "gold_call": "sig(_oracle_evaluate_closure_parameters(pade_coefficients), 1.0)",
        },
        # --- Invalid: a vanishing numerator coefficient, which the conversion
        #     divides by ---
        {
            "setup": """import numpy as np
pade_coefficients = np.array([0.0 + 0.0j, -2.0j, -2.0 + 0.0j])
def run_model():
    try:
        evaluate_closure_parameters(pade_coefficients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_closure_parameters(pade_coefficients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: coefficients that break the parity symmetry, so the closure
        #     parameters come out complex and the heat flux would not be real ---
        {
            "setup": """import numpy as np
pade_coefficients = np.array([0.8 - 1.3j, -2.1j, -2.0 + 0.0j])
def run_model():
    try:
        evaluate_closure_parameters(pade_coefficients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_closure_parameters(pade_coefficients)
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
