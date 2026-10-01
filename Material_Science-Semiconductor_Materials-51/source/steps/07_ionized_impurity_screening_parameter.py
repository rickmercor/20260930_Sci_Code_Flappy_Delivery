"""
Implement ionized_impurity_screening_parameter, which computes the dimensionless Brooks-Herring screening parameter characterizing the strength of Coulomb screening by free carriers around ionized impurities.

The Brooks-Herring model for ionized-impurity scattering characterizes how effectively free carriers screen the Coulomb potential of charged impurity centers via a dimensionless parameter that grows with temperature squared and shrinks with impurity concentration -- stronger screening (larger b) at higher temperature and lower doping reduces the effective scattering cross-section.

Returns
-------
float, the screening parameter b
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ionized_impurity_screening_parameter(C_b: float, T: float, N_I: float) -> float:
    '''Compute the Brooks-Herring screening parameter b.

    Returns
    -------
    b : float
        The dimensionless screening parameter.

    Raises
    ------
    ValueError
        If any input is not finite, or if C_b, T, or N_I is not positive.
    '''
    return b  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_ionized_impurity_screening_parameter(C_b: float, T: float, N_I: float) -> float:
    for name, val in [("C_b", C_b), ("T", T), ("N_I", N_I)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if C_b <= 0 or T <= 0 or N_I <= 0:
        raise ValueError("C_b, T, N_I must be positive")
    return float(C_b * T**2 / N_I)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "C_b = 4.2e14\nT = 370.0\nN_I = 3.1e18", "call": "ionized_impurity_screening_parameter(C_b, T, N_I)", "gold_call": "_oracle_ionized_impurity_screening_parameter(C_b, T, N_I)"},
        {"setup": "C_b = 1.0e14\nT = 300.0\nN_I = 1.0e18", "call": "ionized_impurity_screening_parameter(C_b, T, N_I)", "gold_call": "_oracle_ionized_impurity_screening_parameter(C_b, T, N_I)"},
        {"setup": "C_b = 1.0e15\nT = 500.0\nN_I = 5.0e17", "call": "ionized_impurity_screening_parameter(C_b, T, N_I)", "gold_call": "_oracle_ionized_impurity_screening_parameter(C_b, T, N_I)"},
        {
            "setup": (
                "C_b = -4.2e14\nT = 370.0\nN_I = 3.1e18\n"
                "def run_model():\n    try:\n        ionized_impurity_screening_parameter(C_b, T, N_I)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
                "def run_gold():\n    try:\n        _oracle_ionized_impurity_screening_parameter(C_b, T, N_I)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2"
            ),
            "call": "run_model()", "gold_call": "run_gold()",
        },
    ]
