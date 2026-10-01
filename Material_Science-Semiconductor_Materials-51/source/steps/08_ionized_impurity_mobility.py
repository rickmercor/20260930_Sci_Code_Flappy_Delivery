"""
Implement ionized_impurity_mobility, which computes the ionized-impurity-limited carrier mobility using the Brooks-Herring screening function.

The Brooks-Herring mobility increases with temperature (as T^1.5) and decreases with impurity concentration, moderated by the screening function G(b) = ln(1+b) − b/(1+b), which captures how effectively free-carrier screening reduces the impurity scattering cross-section at large b.

Returns
-------
float, the impurity-limited mobility mu_i
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ionized_impurity_mobility(C_i: float, T: float, N_I: float, b: float) -> float:
    '''Compute the ionized-impurity-limited mobility.

    Returns
    -------
    mu_i : float
        The impurity-limited mobility.

    Raises
    ------
    ValueError
        If any input is not finite, if C_i, T, N_I, or b is not positive,
        or if the resulting screening function G(b) is not positive.
    '''
    return mu_i  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_ionized_impurity_mobility(C_i: float, T: float, N_I: float, b: float) -> float:
    for name, val in [("C_i", C_i), ("T", T), ("N_I", N_I), ("b", b)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if C_i <= 0 or T <= 0 or N_I <= 0 or b <= 0:
        raise ValueError("C_i, T, N_I, b must be positive")
    G = np.log(1 + b) - b / (1 + b)
    if G <= 0:
        raise ValueError("G(b) must be positive")
    return float(C_i * T**1.5 / (N_I * G))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "C_i = 6.5e17\nT = 370.0\nN_I = 3.1e18\nb = 18.54774193548387", "call": "ionized_impurity_mobility(C_i, T, N_I, b)", "gold_call": "_oracle_ionized_impurity_mobility(C_i, T, N_I, b)"},
        {"setup": "C_i = 1.0e17\nT = 300.0\nN_I = 1.0e18\nb = 5.0", "call": "ionized_impurity_mobility(C_i, T, N_I, b)", "gold_call": "_oracle_ionized_impurity_mobility(C_i, T, N_I, b)"},
        {"setup": "C_i = 1.0e18\nT = 500.0\nN_I = 5.0e17\nb = 100.0", "call": "ionized_impurity_mobility(C_i, T, N_I, b)", "gold_call": "_oracle_ionized_impurity_mobility(C_i, T, N_I, b)"},
        {
            "setup": (
                "C_i = 6.5e17\nT = 370.0\nN_I = 3.1e18\nb = -1.0\n"
                "def run_model():\n    try:\n        ionized_impurity_mobility(C_i, T, N_I, b)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
                "def run_gold():\n    try:\n        _oracle_ionized_impurity_mobility(C_i, T, N_I, b)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2"
            ),
            "call": "run_model()", "gold_call": "run_gold()",
        },
    ]
