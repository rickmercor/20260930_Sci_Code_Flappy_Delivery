"""
Implement thermo_params, which computes the inverse thermal energy beta, the barrier ratio gamma, and the bias-limiting factor epsilon from a temperature and a barrier parameter Delta E. These enter the final bias-potential formula to impose a soft ceiling on its own strength.

The OPES-Explore-style bias potential uses beta and a user-chosen barrier energy Delta E to control how strong the maximum bias can become. As the density ratio in the final formula approaches zero (an unexplored region), epsilon prevents the bias potential from diverging to negative infinity, effectively capping the largest bias the method will apply.

Returns
-------
tuple[float, float, float]: (beta, gamma, epsilon), all native Python floats
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def thermo_params(T: float, delta_E: float) -> tuple[float, float, float]:
    '''Compute beta, gamma, and epsilon from temperature and barrier energy.

    Parameters
    ----------
    T : float
        Temperature in Kelvin, must be > 0.
    delta_E : float
        Barrier parameter Delta E in eV, must be > 0.

    Returns
    -------
    beta : float
        Inverse thermal energy, 1 / (k_B * T), in eV^-1, where k_B is the
        Boltzmann constant in eV/K (8.617333262e-5 eV/K).
    gamma : float
        beta * delta_E (dimensionless).
    epsilon : float
        exp(-gamma / (gamma - 1)) (dimensionless).

    Raises
    ------
    ValueError
        If T is not a finite number > 0, if delta_E is not a finite number
        > 0, or if gamma is within 1e-9 of 1.0 (division by zero in epsilon).
    '''
    return beta, gamma, epsilon  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_thermo_params(T: float, delta_E: float) -> tuple[float, float, float]:
    """Reference implementation."""
    _KB_EV_PER_K = 8.617333262e-5  # Boltzmann constant in eV/K
    if not (isinstance(T, (int, float)) and np.isfinite(T) and T > 0.0):
        raise ValueError("T must be a finite number > 0")
    if not (isinstance(delta_E, (int, float)) and np.isfinite(delta_E) and delta_E > 0.0):
        raise ValueError("delta_E must be a finite number > 0")

    beta = 1.0 / (_KB_EV_PER_K * float(T))
    gamma = beta * float(delta_E)
    if abs(gamma - 1.0) < 1e-9:
        raise ValueError("gamma must not be within 1e-9 of 1.0 (division by zero in epsilon)")
    epsilon = float(np.exp(-gamma / (gamma - 1.0)))
    return float(beta), float(gamma), epsilon

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal case: the task's benchmark parameters
            "setup": "T = 300.0\ndelta_E = 15.0",
            "call": "thermo_params(T, delta_E)",
            "gold_call": "_oracle_thermo_params(T, delta_E)",
            "tol": 1e-6,
        },
        {
            # Boundary case: a different temperature and a smaller barrier
            "setup": "T = 500.0\ndelta_E = 5.0",
            "call": "thermo_params(T, delta_E)",
            "gold_call": "_oracle_thermo_params(T, delta_E)",
            "tol": 1e-6,
        },
        {
            # Edge case: delta_E near k_B T, so gamma approaches 1 and epsilon is small
            "setup": "T = 300.0\ndelta_E = 0.03",
            "call": "thermo_params(T, delta_E)",
            "gold_call": "_oracle_thermo_params(T, delta_E)",
            "tol": 1e-6,
        },
        {
            # Invalid-input case: non-positive temperature should raise ValueError
           "setup": (
        "T = 0.0\n"
        "delta_E = 15.0\n"
        "def run_model():\n"
        "    try:\n"
        "        thermo_params(T, delta_E)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold():\n"
        "    try:\n"
        "        _oracle_thermo_params(T, delta_E)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2"
    ),
    "call": "run_model()",
    "gold_call": "run_gold()",
},
    ]
