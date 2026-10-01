"""
Implement bulk_mobility_temperature_scaling, which computes the temperature-corrected bulk carrier mobility from a reference value using the observed power-law temperature dependence.

In bulk materials, phonon-limited mobility decreases with increasing temperature following an approximate power-law relationship over ranges where the dominant scattering mechanism remains unchanged, since higher temperatures increase phonon occupation and thus electron-phonon scattering.

Returns
-------
float, the temperature-scaled bulk mobility
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bulk_mobility_temperature_scaling(mu_bulk_ref: float, T_ref: float, alpha: float, T: float) -> float:
    '''Compute the temperature-scaled bulk mobility.

    Parameters
    ----------
    mu_bulk_ref : float
        Reference bulk mobility at T_ref.
    T_ref : float
        Reference temperature.
    alpha : float
        Power-law temperature-scaling exponent.
    T : float
        Target temperature.

    Returns
    -------
    mu_bulk_T : float
        The temperature-scaled bulk mobility, as a native Python float.

    Raises
    ------
    ValueError
        If any input is not finite, or if mu_bulk_ref, T_ref, or T is not positive.
    '''
    return mu_bulk_T  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_bulk_mobility_temperature_scaling(mu_bulk_ref: float, T_ref: float, alpha: float, T: float) -> float:
    """Reference implementation."""
    for name, val in [("mu_bulk_ref", mu_bulk_ref), ("T_ref", T_ref), ("alpha", alpha), ("T", T)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if mu_bulk_ref <= 0 or T_ref <= 0 or T <= 0:
        raise ValueError("mu_bulk_ref, T_ref, T must be positive")
    return float(mu_bulk_ref * (T / T_ref) ** (-alpha))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "mu_bulk_ref = 1090.0\nT_ref = 245.0\nalpha = 1.58\nT = 370.0", "call": "bulk_mobility_temperature_scaling(mu_bulk_ref, T_ref, alpha, T)", "gold_call": "_oracle_bulk_mobility_temperature_scaling(mu_bulk_ref, T_ref, alpha, T)"},
        {"setup": "mu_bulk_ref = 1400.0\nT_ref = 300.0\nalpha = 1.5\nT = 300.0", "call": "bulk_mobility_temperature_scaling(mu_bulk_ref, T_ref, alpha, T)", "gold_call": "_oracle_bulk_mobility_temperature_scaling(mu_bulk_ref, T_ref, alpha, T)"},
        {"setup": "mu_bulk_ref = 3317.0\nT_ref = 200.0\nalpha = 2.0\nT = 400.0", "call": "bulk_mobility_temperature_scaling(mu_bulk_ref, T_ref, alpha, T)", "gold_call": "_oracle_bulk_mobility_temperature_scaling(mu_bulk_ref, T_ref, alpha, T)"},
        {
            "setup": (
                "mu_bulk_ref = 1400.0\nT_ref = 300.0\nalpha = 1.5\nT = -10.0\n"
                "def run_model():\n    try:\n        bulk_mobility_temperature_scaling(mu_bulk_ref, T_ref, alpha, T)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
                "def run_gold():\n    try:\n        _oracle_bulk_mobility_temperature_scaling(mu_bulk_ref, T_ref, alpha, T)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2"
            ),
            "call": "run_model()", "gold_call": "run_gold()",
        },
    ]
