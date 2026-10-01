"""
Implement diameter_dependent_mobility, which computes the effective 1D nanowire mobility from the temperature-scaled bulk mobility, using the empirical diameter-dependence relation.

Nanowire mobility follows a systematic dependence on diameter relative to the characteristic length scale d0, quantifying the competition between electron-phonon scattering and surface scattering. This relation applies specifically to the temperature-corrected bulk mobility, since the underlying electron-phonon scattering itself depends on temperature; applying it to an uncorrected reference value would conflate two independent physical effects. The relation is only valid when the diameter exceeds d0.

Returns
-------
float, the diameter-dependent nanowire mobility
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def diameter_dependent_mobility(mu_bulk_T: float, d: float, d0: float, beta: float) -> float:
    '''Compute the diameter-dependent nanowire mobility.

    Parameters
    ----------
    mu_bulk_T : float
        Temperature-scaled bulk mobility.
    d : float
        Nanowire diameter.
    d0 : float
        Characteristic length scale.
    beta : float
        Diameter-dependence exponent.

    Returns
    -------
    mu_1D : float
        The diameter-dependent nanowire mobility, as a native Python float.

    Raises
    ------
    ValueError
        If any input is not finite, or if d does not exceed d0.
    '''
    return mu_1D  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_diameter_dependent_mobility(mu_bulk_T: float, d: float, d0: float, beta: float) -> float:
    """Reference implementation."""
    for name, val in [("mu_bulk_T", mu_bulk_T), ("d", d), ("d0", d0), ("beta", beta)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if d <= d0:
        raise ValueError("d must exceed d0 for this formula to apply")
    return float(mu_bulk_T * (1 - (d / d0) ** (-beta)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "mu_bulk_T = 568.2640565437886\nd = 190.0\nd0 = 44.23210184068731\nbeta = 1.44", "call": "diameter_dependent_mobility(mu_bulk_T, d, d0, beta)", "gold_call": "_oracle_diameter_dependent_mobility(mu_bulk_T, d, d0, beta)"},
        {"setup": "mu_bulk_T = 1000.0\nd = 200.0\nd0 = 50.0\nbeta = 2.0", "call": "diameter_dependent_mobility(mu_bulk_T, d, d0, beta)", "gold_call": "_oracle_diameter_dependent_mobility(mu_bulk_T, d, d0, beta)"},
        {"setup": "mu_bulk_T = 500.0\nd = 1000.0\nd0 = 10.0\nbeta = 1.0", "call": "diameter_dependent_mobility(mu_bulk_T, d, d0, beta)", "gold_call": "_oracle_diameter_dependent_mobility(mu_bulk_T, d, d0, beta)"},
        {
            "setup": (
                "mu_bulk_T = 568.0\nd = 30.0\nd0 = 44.2\nbeta = 1.44\n"
                "def run_model():\n    try:\n        diameter_dependent_mobility(mu_bulk_T, d, d0, beta)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
                "def run_gold():\n    try:\n        _oracle_diameter_dependent_mobility(mu_bulk_T, d, d0, beta)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2"
            ),
            "call": "run_model()", "gold_call": "run_gold()",
        },
    ]
