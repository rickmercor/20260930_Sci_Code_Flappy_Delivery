"""
Implement surface_limited_mobility, which extracts the surface-scattering-only contribution to mobility from the combined 1D mobility and the bulk mobility, using Matthiessen's rule.

Matthiessen's rule states that independent scattering mechanisms combine as a reciprocal sum of their individual mobility contributions, analogous to resistors combining in parallel. Isolating the surface-scattering-only contribution requires subtracting the bulk reciprocal from the combined 1D reciprocal, not adding them or inverting the relationship.

Returns
-------
float, the surface-limited mobility
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def surface_limited_mobility(mu_1D: float, mu_bulk_T: float) -> float:
    '''Compute the surface-limited mobility via Matthiessen's rule.

    Parameters
    ----------
    mu_1D : float
        Diameter-dependent 1D nanowire mobility.
    mu_bulk_T : float
        Temperature-scaled bulk mobility.

    Returns
    -------
    mu_s : float
        The surface-limited mobility, as a native Python float.

    Raises
    ------
    ValueError
        If either mobility is non-finite or non-positive, or if mu_1D is
        greater than or equal to mu_bulk_T.
    '''
    return mu_s  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_surface_limited_mobility(mu_1D: float, mu_bulk_T: float) -> float:
    """Reference implementation."""
    for name, val in [("mu_1D", mu_1D), ("mu_bulk_T", mu_bulk_T)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if mu_1D <= 0.0 or mu_bulk_T <= 0.0:
        raise ValueError("mobilities must be positive")
    if mu_1D >= mu_bulk_T:
        raise ValueError("mu_1D must be smaller than mu_bulk_T")
    denom = 1.0 / mu_1D - 1.0 / mu_bulk_T
    return float(1.0 / denom)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "mu_1D = 498.6002642462811\nmu_bulk_T = 568.2640565437886", "call": "surface_limited_mobility(mu_1D, mu_bulk_T)", "gold_call": "_oracle_surface_limited_mobility(mu_1D, mu_bulk_T)"},
        {"setup": "mu_1D = 1767.0\nmu_bulk_T = 3317.0", "call": "surface_limited_mobility(mu_1D, mu_bulk_T)", "gold_call": "_oracle_surface_limited_mobility(mu_1D, mu_bulk_T)"},
        {"setup": "mu_1D = 500.0\nmu_bulk_T = 1000.0", "call": "surface_limited_mobility(mu_1D, mu_bulk_T)", "gold_call": "_oracle_surface_limited_mobility(mu_1D, mu_bulk_T)"},
        {
            "setup": (
                "mu_1D = 500.0\nmu_bulk_T = 500.0\n"
                "def run_model():\n    try:\n        surface_limited_mobility(mu_1D, mu_bulk_T)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
                "def run_gold():\n    try:\n        _oracle_surface_limited_mobility(mu_1D, mu_bulk_T)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2"
            ),
            "call": "run_model()", "gold_call": "run_gold()",
        },
    ]
