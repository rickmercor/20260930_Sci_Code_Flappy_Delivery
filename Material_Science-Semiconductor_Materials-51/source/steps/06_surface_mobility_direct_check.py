"""
Implement surface_mobility_direct_check, which computes the surface-scattering-only mobility using the paper's alternate direct form (Eq. 6), rather than deriving it via Matthiessen's rule, as an independent consistency check on the pipeline.

The paper's diameter-dependent relation (Eq. 5) can be algebraically transformed into an equivalent expression (Eq. 6) that directly isolates the surface-scattering-only mobility contribution: mu_s = mu_bulk * [(d/d0)^beta - 1]. This should agree with the value obtained independently via Matthiessen's rule (Eq. 7) applied to mu_1D and mu_bulk; the two routes are mathematically equivalent, and computing both provides a genuine self-consistency check on the pipeline.

Returns
-------
float, the directly-computed surface-limited mobility.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def surface_mobility_direct_check(mu_bulk_T: float, d: float, d0: float, beta: float) -> float:
    '''Compute surface-scattering-only mobility via the direct Eq. 6 form.

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
    mu_s_direct : float
        The directly computed surface-limited mobility, as a native
        Python float.

    Raises
    ------
    ValueError
        If any input is not finite, or if d does not exceed d0.
    '''
    return mu_s_direct  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_surface_mobility_direct_check(mu_bulk_T: float, d: float, d0: float, beta: float) -> float:
    """Reference implementation."""
    for name, val in [("mu_bulk_T", mu_bulk_T), ("d", d), ("d0", d0), ("beta", beta)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if d <= d0:
        raise ValueError("d must exceed d0 for this formula to apply")
    return float(mu_bulk_T * ((d / d0) ** beta - 1))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "mu_bulk_T = 568.2640565437886\nd = 190.0\nd0 = 44.23210184068731\nbeta = 1.44", "call": "surface_mobility_direct_check(mu_bulk_T, d, d0, beta)", "gold_call": "_oracle_surface_mobility_direct_check(mu_bulk_T, d, d0, beta)"},
        {"setup": "mu_bulk_T = 1000.0\nd = 200.0\nd0 = 50.0\nbeta = 2.0", "call": "surface_mobility_direct_check(mu_bulk_T, d, d0, beta)", "gold_call": "_oracle_surface_mobility_direct_check(mu_bulk_T, d, d0, beta)"},
        {"setup": "mu_bulk_T = 500.0\nd = 1000.0\nd0 = 10.0\nbeta = 1.0", "call": "surface_mobility_direct_check(mu_bulk_T, d, d0, beta)", "gold_call": "_oracle_surface_mobility_direct_check(mu_bulk_T, d, d0, beta)"},
        {
            "setup": (
                "mu_bulk_T = 568.0\nd = 30.0\nd0 = 44.2\nbeta = 1.44\n"
                "def run_model():\n    try:\n        surface_mobility_direct_check(mu_bulk_T, d, d0, beta)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
                "def run_gold():\n    try:\n        _oracle_surface_mobility_direct_check(mu_bulk_T, d, d0, beta)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2"
            ),
            "call": "run_model()", "gold_call": "run_gold()",
        },
    ]
