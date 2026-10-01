"""
Implement d0_from_mfp, which computes the characteristic length scale governing the onset of surface scattering as a fixed proportion of the weighted-average mean free path.

The characteristic length scale d0 is comparable in magnitude to the averaged mean free path but is not necessarily numerically equal to it; the relationship is expressed here as direct proportionality via a fixed constant, reflecting that the two quantities are of the same order but distinct physical quantities.

Returns
-------
float, the characteristic length scale d0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def d0_from_mfp(MFP_avg: float, kappa: float) -> float:
    '''Compute the characteristic length scale from the weighted MFP average.

    Parameters
    ----------
    MFP_avg : float
        The weighted average MFP.
    kappa : float
        The proportionality constant.

    Returns
    -------
    d0 : float
        The characteristic length scale, as a native Python float.

    Raises
    ------
    ValueError
        If MFP_avg or kappa is not finite, or if either is not positive.
    '''
    return d0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_d0_from_mfp(MFP_avg: float, kappa: float) -> float:
    """Reference implementation."""
    for name, val in [("MFP_avg", MFP_avg), ("kappa", kappa)]:
        if not (isinstance(val, (int, float)) and np.isfinite(val)):
            raise ValueError(f"{name} must be finite")
    if MFP_avg <= 0 or kappa <= 0:
        raise ValueError("MFP_avg and kappa must be positive")
    return float(kappa * MFP_avg)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "MFP_avg = 61.43347477873237\nkappa = 0.72", "call": "d0_from_mfp(MFP_avg, kappa)", "gold_call": "_oracle_d0_from_mfp(MFP_avg, kappa)"},
        {"setup": "MFP_avg = 100.0\nkappa = 1.0", "call": "d0_from_mfp(MFP_avg, kappa)", "gold_call": "_oracle_d0_from_mfp(MFP_avg, kappa)"},
        {"setup": "MFP_avg = 30.0\nkappa = 0.5", "call": "d0_from_mfp(MFP_avg, kappa)", "gold_call": "_oracle_d0_from_mfp(MFP_avg, kappa)"},
        {
            "setup": (
                "MFP_avg = -5.0\nkappa = 0.9\n"
                "def run_model():\n    try:\n        d0_from_mfp(MFP_avg, kappa)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
                "def run_gold():\n    try:\n        _oracle_d0_from_mfp(MFP_avg, kappa)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2"
            ),
            "call": "run_model()", "gold_call": "run_gold()",
        },
    ]
