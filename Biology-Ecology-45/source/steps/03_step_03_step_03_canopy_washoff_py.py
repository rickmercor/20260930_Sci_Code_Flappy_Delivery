"""
Compute canopy residue, soil transfer, foliar dissipation, and intercepted rainfall for one day. Passing behavior uses the declared mass-conserving interpretation of the source's decreasing exponential and handles degenerate rainfall or canopy geometry without division by zero; failure corrupts the soil input and canopy mass balance.

The new canopy routine makes transfer depend on ground cover, leaf area, pesticide solubility, and rainfall interception rather than using a fixed wash-off fraction. Foliar residue also decays with its own half-life.

Returns
-------
np.ndarray with shape (4,), containing [remaining canopy mass, soil transfer, foliar dissipation, intercepted rainfall]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def canopy_washoff(
    canopy_mass: float,
    rainfall: float,
    leaf_area_index: float,
    ground_cover_fraction: float,
    solubility_g_l: float,
    foliar_half_life_days: float,
    interception_alpha: float,
) -> "np.ndarray":
    """Apply one day's canopy wash-off followed by foliar decay.
 
    All values are finite. Mass, rain, and LAI are nonnegative; solubility,
    half-life, and alpha are positive; ground cover lies in [0,1]. The source
    exponential is treated as post-wash residue and its complement as transfer.
 
    Returns
    -------
    result : np.ndarray
        [remaining canopy mass, soil transfer, foliar loss, intercepted rain].
 
    Raises
    ------
    ValueError
        If finiteness or any stated range is violated.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
 
def _oracle_canopy_washoff(
    canopy_mass: float,
    rainfall: float,
    leaf_area_index: float,
    ground_cover_fraction: float,
    solubility_g_l: float,
    foliar_half_life_days: float,
    interception_alpha: float,
) -> "np.ndarray":
    values = [canopy_mass, rainfall, leaf_area_index, ground_cover_fraction, solubility_g_l, foliar_half_life_days, interception_alpha]
    if not all(np.isfinite(x) for x in values):
        raise ValueError("all canopy inputs must be finite")
    if canopy_mass < 0.0 or rainfall < 0.0 or leaf_area_index < 0.0 or solubility_g_l <= 0.0 or foliar_half_life_days <= 0.0 or interception_alpha <= 0.0:
        raise ValueError("canopy inputs are outside their allowed ranges")
    if not 0.0 <= ground_cover_fraction <= 1.0:
        raise ValueError("ground_cover_fraction must lie in [0, 1]")
    if rainfall == 0.0 or leaf_area_index == 0.0 or ground_cover_fraction == 0.0:
        intercepted = 0.0
    else:
        x = ground_cover_fraction * rainfall / (interception_alpha * leaf_area_index)
        intercepted = interception_alpha * leaf_area_index * (1.0 - 1.0 / (1.0 + x))
    effective = max(rainfall - intercepted, 0.0)
    extraction = 0.0160 * solubility_g_l ** 0.3832
    post_wash = canopy_mass * math.exp(-extraction * effective)
    transferred = canopy_mass - post_wash
    remaining = post_wash * math.exp(-math.log(2.0) / foliar_half_life_days)
    dissipated = post_wash - remaining
    return np.array([remaining, transferred, dissipated, intercepted], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {"setup": "", "call": "canopy_washoff(100.,18.,2.1,.7,12.,10.6,.25)", "gold_call": "_oracle_canopy_washoff(100.,18.,2.1,.7,12.,10.6,.25)"},
        {"setup": "", "call": "canopy_washoff(100.,0.,2.1,.7,12.,10.6,.25)", "gold_call": "_oracle_canopy_washoff(100.,0.,2.1,.7,12.,10.6,.25)"},
        {"setup": "", "call": "canopy_washoff(0.,20.,0.,0.,12.,10.6,.25)", "gold_call": "_oracle_canopy_washoff(0.,20.,0.,0.,12.,10.6,.25)"},
        {"setup": """def run(fn):
    try: fn(10.,2.,1.,1.2,12.,10.6,.25); return 0
    except ValueError: return 1
""", "call": "run(canopy_washoff)", "gold_call": "run(_oracle_canopy_washoff)"},
        {"setup": """def run(fn):
    try: fn(10.,2.,1.,.5,12.,0.,.25); return 0
    except ValueError: return 1
""", "call": "run(canopy_washoff)", "gold_call": "run(_oracle_canopy_washoff)"},
    ]
