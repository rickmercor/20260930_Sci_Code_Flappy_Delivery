"""
Solve the nonlinear one-layer mass closure for the unique nonnegative aqueous concentration. A passing implementation preserves the declared water and soil mass basis and respects the convergence contract; an incorrect phase balance changes the dissolved mass available for preferential transport.

The updated model replaces an organic-carbon partition approximation with a nonlinear Freundlich isotherm. Combining that isotherm with phase masses gives a monotone scalar root problem for aqueous concentration.

Returns
-------
float, the unique nonnegative aqueous concentration in mg L^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def freundlich_aqueous_concentration(
    total_mass: float,
    water_volume: float,
    soil_mass: float,
    freundlich_coefficient: float,
    freundlich_exponent: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Solve the nonlinear Freundlich phase-mass closure.
 
    Parameters are finite. Total and soil masses and the coefficient are
    nonnegative; water volume, exponent, and tolerance are strictly positive.
    max_iterations must be a positive integer; bool and float are invalid.
 
    Returns
    -------
    concentration : float
        Unique nonnegative aqueous concentration in mg L^-1.
 
    Raises
    ------
    ValueError
        If finiteness, stated ranges, or the integer contract is violated.
    RuntimeError
        If bisection does not converge within max_iterations.
    """
    return concentration

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
 
def _oracle_freundlich_aqueous_concentration(
    total_mass: float,
    water_volume: float,
    soil_mass: float,
    freundlich_coefficient: float,
    freundlich_exponent: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    values = [total_mass, water_volume, soil_mass, freundlich_coefficient, freundlich_exponent, tolerance]
    if not all(np.isfinite(x) for x in values):
        raise ValueError("all numeric inputs must be finite")
    if total_mass < 0.0 or water_volume <= 0.0 or soil_mass < 0.0 or freundlich_coefficient < 0.0 or freundlich_exponent <= 0.0 or tolerance <= 0.0:
        raise ValueError("mass and partition parameters are outside their allowed ranges")
    if not isinstance(max_iterations, (int, np.integer)) or isinstance(max_iterations, (bool, np.bool_)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if total_mass == 0.0:
        return 0.0
    lo = 0.0
    hi = total_mass / water_volume
    for _ in range(int(max_iterations)):
        mid = 0.5 * (lo + hi)
        reconstructed = water_volume * mid + soil_mass * freundlich_coefficient * mid ** freundlich_exponent
        if reconstructed < total_mass:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance * max(1.0, hi):
            return float(0.5 * (lo + hi))
    raise RuntimeError("Freundlich solve did not converge")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {"setup": "", "call": "freundlich_aqueous_concentration(50.,2.5,13.,38.37,1.28,1e-13,300)", "gold_call": "_oracle_freundlich_aqueous_concentration(50.,2.5,13.,38.37,1.28,1e-13,300)"},
        {"setup": "", "call": "freundlich_aqueous_concentration(0.,2.5,13.,38.37,1.28,1e-13,300)", "gold_call": "_oracle_freundlich_aqueous_concentration(0.,2.5,13.,38.37,1.28,1e-13,300)"},
        {"setup": "", "call": "freundlich_aqueous_concentration(5.,2.,0.,0.,1.,1e-13,300)", "gold_call": "_oracle_freundlich_aqueous_concentration(5.,2.,0.,0.,1.,1e-13,300)"},
        {"setup": """def run(fn):
    try: fn(5.,2.,1.,2.,1.2,0.,100); return 0
    except ValueError: return 1
""", "call": "run(freundlich_aqueous_concentration)", "gold_call": "run(_oracle_freundlich_aqueous_concentration)"},
        {"setup": """def run(fn):
    try: fn(5.,2.,1.,2.,1.2,1e-12,20.0); return 0
    except ValueError: return 1
""", "call": "run(freundlich_aqueous_concentration)", "gold_call": "run(_oracle_freundlich_aqueous_concentration)"},
    ]
