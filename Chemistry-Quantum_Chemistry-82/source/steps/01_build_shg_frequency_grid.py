"""
Use the source's second-harmonic-generation frequency prescription. For a positive first-excitation estimate omega_max, evaluate the five integer indices from zero through four at the source denominator eight. Return the numerical frequency vector.

This stage preserves a source-defined scientific quantity used by later parts of the directional convergence audit.

Returns
-------
np.ndarray: Five SHG frequencies.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_shg_frequency_grid(omega_max: float = 0.36) -> np.ndarray:
    """Return the five source-prescribed SHG input frequencies.

    Parameters
    ----------
    omega_max : float
        Positive finite first-excitation estimate.
    Returns
    -------
    np.ndarray
        Five frequencies in ascending order.
    Raises
    ------
    ValueError
        If omega_max is not positive and finite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

def _oracle_build_shg_frequency_grid(omega_max: float = 0.36) -> np.ndarray:
    if not np.isscalar(omega_max) or not np.isfinite(omega_max) or float(omega_max) <= 0.0:
        raise ValueError("omega_max must be a positive finite scalar")
    return float(omega_max) * np.arange(5, dtype=float) / 8.0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"omega=0.36","call":"build_shg_frequency_grid(omega)","gold_call":"_oracle_build_shg_frequency_grid(omega)","tol":1e-12},
        {"setup":"omega=1e-6","call":"build_shg_frequency_grid(omega)","gold_call":"_oracle_build_shg_frequency_grid(omega)","tol":1e-12},
        {"setup":"omega=2.75","call":"build_shg_frequency_grid(omega)","gold_call":"_oracle_build_shg_frequency_grid(omega)","tol":1e-12},
        {"setup":"""def f(g):
 try:g(0.0)
 except ValueError:return 1.0
 return 0.0""","call":"f(build_shg_frequency_grid)","gold_call":"f(_oracle_build_shg_frequency_grid)","tol":1e-12}
    ]
