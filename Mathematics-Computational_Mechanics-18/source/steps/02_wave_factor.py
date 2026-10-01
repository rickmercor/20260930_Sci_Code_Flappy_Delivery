"""
Return the velocity-dependent factor A_I that converts the dynamic energy release rate to the mode-I dynamic stress intensity factor for a crack running at speed V, together with the two elastic wave speeds and the two Rayleigh factors it is built from (the paper's Eqs. 10-14). Plane strain, isotropic. Recover the exact universal function and the two wave-speed expressions from the paper; the quasi-static limit is a different, velocity-independent constant.

A moving crack tip carries a velocity-dependent near-tip field; the universal function A_I(V) is what makes the energy-based SIF reduce to the correct dynamic value, and it diverges as the crack speed approaches the Rayleigh speed.

Returns
-------
return (5,) float64: [A_I, dilatational speed, shear speed, beta1, beta2]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def wave_factor(E, nu, rho, V):
    """E: Young's modulus; nu: Poisson ratio; rho: density; V: crack-tip speed.
    Returns a float64 array [A_I, V1, V2, beta1, beta2] with the mode-I dynamic
    universal factor A_I and the dilatational/shear wave speeds (paper Eqs. 10-14).
    Raises ValueError unless 0 < V < shear wave speed."""
    return np.zeros(5)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 2: velocity-dependent DSIF factor A_I and wave speeds (Eqs. 10-14)."""

import numpy as np


def _oracle_wave_factor(E, nu, rho, V):
    V1 = np.sqrt((1.0 - nu) * E / ((1.0 + nu) * (1.0 - 2.0 * nu) * rho))
    V2 = np.sqrt(E / (2.0 * (1.0 + nu) * rho))
    if not (0.0 < V < V2):
        raise ValueError("crack speed must satisfy 0 < V < shear wave speed")
    b1 = np.sqrt(1.0 - (V / V1) ** 2)
    b2 = np.sqrt(1.0 - (V / V2) ** 2)
    A_I = b1 * (1.0 - b2 ** 2) / (4.0 * b1 * b2 - (1.0 + b2 ** 2) ** 2)
    return np.array([A_I, V1, V2, b1, b2], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'E=1.0;nu=0.3;rho=1.0;V=0.4', "call": 'wave_factor(E, nu, rho, V)', "gold_call": '_oracle_wave_factor(E, nu, rho, V)', "tol": 1e-10},
        {"setup": 'E=2.0;nu=0.25;rho=1.5;V=0.3', "call": 'wave_factor(E, nu, rho, V)', "gold_call": '_oracle_wave_factor(E, nu, rho, V)', "tol": 1e-10},
        {"setup": 'E=1.5;nu=0.35;rho=0.8;V=0.5', "call": 'wave_factor(E, nu, rho, V)', "gold_call": '_oracle_wave_factor(E, nu, rho, V)', "tol": 1e-10},
    ]
