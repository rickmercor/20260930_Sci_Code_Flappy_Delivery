"""
Evaluate the analytic auxiliary-mass and configurational Helmholtz correction.

B' has B mass and A interactions. Relative to pure A, the ideal configurational term and classical kinetic mass term are analytic.

Returns
-------
return np.array([ideal + kinetic, ideal, kinetic], dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def alchemical_mass_correction(x_solute, temperature_k, mass_reference_u, mass_solute_u):
    """Evaluate the auxiliary-system mass and ideal-mixing correction.

    Returns a float64 ndarray of length 3 ordered as
    [F_mass, ideal_configurational_term, kinetic_mass_term], all in eV/atom.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_alchemical_mass_correction(x_solute, temperature_k, mass_reference_u, mass_solute_u):
    """Return total, ideal-configurational, and kinetic-mass Helmholtz terms."""
    import math
    import numpy as np
    x = float(x_solute)
    temperature = float(temperature_k)
    mass_reference = float(mass_reference_u)
    mass_solute = float(mass_solute_u)
    if not all(math.isfinite(v) for v in (x, temperature, mass_reference, mass_solute)):
        raise ValueError("inputs must be finite")
    if x <= 0.0 or x > 1.0 or temperature <= 0.0 or mass_reference <= 0.0 or mass_solute <= 0.0:
        raise ValueError("require 0<x<=1 and positive temperature and masses")
    k_b_ev_per_k = 8.617333262145e-5
    ideal = 0.0 if x == 1.0 else k_b_ev_per_k * temperature * (
        x * math.log(x) + (1.0 - x) * math.log(1.0 - x)
    )
    kinetic = 1.5 * k_b_ev_per_k * temperature * x * math.log(mass_reference / mass_solute)
    return np.array([ideal + kinetic, ideal, kinetic], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'alchemical_mass_correction(0.35,700,22.99,6.94)', 'gold_call': '_oracle_alchemical_mass_correction(0.35,700,22.99,6.94)'}, {'setup': '', 'call': 'alchemical_mass_correction(1.0,520,22.98976928,6.94)', 'gold_call': '_oracle_alchemical_mass_correction(1.0,520,22.98976928,6.94)'}, {'setup': '', 'call': 'alchemical_mass_correction(0.5,300,12,12)', 'gold_call': '_oracle_alchemical_mass_correction(0.5,300,12,12)'}, {'setup': '', 'call': 'alchemical_mass_correction(0.01,2000,55.845,58.693)', 'gold_call': '_oracle_alchemical_mass_correction(0.01,2000,55.845,58.693)'}, {'setup': '', 'call': 'alchemical_mass_correction(0.9,473,22.99,6.94)', 'gold_call': '_oracle_alchemical_mass_correction(0.9,473,22.99,6.94)'}, {'setup': '', 'call': 'alchemical_mass_correction(0.25,6000,55.845,58.693)', 'gold_call': '_oracle_alchemical_mass_correction(0.25,6000,55.845,58.693)'}, {'setup': '', 'call': 'alchemical_mass_correction(0.75,100,6.94,22.99)', 'gold_call': '_oracle_alchemical_mass_correction(0.75,100,6.94,22.99)'}, {'setup': '', 'call': 'alchemical_mass_correction(0.001,1200,39.948,20.18)', 'gold_call': '_oracle_alchemical_mass_correction(0.001,1200,39.948,20.18)'}]
