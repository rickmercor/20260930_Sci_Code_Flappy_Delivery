"""
Evaluate the continuum dopant ionization energy.

Whether a dopant can supply free carriers at operating temperature depends on how the carrier mass is screened by the static lattice dielectric.

Returns
-------
float, continuum ionization energy in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def continuum_ionization_energy(
    m_star_over_me: float,
    epsilon_s: float,
    rydberg_eV: float = 13.6,
) -> float:
    """Return continuum ionization energy in eV.

    Raises
    ------
    ValueError
        If m_star_over_me, epsilon_s, or rydberg_eV is non-positive.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_continuum_ionization_energy(
    m_star_over_me: float,
    epsilon_s: float,
    rydberg_eV: float = 13.6,
) -> float:
    if m_star_over_me <= 0:
        raise ValueError("m_star_over_me must be positive")
    if epsilon_s <= 0:
        raise ValueError("epsilon_s must be positive")
    if rydberg_eV <= 0:
        raise ValueError("rydberg_eV must be positive")
    return float(rydberg_eV * m_star_over_me / (epsilon_s**2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "m_star_over_me = 0.30352146549621584\nepsilon_s = 12.4\nrydberg_eV = 13.6\n",
            "call": "continuum_ionization_energy(m_star_over_me, epsilon_s, rydberg_eV)",
            "gold_call": "_oracle_continuum_ionization_energy(m_star_over_me, epsilon_s, rydberg_eV)",
        },
        {
            "setup": "m_star_over_me = 0.358\nepsilon_s = 9.28\nrydberg_eV = 13.6\n",
            "call": "continuum_ionization_energy(m_star_over_me, epsilon_s, rydberg_eV)",
            "gold_call": "_oracle_continuum_ionization_energy(m_star_over_me, epsilon_s, rydberg_eV)",
        },
        {
            "setup": "m_star_over_me = 1.0\nepsilon_s = 1.0\nrydberg_eV = 13.6\n",
            "call": "continuum_ionization_energy(m_star_over_me, epsilon_s, rydberg_eV)",
            "gold_call": "_oracle_continuum_ionization_energy(m_star_over_me, epsilon_s, rydberg_eV)",
        },
    ]
