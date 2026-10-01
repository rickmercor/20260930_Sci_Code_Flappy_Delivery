"""
Evaluate continuum polaron formation energy.

Even after a dopant ionizes, coupling to lattice polarization can localize the carrier and suppress useful transport in polar oxides.

Returns
-------
float, continuum polaron formation energy in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def continuum_polaron_energy(
    m_star_over_me: float,
    epsilon_eff: float,
    e_ha: float = 27.2,
) -> float:
    """Return continuum polaron formation energy in eV.

    Raises
    ------
    ValueError
        If m_star_over_me, epsilon_eff, or e_ha is non-positive.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math
def _oracle_continuum_polaron_energy(
    m_star_over_me: float,
    epsilon_eff: float,
    e_ha: float = 27.2,
) -> float:
    if m_star_over_me <= 0:
        raise ValueError("m_star_over_me must be positive")
    if epsilon_eff <= 0:
        raise ValueError("epsilon_eff must be positive")
    if e_ha <= 0:
        raise ValueError("e_ha must be positive")
    return float(-(1.0 / (3.0 * math.pi)) * m_star_over_me / (epsilon_eff**2) * e_ha)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "m_star_over_me = 0.30352146549621584\n"
                "epsilon_eff = 8.663013698630134\n"
                "e_ha = 27.2\n"
            ),
            "call": "continuum_polaron_energy(m_star_over_me, epsilon_eff, e_ha)",
            "gold_call": "_oracle_continuum_polaron_energy(m_star_over_me, epsilon_eff, e_ha)",
        },
        {
            "setup": "m_star_over_me = 0.140\nepsilon_eff = 13.920100250626561\ne_ha = 27.2\n",
            "call": "continuum_polaron_energy(m_star_over_me, epsilon_eff, e_ha)",
            "gold_call": "_oracle_continuum_polaron_energy(m_star_over_me, epsilon_eff, e_ha)",
        },
        {
            "setup": "m_star_over_me = 1.0\nepsilon_eff = 1.0\ne_ha = 27.2\n",
            "call": "continuum_polaron_energy(m_star_over_me, epsilon_eff, e_ha)",
            "gold_call": "_oracle_continuum_polaron_energy(m_star_over_me, epsilon_eff, e_ha)",
        },
    ]
