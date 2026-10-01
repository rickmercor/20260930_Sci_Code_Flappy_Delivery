"""
Convert scattering time and mass into drift mobility.

Scattering-limited drift mobility enters the unipolar ranking product linearly once a single DOS mass is chosen.

Returns
-------
float, drift mobility in cm^2/V*s
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scattering_time_mobility(
    m_star_over_me: float,
    tau: float,
    elementary_charge: float = 1.602176634e-19,
    electron_mass_kg: float = 9.1093837015e-31,
) -> float:
    """Return drift mobility in cm^2/V*s.

    Raises
    ------
    ValueError
        If mass, tau, or physical constants are non-positive.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_scattering_time_mobility(
    m_star_over_me: float,
    tau: float,
    elementary_charge: float = 1.602176634e-19,
    electron_mass_kg: float = 9.1093837015e-31,
) -> float:
    if m_star_over_me <= 0:
        raise ValueError("m_star_over_me must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if elementary_charge <= 0 or electron_mass_kg <= 0:
        raise ValueError("physical constants must be positive")
    mu_si = elementary_charge * tau / (m_star_over_me * electron_mass_kg)
    return float(mu_si * 1.0e4)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "m_star_over_me = 0.30352146549621584\n"
                "tau = 1.4e-14\n"
                "elementary_charge = 1.602176634e-19\n"
                "electron_mass_kg = 9.1093837015e-31\n"
            ),
            "call": (
                "scattering_time_mobility(m_star_over_me, tau, "
                "elementary_charge, electron_mass_kg)"
            ),
            "gold_call": (
                "_oracle_scattering_time_mobility(m_star_over_me, tau, "
                "elementary_charge, electron_mass_kg)"
            ),
        },
        {
            "setup": (
                "m_star_over_me = 1.0\ntau = 1.0e-14\n"
                "elementary_charge = 1.602176634e-19\n"
                "electron_mass_kg = 9.1093837015e-31\n"
            ),
            "call": (
                "scattering_time_mobility(m_star_over_me, tau, "
                "elementary_charge, electron_mass_kg)"
            ),
            "gold_call": (
                "_oracle_scattering_time_mobility(m_star_over_me, tau, "
                "elementary_charge, electron_mass_kg)"
            ),
        },
        {
            "setup": (
                "m_star_over_me = 0.1\ntau = 5.0e-15\n"
                "elementary_charge = 1.602176634e-19\n"
                "electron_mass_kg = 9.1093837015e-31\n"
            ),
            "call": (
                "scattering_time_mobility(m_star_over_me, tau, "
                "elementary_charge, electron_mass_kg)"
            ),
            "gold_call": (
                "_oracle_scattering_time_mobility(m_star_over_me, tau, "
                "elementary_charge, electron_mass_kg)"
            ),
        },
    ]
