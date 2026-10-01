"""
Assemble the gated, thermally attenuated prioritization score P.

Extreme-gap candidates are ranked only when ionization stays

below the shallow ceiling and continuum polaron binding stays below thermal energy;

survivors are scored by a permittivity-mobility-field product attenuated by thermal

ionization only.

Returns
-------
float, prioritization score P
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def prioritization_score(
    epsilon_s: float,
    mobility: float,
    e_c: float,
    e_ion: float,
    e_pol: float,
    kT: float = 0.025,
    ion_ceiling_eV: float = 0.21,
    pol_ceiling_eV: float = 0.025,
) -> float:
    """Return prioritization score P.

    Raises
    ------
    ValueError
        If epsilon_s, e_c, kT, or ceilings are non-positive, or mobility is negative.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_prioritization_score(
    epsilon_s: float,
    mobility: float,
    e_c: float,
    e_ion: float,
    e_pol: float,
    kT: float = 0.025,
    ion_ceiling_eV: float = 0.21,
    pol_ceiling_eV: float = 0.025,
) -> float:
    import math

    if epsilon_s <= 0:
        raise ValueError("epsilon_s must be positive")
    if mobility < 0:
        raise ValueError("mobility must be non-negative")
    if e_c <= 0:
        raise ValueError("e_c must be positive")
    if kT <= 0:
        raise ValueError("kT must be positive")
    if ion_ceiling_eV <= 0 or pol_ceiling_eV <= 0:
        raise ValueError("ceilings must be positive")
    if e_ion >= ion_ceiling_eV:
        return 0.0
    if abs(e_pol) >= pol_ceiling_eV:
        return 0.0
    power = epsilon_s * mobility * (e_c**3)
    return float(power * math.exp(-e_ion / kT))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "epsilon_s = 12.4\n"
                "mobility = 81.12599255724561\n"
                "e_c = 3.3116977239237904\n"
                "e_ion = 0.026846331495502956\n"
                "e_pol = -0.011672104974939354\n"
                "kT = 0.025\n"
                "ion_ceiling_eV = 0.21\n"
                "pol_ceiling_eV = 0.025\n"
            ),
            "call": (
                "prioritization_score(epsilon_s, mobility, e_c, e_ion, e_pol, "
                "kT, ion_ceiling_eV, pol_ceiling_eV)"
            ),
            "gold_call": (
                "_oracle_prioritization_score(epsilon_s, mobility, e_c, e_ion, e_pol, "
                "kT, ion_ceiling_eV, pol_ceiling_eV)"
            ),
        },
        {
            "setup": (
                "epsilon_s = 12.4\nmobility = 81.12599255724561\n"
                "e_c = 3.3116977239237904\ne_ion = 0.25\ne_pol = -0.01\n"
                "kT = 0.025\nion_ceiling_eV = 0.21\npol_ceiling_eV = 0.025\n"
            ),
            "call": (
                "prioritization_score(epsilon_s, mobility, e_c, e_ion, e_pol, "
                "kT, ion_ceiling_eV, pol_ceiling_eV)"
            ),
            "gold_call": (
                "_oracle_prioritization_score(epsilon_s, mobility, e_c, e_ion, e_pol, "
                "kT, ion_ceiling_eV, pol_ceiling_eV)"
            ),
        },
        {
            "setup": (
                "epsilon_s = 12.4\nmobility = 81.12599255724561\n"
                "e_c = 3.3116977239237904\ne_ion = 0.02\ne_pol = -0.04\n"
                "kT = 0.025\nion_ceiling_eV = 0.21\npol_ceiling_eV = 0.025\n"
            ),
            "call": (
                "prioritization_score(epsilon_s, mobility, e_c, e_ion, e_pol, "
                "kT, ion_ceiling_eV, pol_ceiling_eV)"
            ),
            "gold_call": (
                "_oracle_prioritization_score(epsilon_s, mobility, e_c, e_ion, e_pol, "
                "kT, ion_ceiling_eV, pol_ceiling_eV)"
            ),
        },
    ]
