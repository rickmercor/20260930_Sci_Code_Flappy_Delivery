"""
Build the effective dielectric from ionic and electronic responses.

Continuum carrier self-trapping depends on how much lattice polarization remains after electronic screening is removed.

Returns
-------
float, effective dielectric constant (dimensionless)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ionic_electronic_dielectric(epsilon_s: float, epsilon_inf: float) -> float:
    """Return the effective dielectric constant from epsilon_s and epsilon_inf.

    Raises
    ------
    ValueError
        If permittivities are non-positive or epsilon_s does not exceed epsilon_inf.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_ionic_electronic_dielectric(epsilon_s: float, epsilon_inf: float) -> float:
    if epsilon_s <= 0 or epsilon_inf <= 0:
        raise ValueError("permittivities must be positive")
    if epsilon_s <= epsilon_inf:
        raise ValueError("epsilon_s must exceed epsilon_inf")
    inv = (1.0 / epsilon_inf) - (1.0 / epsilon_s)
    if inv <= 0:
        raise ValueError("effective dielectric inverse must be positive")
    return float(1.0 / inv)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "epsilon_s = 12.4\nepsilon_inf = 5.1\n",
            "call": "ionic_electronic_dielectric(epsilon_s, epsilon_inf)",
            "gold_call": "_oracle_ionic_electronic_dielectric(epsilon_s, epsilon_inf)",
        },
        {
            "setup": "epsilon_s = 9.28\nepsilon_inf = 3.15\n",
            "call": "ionic_electronic_dielectric(epsilon_s, epsilon_inf)",
            "gold_call": "_oracle_ionic_electronic_dielectric(epsilon_s, epsilon_inf)",
        },
        {
            "setup": "epsilon_s = 6.67\nepsilon_inf = 3.14\n",
            "call": "ionic_electronic_dielectric(epsilon_s, epsilon_inf)",
            "gold_call": "_oracle_ionic_electronic_dielectric(epsilon_s, epsilon_inf)",
        },
    ]
