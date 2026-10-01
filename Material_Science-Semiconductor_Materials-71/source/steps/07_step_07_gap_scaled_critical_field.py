"""
Map band gap onto a critical breakdown field.

Blocking capability of unipolar power devices rises superlinearly with gap, so gap enters materials ranking through the breakdown field.

Returns
-------
float, critical field in MV/cm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gap_scaled_critical_field(
    band_gap_eV: float,
    e_c_ref: float = 1.55,
    e_g_ref: float = 6.2,
    exponent: float = 2.5,
) -> float:
    """Return gap-scaled critical field in MV/cm.

    Raises
    ------
    ValueError
        If band_gap_eV or reference parameters are non-positive.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_gap_scaled_critical_field(
    band_gap_eV: float,
    e_c_ref: float = 1.55,
    e_g_ref: float = 6.2,
    exponent: float = 2.5,
) -> float:
    if band_gap_eV <= 0:
        raise ValueError("band_gap_eV must be positive")
    if e_c_ref <= 0 or e_g_ref <= 0:
        raise ValueError("reference parameters must be positive")
    return float(e_c_ref * (band_gap_eV / e_g_ref) ** exponent)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "band_gap_eV = 8.40\ne_c_ref = 1.55\ne_g_ref = 6.2\nexponent = 2.5\n",
            "call": "gap_scaled_critical_field(band_gap_eV, e_c_ref, e_g_ref, exponent)",
            "gold_call": "_oracle_gap_scaled_critical_field(band_gap_eV, e_c_ref, e_g_ref, exponent)",
        },
        {
            "setup": "band_gap_eV = 6.2\ne_c_ref = 1.55\ne_g_ref = 6.2\nexponent = 2.5\n",
            "call": "gap_scaled_critical_field(band_gap_eV, e_c_ref, e_g_ref, exponent)",
            "gold_call": "_oracle_gap_scaled_critical_field(band_gap_eV, e_c_ref, e_g_ref, exponent)",
        },
        {
            "setup": "band_gap_eV = 1.1\ne_c_ref = 1.55\ne_g_ref = 6.2\nexponent = 2.5\n",
            "call": "gap_scaled_critical_field(band_gap_eV, e_c_ref, e_g_ref, exponent)",
            "gold_call": "_oracle_gap_scaled_critical_field(band_gap_eV, e_c_ref, e_g_ref, exponent)",
        },
    ]
