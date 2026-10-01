"""
Evaluate the BSSE-corrected interfacial binding energy per area.

Vertical contact stability is quantified by the heterostructure binding energy per interfacial area, including a counterpoise correction when an LCAO basis is used.

The binding energy enters the orchestrator contact-selection gate (stable vdW contacts require Eb < 0).

Returns
-------
float — Eb in meV/Å²
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def binding_energy(
    E_het: float,
    E_electrode: float,
    E_channel: float,
    E_cp: float,
    area_A2: float,
) -> float:
    """Return Eb in meV/Å².

    Energies are in eV and area in Å². The returned value is in meV/Å².

    Raises
    ------
    ValueError
        If area_A2 is not positive or any input is non-finite.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_binding_energy(
    E_het: float,
    E_electrode: float,
    E_channel: float,
    E_cp: float,
    area_A2: float,
) -> float:
    vals = [E_het, E_electrode, E_channel, E_cp, area_A2]
    if any(not np.isfinite(float(v)) for v in vals):
        raise ValueError("All inputs must be finite.")
    area_A2 = float(area_A2)
    if area_A2 <= 0.0:
        raise ValueError("area_A2 must be positive.")
    eb_eV = (
        float(E_het) - float(E_electrode) - float(E_channel) + float(E_cp)
    ) / area_A2
    return float(eb_eV * 1000.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "binding_energy(-180.243, -100.0, -80.0, 0.02, 15.21)",
            "gold_call": "_oracle_binding_energy(-180.243, -100.0, -80.0, 0.02, 15.21)",
        },
        {
            "setup": "import numpy as np",
            "call": "binding_energy(-100.0, -50.0, -50.0, 0.0, 10.0)",
            "gold_call": "_oracle_binding_energy(-100.0, -50.0, -50.0, 0.0, 10.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "binding_energy(-10.0, -4.0, -5.0, -0.1, 2.0)",
            "gold_call": "_oracle_binding_energy(-10.0, -4.0, -5.0, -0.1, 2.0)",
        },
    ]
