"""
Rigidly shift zero-gate opposite-spin edges by the effective gate voltage.

An electrostatic gate moves the channel band edges relative to the electrode Fermi level; for this compact instance the shift is rigid with no additional density-of-states lever-arm iteration.

Returns
-------
np.ndarray shape (2,) — [E_vbm, E_cbm] in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gated_band_edges(E_vbm0: float, E_cbm0: float, V_g_eff: float) -> "np.ndarray":
    """Return [E_vbm0 - V_g_eff, E_cbm0 - V_g_eff] in eV.

    Raises
    ------
    ValueError
        If any input is non-finite.
    """
    return [0.0, 0.0]

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_gated_band_edges(E_vbm0: float, E_cbm0: float, V_g_eff: float) -> "np.ndarray":
    E_vbm0 = float(E_vbm0)
    E_cbm0 = float(E_cbm0)
    V_g_eff = float(V_g_eff)
    if not all(np.isfinite(v) for v in (E_vbm0, E_cbm0, V_g_eff)):
        raise ValueError("All inputs must be finite.")
    return np.array([E_vbm0 - V_g_eff, E_cbm0 - V_g_eff], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "gated_band_edges(0.05, 0.58, 0.1)",
            "gold_call": "_oracle_gated_band_edges(0.05, 0.58, 0.1)",
        },
        {
            "setup": "import numpy as np",
            "call": "gated_band_edges(0.05, 0.58, 0.0)",
            "gold_call": "_oracle_gated_band_edges(0.05, 0.58, 0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "gated_band_edges(0.05, 0.58, -0.1)",
            "gold_call": "_oracle_gated_band_edges(0.05, 0.58, -0.1)",
        },
    ]
