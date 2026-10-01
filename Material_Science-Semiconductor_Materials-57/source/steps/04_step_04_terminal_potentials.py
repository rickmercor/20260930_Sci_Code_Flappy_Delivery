"""
Map source-referenced drain and gate voltages onto chemical potentials and an effective gate.

Source-referenced drain and gate voltages must be expressed in the device-Fermi reference before the terminal occupation window and gated channel edges can be combined in the transport calculation.

Returns
-------
np.ndarray shape (3,) — [mu_s, mu_d, V_g_eff]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def terminal_potentials(V_b: float, V_g: float) -> "np.ndarray":
    """Return [mu_s, mu_d, V_g_eff] with mu in eV and V_g_eff in V.

    Raises
    ------
    ValueError
        If V_b or V_g is non-finite.
    """
    return [0.0, 0.0, 0.0]

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_terminal_potentials(V_b: float, V_g: float) -> "np.ndarray":
    V_b = float(V_b)
    V_g = float(V_g)
    if not np.isfinite(V_b) or not np.isfinite(V_g):
        raise ValueError("V_b and V_g must be finite.")
    return np.array([-0.5 * V_b, 0.5 * V_b, V_g + 0.5 * V_b], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "terminal_potentials(-0.2, 0.0)",
            "gold_call": "_oracle_terminal_potentials(-0.2, 0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "terminal_potentials(0.2, 0.0)",
            "gold_call": "_oracle_terminal_potentials(0.2, 0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "terminal_potentials(-0.2, 0.75)",
            "gold_call": "_oracle_terminal_potentials(-0.2, 0.75)",
        },
    ]
