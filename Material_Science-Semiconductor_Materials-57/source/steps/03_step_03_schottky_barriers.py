"""
Convert zero-gate opposite-spin edges into hole and electron Schottky barrier heights.

Relative alignment of the electrode Fermi level with the channel valence and conduction edges decides whether each carrier type sees an Ohmic contact or a finite Schottky barrier.

Barrier heights enter the orchestrator contact-selection decision that chooses which spin current is scored.

Returns
-------
np.ndarray shape (2,) — [phi_p, phi_n] in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def schottky_barriers(E_vbm_up: float, E_cbm_down: float) -> "np.ndarray":
    """Return [phi_p, phi_n], the hole and electron barriers in eV.

    The spin-up valence and spin-down conduction edge energies are in eV
    relative to EF = 0. An Ohmic alignment has zero injection barrier.

    Raises
    ------
    ValueError
        If either edge energy is non-finite.
    """
    return [0.0, 0.0]

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_schottky_barriers(E_vbm_up: float, E_cbm_down: float) -> "np.ndarray":
    E_vbm_up = float(E_vbm_up)
    E_cbm_down = float(E_cbm_down)
    if not np.isfinite(E_vbm_up) or not np.isfinite(E_cbm_down):
        raise ValueError("Edge energies must be finite.")
    phi_p = max(0.0, -E_vbm_up)
    phi_n = max(0.0, E_cbm_down)
    return np.array([phi_p, phi_n], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "schottky_barriers(0.05, 0.58)",
            "gold_call": "_oracle_schottky_barriers(0.05, 0.58)",
        },
        {
            "setup": "import numpy as np",
            "call": "schottky_barriers(-0.2, -0.1)",
            "gold_call": "_oracle_schottky_barriers(-0.2, -0.1)",
        },
        {
            "setup": "import numpy as np",
            "call": "schottky_barriers(0.0, 0.0)",
            "gold_call": "_oracle_schottky_barriers(0.0, 0.0)",
        },
    ]
