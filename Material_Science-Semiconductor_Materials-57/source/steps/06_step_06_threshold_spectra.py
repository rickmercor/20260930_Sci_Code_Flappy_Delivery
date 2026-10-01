"""
Build opposite-spin edge-threshold transmission spectra on a fixed energy grid.

For this compact deterministic instance, spin-up transport opens at and below the valence edge while spin-down opens at and above the conduction edge, encoding opposite-spin band-edge character without a full NEGF rerun.

Returns
-------
np.ndarray shape (2, N) — [T_up; T_down]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def threshold_spectra(
    energies_eV: "np.ndarray",
    E_vbm: float,
    E_cbm: float,
) -> "np.ndarray":
    """Return stacked spectra T_up=1[E<=E_vbm], T_down=1[E>=E_cbm].

    Raises
    ------
    ValueError
        If the energy grid is not 1-D with at least two finite strictly
        increasing points, or either edge is non-finite.
    """
    return [[0.0], [0.0]]

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_threshold_spectra(
    energies_eV: "np.ndarray",
    E_vbm: float,
    E_cbm: float,
) -> "np.ndarray":
    E = np.asarray(energies_eV, dtype=float)
    if E.ndim != 1 or E.size < 2:
        raise ValueError("energies_eV must be 1-D with >= 2 points.")
    if not np.all(np.isfinite(E)):
        raise ValueError("energies_eV must be finite.")
    if np.any(np.diff(E) <= 0.0):
        raise ValueError("energies_eV must be strictly increasing.")
    E_vbm = float(E_vbm)
    E_cbm = float(E_cbm)
    if not np.isfinite(E_vbm) or not np.isfinite(E_cbm):
        raise ValueError("Edge energies must be finite.")
    T_up = (E <= E_vbm).astype(float)
    T_dn = (E >= E_cbm).astype(float)
    return np.vstack([T_up, T_dn])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np; E=np.linspace(-1.0,1.0,21)",
            "call": "threshold_spectra(E.copy(), 0.05, 0.58)",
            "gold_call": "_oracle_threshold_spectra(E.copy(), 0.05, 0.58)",
        },
        {
            "setup": "import numpy as np",
            "call": "threshold_spectra(np.linspace(-0.5,0.5,11), -0.4, 0.4)",
            "gold_call": "_oracle_threshold_spectra(np.linspace(-0.5,0.5,11), -0.4, 0.4)",
        },
        {
            "setup": "import numpy as np",
            "call": "threshold_spectra(np.array([-1.0,0.0,1.0]), 0.0, 0.0)",
            "gold_call": "_oracle_threshold_spectra(np.array([-1.0,0.0,1.0]), 0.0, 0.0)",
        },
    ]
