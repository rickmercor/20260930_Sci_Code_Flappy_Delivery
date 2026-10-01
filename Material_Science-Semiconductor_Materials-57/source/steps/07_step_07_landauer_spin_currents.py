"""
Integrate spin-resolved Landauer–Büttiker currents from spectra and terminal chemical potentials.

Finite bias opens an energy window between source and drain occupations; each spin channel contributes current only where its transmission overlaps that window.

Returns
-------
np.ndarray shape (2,) — [I_up, I_down] in µA/µm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def landauer_spin_currents(
    energies_eV: "np.ndarray",
    T_spin: "np.ndarray",
    mu_s: float,
    mu_d: float,
    temperature_K: float,
    I0: float,
) -> "np.ndarray":
    """Return signed [I_up, I_down] in µA/µm.

    Raises
    ------
    ValueError
        If the energy grid is invalid, T_spin shape mismatches, temperature
        is not positive, or inputs are non-finite.
    """
    return [0.0, 0.0]

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _fermi_dirac(E, mu, temperature_K):
    kb_ev = 8.617333262145e-5
    x = (E - mu) / (kb_ev * temperature_K)
    out = np.empty_like(x, dtype=float)
    out[x > 40.0] = 0.0
    out[x < -40.0] = 1.0
    mid = (x >= -40.0) & (x <= 40.0)
    out[mid] = 1.0 / (1.0 + np.exp(x[mid]))
    return out

def _oracle_landauer_spin_currents(
    energies_eV: "np.ndarray",
    T_spin: "np.ndarray",
    mu_s: float,
    mu_d: float,
    temperature_K: float,
    I0: float,
) -> "np.ndarray":
    E = np.asarray(energies_eV, dtype=float)
    T = np.asarray(T_spin, dtype=float)
    mu_s = float(mu_s)
    mu_d = float(mu_d)
    temperature_K = float(temperature_K)
    I0 = float(I0)
    if E.ndim != 1 or E.size < 2:
        raise ValueError("energies_eV must be 1-D with >= 2 points.")
    if T.shape != (2, E.size):
        raise ValueError("T_spin must have shape (2, N) matching energies_eV.")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be > 0.")
    if not np.all(np.isfinite(E)) or not np.all(np.isfinite(T)):
        raise ValueError("energies and spectra must be finite.")
    if np.any(np.diff(E) <= 0.0):
        raise ValueError("energies_eV must be strictly increasing.")
    df = _fermi_dirac(E, mu_d, temperature_K) - _fermi_dirac(E, mu_s, temperature_K)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    I_up = I0 * trapz(T[0] * df, E)
    I_dn = I0 * trapz(T[1] * df, E)
    return np.array([I_up, I_dn], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np; "
                "E=np.linspace(-1.5,1.5,401); "
                "T=np.vstack([(E<=0.05).astype(float),(E>=0.58).astype(float)])"
            ),
            "call": "landauer_spin_currents(E.copy(), T.copy(), 0.1, -0.1, 300.0, 4300.0)",
            "gold_call": "_oracle_landauer_spin_currents(E.copy(), T.copy(), 0.1, -0.1, 300.0, 4300.0)",
        },
        {
            "setup": (
                "import numpy as np; "
                "E=np.linspace(-1.0,1.0,201); "
                "T=np.vstack([np.ones_like(E), np.ones_like(E)])"
            ),
            "call": "landauer_spin_currents(E.copy(), T.copy(), 0.0, 0.0, 300.0, 4300.0)",
            "gold_call": "_oracle_landauer_spin_currents(E.copy(), T.copy(), 0.0, 0.0, 300.0, 4300.0)",
        },
        {
            "setup": (
                "import numpy as np; "
                "E=np.linspace(-1.5,1.5,401); "
                "T=np.vstack([np.zeros_like(E),(E>=0.05).astype(float)])"
            ),
            "call": "landauer_spin_currents(E.copy(), T.copy(), -0.1, 0.1, 300.0, 4300.0)",
            "gold_call": "_oracle_landauer_spin_currents(E.copy(), T.copy(), -0.1, 0.1, 300.0, 4300.0)",
        },
    ]
