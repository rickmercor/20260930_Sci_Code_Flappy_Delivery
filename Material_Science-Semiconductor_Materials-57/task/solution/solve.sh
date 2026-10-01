#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

def lattice_mismatch(
    a_electrode: float,
    n_electrode: int,
    a_channel: float,
    n_channel: int,
) -> float:
    a_electrode = float(a_electrode)
    a_channel = float(a_channel)
    n_electrode = int(n_electrode)
    n_channel = int(n_channel)
    if a_electrode <= 0.0 or a_channel <= 0.0:
        raise ValueError("Lattice constants must be positive.")
    if n_electrode < 1 or n_channel < 1:
        raise ValueError("Supercell repeats must be >= 1.")
    if not all(np.isfinite(v) for v in (a_electrode, a_channel)):
        raise ValueError("Lattice constants must be finite.")
    num = abs(n_electrode * a_electrode - n_channel * a_channel)
    den = n_channel * a_channel
    return float(100.0 * num / den)

import numpy as np

def binding_energy(
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

import numpy as np

def schottky_barriers(E_vbm_up: float, E_cbm_down: float) -> "np.ndarray":
    E_vbm_up = float(E_vbm_up)
    E_cbm_down = float(E_cbm_down)
    if not np.isfinite(E_vbm_up) or not np.isfinite(E_cbm_down):
        raise ValueError("Edge energies must be finite.")
    phi_p = max(0.0, -E_vbm_up)
    phi_n = max(0.0, E_cbm_down)
    return np.array([phi_p, phi_n], dtype=float)

import numpy as np

def terminal_potentials(V_b: float, V_g: float) -> "np.ndarray":
    V_b = float(V_b)
    V_g = float(V_g)
    if not np.isfinite(V_b) or not np.isfinite(V_g):
        raise ValueError("V_b and V_g must be finite.")
    return np.array([-0.5 * V_b, 0.5 * V_b, V_g + 0.5 * V_b], dtype=float)

import numpy as np

def gated_band_edges(E_vbm0: float, E_cbm0: float, V_g_eff: float) -> "np.ndarray":
    E_vbm0 = float(E_vbm0)
    E_cbm0 = float(E_cbm0)
    V_g_eff = float(V_g_eff)
    if not all(np.isfinite(v) for v in (E_vbm0, E_cbm0, V_g_eff)):
        raise ValueError("All inputs must be finite.")
    return np.array([E_vbm0 - V_g_eff, E_cbm0 - V_g_eff], dtype=float)

import numpy as np

def threshold_spectra(
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

def landauer_spin_currents(
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

import numpy as np

def spin_filtering_efficiency(I_up: float, I_down: float) -> float:
    I_up = float(I_up)
    I_down = float(I_down)
    if not np.isfinite(I_up) or not np.isfinite(I_down):
        raise ValueError("Currents must be finite.")
    denom = I_up + I_down
    if denom == 0.0:
        raise ValueError("Total I_up + I_down must be nonzero.")
    return float((I_up - I_down) / denom * 100.0)

import numpy as np

def decade_swing(V_g1: float, V_g2: float, I1: float, I2: float) -> float:
    V_g1 = float(V_g1)
    V_g2 = float(V_g2)
    I1 = float(I1)
    I2 = float(I2)
    if not all(np.isfinite(v) for v in (V_g1, V_g2, I1, I2)):
        raise ValueError("All inputs must be finite.")
    if V_g1 == V_g2:
        raise ValueError("V_g1 and V_g2 must differ.")
    if I1 == 0.0 or I2 == 0.0:
        raise ValueError("Currents must be nonzero.")
    dlog = np.log10(abs(I1)) - np.log10(abs(I2))
    if dlog == 0.0:
        raise ValueError("log10|I1| and log10|I2| must differ.")
    return float(abs(V_g2 - V_g1) / abs(dlog) * 1000.0)

import numpy as np

def _select_scored_spin(
    delta_pct: float,
    eb_meV_A2: float,
    phi_p: float,
    phi_n: float,
) -> int:
    """Return 0 (spin-up) or 1 (spin-down) from compact admissibility and contact type.

    The task-defined compact admissibility conditions require negative
    binding energy and lattice mismatch no larger than 2.1%. A p-type Ohmic
    hole contact (phi_p = 0, phi_n > 0) injects the spin-up valence channel;
    an n-type Ohmic electron contact injects spin-down.
    """
    delta_pct = float(delta_pct)
    eb_meV_A2 = float(eb_meV_A2)
    phi_p = float(phi_p)
    phi_n = float(phi_n)
    if not (eb_meV_A2 < 0.0 and delta_pct <= 2.1):
        raise ValueError("Interface not valid for compact spin-FET model.")
    if phi_p <= 0.0 and phi_n > 0.0:
        return 0
    if phi_n <= 0.0 and phi_p > 0.0:
        return 1
    raise ValueError("No Ohmic spin channel for scored swing.")

def orchestrate_device_swing(
    a_electrode: float = 4.127,
    n_electrode: int = 1,
    a_channel: float = 4.19,
    n_channel: int = 1,
    E_het: float = -180.243,
    E_electrode: float = -100.0,
    E_channel: float = -80.0,
    E_cp: float = 0.02,
    area_A2: float = 15.21,
    E_vbm_up: float = 0.05,
    E_cbm_down: float = 0.58,
    V_b: float = -0.2,
    temperature_K: float = 300.0,
    I0: float = 4300.0,
    E_min: float = -1.5,
    E_max: float = 1.5,
    n_energy: int = 4001,
    V_g1: float = 0.20,
    V_g2: float = 0.32,
) -> float:
    delta = lattice_mismatch(a_electrode, n_electrode, a_channel, n_channel)
    eb = binding_energy(E_het, E_electrode, E_channel, E_cp, area_A2)
    phi = schottky_barriers(E_vbm_up, E_cbm_down)
    spin_idx = _select_scored_spin(delta, eb, float(phi[0]), float(phi[1]))

    E = np.linspace(float(E_min), float(E_max), int(n_energy))

    def _scored_spin_current(V_g: float):
        pots = terminal_potentials(V_b, V_g)
        edges = gated_band_edges(E_vbm_up, E_cbm_down, pots[2])
        Tsp = threshold_spectra(E, edges[0], edges[1])
        I = landauer_spin_currents(
            E, Tsp, pots[0], pots[1], temperature_K, I0
        )
        sfe = spin_filtering_efficiency(I[0], I[1])
        return float(I[spin_idx]), float(sfe)

    I1, sfe1 = _scored_spin_current(V_g1)
    I2, sfe2 = _scored_spin_current(V_g2)
    if spin_idx == 0:
        if sfe1 <= 0.0 or sfe2 <= 0.0:
            raise ValueError("SFE must be positive on the scored spin-up flank.")
    else:
        if sfe1 >= 0.0 or sfe2 >= 0.0:
            raise ValueError("SFE must be negative on the scored spin-down flank.")
    return float(decade_swing(V_g1, V_g2, I1, I2))
SCICODE_GOLD_EOF
