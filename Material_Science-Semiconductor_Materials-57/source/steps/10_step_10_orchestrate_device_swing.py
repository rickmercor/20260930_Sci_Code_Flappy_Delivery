"""
Assemble the full paper-equation pipeline and return the selected-spin subthreshold swing between two gate points for the deterministic instance.

Lattice mismatch, binding energy, and Schottky barriers select the scored spin channel for a stable vdW interface. Source-referenced terminals, rigid gate shifts, opposite-spin threshold spectra, Landauer currents, SFE branch check, and decade swing then yield the device metric.

Returns
-------
float — positive SS magnitude in mV/dec
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    """Return the positive selected-spin subthreshold-swing magnitude in mV/dec.

    Assembled by calling the earlier sub-problem functions:
    lattice_mismatch, binding_energy, schottky_barriers, terminal_potentials,
    gated_band_edges, threshold_spectra, landauer_spin_currents,
    spin_filtering_efficiency, and decade_swing.

    Raises
    ------
    ValueError
        If any intermediate step raises, the interface/contact selection
        rejects the instance, or SFE does not match the scored branch.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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

def _oracle_orchestrate_device_swing(
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
    delta = _oracle_lattice_mismatch(a_electrode, n_electrode, a_channel, n_channel)
    eb = _oracle_binding_energy(E_het, E_electrode, E_channel, E_cp, area_A2)
    phi = _oracle_schottky_barriers(E_vbm_up, E_cbm_down)
    spin_idx = _select_scored_spin(delta, eb, float(phi[0]), float(phi[1]))

    E = np.linspace(float(E_min), float(E_max), int(n_energy))

    def _scored_spin_current(V_g: float):
        pots = _oracle_terminal_potentials(V_b, V_g)
        edges = _oracle_gated_band_edges(E_vbm_up, E_cbm_down, pots[2])
        Tsp = _oracle_threshold_spectra(E, edges[0], edges[1])
        I = _oracle_landauer_spin_currents(
            E, Tsp, pots[0], pots[1], temperature_K, I0
        )
        sfe = _oracle_spin_filtering_efficiency(I[0], I[1])
        return float(I[spin_idx]), float(sfe)

    I1, sfe1 = _scored_spin_current(V_g1)
    I2, sfe2 = _scored_spin_current(V_g2)
    if spin_idx == 0:
        if sfe1 <= 0.0 or sfe2 <= 0.0:
            raise ValueError("SFE must be positive on the scored spin-up flank.")
    else:
        if sfe1 >= 0.0 or sfe2 >= 0.0:
            raise ValueError("SFE must be negative on the scored spin-down flank.")
    return float(_oracle_decade_swing(V_g1, V_g2, I1, I2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "orchestrate_device_swing()",
            "gold_call": "_oracle_orchestrate_device_swing()",
        },
        {
            "setup": "import numpy as np",
            "call": "orchestrate_device_swing(V_g1=0.18, V_g2=0.30)",
            "gold_call": "_oracle_orchestrate_device_swing(V_g1=0.18, V_g2=0.30)",
        },
        {
            "setup": "import numpy as np",
            "call": "orchestrate_device_swing(temperature_K=200.0)",
            "gold_call": "_oracle_orchestrate_device_swing(temperature_K=200.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "orchestrate_device_swing(E_vbm_up=0.0)",
            "gold_call": "_oracle_orchestrate_device_swing(E_vbm_up=0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "orchestrate_device_swing(E_vbm_up=-0.58, E_cbm_down=-0.05, V_b=0.2, V_g1=-0.32, V_g2=-0.20)",
            "gold_call": "_oracle_orchestrate_device_swing(E_vbm_up=-0.58, E_cbm_down=-0.05, V_b=0.2, V_g1=-0.32, V_g2=-0.20)",
        },
    ]
