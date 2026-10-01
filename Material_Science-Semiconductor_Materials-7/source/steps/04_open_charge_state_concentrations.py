"""
Compute the equilibrium concentration of every charge state of every defect at a given temperature, Fermi level and dopant chemical potential, for defects that are free to exchange atoms with their surroundings.

In the dilute limit each charge state of a defect is populated in proportion to its site density and to a Boltzmann factor of its formation free energy. That free energy depends on the Fermi level through the defect charge, on the chemical potentials of the atoms exchanged with the reservoirs, and, at finite temperature, on the vibrational contribution of the atoms the defect adds to or removes from the lattice. Concentrations of this kind describe every defect that is still able to equilibrate at the current temperature.

Returns
-------
np.ndarray of float, the dilute-limit equilibrium concentration (cm^-3) of each charge state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def open_charge_state_concentrations(temperature: float, fermi_level: float, mu_dopant: float, host: dict, defects: dict) -> "np.ndarray":
    '''Dilute-limit equilibrium concentrations of all charge states.

    Parameters
    ----------
    temperature : float
        Absolute temperature (K), strictly positive.
    fermi_level : float
        Fermi level (eV) on the fixed energy scale whose zero is the valence-band maximum at
        0 K.
    mu_dopant : float
        Dopant chemical potential (eV) relative to the reference used for the tabulated
        formation energies.
    host : dict
        Host description; only "hw0", the representative vibrational quantum of the lattice
        (eV), is used.
    defects : dict
        Defect table with per-defect arrays of length n_def:
        "site" : site density (cm^-3), shared by all charge states of the defect;
        "n_added" : net number of atoms the defect adds to the crystal (integer);
        "n_dopant" : number of dopant atoms the defect contains (integer, 0 or 1);
        and per-charge-state arrays of length n_cs:
        "cs_def" : index of the defect each charge state belongs to;
        "cs_q" : charge of the state (units of the elementary charge);
        "cs_e" : formation energy (eV) at a Fermi level of zero and dopant chemical potential
        zero, with host chemical potentials already included.
        Other keys are ignored.

    Returns
    -------
    concentrations : np.ndarray
        Float array of length n_cs: concentration (cm^-3) of each charge state, including the
        vibrational free-energy contribution of the source framework for the atoms each defect
        adds or removes.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If temperature is not positive or the per-charge-state arrays have unequal lengths.
    '''
    return concentrations

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_open_charge_state_concentrations(temperature: float, fermi_level: float, mu_dopant: float, host: dict, defects: dict) -> "np.ndarray":
    kb_ev = 8.617333262e-5  # Boltzmann constant (eV/K)
    t = float(temperature)
    if not t > 0.0:
        raise ValueError("temperature must be positive")
    cs_def = np.asarray(defects["cs_def"], dtype=int).ravel()
    cs_q = np.asarray(defects["cs_q"], dtype=float).ravel()
    cs_e = np.asarray(defects["cs_e"], dtype=float).ravel()
    if not (cs_def.size == cs_q.size == cs_e.size):
        raise ValueError("per-charge-state arrays must have equal lengths")
    site = np.asarray(defects["site"], dtype=float).ravel()
    n_dop = np.asarray(defects["n_dopant"], dtype=float).ravel()
    g_vib = _oracle_vibrational_free_energy_shift(t, host["hw0"], defects["n_added"])
    g = cs_e + cs_q * float(fermi_level) - n_dop[cs_def] * float(mu_dopant) + g_vib[cs_def]
    return site[cs_def] * np.exp(-g / (kb_ev * t))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
host = {"hw0": 0.0119}
defects = {
    "site": np.array([1.484e22, 1.484e22, 1.484e22, 1.484e22, 5.936e22]),
    "n_added": np.array([1, -1, 0, 0, 1]),
    "n_dopant": np.array([0, 0, 1, 1, 1]),
    "em": np.array([0.97, 1.43, 1.62, 1.77, 1.86]),
    "cs_def": np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4]),
    "cs_q": np.array([2, 1, 0, 0, -1, -2, 0, -1, 1, 0, 1, 0]),
    "cs_e": np.array([0.29, 1.66, 3.15, 2.61, 2.92, 3.66, 1.21, 1.34, 0.01, 1.30, 0.66, 1.84]),
}
def fresh():
    return {k: v.copy() for k, v in defects.items()}
"""
    return [
        # --- Valid: high temperature, near-midgap Fermi level ---
        {
            "setup": base,
            "call": "np.log10(open_charge_state_concentrations(1081.0, 0.89, -0.35, dict(host), fresh()))",
            "gold_call": "np.log10(_oracle_open_charge_state_concentrations(1081.0, 0.89, -0.35, dict(host), fresh()))",
            "tol": 1e-7,
        },
        # --- Valid: intermediate temperature, p-type Fermi level ---
        {
            "setup": base,
            "call": "np.log10(open_charge_state_concentrations(581.0, 0.43, -0.12, dict(host), fresh()))",
            "gold_call": "np.log10(_oracle_open_charge_state_concentrations(581.0, 0.43, -0.12, dict(host), fresh()))",
            "tol": 1e-7,
        },
        # --- Boundary: room temperature with the Fermi level near the valence band ---
        {
            "setup": base,
            "call": "np.log10(open_charge_state_concentrations(296.0, 0.19, 0.0, dict(host), fresh()))",
            "gold_call": "np.log10(_oracle_open_charge_state_concentrations(296.0, 0.19, 0.0, dict(host), fresh()))",
            "tol": 1e-7,
        },
        # --- Edge: single neutral substitutional (no charge, no vibrational term) ---
        {
            "setup": """import numpy as np
host = {"hw0": 0.0119}
single = {"site": np.array([1.484e22]), "n_added": np.array([0]), "n_dopant": np.array([1]),
          "em": np.array([1.62]), "cs_def": np.array([0]), "cs_q": np.array([0]), "cs_e": np.array([1.21])}
""",
            "call": "np.log10(open_charge_state_concentrations(947.0, 0.8, -0.4, dict(host), {k: v.copy() for k, v in single.items()}))",
            "gold_call": "np.log10(_oracle_open_charge_state_concentrations(947.0, 0.8, -0.4, dict(host), {k: v.copy() for k, v in single.items()}))",
            "tol": 1e-7,
        },
    ]
