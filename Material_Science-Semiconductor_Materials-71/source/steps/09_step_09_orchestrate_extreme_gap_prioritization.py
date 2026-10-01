"""
Run the full continuum prioritization pipeline end to end.

Ranking an extreme-gap oxide requires chaining band-derived

masses through ionization, localization, mobility, breakdown field, and gated scoring

into one comparable prioritization number.

Returns
-------
float, prioritization score P from the full pipeline
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orchestrate_extreme_gap_prioritization(
    k_values: "np.ndarray",
    energy_values: "np.ndarray",
    alpha: float,
    epsilon_s: float,
    epsilon_inf: float,
    band_gap_eV: float,
    tau: float,
    kT: float = 0.025,
    hbar2_over_me: float = 7.6199642,
    rydberg_eV: float = 13.6,
    e_ha: float = 27.2,
    elementary_charge: float = 1.602176634e-19,
    electron_mass_kg: float = 9.1093837015e-31,
    e_c_ref: float = 1.55,
    e_g_ref: float = 6.2,
    exponent: float = 2.5,
    ion_ceiling_eV: float = 0.21,
    pol_ceiling_eV: float = 0.025,
) -> float:
    """Return P assembled by calling the earlier sub-problem functions.

    Raises
    ------
    ValueError
        Propagated from invalid upstream inputs.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_orchestrate_extreme_gap_prioritization(
    k_values: "np.ndarray",
    energy_values: "np.ndarray",
    alpha: float,
    epsilon_s: float,
    epsilon_inf: float,
    band_gap_eV: float,
    tau: float,
    kT: float = 0.025,
    hbar2_over_me: float = 7.6199642,
    rydberg_eV: float = 13.6,
    e_ha: float = 27.2,
    elementary_charge: float = 1.602176634e-19,
    electron_mass_kg: float = 9.1093837015e-31,
    e_c_ref: float = 1.55,
    e_g_ref: float = 6.2,
    exponent: float = 2.5,
    ion_ceiling_eV: float = 0.21,
    pol_ceiling_eV: float = 0.025,
) -> float:
    masses = _oracle_directional_band_masses(
        k_values, energy_values, alpha, hbar2_over_me
    )
    m_dos = _oracle_dos_geometric_mass(masses)
    e_ion = _oracle_continuum_ionization_energy(m_dos, epsilon_s, rydberg_eV)
    eps_eff = _oracle_ionic_electronic_dielectric(epsilon_s, epsilon_inf)
    e_pol = _oracle_continuum_polaron_energy(m_dos, eps_eff, e_ha)
    mu = _oracle_scattering_time_mobility(
        m_dos, tau, elementary_charge, electron_mass_kg
    )
    e_c = _oracle_gap_scaled_critical_field(band_gap_eV, e_c_ref, e_g_ref, exponent)
    return _oracle_prioritization_score(
        epsilon_s,
        mu,
        e_c,
        e_ion,
        e_pol,
        kT,
        ion_ceiling_eV,
        pol_ceiling_eV,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "k_values = np.array([0.048, 0.052, 0.045])\n"
                "energy_values = np.array(["
                "0.0393135912340984, 0.024891847042767198, 0.02465676303235815])\n"
                "alpha = 0.38\nepsilon_s = 12.4\nepsilon_inf = 5.1\n"
                "band_gap_eV = 8.40\ntau = 1.4e-14\n"
            ),
            "call": (
                "orchestrate_extreme_gap_prioritization(k_values, energy_values, alpha, "
                "epsilon_s, epsilon_inf, band_gap_eV, tau)"
            ),
            "gold_call": (
                "_oracle_orchestrate_extreme_gap_prioritization(k_values, energy_values, alpha, "
                "epsilon_s, epsilon_inf, band_gap_eV, tau)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "k_values = np.array([0.048, 0.052, 0.045])\n"
                "energy_values = np.array(["
                "0.0393135912340984, 0.024891847042767198, 0.02465676303235815])\n"
                "alpha = 0.38\nepsilon_s = 12.4\nepsilon_inf = 5.1\n"
                "band_gap_eV = 6.2\ntau = 1.4e-14\n"
            ),
            "call": (
                "orchestrate_extreme_gap_prioritization(k_values, energy_values, alpha, "
                "epsilon_s, epsilon_inf, band_gap_eV, tau)"
            ),
            "gold_call": (
                "_oracle_orchestrate_extreme_gap_prioritization(k_values, energy_values, alpha, "
                "epsilon_s, epsilon_inf, band_gap_eV, tau)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "k_values = np.array([0.048, 0.052, 0.045])\n"
                "energy_values = np.array(["
                "0.0393135912340984, 0.024891847042767198, 0.02465676303235815])\n"
                "alpha = 0.38\nepsilon_s = 12.4\nepsilon_inf = 5.1\n"
                "band_gap_eV = 8.40\ntau = 1.4e-14\npol_ceiling_eV = 0.01\n"
            ),
            "call": (
                "orchestrate_extreme_gap_prioritization(k_values, energy_values, alpha, "
                "epsilon_s, epsilon_inf, band_gap_eV, tau, pol_ceiling_eV=pol_ceiling_eV)"
            ),
            "gold_call": (
                "_oracle_orchestrate_extreme_gap_prioritization(k_values, energy_values, alpha, "
                "epsilon_s, epsilon_inf, band_gap_eV, tau, pol_ceiling_eV=pol_ceiling_eV)"
            ),
        },
    ]
