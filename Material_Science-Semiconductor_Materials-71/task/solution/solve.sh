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


def directional_band_masses(
    k_values: "np.ndarray",
    energy_values: "np.ndarray",
    alpha: float,
    hbar2_over_me: float = 7.6199642,
) -> "np.ndarray":
    k = np.asarray(k_values, dtype=float)
    e = np.asarray(energy_values, dtype=float)
    if k.ndim != 1 or e.ndim != 1 or k.shape != e.shape:
        raise ValueError("k_values and energy_values must be 1D arrays of equal length")
    if k.size == 0:
        raise ValueError("at least one band sample is required")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if hbar2_over_me <= 0:
        raise ValueError("hbar2_over_me must be positive")
    if np.any(k <= 0):
        raise ValueError("all k_values must be positive")
    if np.any(e <= 0):
        raise ValueError("all energy_values must be positive")
    denom = (2.0 * alpha * e + 1.0) ** 2 - 1.0
    if np.any(denom <= 0):
        raise ValueError("inversion denominator must be positive")
    return (2.0 * alpha * hbar2_over_me * k**2) / denom

import numpy as np
def dos_geometric_mass(directional_masses: "np.ndarray") -> float:
    masses = np.asarray(directional_masses, dtype=float)
    if masses.ndim != 1 or masses.size == 0:
        raise ValueError("directional_masses must be a non-empty 1D array")
    if np.any(masses <= 0):
        raise ValueError("all directional masses must be positive")
    return float(np.prod(masses) ** (1.0 / masses.size))

import numpy as np

def continuum_ionization_energy(
    m_star_over_me: float,
    epsilon_s: float,
    rydberg_eV: float = 13.6,
) -> float:
    if m_star_over_me <= 0:
        raise ValueError("m_star_over_me must be positive")
    if epsilon_s <= 0:
        raise ValueError("epsilon_s must be positive")
    if rydberg_eV <= 0:
        raise ValueError("rydberg_eV must be positive")
    return float(rydberg_eV * m_star_over_me / (epsilon_s**2))

import numpy as np

def ionic_electronic_dielectric(epsilon_s: float, epsilon_inf: float) -> float:
    if epsilon_s <= 0 or epsilon_inf <= 0:
        raise ValueError("permittivities must be positive")
    if epsilon_s <= epsilon_inf:
        raise ValueError("epsilon_s must exceed epsilon_inf")
    inv = (1.0 / epsilon_inf) - (1.0 / epsilon_s)
    if inv <= 0:
        raise ValueError("effective dielectric inverse must be positive")
    return float(1.0 / inv)

import numpy as np
import math
def continuum_polaron_energy(
    m_star_over_me: float,
    epsilon_eff: float,
    e_ha: float = 27.2,
) -> float:
    if m_star_over_me <= 0:
        raise ValueError("m_star_over_me must be positive")
    if epsilon_eff <= 0:
        raise ValueError("epsilon_eff must be positive")
    if e_ha <= 0:
        raise ValueError("e_ha must be positive")
    return float(-(1.0 / (3.0 * math.pi)) * m_star_over_me / (epsilon_eff**2) * e_ha)

import numpy as np
def scattering_time_mobility(
    m_star_over_me: float,
    tau: float,
    elementary_charge: float = 1.602176634e-19,
    electron_mass_kg: float = 9.1093837015e-31,
) -> float:
    if m_star_over_me <= 0:
        raise ValueError("m_star_over_me must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if elementary_charge <= 0 or electron_mass_kg <= 0:
        raise ValueError("physical constants must be positive")
    mu_si = elementary_charge * tau / (m_star_over_me * electron_mass_kg)
    return float(mu_si * 1.0e4)

import numpy as np

def gap_scaled_critical_field(
    band_gap_eV: float,
    e_c_ref: float = 1.55,
    e_g_ref: float = 6.2,
    exponent: float = 2.5,
) -> float:
    if band_gap_eV <= 0:
        raise ValueError("band_gap_eV must be positive")
    if e_c_ref <= 0 or e_g_ref <= 0:
        raise ValueError("reference parameters must be positive")
    return float(e_c_ref * (band_gap_eV / e_g_ref) ** exponent)

import numpy as np
def prioritization_score(
    epsilon_s: float,
    mobility: float,
    e_c: float,
    e_ion: float,
    e_pol: float,
    kT: float = 0.025,
    ion_ceiling_eV: float = 0.21,
    pol_ceiling_eV: float = 0.025,
) -> float:
    import math

    if epsilon_s <= 0:
        raise ValueError("epsilon_s must be positive")
    if mobility < 0:
        raise ValueError("mobility must be non-negative")
    if e_c <= 0:
        raise ValueError("e_c must be positive")
    if kT <= 0:
        raise ValueError("kT must be positive")
    if ion_ceiling_eV <= 0 or pol_ceiling_eV <= 0:
        raise ValueError("ceilings must be positive")
    if e_ion >= ion_ceiling_eV:
        return 0.0
    if abs(e_pol) >= pol_ceiling_eV:
        return 0.0
    power = epsilon_s * mobility * (e_c**3)
    return float(power * math.exp(-e_ion / kT))

import numpy as np


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
    masses = directional_band_masses(
        k_values, energy_values, alpha, hbar2_over_me
    )
    m_dos = dos_geometric_mass(masses)
    e_ion = continuum_ionization_energy(m_dos, epsilon_s, rydberg_eV)
    eps_eff = ionic_electronic_dielectric(epsilon_s, epsilon_inf)
    e_pol = continuum_polaron_energy(m_dos, eps_eff, e_ha)
    mu = scattering_time_mobility(
        m_dos, tau, elementary_charge, electron_mass_kg
    )
    e_c = gap_scaled_critical_field(band_gap_eV, e_c_ref, e_g_ref, exponent)
    return prioritization_score(
        epsilon_s,
        mu,
        e_c,
        e_ion,
        e_pol,
        kT,
        ion_ceiling_eV,
        pol_ceiling_eV,
    )
SCICODE_GOLD_EOF
