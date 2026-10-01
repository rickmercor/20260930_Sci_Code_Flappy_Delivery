"""
Convert a macroscopic reference state of a simple fluid into the reduced units of the mesoscopic model: the inputs are the temperature in K, the mass density in kg/m^3, the pressure in Pa, the specific isochoric heat capacity in J/(kg K), the isothermal compressibility in 1/Pa, the thermal expansion coefficient in 1/K, the molar mass in kg/mol and the coarse-graining degree phi, the integer number of physical molecules carried by one mesoparticle. The mesoparticle number density is c0 = rho_mass N_A / (M_w phi) and the mesoparticle mass is m = phi M_w / N_A. The reference length is the mean mesoparticle spacing L_ref = c0^(-1/3), so that the reduced bulk density is one by construction; the reference energy is u_ref = L_ref^3 / kappa_T, the work of compressing one mesoparticle volume against the macroscopic compressibility, so that the reduced compressibility is one by construction; the reference time is t_ref = sqrt(m L_ref^2 / u_ref). Reduced quantities carry a star: T* = k_B T / u_ref, P* = P kappa_T, the macroscopic isochoric heat capacity per mesoparticle in units of k_B, C_V* = c_V m / k_B, and the thermal expansion coefficient alpha* = alpha u_ref / k_B. Use k_B = 1.380649e-23 J/K and N_A = 6.02214076e23 1/mol. Raise ValueError if any physical input is not positive or if phi is not a positive integer.

In the generalised energy-conserving dissipative particle dynamics of the source each mesoparticle carries a mass m, a position, a momentum, an internal energy and a volume; its thermodynamics is a local model parametrised from a few macroscopic properties of the fluid at a chosen reference state. The choice of the energy scale from the compressibility rather than from k_B T (the usual DPD choice) makes the reduced temperature of a liquid a small number, about 0.011 for liquid argon at 125.7 K, 1419.7 kg/m^3 and 85.31 MPa, and the reduced pressure P* = P kappa_T of order 0.13, which is the regime in which the interparticle forces dominate over thermal agitation.

Returns
-------
A (9,) float64 array [T*, P*, C_V*, alpha*, kappa_T* = 1, c* = 1, L_ref in nm, u_ref in zJ (1e-21 J), t_ref in ps].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reduced_state(temperature, mass_density, pressure, cv_specific, compressibility, expansion, molar_mass, cg_degree):
    """Convert a macroscopic reference state of a simple fluid into the reduced units of the
    mesoscopic model: the inputs are the temperature in K, the mass density in kg/m^3, the
    pressure in Pa, the specific isochoric heat capacity in J/(kg K), the isothermal
    compressibility in 1/Pa, the thermal expansion coefficient in 1/K, the molar mass in
    kg/mol and the coarse-graining degree phi, the integer number of physical molecules
    carried by one mesoparticle. A (9,) float64 array [T*, P*, C_V*, alpha*, kappa_T* = 1,
    c* = 1, L_ref in nm, u_ref in zJ (1e-21 J), t_ref in ps]."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reduced_state(temperature, mass_density, pressure, cv_specific, compressibility, expansion, molar_mass, cg_degree):
    K_B = 1.380649e-23
    N_A = 6.02214076e23
    if temperature <= 0 or mass_density <= 0 or pressure <= 0 or cv_specific <= 0 or compressibility <= 0 or molar_mass <= 0:
        raise ValueError("state inputs must be positive")
    if int(cg_degree) != cg_degree or cg_degree < 1:
        raise ValueError("coarse-graining degree must be a positive integer")
    rho_atoms = mass_density*N_A/molar_mass          # atoms per m^3
    c0 = rho_atoms/cg_degree                          # mesoparticles per m^3
    l_ref = c0**(-1.0/3.0)
    u_ref = l_ref**3/compressibility
    m_ref = cg_degree*molar_mass/N_A
    t_ref = np.sqrt(m_ref*l_ref**2/u_ref)
    t_star = K_B*temperature/u_ref
    p_star = pressure*compressibility
    cv_star = cv_specific*m_ref/K_B
    alpha_star = expansion*u_ref/K_B
    return np.array([t_star, p_star, cv_star, alpha_star, 1.0, 1.0, l_ref*1.0e9, u_ref*1.0e21, t_ref*1.0e12])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n# normal: benchmark liquid-argon state\ntemperature = 125.7\nmass_density = 1419.7\npressure = 85.31e6\ncv_specific = 520.0\ncompressibility = 1.49e-9\nexpansion = 2.64e-3\nmolar_mass = 0.040\ncg_degree = 5\n',
         'call': 'reduced_state(temperature, mass_density, pressure, cv_specific, compressibility, expansion, molar_mass, cg_degree)',
         'gold_call': '_oracle_reduced_state(temperature, mass_density, pressure, cv_specific, compressibility, expansion, molar_mass, cg_degree)'},
        {'setup': 'import numpy as np\n# boundary: minimum valid positive-integer coarse-graining degree\ntemperature = 418.8\nmass_density = 695.99\npressure = 85.31e6\ncv_specific = 356.0\ncompressibility = 6.83e-9\nexpansion = 1.97e-3\nmolar_mass = 0.040\ncg_degree = 1\n',
         'call': 'reduced_state(temperature, mass_density, pressure, cv_specific, compressibility, expansion, molar_mass, cg_degree)',
         'gold_call': '_oracle_reduced_state(temperature, mass_density, pressure, cv_specific, compressibility, expansion, molar_mass, cg_degree)'},
        {'setup': 'import numpy as np\n# edge: low-pressure, low-expansion state at a different molecular scale\ntemperature = 300.0\nmass_density = 997.0\npressure = 1.0e5\ncv_specific = 4130.0\ncompressibility = 4.5e-10\nexpansion = 2.6e-4\nmolar_mass = 0.018015\ncg_degree = 3\n',
         'call': 'reduced_state(temperature, mass_density, pressure, cv_specific, compressibility, expansion, molar_mass, cg_degree)',
         'gold_call': '_oracle_reduced_state(temperature, mass_density, pressure, cv_specific, compressibility, expansion, molar_mass, cg_degree)'},
    ]
