"""
Convert the physical description of an undoped thin-film (metal-insulator-metal) diode into the parameter vector used by the rest of the chain. The active layer of thickness d (nm) and relative permittivity eps is sandwiched between a hole-injecting anode at x = 0 and an electron-injecting cathode at x = d; the temperature is T (K); the effective densities of states are N_c and N_v (1/m^3), the energy gap E_g (eV), the injection barriers phi_an at the anode and phi_cat at the cathode (eV), the mobilities mu_n and mu_p (m^2/(V s)) and the reduced Langevin factor zeta. The carrier densities at the contacts follow Boltzmann statistics with the contact Fermi level: p_an = N_v exp(-phi_an / kT) and the corresponding electron density n_an = N_c exp(-(E_g - phi_an) / kT) at the anode, n_cat = N_c exp(-phi_cat / kT) and p_cat = N_v exp(-(E_g - phi_cat) / kT) at the cathode, so that n p = n_i^2 = N_c N_v exp(-E_g / kT) at both contacts. The nominal built-in potential is V_bi,0 = (E_g - phi_an - phi_cat) / q, the geometric capacitance C_geo = eps eps0 / d, the Debye screening lengths of the injected carriers at the contacts are lambda_an = sqrt(2 eps eps0 kT / (q^2 p_an)) and lambda_cat = sqrt(2 eps eps0 kT / (q^2 n_cat)), and the bimolecular recombination coefficient is the reduced Langevin value gamma = zeta q (mu_n + mu_p) / (eps eps0). Use q = 1.602176634e-19 C, k = 1.380649e-23 J/K, eps0 = 8.8541878128e-12 F/m. Raise ValueError if d, eps, T, N_c or N_v is not positive, if a barrier is negative or the barriers together reach the gap, or if a mobility or zeta is not positive.

Thin-film diodes based on undoped organic or perovskite semiconductors behave as metal-insulator-metal devices: the active layer is fully depleted of doping-induced carriers and all carriers are injected from the contacts. The source describes their low-frequency capacitance through the accumulation layers of injected carriers at the contacts, whose thickness is set by the Debye length of the carrier density at the contact; ohmic contacts have Debye lengths far below the layer thickness, weakly injecting contacts Debye lengths above it.

Returns
-------
A (14,) float64 array [kT/q in V, V_bi,0 in V, C_geo in nF/cm^2, lambda_an in nm, lambda_cat in nm, ln p_an, ln n_cat, ln n_an, ln p_cat (natural logarithms of the densities in 1/m^3), gamma in units of 1e-18 m^3/s, d in nm, eps, mu_n, mu_p in m^2/(V s)].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def diode_parameters(thickness: float, permittivity: float, temperature: float, nc: float, nv: float, gap: float, barrier_anode: float, barrier_cathode: float, mobility_n: float, mobility_p: float, langevin_factor: float) -> "np.ndarray":
    """Convert the physical description of an undoped thin-film (metal-insulator-metal) diode
    into the parameter vector used by the rest of the chain. A (14,) float64 array [kT/q in
    V, V_bi,0 in V, C_geo in nF/cm^2, lambda_an in nm, lambda_cat in nm, ln p_an, ln n_cat,
    ln n_an, ln p_cat (natural logarithms of the densities in 1/m^3), gamma in units of
    1e-18 m^3/s, d in nm, eps, mu_n, mu_p in m^2/(V s)].
    Parameters
    ----------
    thickness : float
        Active-layer thickness d in nm.
    permittivity : float
        Relative permittivity of the active layer.
    temperature : float
        Temperature in K.
    nc : float
        Conduction-band density of states in 1/m^3.
    nv : float
        Valence-band density of states in 1/m^3.
    gap : float
        Transport gap in eV.
    barrier_anode : float
        Hole injection barrier at the anode in eV.
    barrier_cathode : float
        Electron injection barrier at the cathode in eV.
    mobility_n : float
        Electron mobility in m^2/(V s).
    mobility_p : float
        Hole mobility in m^2/(V s).
    langevin_factor : float
        Reduction factor of the Langevin recombination coefficient.

    Raises
    ------
    ValueError
        If thickness, permittivity, temperature or either density of states is
        not positive; if a barrier is negative or the two barriers together
        reach the gap; or if a mobility or the Langevin factor is not positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_Q = 1.602176634e-19

_KB = 1.380649e-23

_E0 = 8.8541878128e-12

def _oracle_diode_parameters(thickness: float, permittivity: float, temperature: float, nc: float, nv: float, gap: float, barrier_anode: float, barrier_cathode: float, mobility_n: float, mobility_p: float, langevin_factor: float) -> "np.ndarray":
    d_nm = float(thickness); eps = float(permittivity); T = float(temperature); Nc = float(nc); Nv = float(nv); Eg = float(gap)
    pa = float(barrier_anode); pc = float(barrier_cathode); mun = float(mobility_n); mup = float(mobility_p); zeta = float(langevin_factor)
    if not (d_nm > 0.0) or not (eps > 0.0) or not (T > 0.0) or not (Nc > 0.0) or not (Nv > 0.0):
        raise ValueError("thickness, permittivity, temperature and densities of states must be positive")
    if not (pa >= 0.0) or not (pc >= 0.0) or not (Eg > pa + pc):
        raise ValueError("barriers must be non-negative and smaller than the gap in total")
    if not (mun > 0.0) or not (mup > 0.0) or not (zeta > 0.0):
        raise ValueError("mobilities and the Langevin factor must be positive")
    VT = _KB * T / _Q
    p_an = Nv * np.exp(-pa / VT); n_cat = Nc * np.exp(-pc / VT)
    n_an = Nc * np.exp(-(Eg - pa) / VT); p_cat = Nv * np.exp(-(Eg - pc) / VT)
    Vbi0 = Eg - pa - pc
    d = d_nm * 1e-9; ee = eps * _E0
    cgeo = ee / d * 1e5                                             # nF/cm^2
    lam_an = np.sqrt(2.0 * ee * VT / (_Q * p_an)) * 1e9; lam_cat = np.sqrt(2.0 * ee * VT / (_Q * n_cat)) * 1e9
    gam = zeta * _Q * (mun + mup) / ee * 1e18                        # 1e-18 m^3/s
    return np.array([VT, Vbi0, cgeo, lam_an, lam_cat, np.log(p_an), np.log(n_cat), np.log(n_an), np.log(p_cat), gam, d_nm, eps, mun, mup], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nthickness = 100.0\npermittivity = 3.5\ntemperature = 300.0\nnc = 1.0e26\nnv = 1.0e26\ngap = 1.5\nbarrier_anode = 0.02\nbarrier_cathode = 0.02\nmobility_n = 1.0e-8\nmobility_p = 5.0e-9\nlangevin_factor = 0.1\n',
         'call': 'diode_parameters(thickness, permittivity, temperature, nc, nv, gap, barrier_anode, barrier_cathode, mobility_n, mobility_p, langevin_factor)',
         'gold_call': '_oracle_diode_parameters(thickness, permittivity, temperature, nc, nv, gap, barrier_anode, barrier_cathode, mobility_n, mobility_p, langevin_factor)'},
        {'setup': 'import numpy as np\nthickness = 100.0\npermittivity = 3.5\ntemperature = 300.0\nnc = 1.0e26\nnv = 1.0e26\ngap = 1.5\nbarrier_anode = 0.02\nbarrier_cathode = 0.50\nmobility_n = 1.0e-8\nmobility_p = 5.0e-9\nlangevin_factor = 0.1\n',
         'call': 'diode_parameters(thickness, permittivity, temperature, nc, nv, gap, barrier_anode, barrier_cathode, mobility_n, mobility_p, langevin_factor)',
         'gold_call': '_oracle_diode_parameters(thickness, permittivity, temperature, nc, nv, gap, barrier_anode, barrier_cathode, mobility_n, mobility_p, langevin_factor)'},
        {'setup': 'import numpy as np\nthickness = 80.0\npermittivity = 3.0\ntemperature = 320.0\nnc = 5.0e25\nnv = 2.0e26\ngap = 1.7\nbarrier_anode = 0.10\nbarrier_cathode = 0.15\nmobility_n = 2.0e-8\nmobility_p = 1.0e-8\nlangevin_factor = 0.05\n',
         'call': 'diode_parameters(thickness, permittivity, temperature, nc, nv, gap, barrier_anode, barrier_cathode, mobility_n, mobility_p, langevin_factor)',
         'gold_call': '_oracle_diode_parameters(thickness, permittivity, temperature, nc, nv, gap, barrier_anode, barrier_cathode, mobility_n, mobility_p, langevin_factor)'},
        {'setup': 'import numpy as np\nthickness = 100.0\npermittivity = 3.5\ntemperature = 300.0\nnc = 1.0e26\nnv = 1.0e26\ngap = 1.5\nbarrier_anode = 0.25\nbarrier_cathode = 0.25\nmobility_n = 1.0e-8\nmobility_p = 5.0e-9\nlangevin_factor = 0.1\n',
         'call': 'diode_parameters(thickness, permittivity, temperature, nc, nv, gap, barrier_anode, barrier_cathode, mobility_n, mobility_p, langevin_factor)',
         'gold_call': '_oracle_diode_parameters(thickness, permittivity, temperature, nc, nv, gap, barrier_anode, barrier_cathode, mobility_n, mobility_p, langevin_factor)'},
        {'setup': 'import numpy as np\nthickness = 100.0\npermittivity = 3.5\ntemperature = 300.0\nnc = 1.0e26\nnv = 1.0e26\ngap = 1.5\nbarrier_anode = 0.0\nbarrier_cathode = 0.0\nmobility_n = 1.0e-8\nmobility_p = 5.0e-9\nlangevin_factor = 0.1\n',
         'call': 'diode_parameters(thickness, permittivity, temperature, nc, nv, gap, barrier_anode, barrier_cathode, mobility_n, mobility_p, langevin_factor)',
         'gold_call': '_oracle_diode_parameters(thickness, permittivity, temperature, nc, nv, gap, barrier_anode, barrier_cathode, mobility_n, mobility_p, langevin_factor)'},
    ]
