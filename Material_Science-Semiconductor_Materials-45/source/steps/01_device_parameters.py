"""
Assemble the SI parameter vector of an undoped thin-film solar cell. The device is an undoped active layer of thickness d (nm) and relative permittivity eps between a hole-injecting anode at x = 0 and an electron-injecting cathode at x = d, with effective densities of states N_c and N_v (1/cm^3), energy gap E_g (eV), injection barriers phi_an and phi_cat (eV, measured from the transport levels to the contact Fermi levels), balanced or unbalanced mobilities mu_n and mu_p (cm^2/(V s)), the reduced Langevin recombination coefficient gamma_L = zeta q (mu_n + mu_p) / (eps eps0) with the reduction factor zeta, and a uniform exciton generation rate G_ex (1/(cm^3 s)). The contacts hold Boltzmann densities of both carriers at all times: at the anode p_an = N_v exp(-phi_an/kT) and n_an = N_c exp(-(E_g - phi_an)/kT), at the cathode n_cat = N_c exp(-phi_cat/kT) and p_cat = N_v exp(-(E_g - phi_cat)/kT), so that n p = n_i^2 = N_c N_v exp(-E_g/kT) at both contacts and minority carriers reaching a contact are extracted. The built-in voltage is V_bi = (E_g - phi_an - phi_cat)/q, the electrostatic potential obeys psi(0) = 0 and psi(d) = V_bi - V for the applied voltage V (forward bias positive), and the generation current is q G_ex d. Return the 14-entry vector [kT/q (V), V_bi (V), d (m), eps eps0 (F/m), p_an, n_an, n_cat, p_cat (1/m^3), n_i^2 (1/m^6), gamma_L (m^3/s), D_n, D_p (m^2/s), G_ex (1/(m^3 s)), q G_ex d (A/m^2)] with q = 1.602176634e-19 C, k = 1.380649e-23 J/K and eps0 = 8.8541878128e-12 F/m. Raise ValueError if a length, permittivity, temperature, density of states, gap or mobility is not positive, if the reduction factor or generation rate is negative, or if a barrier is negative or the two barriers do not sum to less than the gap.

Thin-film organic solar cells are undoped metal-insulator-metal structures whose electric field and dark carrier populations are set entirely by the contacts, so the injection barriers fix both the built-in voltage and the equilibrium charge that photogenerated carriers can recombine with.

Returns
-------
A (14,) float64 array [kT/q, V_bi, d, eps eps0, p_an, n_an, n_cat, p_cat, n_i^2, gamma_L, D_n, D_p, G_ex, q G_ex d] in SI units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def device_parameters(thickness: float, permittivity: float, temperature: float, nc: float, nv: float, gap: float, barrier_anode: float, barrier_cathode: float, mobility_n: float, mobility_p: float, langevin_factor: float, generation_rate: float) -> "np.ndarray":
    """Assemble the SI parameter vector of an undoped thin-film solar cell. A (14,) float64
        array [kT/q, V_bi, d, eps eps0, p_an, n_an, n_cat, p_cat, n_i^2, gamma_L, D_n, D_p,
        G_ex, q G_ex d] in SI units.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _Q():
    return 1.602176634e-19


def _KB():
    return 1.380649e-23


def _E0():
    return 8.8541878128e-12


def _oracle_device_parameters(
    thickness: float,
    permittivity: float,
    temperature: float,
    nc: float,
    nv: float,
    gap: float,
    barrier_anode: float,
    barrier_cathode: float,
    mobility_n: float,
    mobility_p: float,
    langevin_factor: float,
    generation_rate: float,
) -> "np.ndarray":
    """Parameter vector (SI) of the undoped thin-film cell with symmetric or asymmetric injection barriers."""
    d_nm = float(thickness)
    eps = float(permittivity)
    T = float(temperature)
    Nc = float(nc)
    Nv = float(nv)
    Eg = float(gap)

    pa = float(barrier_anode)
    pc = float(barrier_cathode)
    mun = float(mobility_n)
    mup = float(mobility_p)
    gam = float(langevin_factor)
    Gex = float(generation_rate)

    if not (
        d_nm > 0
        and eps > 0
        and T > 0
        and Nc > 0
        and Nv > 0
        and Eg > 0
        and mun > 0
        and mup > 0
        and gam >= 0
        and Gex >= 0
    ):
        raise ValueError(
            "thickness, permittivity, temperature, densities of states, gap "
            "and mobilities must be positive; the reduction factor and "
            "generation rate non-negative"
        )

    if not (0.0 <= pa and 0.0 <= pc and pa + pc < Eg):
        raise ValueError(
            "injection barriers must be non-negative and sum to less than the gap"
        )

    VT = _KB() * T / _Q()
    d = d_nm * 1e-9
    ee = eps * _E0()

    Nc_m = Nc * 1e6
    Nv_m = Nv * 1e6

    p_an = Nv_m * np.exp(-pa / VT)
    n_an = Nc_m * np.exp(-(Eg - pa) / VT)
    n_cat = Nc_m * np.exp(-pc / VT)
    p_cat = Nv_m * np.exp(-(Eg - pc) / VT)

    ni2 = Nc_m * Nv_m * np.exp(-Eg / VT)

    mun_m = mun * 1e-4
    mup_m = mup * 1e-4

    gb = gam * _Q() * (mun_m + mup_m) / ee
    G = Gex * 1e6

    return np.array(
        [
            VT,
            Eg - pa - pc,
            d,
            ee,
            p_an,
            n_an,
            n_cat,
            p_cat,
            ni2,
            gb,
            mun_m * VT,
            mup_m * VT,
            G,
            _Q() * G * d,
        ],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n',
         'call': 'device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)',
         'gold_call': '_oracle_device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)'},
        {'setup': 'import numpy as np\n',
         'call': 'device_parameters(150.0, 3.0, 290.0, 1.0e21, 5.0e20, 1.5, 0.30, 0.15, 5.0e-4, 1.0e-4, 0.2, 5.0e21)',
         'gold_call': '_oracle_device_parameters(150.0, 3.0, 290.0, 1.0e21, 5.0e20, 1.5, 0.30, 0.15, 5.0e-4, 1.0e-4, 0.2, 5.0e21)'},
        {'setup': 'import numpy as np\n',
         'call': 'device_parameters(80.0, 4.0, 320.0, 2.0e20, 1.0e20, 1.2, 0.0, 0.40, 1.0e-3, 2.0e-3, 0.0, 0.0)',
         'gold_call': '_oracle_device_parameters(80.0, 4.0, 320.0, 2.0e20, 1.0e20, 1.2, 0.0, 0.40, 1.0e-3, 2.0e-3, 0.0, 0.0)'},
    ]
