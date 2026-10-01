"""
Perpendicular proton heating rate at one scale (final orchestrator step).

Perpendicular proton heating rate at one scale (final orchestrator step).


Thermal protons in the plasma of step 01 meet the turbulent fluctuations of the measured

amplitude spectrum of step 02. At each perpendicular scale the fluctuations act on them as

structures coherent in both space and time, so repeated uncorrelated encounters break the

proton magnetic moment and spread the protons in perpendicular energy. This step returns

the resulting growth rate of the perpendicular kinetic energy per unit mass of the protons

due to the fluctuations at one perpendicular wavenumber, in the low-beta model of steps

03-06, with c1 and c2 the two dimensionless constants of order unity that this

perpendicular ion-heating estimate carries. The defaults are the benchmark configuration.



The encounters recur at the fluctuation frequency omega_k, which is the step-05 ratio

times Omega_p of step 01. Each encounter changes a thermal proton's perpendicular energy

by a random amount whose mean square scales as the step-06 weight W(k_perp rho_p) times

epsilon_k^2 of step 04, and this change is exponentially small in Omega_p / omega_k. The

two constants are normalised so that the rate is

    Q = 4 c1 W(k_perp rho_p) epsilon_k^2 v_thp^2 omega_k exp(-c2 Omega_p / omega_k),

with v_thp in m/s and omega_k in rad/s, which gives Q in W/kg. The factor 4 cancels the

1/4 long-wavelength limit of W, so as k_perp rho_p -> 0 the rate tends to

c1 k_perp delta_b_k^3 exp(-c2 Omega_p / (k_perp delta_b_k)) in SI units.



Returns

-------

The perpendicular proton heating rate in W/kg.

Returns
-------
The perpendicular proton heating rate in W/kg.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def proton_perpendicular_heating_rate(
    k_perp: float = 0.280,
    B0_nT: float = 180.0,
    n_p_cm3: float = 90.0,
    v_thp_kms: float = 92.0,
    te_over_tp: float = 2.0,
    dB_ref_nT: float = 12.0,
    k_ref: float = 1.0e-3,
    k_break: float = 0.150,
    slope_large: float = 1.0 / 3.0,
    slope_small: float = 0.70,
    c1: float = 0.75,
    c2: float = 0.34,
) -> float:
    """Return the perpendicular proton heating rate from the fluctuations at k_perp.

    Parameters
    ----------
    k_perp : float, optional
        Perpendicular wavenumber of the fluctuations in rad/km. Finite and strictly positive.
    B0_nT : float, optional
        Mean magnetic field strength in nT (step 01). Finite and strictly positive.
    n_p_cm3 : float, optional
        Proton number density in cm^-3 (step 01). Finite and strictly positive.
    v_thp_kms : float, optional
        Proton thermal speed sqrt(2 k_B T_p / m_p) in km/s (step 01). Finite and strictly
        positive.
    te_over_tp : float, optional
        Electron-to-proton temperature ratio T_e / T_p. Finite and non-negative.
    dB_ref_nT, k_ref, k_break, slope_large, slope_small : float, optional
        Parameters of the measured amplitude spectrum of step 02 (nT, rad/km, rad/km and
        two finite indices).
    c1, c2 : float, optional
        The two dimensionless order-unity constants of the heating estimate. Finite and
        strictly positive.

    Returns
    -------
    float
        Rate of growth of the perpendicular kinetic energy per unit mass of thermal protons
        caused by the fluctuations at ``k_perp`` breaking their magnetic moment, in W/kg.

    Raises
    ------
    ValueError
        If ``k_perp``, ``c1`` or ``c2`` is not a finite, strictly positive real scalar, or
        for any argument that steps 01-06 reject.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from scipy.special import i0e, j1


def _oracle_proton_perpendicular_heating_rate(
    k_perp: float = 0.280,
    B0_nT: float = 180.0,
    n_p_cm3: float = 90.0,
    v_thp_kms: float = 92.0,
    te_over_tp: float = 2.0,
    dB_ref_nT: float = 12.0,
    k_ref: float = 1.0e-3,
    k_break: float = 0.150,
    slope_large: float = 1.0 / 3.0,
    slope_small: float = 0.70,
    c1: float = 0.75,
    c2: float = 0.34,
) -> float:
    k = _kmb_scalar("k_perp", k_perp)
    c1v = _kmb_scalar("c1", c1)
    c2v = _kmb_scalar("c2", c2)
    omega_p, rho_p, v_a = _oracle_plasma_reference_scales(B0_nT, n_p_cm3, v_thp_kms)
    kk = np.array([k])
    delta_b = _oracle_perpendicular_velocity_amplitude(kk, B0_nT, v_a, dB_ref_nT, k_ref, k_break,
                                                        slope_large, slope_small)
    k_rho = kk * rho_p
    epsilon = float(_oracle_electric_amplitude_parameter(k_rho, delta_b, v_thp_kms, te_over_tp)[0])
    ratio = float(_oracle_fluctuation_frequency_ratio(kk, delta_b, omega_p, rho_p, te_over_tp)[0])
    weight = float(_oracle_gyroaveraging_weight(k_rho)[0])
    v_th = float(v_thp_kms) * 1.0e3
    return float(4.0 * c1v * omega_p * v_th * v_th * weight * epsilon * epsilon * ratio
                 * math.exp(-c2v / ratio))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic whole-pipeline cases (candidate against oracle)."""
    imports = "import numpy as np\nimport math\nfrom scipy.special import i0e, j1\n"
    probe = (
        imports
        + "def _probe_call():\n"
        "    try:\n"
        "        proton_perpendicular_heating_rate(k_perp=0.28, c2=0.0)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
        "def _probe_gold():\n"
        "    try:\n"
        "        _oracle_proton_perpendicular_heating_rate(k_perp=0.28, c2=0.0)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )
    return [
        # the benchmark configuration
        {
            "setup": imports,
            "call": "proton_perpendicular_heating_rate()",
            "gold_call": "_oracle_proton_perpendicular_heating_rate()",
        },
        # a smaller scale and cooler electrons in the same stream
        {
            "setup": imports,
            "call": "proton_perpendicular_heating_rate(k_perp=0.45, te_over_tp=0.5)",
            "gold_call": "_oracle_proton_perpendicular_heating_rate(k_perp=0.45, te_over_tp=0.5)",
        },
        # a weaker, cooler 0.3 au-like plasma with a different spectrum, at a large scale
        {
            "setup": imports,
            "call": ("proton_perpendicular_heating_rate(k_perp=0.05, B0_nT=50.0, n_p_cm3=10.0, "
                     "v_thp_kms=50.0, te_over_tp=1.0, dB_ref_nT=3.0, k_ref=1.0e-3, k_break=0.03, "
                     "slope_large=0.25, slope_small=0.8)"),
            "gold_call": ("_oracle_proton_perpendicular_heating_rate(k_perp=0.05, B0_nT=50.0, n_p_cm3=10.0, "
                          "v_thp_kms=50.0, te_over_tp=1.0, dB_ref_nT=3.0, k_ref=1.0e-3, k_break=0.03, "
                          "slope_large=0.25, slope_small=0.8)"),
        },
        # a zero order-unity constant is rejected
        {
            "setup": probe,
            "call": "_probe_call()",
            "gold_call": "_probe_gold()",
        },
    ]
