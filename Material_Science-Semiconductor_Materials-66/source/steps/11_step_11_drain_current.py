"""
Return the magnitude of the drain current in microamperes for the supplied device, bias and model parameters.

This final orchestrator composes the preceding scientific steps for the two channel endpoints and source-side transport coefficients. Geometry is supplied in centimetres, potentials in volts and temperature in kelvin. Apply the supplied quadrature and bisection settings throughout and return one scalar current magnitude in microamperes.

Returns
-------
float, the magnitude of the drain current IDS in microamperes as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def drain_current_microamp(T: float, NA: float, tox: float, Wch: float, Lch: float,
                           VGB: float, VDS: float, phi_m: float, chi_si: float,
                           DitC: float, Edecay: float, UBT: float, Usp: float,
                           Voffset: float, mu0: float, theta1: float, theta2: float,
                           eta: float, gt: float, gA: float, EA_above_EV: float,
                           n_nodes: int, tol: float) -> float:
    '''Drain current of the cryogenic FET at one bias point, in microamperes.

    Parameters
    ----------
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    tox : float
        Physical gate-oxide thickness in cm, strictly positive.
    Wch : float
        Channel width in cm, strictly positive.
    Lch : float
        Channel length in cm, strictly positive.
    VGB : float
        Gate-to-body voltage in volts, finite.
    VDS : float
        Drain-to-source voltage in volts, finite and non-negative.
    phi_m : float
        Gate work function in volts, strictly positive.
    chi_si : float
        Electron affinity of silicon in volts, strictly positive.
    DitC : float
        Interface-trap density scale at the conduction-band edge, in cm^-2 eV^-1,
        strictly positive.
    Edecay : float
        Energy decay scale of the trap spectrum, in eV, strictly positive.
    UBT : float
        Characteristic band-tail statistical voltage in volts, strictly positive.
    Usp : float
        Smoothing voltage of the statistical crossover, in volts, strictly positive.
    Voffset : float
        Offset in volts locating the crossover. Finite, may be negative.
    mu0 : float
        Low-field electron mobility at T, in cm^2 V^-1 s^-1, strictly positive.
    theta1 : float
        Linear mobility-degradation coefficient, non-negative.
    theta2 : float
        Quadratic mobility-degradation coefficient, non-negative.
    eta : float
        Weight of the mobile charge in the effective field, non-negative.
    gt : float
        Trap degeneracy factor, strictly positive.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level above the valence-band edge, in eV, strictly positive.
    n_nodes : int
        Number of Gauss-Legendre nodes, an integer of at least 2.
    tol : float
        Bracket width at which the surface-potential bisection stops, in volts,
        strictly positive.

    Returns
    -------
    IDS : float
        Magnitude of the drain current in microamperes, as a native Python float.

    Raises
    ------
    ValueError
        If `VGB` or `Voffset` is not a finite scalar, if `VDS` is not a finite
        non-negative scalar, if any of `T`, `NA`, `tox`, `Wch`, `Lch`, `phi_m`,
        `chi_si`, `DitC`, `Edecay`, `UBT`, `Usp`, `mu0`, `gt`, `gA`,
        `EA_above_EV`, `tol` is not a finite strictly positive scalar, if either
        of `theta1`, `theta2`, `eta` is not a finite non-negative scalar, or if
        `n_nodes` is not an integer greater than or equal to 2.
'''
    return IDS

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_drain_current_microamp(T: float, NA: float, tox: float, Wch: float, Lch: float,
                           VGB: float, VDS: float, phi_m: float, chi_si: float,
                           DitC: float, Edecay: float, UBT: float, Usp: float,
                           Voffset: float, mu0: float, theta1: float, theta2: float,
                           eta: float, gt: float, gA: float, EA_above_EV: float,
                           n_nodes: int, tol: float) -> float:
    for name, v in (("VGB", VGB), ("Voffset", Voffset)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    if not np.isscalar(VDS) or not np.isfinite(float(VDS)) or float(VDS) < 0.0:
        raise ValueError("VDS must be a finite non-negative scalar")
    for name, v in (("T", T), ("NA", NA), ("tox", tox), ("Wch", Wch), ("Lch", Lch),
                    ("phi_m", phi_m), ("chi_si", chi_si), ("DitC", DitC),
                    ("Edecay", Edecay), ("UBT", UBT), ("Usp", Usp), ("mu0", mu0),
                    ("gt", gt), ("gA", gA), ("EA_above_EV", EA_above_EV), ("tol", tol)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    for name, v in (("theta1", theta1), ("theta2", theta2), ("eta", eta)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) < 0.0:
            raise ValueError("%s must be a finite non-negative scalar" % name)
    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2")

    KB = 1.380649e-23
    Q = 1.602176634e-19
    EPS_OX = 3.9 * 8.8541878128e-14
    T = float(T); VDS = float(VDS); n_nodes = int(n_nodes)
    ut = KB * T / Q
    Cox = EPS_OX / float(tox)

    phi_ms = _oracle_work_function_difference(T, NA, phi_m, chi_si, gA, EA_above_EV)

    psi_S = _oracle_surface_potential(VGB, 0.0, T, NA, Cox, phi_ms, DitC, Edecay,
                                      UBT, Usp, Voffset, gt, gA, EA_above_EV, n_nodes, tol)
    psi_D = _oracle_surface_potential(VGB, VDS, T, NA, Cox, phi_ms, DitC, Edecay,
                                      UBT, Usp, Voffset, gt, gA, EA_above_EV, n_nodes, tol)

    Qm_S = _oracle_mobile_charge_density(psi_S, 0.0, T, NA, UBT, Usp, Voffset,
                                         gA, EA_above_EV, n_nodes)
    Qm_D = _oracle_mobile_charge_density(psi_D, VDS, T, NA, UBT, Usp, Voffset,
                                         gA, EA_above_EV, n_nodes)

    Qdep_S = _oracle_depletion_charge_density(psi_S, 0.0, T, NA, gA, EA_above_EV)
    mu_n = _oracle_effective_mobility(Qdep_S, Qm_S, mu0, theta1, theta2, eta)
    m = _oracle_coupling_factor(psi_S, 0.0, T, NA, Cox, DitC, Edecay, gt, gA,
                                EA_above_EV, n_nodes)

    I = (float(Wch) / float(Lch)) * mu_n * (
        -(Qm_S ** 2 - Qm_D ** 2) / (2.0 * m * Cox) + ut * (Qm_S - Qm_D))
    return float(abs(I) * 1.0e6)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        drain_current_microamp(-1.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.62, 0.1, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_drain_current_microamp(-1.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.62, 0.1, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np",
            "call": "drain_current_microamp(12.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.62, 0.1, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)",
            "gold_call": "_oracle_drain_current_microamp(12.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.62, 0.1, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)",
        },
        {
            "setup": "import numpy as np",
            "call": "drain_current_microamp(12.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.90, 0.1, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)",
            "gold_call": "_oracle_drain_current_microamp(12.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.90, 0.1, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)",
        },
        {
            "setup": "import numpy as np",
            "call": "drain_current_microamp(12.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.62, 0.0, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)",
            "gold_call": "_oracle_drain_current_microamp(12.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.62, 0.0, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)",
        },
        {
            "setup": "import numpy as np",
            "call": "drain_current_microamp(250.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.62, 0.1, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 314.822, 0.37306, 0.57612, 0.5, 2.0, 4.0, 0.045, 200, 1e-12)",
            "gold_call": "_oracle_drain_current_microamp(250.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.62, 0.1, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 314.822, 0.37306, 0.57612, 0.5, 2.0, 4.0, 0.045, 200, 1e-12)",
        },
        {
            "setup": "import numpy as np",
            "call": "drain_current_microamp(12.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.40, 0.1, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)",
            "gold_call": "_oracle_drain_current_microamp(12.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.40, 0.1, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)",
        },
            {   # the same device and bias at 4.2 K, where the band tail dominates the turn-on
            "setup": 'import numpy as np',
            "call": "drain_current_microamp(4.2, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.62, 0.1, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)",
            "gold_call": "_oracle_drain_current_microamp(4.2, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 0.62, 0.1, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)",
        },
        {   # strong inversion at a higher drain bias, where mobility degradation sets the level
            "setup": 'import numpy as np',
            "call": "drain_current_microamp(12.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 1.20, 0.3, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)",
            "gold_call": "_oracle_drain_current_microamp(12.0, 3.0e17, 2.6e-7, 2.0e-4, 6.5e-6, 1.20, 0.3, 4.35, 4.05, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 395.655, 0.55428, 0.10007, 0.5, 2.0, 4.0, 0.045, 400, 1e-14)",
        },
]
