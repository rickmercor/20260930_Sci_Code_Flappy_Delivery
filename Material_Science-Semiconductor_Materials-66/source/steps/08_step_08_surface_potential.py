"""
Return the surface potential corresponding to the supplied gate and channel voltages.

The surface potential is the electrostatic state required by the remaining charge and transport steps. Follow the task's numerical prescription: bisect [psi_b+1e-9, Eg(T)/2+Vch+0.20] V until its width is below tol, then return its midpoint. The supplied n_nodes applies to the energy and potential quadratures

Returns
-------
float, the surface potential psi_s in volts as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def surface_potential(VGB: float, Vch: float, T: float, NA: float, Cox: float,
                      phi_ms: float, DitC: float, Edecay: float, UBT: float,
                      Usp: float, Voffset: float, gt: float, gA: float,
                      EA_above_EV: float, n_nodes: int, tol: float) -> float:
    '''Surface potential solving the gate boundary condition at one bias point.

    Parameters
    ----------
    VGB : float
        Gate-to-body voltage in volts, finite.
    Vch : float
        Channel potential in volts, finite.
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    Cox : float
        Oxide capacitance per unit area in F/cm^2, strictly positive.
    phi_ms : float
        Metal-semiconductor work-function difference in volts, finite.
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
    gt : float
        Trap degeneracy factor, strictly positive.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level above the valence-band edge, in eV, strictly positive.
    n_nodes : int
        Number of Gauss-Legendre nodes, an integer of at least 2.
    tol : float
        Absolute bracket width at which the bisection stops, in volts, strictly
        positive.

    Returns
    -------
    psi_s : float
        Surface potential in volts, as a native Python float.

    Raises
    ------
    ValueError
        If `VGB`, `Vch`, `phi_ms` or `Voffset` is not a finite scalar, if any of
        `T`, `NA`, `Cox`, `DitC`, `Edecay`, `UBT`, `Usp`, `gt`, `gA`,
        `EA_above_EV`, `tol` is not a finite strictly positive scalar, if
        `n_nodes` is not an integer greater than or equal to 2, or if the residual
        does not change sign across the initial bracket.
'''
    return psi_s

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _bandgap(T):
    """Silicon band gap in eV at temperature T in kelvin."""
    return 1.170 - 4.730e-4 * T * T / (T + 636.0)


def _oracle_surface_potential(VGB: float, Vch: float, T: float, NA: float, Cox: float,
                      phi_ms: float, DitC: float, Edecay: float, UBT: float,
                      Usp: float, Voffset: float, gt: float, gA: float,
                      EA_above_EV: float, n_nodes: int, tol: float) -> float:
    for name, v in (("VGB", VGB), ("Vch", Vch), ("phi_ms", phi_ms), ("Voffset", Voffset)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    for name, v in (("T", T), ("NA", NA), ("Cox", Cox), ("DitC", DitC), ("Edecay", Edecay),
                    ("UBT", UBT), ("Usp", Usp), ("gt", gt), ("gA", gA),
                    ("EA_above_EV", EA_above_EV), ("tol", tol)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2")

    T = float(T); NA = float(NA); Cox = float(Cox); Vch = float(Vch)
    gA = float(gA); EA = float(EA_above_EV); n_nodes = int(n_nodes)
    psib = _oracle_bulk_potential(T, NA, gA, EA)

    def _residual(ps):
        Qm = _oracle_mobile_charge_density(ps, Vch, T, NA, UBT, Usp, Voffset, gA, EA, n_nodes)
        Qd = _oracle_depletion_charge_density(ps, Vch, T, NA, gA, EA)
        Qit = _oracle_interface_trap_charge(ps, Vch, T, DitC, Edecay, gt, n_nodes)
        return (float(phi_ms) - Qit / Cox) + (ps - psib) - (Qm + Qd) / Cox - float(VGB)

    lo = psib + 1e-9
    hi = _bandgap(T) / 2.0 + Vch + 0.20
    flo, fhi = _residual(lo), _residual(hi)
    if flo * fhi > 0.0:
        raise ValueError("surface potential is not bracketed on the search interval")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = _residual(mid)
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if hi - lo < float(tol):
            break
    return float(0.5 * (lo + hi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\nPMS = -0.8468369446207342\ndef run_model():\n    try:\n        surface_potential(0.62, 0.0, -1.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_surface_potential(0.62, 0.0, -1.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\nPMS = -0.8468369446207342\n",
            "call": "surface_potential(0.62, 0.0, 12.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)",
            "gold_call": "_oracle_surface_potential(0.62, 0.0, 12.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)",
        },
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\nPMS = -0.8468369446207342\n",
            "call": "surface_potential(0.62, 0.1, 12.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)",
            "gold_call": "_oracle_surface_potential(0.62, 0.1, 12.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)",
        },
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\nPMS = -0.8468369446207342\n",
            "call": "surface_potential(0.20, 0.0, 12.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)",
            "gold_call": "_oracle_surface_potential(0.20, 0.0, 12.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)",
        },
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\nPMS = -0.8468369446207342\n",
            "call": "surface_potential(1.00, 0.0, 12.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)",
            "gold_call": "_oracle_surface_potential(1.00, 0.0, 12.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)",
        },
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\n",
            "call": "surface_potential(0.62, 0.0, 250.0, 3.0e17, COX, -0.90, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 200, 1e-12)",
            "gold_call": "_oracle_surface_potential(0.62, 0.0, 250.0, 3.0e17, COX, -0.90, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 200, 1e-12)",
        },
            {   # far into strong inversion, where the solution approaches the bracket top
            "setup": 'import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\nPMS = -0.8468369446207342',
            "call": "surface_potential(1.50, 0.0, 12.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)",
            "gold_call": "_oracle_surface_potential(1.50, 0.0, 12.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)",
        },
        {   # deep subthreshold, where the mobile charge is negligible against the depletion charge
            "setup": 'import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\nPMS = -0.8468369446207342',
            "call": "surface_potential(0.05, 0.0, 12.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)",
            "gold_call": "_oracle_surface_potential(0.05, 0.0, 12.0, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 400, 1e-14)",
        },
        {   # deep cryogenic with a coarser quadrature and a looser bracket tolerance
            "setup": 'import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\nPMS = -0.8468369446207342',
            "call": "surface_potential(0.62, 0.0, 4.2, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 64, 1e-12)",
            "gold_call": "_oracle_surface_potential(0.62, 0.0, 4.2, 3.0e17, COX, PMS, 1.833e12, 0.2943, 0.0291, 0.0998, 0.004088, 2.0, 4.0, 0.045, 64, 1e-12)",
        },
]
