"""
Return the mobile inversion charge density at the supplied surface and channel potentials.

This quantity is the semiconductor's mobile-charge contribution used by the surface-potential solve and transport calculation. Follow the task's omission of holes and its depletion-charge convention. Use n_nodes for Gauss-Legendre potential integration and return an areal charge density in C/cm^2.

Returns
-------
float, the mobile inversion charge density Qm in C/cm^2 as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mobile_charge_density(psi_s: float, Vch: float, T: float, NA: float, UBT: float,
                          Usp: float, Voffset: float, gA: float, EA_above_EV: float,
                          n_nodes: int) -> float:
    '''Mobile inversion charge density at a given surface and channel potential.

    Parameters
    ----------
    psi_s : float
        Surface potential in volts, referenced to the intrinsic level. Must not
        lie below the bulk potential.
    Vch : float
        Channel potential in volts, finite.
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    UBT : float
        Characteristic band-tail statistical voltage in volts, strictly positive.
    Usp : float
        Smoothing voltage of the statistical crossover, in volts, strictly positive.
    Voffset : float
        Offset in volts locating the crossover. Finite, may be negative.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level above the valence-band edge, in eV, strictly positive.
    n_nodes : int
        Number of Gauss-Legendre nodes, an integer of at least 2.

    Returns
    -------
    Qm : float
        Mobile inversion charge density in C/cm^2, negative or zero, as a native
        Python float.

    Raises
    ------
    ValueError
        If `psi_s`, `Vch` or `Voffset` is not a finite scalar, if any of `T`,
        `NA`, `UBT`, `Usp`, `gA`, `EA_above_EV` is not a finite strictly positive
        scalar, if `n_nodes` is not an integer greater than or equal to 2, or if
        `psi_s` lies below the bulk potential.
'''
    return Qm

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _bandgap(T):
    """Silicon band gap in eV at temperature T in kelvin."""
    return 1.170 - 4.730e-4 * T * T / (T + 636.0)


def _thermal_voltage(T):
    """Thermal voltage kT/q in volts."""
    return 1.380649e-23 * T / 1.602176634e-19


def _occupation(z, g):
    """Overflow-free 1 / (1 + g * exp(z))."""
    u = np.clip(np.asarray(z, dtype=float) + np.log(g), -700.0, 700.0)
    e = np.exp(-np.abs(u))
    return np.where(u > 0.0, e / (1.0 + e), 1.0 / (1.0 + e))


def _gauss_legendre(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def _oracle_mobile_charge_density(psi_s: float, Vch: float, T: float, NA: float, UBT: float,
                          Usp: float, Voffset: float, gA: float, EA_above_EV: float,
                          n_nodes: int) -> float:
    for name, v in (("psi_s", psi_s), ("Vch", Vch), ("Voffset", Voffset)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    for name, v in (("T", T), ("NA", NA), ("UBT", UBT), ("Usp", Usp), ("gA", gA),
                    ("EA_above_EV", EA_above_EV)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2")

    Q = 1.602176634e-19
    EPS_SI = 11.7 * 8.8541878128e-14
    T = float(T); NA = float(NA); gA = float(gA); EA = float(EA_above_EV)
    psi_s = float(psi_s); Vch = float(Vch); n_nodes = int(n_nodes)
    ut = _thermal_voltage(T)
    psib = _oracle_bulk_potential(T, NA, gA, EA)
    if psi_s < psib:
        raise ValueError("psi_s must not lie below the bulk potential")

    t, w = _gauss_legendre(psib, psi_s, n_nodes)
    n = _oracle_band_tail_carrier_density(t, Vch, T, UBT, Usp, Voffset, n_nodes)
    NAm = NA * _occupation(((EA - _bandgap(T) / 2.0) - (t - Vch)) / ut, gA)
    Es2 = max(0.0, (2.0 * Q / EPS_SI) * float(np.sum(w * (n + NAm))))
    Qsc = -EPS_SI * np.sqrt(Es2)
    return float(Qsc - _oracle_depletion_charge_density(psi_s, Vch, T, NA, gA, EA))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        mobile_charge_density(0.5739, 0.0, -1.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_mobile_charge_density(0.5739, 0.0, -1.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np",
            "call": "mobile_charge_density(0.5739, 0.0, 12.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
            "gold_call": "_oracle_mobile_charge_density(0.5739, 0.0, 12.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "mobile_charge_density(0.6168, 0.1, 12.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
            "gold_call": "_oracle_mobile_charge_density(0.6168, 0.1, 12.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "mobile_charge_density(0.30, 0.0, 12.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
            "gold_call": "_oracle_mobile_charge_density(0.30, 0.0, 12.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "mobile_charge_density(-0.5618895001762898, 0.0, 12.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
            "gold_call": "_oracle_mobile_charge_density(-0.5618895001762898, 0.0, 12.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "mobile_charge_density(0.68, 0.0, 12.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
            "gold_call": "_oracle_mobile_charge_density(0.68, 0.0, 12.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "mobile_charge_density(0.55, 0.0, 250.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 200)",
            "gold_call": "_oracle_mobile_charge_density(0.55, 0.0, 250.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 200)",
        },
            {   # heavier doping driven well into inversion
            "setup": 'import numpy as np',
            "call": "mobile_charge_density(0.70, 0.0, 12.0, 1.0e18, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
            "gold_call": "_oracle_mobile_charge_density(0.70, 0.0, 12.0, 1.0e18, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
        },
        {   # intermediate temperature near the onset of inversion
            "setup": 'import numpy as np',
            "call": "mobile_charge_density(0.55, 0.0, 77.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
            "gold_call": "_oracle_mobile_charge_density(0.55, 0.0, 77.0, 3.0e17, 0.0291, 0.0998, 0.004088, 4.0, 0.045, 400)",
        },
]
