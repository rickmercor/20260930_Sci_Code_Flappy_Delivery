"""
Return the depletion charge density for the supplied cryogenic electrostatic state.

Source-convention clarification: this task defines depletion charge as the exact analytic integral of the same varying ionised-acceptor profile used in the Poisson first integral, with both endpoint occupations evaluated at the same Vch. This resolves the source's bulk-occupation notation when Vch is nonzero. Thus, for fA(psi)=1/[1+gA exp((EA_above_EV-Eg(T)/2-(psi-Vch))/UT)], use Qdep=-sqrt(2 q NA eps_si [(psi_s-psi_b)-UT ln(fA(psi_s)/fA(psi_b))]). Potentials are in volts and charge density in C/cm^2.

Returns
-------
float, the depletion charge density Qdep in C/cm^2 as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def depletion_charge_density(psi_s: float, Vch: float, T: float, NA: float,
                             gA: float, EA_above_EV: float) -> float:
    '''Depletion charge density including incomplete dopant ionisation.

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
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level above the valence-band edge, in eV, strictly positive.

    Returns
    -------
    Qdep : float
        Depletion charge density in C/cm^2, negative or zero, as a native Python
        float.

    Raises
    ------
    ValueError
        If `psi_s` or `Vch` is not a finite scalar, if any of `T`, `NA`, `gA`,
        `EA_above_EV` is not a finite strictly positive scalar, or if `psi_s`
        lies below the bulk potential.
'''
    return Qdep

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


def _oracle_depletion_charge_density(psi_s: float, Vch: float, T: float, NA: float,
                             gA: float, EA_above_EV: float) -> float:
    for name, v in (("psi_s", psi_s), ("Vch", Vch)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    for name, v in (("T", T), ("NA", NA), ("gA", gA), ("EA_above_EV", EA_above_EV)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    Q = 1.602176634e-19
    EPS_SI = 11.7 * 8.8541878128e-14
    T = float(T); NA = float(NA); gA = float(gA); EA = float(EA_above_EV)
    psi_s = float(psi_s); Vch = float(Vch)
    ut = _thermal_voltage(T)
    psib = _oracle_bulk_potential(T, NA, gA, EA)
    if psi_s < psib:
        raise ValueError("psi_s must not lie below the bulk potential")
    psiA = EA - _bandgap(T) / 2.0
    fs = _occupation((psiA - (psi_s - Vch)) / ut, gA)
    fb = _occupation((psiA - (psib - Vch)) / ut, gA)
    G = (psi_s - psib) - ut * np.log(fs / fb)
    return float(-np.sqrt(2.0 * Q * NA * EPS_SI * max(0.0, G)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        depletion_charge_density(0.5739, 0.0, -1.0, 3.0e17, 4.0, 0.045)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_depletion_charge_density(0.5739, 0.0, -1.0, 3.0e17, 4.0, 0.045)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np",
            "call": "depletion_charge_density(0.5739, 0.0, 12.0, 3.0e17, 4.0, 0.045)",
            "gold_call": "_oracle_depletion_charge_density(0.5739, 0.0, 12.0, 3.0e17, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "depletion_charge_density(0.6168, 0.1, 12.0, 3.0e17, 4.0, 0.045)",
            "gold_call": "_oracle_depletion_charge_density(0.6168, 0.1, 12.0, 3.0e17, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "depletion_charge_density(-0.5618895001762898, 0.0, 12.0, 3.0e17, 4.0, 0.045)",
            "gold_call": "_oracle_depletion_charge_density(-0.5618895001762898, 0.0, 12.0, 3.0e17, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "depletion_charge_density(0.90, 0.0, 12.0, 3.0e17, 4.0, 0.045)",
            "gold_call": "_oracle_depletion_charge_density(0.90, 0.0, 12.0, 3.0e17, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "depletion_charge_density(0.55, 0.0, 300.0, 3.0e17, 4.0, 0.045)",
            "gold_call": "_oracle_depletion_charge_density(0.55, 0.0, 300.0, 3.0e17, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "depletion_charge_density(0.40, 0.0, 12.0, 1.0e18, 2.0, 0.070)",
            "gold_call": "_oracle_depletion_charge_density(0.40, 0.0, 12.0, 1.0e18, 2.0, 0.070)",
        },
            {   # heavy doping, where the ionisation profile varies most across the region
            "setup": 'import numpy as np',
            "call": "depletion_charge_density(0.50, 0.0, 12.0, 5.0e18, 4.0, 0.045)",
            "gold_call": "_oracle_depletion_charge_density(0.50, 0.0, 12.0, 5.0e18, 4.0, 0.045)",
        },
        {   # deep cryogenic, where almost every acceptor in the bulk is frozen out
            "setup": 'import numpy as np',
            "call": "depletion_charge_density(0.60, 0.0, 4.2, 3.0e17, 4.0, 0.045)",
            "gold_call": "_oracle_depletion_charge_density(0.60, 0.0, 4.2, 3.0e17, 4.0, 0.045)",
        },
]
