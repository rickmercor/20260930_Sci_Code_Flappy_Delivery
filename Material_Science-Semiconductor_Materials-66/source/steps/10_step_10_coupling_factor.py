"""
Return the electrostatic coupling factor for the supplied surface and channel potentials.

This dimensionless electrostatic response enters the charge-based transport calculation. Use the task's material definitions, depletion convention and continuous trap spectrum, with n_nodes controlling energy quadrature. Capacitance inputs are per unit area in F/cm^2.

Returns
-------
float, the dimensionless electrostatic coupling factor m as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coupling_factor(psi_s: float, Vch: float, T: float, NA: float, Cox: float,
                    DitC: float, Edecay: float, gt: float, gA: float,
                    EA_above_EV: float, n_nodes: int) -> float:
    '''Electrostatic coupling factor at a given surface and channel potential.

    Parameters
    ----------
    psi_s : float
        Surface potential in volts. Must lie strictly above the bulk potential.
    Vch : float
        Channel potential in volts, finite.
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    Cox : float
        Oxide capacitance per unit area in F/cm^2, strictly positive.
    DitC : float
        Interface-trap density scale at the conduction-band edge, in cm^-2 eV^-1,
        strictly positive.
    Edecay : float
        Energy decay scale of the trap spectrum, in eV, strictly positive.
    gt : float
        Trap degeneracy factor, strictly positive.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level above the valence-band edge, in eV, strictly positive.
    n_nodes : int
        Number of Gauss-Legendre nodes, an integer of at least 2.

    Returns
    -------
    m : float
        Dimensionless electrostatic coupling factor, greater than one, as a
        native Python float.

    Raises
    ------
    ValueError
        If `psi_s` or `Vch` is not a finite scalar, if any of `T`, `NA`, `Cox`,
        `DitC`, `Edecay`, `gt`, `gA`, `EA_above_EV` is not a finite strictly
        positive scalar, if `n_nodes` is not an integer greater than or equal to
        2, or if `psi_s` does not lie strictly above the bulk potential.
'''
    return m

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


def _oracle_coupling_factor(psi_s: float, Vch: float, T: float, NA: float, Cox: float,
                    DitC: float, Edecay: float, gt: float, gA: float,
                    EA_above_EV: float, n_nodes: int) -> float:
    for name, v in (("psi_s", psi_s), ("Vch", Vch)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    for name, v in (("T", T), ("NA", NA), ("Cox", Cox), ("DitC", DitC),
                    ("Edecay", Edecay), ("gt", gt), ("gA", gA), ("EA_above_EV", EA_above_EV)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2")

    Q = 1.602176634e-19
    EPS_SI = 11.7 * 8.8541878128e-14
    T = float(T); NA = float(NA); gA = float(gA); EA = float(EA_above_EV)
    psi_s = float(psi_s); Vch = float(Vch); n_nodes = int(n_nodes)
    ut = _thermal_voltage(T)
    eg = _bandgap(T)
    psib = _oracle_bulk_potential(T, NA, gA, EA)
    if psi_s <= psib:
        raise ValueError("psi_s must lie strictly above the bulk potential")

    psiA = EA - eg / 2.0
    fs = _occupation((psiA - (psi_s - Vch)) / ut, gA)
    fb = _occupation((psiA - (psib - Vch)) / ut, gA)
    G = (psi_s - psib) - ut * np.log(fs / fb)
    Cdep = Q * NA * EPS_SI * fs / np.sqrt(2.0 * Q * NA * EPS_SI * G)

    dE, w = _gauss_legendre(0.0, eg / 2.0, n_nodes)
    ft = _occupation(((eg / 2.0 - dE) - (psi_s - Vch)) / ut, float(gt))
    Cit = (Q / ut) * np.sum(w * float(DitC) * np.exp(-dE / float(Edecay)) * ft * (1.0 - ft))

    return float(1.0 + (float(Cdep) + float(Cit)) / float(Cox))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\ndef run_model():\n    try:\n        coupling_factor(0.5739, 0.0, -1.0, 3.0e17, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 400)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_coupling_factor(0.5739, 0.0, -1.0, 3.0e17, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 400)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\n",
            "call": "coupling_factor(0.5739, 0.0, 12.0, 3.0e17, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 400)",
            "gold_call": "_oracle_coupling_factor(0.5739, 0.0, 12.0, 3.0e17, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 400)",
        },
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\n",
            "call": "coupling_factor(0.6168, 0.1, 12.0, 3.0e17, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 400)",
            "gold_call": "_oracle_coupling_factor(0.6168, 0.1, 12.0, 3.0e17, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 400)",
        },
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\n",
            "call": "coupling_factor(0.90, 0.0, 12.0, 3.0e17, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 400)",
            "gold_call": "_oracle_coupling_factor(0.90, 0.0, 12.0, 3.0e17, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 400)",
        },
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\n",
            "call": "coupling_factor(0.20, 0.0, 12.0, 3.0e17, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 400)",
            "gold_call": "_oracle_coupling_factor(0.20, 0.0, 12.0, 3.0e17, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 400)",
        },
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7\n",
            "call": "coupling_factor(0.55, 0.0, 250.0, 3.0e17, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 200)",
            "gold_call": "_oracle_coupling_factor(0.55, 0.0, 250.0, 3.0e17, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 200)",
        },
        {
            "setup": "import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 1.5e-7\n",
            "call": "coupling_factor(0.5739, 0.0, 12.0, 1.0e18, COX, 5.0e11, 0.1200, 4.0, 2.0, 0.070, 200)",
            "gold_call": "_oracle_coupling_factor(0.5739, 0.0, 12.0, 1.0e18, COX, 5.0e11, 0.1200, 4.0, 2.0, 0.070, 200)",
        },
            {   # a dense, slowly decaying trap spectrum, where the trap capacitance dominates
            "setup": 'import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7',
            "call": "coupling_factor(0.5739, 0.0, 12.0, 3.0e17, COX, 5.0e12, 0.4000, 2.0, 4.0, 0.045, 400)",
            "gold_call": "_oracle_coupling_factor(0.5739, 0.0, 12.0, 3.0e17, COX, 5.0e12, 0.4000, 2.0, 4.0, 0.045, 400)",
        },
        {   # heavier doping in inversion, where the depletion capacitance dominates
            "setup": 'import numpy as np\nCOX = 3.9 * 8.8541878128e-14 / 2.6e-7',
            "call": "coupling_factor(0.70, 0.0, 12.0, 1.0e18, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 400)",
            "gold_call": "_oracle_coupling_factor(0.70, 0.0, 12.0, 1.0e18, COX, 1.833e12, 0.2943, 2.0, 4.0, 0.045, 400)",
        },
]
