"""
Return the occupied interface-trap charge density at the specified surface and channel potentials.

The occupied continuous interface-trap spectrum contributes to the gate electrostatics. Trap energies are expressed in eV and the returned areal charge in C/cm^2. The supplied n_nodes controls Gauss-Legendre quadrature; the remaining parameters describe the task's trap spectrum and occupation.

Returns
-------
float, the occupied interface-trap charge density Qit in C/cm^2 as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interface_trap_charge(psi_s: float, Vch: float, T: float, DitC: float,
                          Edecay: float, gt: float, n_nodes: int) -> float:
    '''Occupied interface-trap charge density for the continuous trap spectrum.

    Parameters
    ----------
    psi_s : float
        Surface potential in volts, referenced to the intrinsic level.
    Vch : float
        Channel potential in volts, finite.
    T : float
        Lattice temperature in kelvin, strictly positive.
    DitC : float
        Interface-trap density scale at the conduction-band edge, in
        cm^-2 eV^-1, strictly positive.
    Edecay : float
        Characteristic energy decay scale of the trap spectrum, in eV,
        strictly positive.
    gt : float
        Trap degeneracy factor, strictly positive.
    n_nodes : int
        Number of Gauss-Legendre nodes, an integer of at least 2.

    Returns
    -------
    Qit : float
        Occupied interface-trap charge density in C/cm^2, negative or zero,
        as a native Python float.

    Raises
    ------
    ValueError
        If `psi_s` or `Vch` is not a finite scalar, if any of `T`, `DitC`,
        `Edecay`, `gt` is not a finite strictly positive scalar, or if
        `n_nodes` is not an integer greater than or equal to 2.
'''
    return Qit

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


def _oracle_interface_trap_charge(psi_s: float, Vch: float, T: float, DitC: float,
                          Edecay: float, gt: float, n_nodes: int) -> float:
    for name, v in (("psi_s", psi_s), ("Vch", Vch)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    for name, v in (("T", T), ("DitC", DitC), ("Edecay", Edecay), ("gt", gt)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2")
    Q = 1.602176634e-19
    T = float(T)
    eg = _bandgap(T)
    ut = _thermal_voltage(T)
    dE, w = _gauss_legendre(0.0, eg / 2.0, int(n_nodes))
    psi_t = eg / 2.0 - dE
    f = _occupation((psi_t - (float(psi_s) - float(Vch))) / ut, float(gt))
    return float(-Q * np.sum(w * float(DitC) * np.exp(-dE / float(Edecay)) * f))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        interface_trap_charge(0.5739, 0.0, -1.0, 1.833e12, 0.2943, 2.0, 400)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_interface_trap_charge(0.5739, 0.0, -1.0, 1.833e12, 0.2943, 2.0, 400)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np",
            "call": "interface_trap_charge(0.5739, 0.0, 12.0, 1.833e12, 0.2943, 2.0, 400)",
            "gold_call": "_oracle_interface_trap_charge(0.5739, 0.0, 12.0, 1.833e12, 0.2943, 2.0, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "interface_trap_charge(0.6168, 0.1, 12.0, 1.833e12, 0.2943, 2.0, 400)",
            "gold_call": "_oracle_interface_trap_charge(0.6168, 0.1, 12.0, 1.833e12, 0.2943, 2.0, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "interface_trap_charge(-0.40, 0.0, 12.0, 1.833e12, 0.2943, 2.0, 400)",
            "gold_call": "_oracle_interface_trap_charge(-0.40, 0.0, 12.0, 1.833e12, 0.2943, 2.0, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "interface_trap_charge(0.90, 0.0, 12.0, 1.833e12, 0.2943, 2.0, 400)",
            "gold_call": "_oracle_interface_trap_charge(0.90, 0.0, 12.0, 1.833e12, 0.2943, 2.0, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "interface_trap_charge(0.55, 0.0, 250.0, 1.833e12, 0.2943, 2.0, 400)",
            "gold_call": "_oracle_interface_trap_charge(0.55, 0.0, 250.0, 1.833e12, 0.2943, 2.0, 400)",
        },
        {
            "setup": "import numpy as np",
            "call": "interface_trap_charge(0.5739, 0.0, 12.0, 5.0e11, 0.1200, 4.0, 200)",
            "gold_call": "_oracle_interface_trap_charge(0.5739, 0.0, 12.0, 5.0e11, 0.1200, 4.0, 200)",
        },
            {   # a coarse quadrature of the same trap integral
            "setup": 'import numpy as np',
            "call": "interface_trap_charge(0.5739, 0.0, 12.0, 1.833e12, 0.2943, 2.0, 16)",
            "gold_call": "_oracle_interface_trap_charge(0.5739, 0.0, 12.0, 1.833e12, 0.2943, 2.0, 16)",
        },
        {   # a dense, sharply decaying trap spectrum at intermediate temperature
            "setup": 'import numpy as np',
            "call": "interface_trap_charge(0.62, 0.0, 77.0, 5.0e12, 0.1200, 2.0, 400)",
            "gold_call": "_oracle_interface_trap_charge(0.62, 0.0, 77.0, 5.0e12, 0.1200, 2.0, 400)",
        },
]
