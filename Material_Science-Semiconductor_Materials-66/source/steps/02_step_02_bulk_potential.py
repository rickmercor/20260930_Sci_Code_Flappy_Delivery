"""
Return the equilibrium bulk potential of the p-type body including incomplete dopant ionisation.

The bulk potential fixes the field-free electrostatic reference used throughout the model. Potentials are referenced to the intrinsic level under the task's midgap approximation. The supplied acceptor level is measured upward from the valence-band edge

Returns
-------
float, the bulk potential psi_b in volts referenced to the intrinsic level, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bulk_potential(T: float, NA: float, gA: float, EA_above_EV: float) -> float:
    '''Equilibrium bulk potential of the p-type body including incomplete ionisation.

    Parameters
    ----------
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level measured upward from the valence-band edge, in eV,
        strictly positive.

    Returns
    -------
    psi_b : float
        Bulk potential in volts referenced to the intrinsic level, as a native
        Python float. Negative for a p-type body.

    Raises
    ------
    ValueError
        If any of `T`, `NA`, `gA`, `EA_above_EV` is not a finite strictly
        positive scalar.
'''
    return psi_b

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


def _psi_bulk(T, NA, gA, EA):
    """Bulk potential from charge neutrality, evaluated stably in log space."""
    ut = _thermal_voltage(T)
    lni = _oracle_log_intrinsic_carrier_density(T, 2.86e19, 2.66e19)
    psiA = EA - _bandgap(T) / 2.0
    S = np.log(4.0 * gA * NA) + psiA / ut - lni
    corr = np.logaddexp(0.0, 0.5 * np.logaddexp(0.0, S)) - np.log(2.0)
    return float(ut * ((lni - np.log(NA)) + corr))


def _oracle_bulk_potential(T: float, NA: float, gA: float, EA_above_EV: float) -> float:
    for name, v in (("T", T), ("NA", NA), ("gA", gA), ("EA_above_EV", EA_above_EV)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    return _psi_bulk(float(T), float(NA), float(gA), float(EA_above_EV))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        bulk_potential(-1.0, 3.0e17, 4.0, 0.045)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_bulk_potential(-1.0, 3.0e17, 4.0, 0.045)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np",
            "call": "bulk_potential(12.0, 3.0e17, 4.0, 0.045)",
            "gold_call": "_oracle_bulk_potential(12.0, 3.0e17, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "bulk_potential(300.0, 3.0e17, 4.0, 0.045)",
            "gold_call": "_oracle_bulk_potential(300.0, 3.0e17, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "bulk_potential(4.2, 3.0e17, 4.0, 0.045)",
            "gold_call": "_oracle_bulk_potential(4.2, 3.0e17, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "bulk_potential(12.0, 1.0e15, 4.0, 0.045)",
            "gold_call": "_oracle_bulk_potential(12.0, 1.0e15, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "bulk_potential(75.0, 3.0e17, 2.0, 0.070)",
            "gold_call": "_oracle_bulk_potential(75.0, 3.0e17, 2.0, 0.070)",
        },
            {   # heavy doping deep in freeze-out, where the correction is largest
            "setup": 'import numpy as np',
            "call": "bulk_potential(4.2, 5.0e18, 4.0, 0.045)",
            "gold_call": "_oracle_bulk_potential(4.2, 5.0e18, 4.0, 0.045)",
        },
        {   # light doping near complete ionisation, where it nearly vanishes
            "setup": 'import numpy as np',
            "call": "bulk_potential(250.0, 1.0e15, 4.0, 0.045)",
            "gold_call": "_oracle_bulk_potential(250.0, 1.0e15, 4.0, 0.045)",
        },
]
