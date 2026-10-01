"""
Return the temperature-dependent metal-semiconductor work-function difference.

This reference quantity fixes electrostatic alignment between the supplied gate material and silicon body. Use volts for the work functions, electron affinity and returned difference, with the same intrinsic-level convention as the rest of the task.

Returns
-------
float, the metal-semiconductor work-function difference phi_ms in volts as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def work_function_difference(T: float, NA: float, phi_m: float, chi_si: float,
                             gA: float, EA_above_EV: float) -> float:
    '''Metal-semiconductor work-function difference at temperature T.

    Parameters
    ----------
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    phi_m : float
        Gate work function in volts, treated as temperature independent,
        strictly positive.
    chi_si : float
        Electron affinity of silicon in volts, strictly positive.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level measured upward from the valence-band edge, in eV,
        strictly positive.

    Returns
    -------
    phi_ms : float
        Work-function difference in volts, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is not a finite strictly positive scalar.
'''
    return phi_ms

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


def _oracle_work_function_difference(T: float, NA: float, phi_m: float, chi_si: float,
                             gA: float, EA_above_EV: float) -> float:
    for name, v in (("T", T), ("NA", NA), ("phi_m", phi_m), ("chi_si", chi_si),
                    ("gA", gA), ("EA_above_EV", EA_above_EV)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    T = float(T)
    psib = _oracle_bulk_potential(T, float(NA), float(gA), float(EA_above_EV))
    return float(float(phi_m) - (float(chi_si) + _bandgap(T) / 2.0 - psib))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        work_function_difference(-1.0, 3.0e17, 4.35, 4.05, 4.0, 0.045)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_work_function_difference(-1.0, 3.0e17, 4.35, 4.05, 4.0, 0.045)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np",
            "call": "work_function_difference(12.0, 3.0e17, 4.35, 4.05, 4.0, 0.045)",
            "gold_call": "_oracle_work_function_difference(12.0, 3.0e17, 4.35, 4.05, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "work_function_difference(300.0, 3.0e17, 4.35, 4.05, 4.0, 0.045)",
            "gold_call": "_oracle_work_function_difference(300.0, 3.0e17, 4.35, 4.05, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "work_function_difference(4.2, 3.0e17, 4.35, 4.05, 4.0, 0.045)",
            "gold_call": "_oracle_work_function_difference(4.2, 3.0e17, 4.35, 4.05, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "work_function_difference(12.0, 1.0e18, 4.10, 4.05, 4.0, 0.045)",
            "gold_call": "_oracle_work_function_difference(12.0, 1.0e18, 4.10, 4.05, 4.0, 0.045)",
        },
        {
            "setup": "import numpy as np",
            "call": "work_function_difference(150.0, 3.0e17, 4.35, 4.05, 2.0, 0.070)",
            "gold_call": "_oracle_work_function_difference(150.0, 3.0e17, 4.35, 4.05, 2.0, 0.070)",
        },
            {   # a different gate metal and dopant species when cold
            "setup": 'import numpy as np',
            "call": "work_function_difference(4.2, 1.0e18, 4.10, 4.05, 2.0, 0.070)",
            "gold_call": "_oracle_work_function_difference(4.2, 1.0e18, 4.10, 4.05, 2.0, 0.070)",
        },
]
