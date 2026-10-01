"""
Return the natural logarithm of silicon's intrinsic carrier concentration for the supplied temperature and band-edge density-of-states references.

The logarithmic carrier-density reference supports the subsequent cryogenic electrostatics without requiring a representable carrier concentration. Use the task's prescribed band-gap and temperature-scaling definitions. Concentrations are expressed in cm^-3 and temperature in kelvin.

Returns
-------
float, the natural logarithm of the intrinsic carrier concentration n_i in cm^-3 as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def log_intrinsic_carrier_density(T: float, NC300: float, NV300: float) -> float:
    '''Natural logarithm of the intrinsic carrier concentration of silicon.

    Parameters
    ----------
    T : float
        Lattice temperature in kelvin, strictly positive.
    NC300 : float
        Conduction-band effective density of states at 300 K, in cm^-3,
        strictly positive.
    NV300 : float
        Valence-band effective density of states at 300 K, in cm^-3,
        strictly positive.

    Returns
    -------
    log_ni : float
        Natural logarithm of the intrinsic carrier concentration in cm^-3, as a
        native Python float. Always finite; large and negative at cryogenic
        temperature.

    Raises
    ------
    ValueError
        If `T` is not a finite strictly positive scalar, or if either of `NC300`
        and `NV300` is not a finite strictly positive scalar.
'''
    return log_ni

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


def _oracle_log_intrinsic_carrier_density(T: float, NC300: float, NV300: float) -> float:
    for name, v in (("T", T), ("NC300", NC300), ("NV300", NV300)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    T = float(T)
    s = (T / 300.0) ** 1.5
    return float(0.5 * (np.log(float(NC300) * s) + np.log(float(NV300) * s))
                 - _bandgap(T) / (2.0 * _thermal_voltage(T)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        log_intrinsic_carrier_density(-1.0, 2.86e19, 2.66e19)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_log_intrinsic_carrier_density(-1.0, 2.86e19, 2.66e19)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np",
            "call": "log_intrinsic_carrier_density(12.0, 2.86e19, 2.66e19)",
            "gold_call": "_oracle_log_intrinsic_carrier_density(12.0, 2.86e19, 2.66e19)",
        },
        {
            "setup": "import numpy as np",
            "call": "log_intrinsic_carrier_density(300.0, 2.86e19, 2.66e19)",
            "gold_call": "_oracle_log_intrinsic_carrier_density(300.0, 2.86e19, 2.66e19)",
        },
        {
            "setup": "import numpy as np",
            "call": "log_intrinsic_carrier_density(75.0, 2.86e19, 2.66e19)",
            "gold_call": "_oracle_log_intrinsic_carrier_density(75.0, 2.86e19, 2.66e19)",
        },
        {
            "setup": "import numpy as np",
            "call": "log_intrinsic_carrier_density(1.0, 2.86e19, 2.66e19)",
            "gold_call": "_oracle_log_intrinsic_carrier_density(1.0, 2.86e19, 2.66e19)",
        },
        {
            "setup": "import numpy as np",
            "call": "log_intrinsic_carrier_density(250.0, 3.20e19, 1.80e19)",
            "gold_call": "_oracle_log_intrinsic_carrier_density(250.0, 3.20e19, 1.80e19)",
        },
            {   # deep cryogenic with an asymmetric band-edge pair
            "setup": 'import numpy as np',
            "call": "log_intrinsic_carrier_density(4.2, 3.20e19, 1.80e19)",
            "gold_call": "_oracle_log_intrinsic_carrier_density(4.2, 3.20e19, 1.80e19)",
        },
]
