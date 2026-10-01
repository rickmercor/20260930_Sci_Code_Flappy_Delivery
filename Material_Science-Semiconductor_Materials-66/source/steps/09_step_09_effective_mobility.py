"""
Return the effective electron mobility for the supplied depletion and mobile charges.

This transport coefficient is evaluated from the source-side state for the task's low-drain-bias calculation. Charges are in C/cm^2, mobility is in cm^2/(V s), the field normaliser is the task's fixed E0=1e6 V/cm, and eta is supplied explicitly.

Returns
-------
float, the effective electron mobility mu_n in cm^2 V^-1 s^-1 as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_mobility(Qdep: float, Qm: float, mu0: float, theta1: float,
                       theta2: float, eta: float) -> float:
    '''Electron mobility degraded by the effective vertical field.

    Parameters
    ----------
    Qdep : float
        Depletion charge density in C/cm^2, finite.
    Qm : float
        Mobile inversion charge density in C/cm^2, finite.
    mu0 : float
        Low-field electron mobility at the operating temperature, in
        cm^2 V^-1 s^-1, strictly positive.
    theta1 : float
        Dimensionless linear mobility-degradation coefficient, non-negative.
    theta2 : float
        Dimensionless quadratic mobility-degradation coefficient, non-negative.
    eta : float
        Weight of the mobile inversion charge in the effective field,
        non-negative.

    Returns
    -------
    mu_n : float
        Effective electron mobility in cm^2 V^-1 s^-1, as a native Python float.

    Raises
    ------
    ValueError
        If `Qdep` or `Qm` is not a finite scalar, if `mu0` is not a finite
        strictly positive scalar, or if any of `theta1`, `theta2`, `eta` is not
        a finite non-negative scalar.
'''
    return mu_n

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_effective_mobility(Qdep: float, Qm: float, mu0: float, theta1: float,
                       theta2: float, eta: float) -> float:
    for name, v in (("Qdep", Qdep), ("Qm", Qm)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    if not np.isscalar(mu0) or not np.isfinite(float(mu0)) or float(mu0) <= 0.0:
        raise ValueError("mu0 must be a finite strictly positive scalar")
    for name, v in (("theta1", theta1), ("theta2", theta2), ("eta", eta)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) < 0.0:
            raise ValueError("%s must be a finite non-negative scalar" % name)
    EPS_SI = 11.7 * 8.8541878128e-14
    E0 = 1.0e6
    Eeff = (abs(float(Qdep)) + float(eta) * abs(float(Qm))) / EPS_SI
    F = Eeff / E0
    return float(float(mu0) / (1.0 + float(theta1) * F + float(theta2) * F * F))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        effective_mobility(-3.2833e-07, -3.5651e-08, -1.0, 0.55428, 0.10007, 0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_effective_mobility(-3.2833e-07, -3.5651e-08, -1.0, 0.55428, 0.10007, 0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np",
            "call": "effective_mobility(-3.2833e-07, -3.5651e-08, 395.655, 0.55428, 0.10007, 0.5)",
            "gold_call": "_oracle_effective_mobility(-3.2833e-07, -3.5651e-08, 395.655, 0.55428, 0.10007, 0.5)",
        },
        {
            "setup": "import numpy as np",
            "call": "effective_mobility(-3.2833e-07, 0.0, 395.655, 0.55428, 0.10007, 0.5)",
            "gold_call": "_oracle_effective_mobility(-3.2833e-07, 0.0, 395.655, 0.55428, 0.10007, 0.5)",
        },
        {
            "setup": "import numpy as np",
            "call": "effective_mobility(-3.5821e-07, -9.7498e-04, 395.655, 0.55428, 0.10007, 0.5)",
            "gold_call": "_oracle_effective_mobility(-3.5821e-07, -9.7498e-04, 395.655, 0.55428, 0.10007, 0.5)",
        },
        {
            "setup": "import numpy as np",
            "call": "effective_mobility(0.0, 0.0, 395.655, 0.55428, 0.10007, 0.5)",
            "gold_call": "_oracle_effective_mobility(0.0, 0.0, 395.655, 0.55428, 0.10007, 0.5)",
        },
        {
            "setup": "import numpy as np",
            "call": "effective_mobility(-3.2833e-07, -3.5651e-08, 314.822, 0.37306, 0.57612, 0.5)",
            "gold_call": "_oracle_effective_mobility(-3.2833e-07, -3.5651e-08, 314.822, 0.37306, 0.57612, 0.5)",
        },
        {
            "setup": "import numpy as np",
            "call": "effective_mobility(-3.2833e-07, -3.5651e-08, 395.655, 0.0, 0.0, 1.0)",
            "gold_call": "_oracle_effective_mobility(-3.2833e-07, -3.5651e-08, 395.655, 0.0, 0.0, 1.0)",
        },
            {   # the heavily doped device in strong inversion, deep into degradation
            "setup": 'import numpy as np',
            "call": "effective_mobility(-1.312891e-06, -2.753406e-05, 395.655, 0.55428, 0.10007, 0.5)",
            "gold_call": "_oracle_effective_mobility(-1.312891e-06, -2.753406e-05, 395.655, 0.55428, 0.10007, 0.5)",
        },
]
