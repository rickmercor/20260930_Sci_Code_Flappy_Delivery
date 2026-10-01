"""
Collect the four constant coefficients of the radial stretch term of the model potential from its four published parameters, so that the potential can afterwards be evaluated as a fixed linear combination of an exponential and two inverse powers. The radial term is the common prefactor de/(c1 - 6) multiplying a bracket of three contributions in the scaled separation: plus 2(3 - c2) times the exponential, minus (4c2 - c1c2 + c1) times the inverse sixth power, and minus (c1 - 6)c2 times the inverse fourth power. Return the prefactor first, then those three bracket coefficients in that order, each multiplied by the prefactor.

The radial term of this potential is written as a single prefactor times a bracket holding an exponential and two inverse powers of the scaled separation. The exponential contribution enters positively and the two inverse powers enter negatively, so the repulsive wall and the long-range attraction have opposite signs. Splitting the constants out once keeps the later evaluations cheap and makes the sign of each contribution explicit; this step is bookkeeping, and the science of the chain sits in the steps that follow.

Returns
-------
ndarray of shape (4,): de/(c1-6), then 2(3-c2)*de/(c1-6), then -(4c2-c1c2+c1)*de/(c1-6), then -(c1-6)c2*de/(c1-6).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def extract_radial_parameters(de: float, re: float, c1: float, c2: float) -> "np.ndarray":
    """Collect the four constant coefficients of the radial stretch term of the model potential from its four published parameters, so that the potential can afterwards be evaluated as a fixed linear combination of an exponential and two inverse powers. The radial term is the common prefactor de/(c1 - 6) multiplying a bracket of three contributions in the scaled separation: plus 2(3 - c2) times the exponential, minus (4c2 - c1c2 + c1) times the inverse sixth power, and minus (c1 - 6)c2 times the inverse fourth power. Return the prefactor first, then those three bracket coefficients in that order, each multiplied by the prefactor.

    Parameters
    ----------
    de : float
        Positive well-depth scale of the radial term, in kcal/mol.
    re : float
        Positive reference separation, in angstroms.
    c1 : float
        Positive shape exponent of the repulsive block; must not equal six.
    c2 : float
        Positive weight of the inverse-fourth block.

    Returns
    -------
    coefficients : np.ndarray
        ndarray of shape (4,): de/(c1-6), then 2(3-c2)*de/(c1-6), then -(4c2-c1c2+c1)*de/(c1-6), then -(c1-6)c2*de/(c1-6).

    Raises
    ------
    ValueError
        if any parameter is not a positive finite number, or if c1 equals six, which makes the prefactor singular.
    """
    return coefficients

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp


def _oracle_extract_radial_parameters(de: float, re: float, c1: float, c2: float) -> "np.ndarray":
    for nm, v in (("de", de), ("re", re), ("c1", c1), ("c2", c2)):
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"{nm} must be a positive finite number")
    if abs(c1 - 6.0) < 1e-12:
        raise ValueError("c1 must not equal 6: the prefactor de/(c1-6) is singular")
    pref = de / (c1 - 6.0)
    return np.array([pref,
                     2.0 * (3.0 - c2) * pref,
                     -(4.0 * c2 - c1 * c2 + c1) * pref,
                     -(c1 - 6.0) * c2 * pref], float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test-case specifications for this step."""
    return [
        {"setup": "import numpy as np",
         "call": "extract_radial_parameters(47.0, 1.1, 7.37, 1.61)",
         "gold_call": "_oracle_extract_radial_parameters(47.0, 1.1, 7.37, 1.61)"},   # normal
        {"setup": "import numpy as np",
         "call": "extract_radial_parameters(50.0, 1.0, 8.0, 2.0)",
         "gold_call": "_oracle_extract_radial_parameters(50.0, 1.0, 8.0, 2.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "extract_radial_parameters(1.0, 0.5, 7.0, 0.25)",
         "gold_call": "_oracle_extract_radial_parameters(1.0, 0.5, 7.0, 0.25)"},   # boundary
        {"setup": "import numpy as np",
         "call": "extract_radial_parameters(47.0, 1.1, 6.5, 1.61)",
         "gold_call": "_oracle_extract_radial_parameters(47.0, 1.1, 6.5, 1.61)"},   # edge
        {"setup": "import numpy as np\ndef _c(fn):\n    try:\n        fn(47.0, 1.1, 6.0, 1.61)\n    except ValueError:\n        return 1\n    except Exception:\n        return 0\n    return 0",
         "call": "_c(extract_radial_parameters)", "gold_call": "_c(_oracle_extract_radial_parameters)"},   # exception contract
    ]
