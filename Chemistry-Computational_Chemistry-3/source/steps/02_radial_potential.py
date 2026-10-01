"""
Evaluate the radial stretch term of the model potential at one or more internuclear separations, using the coefficients of the previous step. The separation enters only through its ratio to the reference length.

This term carries the covalent stretch of the breaking bond: an exponential repulsion at short range against the two attractive inverse powers that produce the long-range tail.

Returns
-------
ndarray with the same shape as r: the radial potential in kcal/mol.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radial_potential(r: "np.ndarray", de: float, re: float, c1: float, c2: float) -> "np.ndarray":
    """Evaluate the radial stretch term of the model potential at one or more internuclear separations, using the coefficients of the previous step. The separation enters only through its ratio to the reference length.

    Parameters
    ----------
    r : np.ndarray
        Strictly positive internuclear separation(s), in angstroms.
    de : float
        Positive well-depth scale, in kcal/mol.
    re : float
        Positive reference separation, in angstroms.
    c1 : float
        Positive shape exponent; must not equal six.
    c2 : float
        Positive weight of the inverse-fourth block.

    Returns
    -------
    v_radial : np.ndarray
        ndarray with the same shape as r: the radial potential in kcal/mol.

    Raises
    ------
    ValueError
        if any separation is not strictly positive, or if the parameters are invalid for the previous step.
    """
    return v_radial

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp


def _oracle_radial_potential(r: "np.ndarray", de: float, re: float, c1: float, c2: float) -> "np.ndarray":
    a = _oracle_extract_radial_parameters(de, re, c1, c2)
    r = np.asarray(r)
    if not np.iscomplexobj(r):
        r = r.astype(float)
        if np.any(r <= 0.0):
            raise ValueError("r must be strictly positive")
    x = r / re
    return a[1] * np.exp(c1 * (1.0 - x)) + a[2] * x ** -6 + a[3] * x ** -4

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test-case specifications for this step."""
    return [
        {"setup": "import numpy as np",
         "call": "radial_potential(3.4, 47.0, 1.1, 7.37, 1.61)",
         "gold_call": "_oracle_radial_potential(3.4, 47.0, 1.1, 7.37, 1.61)"},   # normal
        {"setup": "import numpy as np",
         "call": "radial_potential(np.array([2.0, 3.0, 4.0]), 47.0, 1.1, 7.37, 1.61)",
         "gold_call": "_oracle_radial_potential(np.array([2.0, 3.0, 4.0]), 47.0, 1.1, 7.37, 1.61)"},   # normal
        {"setup": "import numpy as np",
         "call": "radial_potential(1.1, 47.0, 1.1, 7.37, 1.61)",
         "gold_call": "_oracle_radial_potential(1.1, 47.0, 1.1, 7.37, 1.61)"},   # boundary
        {"setup": "import numpy as np",
         "call": "radial_potential(12.0, 47.0, 1.1, 7.37, 1.61)",
         "gold_call": "_oracle_radial_potential(12.0, 47.0, 1.1, 7.37, 1.61)"},   # edge
        {"setup": "import numpy as np\ndef _c(fn):\n    try:\n        fn(-1.0, 47.0, 1.1, 7.37, 1.61)\n    except ValueError:\n        return 1\n    except Exception:\n        return 0\n    return 0",
         "call": "_c(radial_potential)", "gold_call": "_c(_oracle_radial_potential)"},   # exception contract
    ]
