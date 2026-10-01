"""
Compute the two primitive reciprocal-lattice vectors of the moire superlattice formed by twisting two layers with a common lattice constant.

A small twist theta between two hexagonal lattices of lattice constant a produces a hexagonal moire pattern
whose period follows from the mismatch of the two layers' reciprocal lattices; the lattice-constant mismatch
of the two materials is neglected and the lattice constant of the electron layer is used. The moire
reciprocal vectors have a common length G and enclose 120 degrees. The orientation used throughout the
task is b1 = G (sqrt(3)/2, -1/2) and b2 = G (0, 1), in inverse nanometres.

Returns
-------
b_vectors : np.ndarray -- Float array of shape (2, 2); row 0 is b1 = G (sqrt(3)/2, -1/2), row 1 is b2 = G (0, 1),
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def moire_reciprocal_lattice(theta_deg: float, a_lattice: float) -> "np.ndarray":
    """Return the moire reciprocal-lattice vectors for a twisted hexagonal bilayer.

    Parameters
    ----------
    theta_deg : float
        Twist angle in degrees, 0 < theta_deg < 60.
    a_lattice : float
        Monolayer lattice constant in nm, positive.

    Returns
    -------
    b_vectors : np.ndarray
        Float array of shape (2, 2); row 0 is b1 = G (sqrt(3)/2, -1/2), row 1 is b2 = G (0, 1),
        where G is the length of a primitive moire reciprocal vector, in nm^-1.

    Raises
    ------
    ValueError
        If theta_deg is not strictly between 0 and 60 or a_lattice is not positive.
    """
    return b_vectors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_moire_reciprocal_lattice(theta_deg: float, a_lattice: float) -> "np.ndarray":
    if not (0.0 < theta_deg < 60.0) or a_lattice <= 0.0:
        raise ValueError("need 0 < theta_deg < 60 and a_lattice > 0")
    a_m = a_lattice / (2.0 * np.sin(np.deg2rad(theta_deg) / 2.0))
    g = 4.0 * np.pi / (np.sqrt(3.0) * a_m)
    return np.array([[g * np.sqrt(3.0) / 2.0, -g / 2.0], [0.0, g]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    invalid = """import numpy as np
def run(fn):
    try:
        fn(0.0, 0.327)
        return 0
    except ValueError:
        return 1
"""
    return [
        {"setup": "import numpy as np",
         "call": "moire_reciprocal_lattice(3.0, 0.327)",
         "gold_call": "_oracle_moire_reciprocal_lattice(3.0, 0.327)"},
        {"setup": "import numpy as np",
         "call": "moire_reciprocal_lattice(0.5, 0.325)",
         "gold_call": "_oracle_moire_reciprocal_lattice(0.5, 0.325)"},
        {"setup": "import numpy as np",
         "call": "moire_reciprocal_lattice(21.8, 0.3288)",
         "gold_call": "_oracle_moire_reciprocal_lattice(21.8, 0.3288)"},
        {"setup": invalid,
         "call": "run(moire_reciprocal_lattice)",
         "gold_call": "run(_oracle_moire_reciprocal_lattice)"},
    ]
