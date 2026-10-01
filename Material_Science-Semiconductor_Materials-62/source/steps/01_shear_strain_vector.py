"""
Return the vector of the three independent shear strains of a symmetric 3x3 strain tensor, with the components in the order the source uses. Raise ValueError if e is not 3x3, not finite, or not symmetric.

A general deformation of a cubic cell has three normal strains on the diagonal and three shear strains off it. The source collects the shears into one vector that enters its strained band gaps only through its magnitude.

Returns
-------
numpy.ndarray of float64 with shape (3,): the shear-strain vector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def shear_strain_vector(e: "np.ndarray") -> "np.ndarray":
    """Return the vector of the three independent shear strains of a symmetric 3x3 strain tensor, with the components in the order the source uses. Raise ValueError if e is not 3x3, not finite, or not symmetric.

    Parameters
    ----------
    e : numpy.ndarray
        Symmetric strain tensor of shape (3, 3) in the crystal frame (x, y, z along the cubic axes, z the (001) growth axis), dimensionless.

    Returns
    -------
    e_s : numpy.ndarray
        Array of shape (3,) holding the three shear strains in the source's order (float64).

    Raises
    ------
    ValueError
        If e is not of shape (3, 3), contains a non-finite entry, or is not symmetric.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_shear_strain_vector(e: "np.ndarray") -> "np.ndarray":
    """The shear-strain vector e_s = (e_yz, e_zx, e_xy) of a symmetric strain tensor."""
    e = np.asarray(e, dtype=np.float64)
    if e.shape != (3, 3):
        raise ValueError("strain tensor must be 3x3")
    if not np.all(np.isfinite(e)):
        raise ValueError("strain tensor must be finite")
    if not np.allclose(e, e.T, rtol=0.0, atol=1e-12):
        raise ValueError("strain tensor must be symmetric")
    return np.array([e[1, 2], e[2, 0], e[0, 1]], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ne = np.array([[0.020, 0.006, 0.004], [0.006, 0.015, 0.008], [0.004, 0.008, -0.018]])\n",
            "call": "shear_strain_vector(e)",
            "gold_call": "_oracle_shear_strain_vector(e)",
        },
        {
            "setup": "import numpy as np\ne = np.diag([0.01, 0.01, -0.009])\n",
            "call": "shear_strain_vector(e)",
            "gold_call": "_oracle_shear_strain_vector(e)",
        },
        {
            "setup": "import numpy as np\ne = np.array([[0.0, 0.004, 0.0], [0.004, 0.0, 0.006], [0.0, 0.006, 0.0]])\n",
            "call": "shear_strain_vector(e)",
            "gold_call": "_oracle_shear_strain_vector(e)",
        },
        {
            "setup": "import numpy as np\ne = np.array([[0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])\ndef run_model():\n    try:\n        shear_strain_vector(e)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_shear_strain_vector(e)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
