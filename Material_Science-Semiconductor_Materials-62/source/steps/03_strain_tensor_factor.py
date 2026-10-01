"""
Return the 3x3 strain-dependent tensor factor that the source's effective conduction-band parameters carry in place of the identity, evaluated through first order in the strain with all O(e^2) terms discarded, as the problem statement prescribes. It must reduce to the identity at zero strain. Raise ValueError if e is not 3x3, not finite, or not symmetric.

Folding the valence bands into the conduction band under strain leaves the interband coupling multiplied by a tensor built from the strain. The source names that tensor and gives its first-order form; which entries carry which strains is what the source states.

Returns
-------
numpy.ndarray of float64 with shape (3, 3), symmetric.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def strain_tensor_factor(e: "np.ndarray") -> "np.ndarray":
    """Return the 3x3 strain-dependent tensor factor that the source's effective conduction-band parameters carry in place of the identity, evaluated through first order in the strain with all O(e^2) terms discarded, as the problem statement prescribes. It must reduce to the identity at zero strain. Raise ValueError if e is not 3x3, not finite, or not symmetric.

    Parameters
    ----------
    e : numpy.ndarray
        Symmetric strain tensor of shape (3, 3) in the crystal frame (x, y, z along the cubic axes, z the (001) growth axis), dimensionless.

    Returns
    -------
    adj : numpy.ndarray
        Array of shape (3, 3), symmetric, equal to the identity at zero strain (float64).

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


def _oracle_strain_tensor_factor(e: "np.ndarray") -> "np.ndarray":
    """(A38): adj{1 - e} through FIRST order in the strain, O(e^2) discarded.

    adj{1-e} = [[1-e_yy-e_zz,  e_xy,        e_xz     ],
                [e_xy,         1-e_xx-e_zz, e_yz     ],
                [e_xz,         e_yz,        1-e_xx-e_yy]]
    The prompt declares the truncation; the source fixes which entries carry which strains.
    """
    e = np.asarray(e, dtype=np.float64)
    if e.shape != (3, 3):
        raise ValueError("strain tensor must be 3x3")
    if not np.all(np.isfinite(e)):
        raise ValueError("strain tensor must be finite")
    if not np.allclose(e, e.T, rtol=0.0, atol=1e-12):
        raise ValueError("strain tensor must be symmetric")
    exx, eyy, ezz = e[0, 0], e[1, 1], e[2, 2]
    exy, exz, eyz = e[0, 1], e[0, 2], e[1, 2]
    return np.array([[1.0 - eyy - ezz, exy, exz],
                     [exy, 1.0 - exx - ezz, eyz],
                     [exz, eyz, 1.0 - exx - eyy]], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ne = np.array([[0.020, 0.006, 0.004], [0.006, 0.015, 0.008], [0.004, 0.008, -0.018]])\n",
            "call": "strain_tensor_factor(e)",
            "gold_call": "_oracle_strain_tensor_factor(e)",
        },
        {
            "setup": "import numpy as np\ne = np.zeros((3, 3))\n",
            "call": "strain_tensor_factor(e)",
            "gold_call": "_oracle_strain_tensor_factor(e)",
        },
        {
            "setup": "import numpy as np\ne = np.array([[0.0, 0.004, 0.0], [0.004, 0.0, 0.006], [0.0, 0.006, 0.0]])\n",
            "call": "strain_tensor_factor(e)",
            "gold_call": "_oracle_strain_tensor_factor(e)",
        },
        {
            "setup": "import numpy as np\ne = np.array([[0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])\ndef run_model():\n    try:\n        strain_tensor_factor(e)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_strain_tensor_factor(e)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
