"""
Return the effective conduction-band g-tensor G obtained by rewriting the sum of the source's isotropic and anisotropic Zeeman terms as one term of the form (mu_B/2) B . G . sigma, with B the magnetic field and sigma the vector of Pauli matrices. Follow exactly which spin components the source's anisotropic term couples to; G is not required to be symmetric. Raise ValueError if either input is not 3x3.

The two contributions do not simply add. The isotropic term acts on all three spin components, whereas the anisotropic term the source derives is tied to the growth axis of the zincblende cell, and that determines which entries of the effective tensor it can reach.

Returns
-------
numpy.ndarray of float64 with shape (3, 3), the effective g-tensor.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_g_tensor(g_iso: "np.ndarray", g_ani: "np.ndarray") -> "np.ndarray":
    """Return the effective conduction-band g-tensor G obtained by rewriting the sum of the source's isotropic and anisotropic Zeeman terms as one term of the form (mu_B/2) B . G . sigma, with B the magnetic field and sigma the vector of Pauli matrices. Follow exactly which spin components the source's anisotropic term couples to; G is not required to be symmetric. Raise ValueError if either input is not 3x3.

    Parameters
    ----------
    g_iso : numpy.ndarray
        Isotropic effective g-tensor of shape (3, 3) from step 04.
    g_ani : numpy.ndarray
        Anisotropic g correction of shape (3, 3) from step 05.

    Returns
    -------
    g : numpy.ndarray
        Array of shape (3, 3): the effective g-tensor G, dimensionless (float64), not necessarily symmetric.

    Raises
    ------
    ValueError
        If g_iso or g_ani is not of shape (3, 3).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_effective_g_tensor(g_iso: "np.ndarray", g_ani: "np.ndarray") -> "np.ndarray":
    """(A28)-(A29): the two Zeeman terms are
           (mu_B/2) B . g_iso . sigma      and      (mu_B/2) B . g_ani . (0, 0, sigma_z)^T.
    Rewriting their sum as a single (mu_B/2) B . G . sigma gives
           G_jk = g_iso_jk + delta_kz g_ani_jz ,
    i.e. g_ani contributes ONLY the z-COLUMN of G (every B_j couples to sigma_z through
    g_ani_jz) and nothing else. G is in general not symmetric; that is allowed for a
    g-tensor, only G G^T is observable.
    """
    g_iso = np.asarray(g_iso, dtype=np.float64)
    g_ani = np.asarray(g_ani, dtype=np.float64)
    if g_iso.shape != (3, 3) or g_ani.shape != (3, 3):
        raise ValueError("both g tensors must be 3x3")
    g = g_iso.copy()
    g[:, 2] += g_ani[:, 2]
    return g

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ng_iso = np.array([[-24.0, 0.1, 0.2], [0.1, -23.9, 0.3], [0.2, 0.3, -23.0]])\ng_ani = np.array([[1.0, 0.05, 0.06], [0.05, 1.1, 0.07], [0.06, 0.07, 2.3]])\n",
            "call": "effective_g_tensor(g_iso, g_ani)",
            "gold_call": "_oracle_effective_g_tensor(g_iso, g_ani)",
        },
        {
            "setup": "import numpy as np\ng_iso = np.diag([-15.0, -15.0, -15.0])\ng_ani = np.zeros((3, 3))\n",
            "call": "effective_g_tensor(g_iso, g_ani)",
            "gold_call": "_oracle_effective_g_tensor(g_iso, g_ani)",
        },
        {
            "setup": "import numpy as np\ng_iso = np.eye(3) * 2.0\ng_ani = np.full((3, 3), 0.5)\n",
            "call": "effective_g_tensor(g_iso, g_ani)",
            "gold_call": "_oracle_effective_g_tensor(g_iso, g_ani)",
        },
        {
            "setup": "import numpy as np\ng_iso = np.eye(2)\ng_ani = np.eye(3)\ndef run_model():\n    try:\n        effective_g_tensor(g_iso, g_ani)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_effective_g_tensor(g_iso, g_ani)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
