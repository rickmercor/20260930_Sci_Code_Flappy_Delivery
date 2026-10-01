"""
Return the anisotropic correction to the effective g-tensor that the source derives from the strain-induced splitting of the light and heavy holes, from P in eV nm, the tensor factor, and the strained conduction-to-light-hole and conduction-to-heavy-hole gaps in eV, using hbar^2/(2 m_e) = 0.0380998212 eV nm^2 as given in the problem statement. It must vanish identically when those two gaps are equal. Raise ValueError if P or either gap is not positive, or if adj is not 3x3.

Without strain the light and heavy holes are degenerate at the zone centre and the conduction band sees them symmetrically. Strain lifts that degeneracy, and the source shows the imbalance appears as a separate anisotropic Zeeman term with its own prefactor and its own gap combination.

Returns
-------
numpy.ndarray of float64 with shape (3, 3), the anisotropic g correction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def anisotropic_g_tensor(P: float, adj: "np.ndarray", delta_lhb: float, delta_hhb: float) -> "np.ndarray":
    """Return the anisotropic correction to the effective g-tensor that the source derives from the strain-induced splitting of the light and heavy holes, from P in eV nm, the tensor factor, and the strained conduction-to-light-hole and conduction-to-heavy-hole gaps in eV, using hbar^2/(2 m_e) = 0.0380998212 eV nm^2 as given in the problem statement. It must vanish identically when those two gaps are equal. Raise ValueError if P or either gap is not positive, or if adj is not 3x3.

    Parameters
    ----------
    P : float
        Interband coupling P in eV nm, positive.
    adj : numpy.ndarray
        The 3x3 strain-dependent tensor factor of step 03.
    delta_lhb : float
        Strained conduction-to-light-hole gap in eV, positive.
    delta_hhb : float
        Strained conduction-to-heavy-hole gap in eV, positive.

    Returns
    -------
    g_ani : numpy.ndarray
        Array of shape (3, 3): the anisotropic g correction, dimensionless (float64).

    Raises
    ------
    ValueError
        If P, delta_lhb or delta_hhb is not positive, or adj is not of shape (3, 3).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_anisotropic_g_tensor(P: float, adj: "np.ndarray", delta_lhb: float, delta_hhb: float) -> "np.ndarray":
    """(A34) bottom: g_ani = (2 m_e P^2 / hbar^2) adj{1-e} (1/D_LHB - 1/D_HHB).

    No 2/3 prefactor here, and the gap difference is LHB-HHB, so it vanishes when
    strain does not split the light and heavy holes.
    """
    hb2_2me = 0.0380998212        # hbar^2 / (2 m_e)  [eV nm^2]   (given in the prompt)
    if P <= 0.0:
        raise ValueError("P must be positive")
    if delta_lhb <= 0.0 or delta_hhb <= 0.0:
        raise ValueError("gaps must be positive")
    adj = np.asarray(adj, dtype=np.float64)
    if adj.shape != (3, 3):
        raise ValueError("adj must be 3x3")
    kane = P * P / hb2_2me
    return kane * adj * (1.0 / delta_lhb - 1.0 / delta_hhb)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 0.417, 0.390, 0.905, 6.0, 1.8, 3.6\ne = np.array([[0.020, 0.006, 0.004], [0.006, 0.015, 0.008], [0.004, 0.008, -0.018]])\ne_s = _oracle_shear_strain_vector(e)\ngaps = _oracle_strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)\nadj = _oracle_strain_tensor_factor(e)\ndelta_lhb, delta_hhb = gaps[0], gaps[1]\n",
            "call": "anisotropic_g_tensor(P, adj, delta_lhb, delta_hhb)",
            "gold_call": "_oracle_anisotropic_g_tensor(P, adj, delta_lhb, delta_hhb)",
        },
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 0.417, 0.390, 0.905, 6.0, 1.8, 3.6\nadj = np.eye(3)\ndelta_lhb, delta_hhb = delta_g, delta_g\n",
            "call": "anisotropic_g_tensor(P, adj, delta_lhb, delta_hhb)",
            "gold_call": "_oracle_anisotropic_g_tensor(P, adj, delta_lhb, delta_hhb)",
        },
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 1.519, 0.341, 1.047, 8.33, 2.0, 4.8\ne = np.diag([0.01, 0.01, -0.009])\ne_s = _oracle_shear_strain_vector(e)\ngaps = _oracle_strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)\nadj = _oracle_strain_tensor_factor(e)\ndelta_lhb, delta_hhb = gaps[0], gaps[1]\n",
            "call": "anisotropic_g_tensor(P, adj, delta_lhb, delta_hhb)",
            "gold_call": "_oracle_anisotropic_g_tensor(P, adj, delta_lhb, delta_hhb)",
        },
        {
            "setup": "import numpy as np\nP, adj, delta_lhb, delta_hhb = 0.905, np.eye(2), 0.31, 0.32\ndef run_model():\n    try:\n        anisotropic_g_tensor(P, adj, delta_lhb, delta_hhb)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_anisotropic_g_tensor(P, adj, delta_lhb, delta_hhb)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
