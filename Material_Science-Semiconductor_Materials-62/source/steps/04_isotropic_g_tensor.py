"""
Return the isotropic strain-renormalized effective g-tensor of the conduction band as the source defines it, from the interband coupling P in eV nm, the tensor factor of the previous step, and the strained conduction-to-light-hole and conduction-to-split-off gaps in eV. Use hbar^2/(2 m_e) = 0.0380998212 eV nm^2 and the free-electron g-factor g_e = 2.0023193, both given in the problem statement. Raise ValueError if P or either gap is not positive, or if adj is not 3x3.

The conduction-band g-factor of a narrow-gap III-V compound is dominated by the spin-orbit-split valence bands it couples to, which is why it departs so far from 2. Strain moves those bands and the tensor factor makes the departure direction dependent.

Returns
-------
numpy.ndarray of float64 with shape (3, 3), the isotropic effective g-tensor.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def isotropic_g_tensor(P: float, adj: "np.ndarray", delta_lhb: float, delta_sob: float) -> "np.ndarray":
    """Return the isotropic strain-renormalized effective g-tensor of the conduction band as the source defines it, from the interband coupling P in eV nm, the tensor factor of the previous step, and the strained conduction-to-light-hole and conduction-to-split-off gaps in eV. Use hbar^2/(2 m_e) = 0.0380998212 eV nm^2 and the free-electron g-factor g_e = 2.0023193, both given in the problem statement. Raise ValueError if P or either gap is not positive, or if adj is not 3x3.

    Parameters
    ----------
    P : float
        Interband coupling P in eV nm, positive.
    adj : numpy.ndarray
        The 3x3 strain-dependent tensor factor of step 03.
    delta_lhb : float
        Strained conduction-to-light-hole gap in eV, positive.
    delta_sob : float
        Strained conduction-to-split-off gap in eV, positive.

    Returns
    -------
    g_iso : numpy.ndarray
        Array of shape (3, 3): the isotropic effective g-tensor, dimensionless (float64).

    Raises
    ------
    ValueError
        If P, delta_lhb or delta_sob is not positive, or adj is not of shape (3, 3).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_isotropic_g_tensor(P: float, adj: "np.ndarray", delta_lhb: float, delta_sob: float) -> "np.ndarray":
    """(A34) top: g_iso = g_e 1 - (2/3) (2 m_e P^2 / hbar^2) adj{1-e} (1/D_LHB - 1/D_SOB).

    With hbar^2/(2 m_e) = 0.0380998212 eV nm^2 as the unit, 2 m_e P^2 / hbar^2 = P^2 / 0.0380998212.
    """
    hb2_2me = 0.0380998212        # hbar^2 / (2 m_e)  [eV nm^2]   (given in the prompt)
    g_e = 2.0023193               # free-electron g-factor         (given in the prompt)
    if P <= 0.0:
        raise ValueError("P must be positive")
    if delta_lhb <= 0.0 or delta_sob <= 0.0:
        raise ValueError("gaps must be positive")
    adj = np.asarray(adj, dtype=np.float64)
    if adj.shape != (3, 3):
        raise ValueError("adj must be 3x3")
    kane = P * P / hb2_2me
    return g_e * np.eye(3) - (2.0 / 3.0) * kane * adj * (1.0 / delta_lhb - 1.0 / delta_sob)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 0.417, 0.390, 0.905, 6.0, 1.8, 3.6\ne = np.array([[0.020, 0.006, 0.004], [0.006, 0.015, 0.008], [0.004, 0.008, -0.018]])\ne_s = _oracle_shear_strain_vector(e)\ngaps = _oracle_strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)\nadj = _oracle_strain_tensor_factor(e)\ndelta_lhb, delta_sob = gaps[0], gaps[2]\n",
            "call": "isotropic_g_tensor(P, adj, delta_lhb, delta_sob)",
            "gold_call": "_oracle_isotropic_g_tensor(P, adj, delta_lhb, delta_sob)",
        },
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 0.417, 0.390, 0.905, 6.0, 1.8, 3.6\nadj = np.eye(3)\ndelta_lhb, delta_sob = delta_g, delta_g + delta_soff\n",
            "call": "isotropic_g_tensor(P, adj, delta_lhb, delta_sob)",
            "gold_call": "_oracle_isotropic_g_tensor(P, adj, delta_lhb, delta_sob)",
        },
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 1.519, 0.341, 1.047, 8.33, 2.0, 4.8\ne = np.array([[0.0, 0.004, 0.0], [0.004, 0.0, 0.006], [0.0, 0.006, 0.0]])\ne_s = _oracle_shear_strain_vector(e)\ngaps = _oracle_strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)\nadj = _oracle_strain_tensor_factor(e)\ndelta_lhb, delta_sob = gaps[0], gaps[2]\n",
            "call": "isotropic_g_tensor(P, adj, delta_lhb, delta_sob)",
            "gold_call": "_oracle_isotropic_g_tensor(P, adj, delta_lhb, delta_sob)",
        },
        {
            "setup": "import numpy as np\nP, adj, delta_lhb, delta_sob = 0.905, np.eye(3), -0.3, 0.7\ndef run_model():\n    try:\n        isotropic_g_tensor(P, adj, delta_lhb, delta_sob)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_isotropic_g_tensor(P, adj, delta_lhb, delta_sob)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
