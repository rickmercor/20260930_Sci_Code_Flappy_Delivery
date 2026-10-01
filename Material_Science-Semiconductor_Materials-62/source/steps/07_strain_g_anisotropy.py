"""
Orchestrator. Build the shear-strain vector, the three strained gaps, the tensor factor, the isotropic g-tensor, the anisotropic correction and the effective g-tensor, in that order, and return the strain-induced anisotropy G_zz minus G_xx of the effective tensor. Call the earlier step functions rather than reimplementing any of them. Raise ValueError if e is not 3x3.

An isotropic g-factor has no anisotropy whatever the strain, so this quantity is zero unless the construction the source prescribes is followed. It is the cleanest single number that separates the strained effective model from the unstrained textbook one.

Returns
-------
float, G_zz - G_xx of the effective conduction-band g-tensor.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def strain_g_anisotropy(delta_g: float, delta_soff: float, P: float, a: float, b: float, d_cv: float,
                                e: "np.ndarray") -> float:
    """Orchestrator. Build the shear-strain vector, the three strained gaps, the tensor factor, the isotropic g-tensor, the anisotropic correction and the effective g-tensor, in that order, and return the strain-induced anisotropy G_zz minus G_xx of the effective tensor. Call the earlier step functions rather than reimplementing any of them. Raise ValueError if e is not 3x3.

    Parameters
    ----------
    delta_g : float
        Unstrained band gap Delta_g in eV, positive.
    delta_soff : float
        Unstrained split-off gap Delta_soff in eV, positive.
    P : float
        Interband coupling P in eV nm, positive.
    a : float
        Hydrostatic deformation potential a in eV.
    b : float
        Uniaxial (shear) deformation potential b in eV.
    d_cv : float
        Strain correction to the interband coupling d_cv in eV, nonnegative.
    e : numpy.ndarray
        Symmetric strain tensor of shape (3, 3) in the crystal frame (x, y, z along the cubic axes, z the (001) growth axis), dimensionless.

    Returns
    -------
    dg : float
        G_zz - G_xx of the effective conduction-band g-tensor, dimensionless.

    Raises
    ------
    ValueError
        If e is not of shape (3, 3), or any earlier step rejects its input (closed gap, non-symmetric strain, non-positive parameter).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_strain_g_anisotropy(delta_g: float, delta_soff: float, P: float, a: float, b: float, d_cv: float,
                                e: "np.ndarray") -> float:
    """ORCHESTRATOR. dg = G_zz - G_xx of the effective CB g-tensor under strain e.

    Calls every earlier step directly and uses each return value.
    """
    e = np.asarray(e, dtype=np.float64)
    if e.shape != (3, 3):
        raise ValueError("strain tensor must be 3x3")
    e_s = _oracle_shear_strain_vector(e)
    gaps = _oracle_strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)
    adj = _oracle_strain_tensor_factor(e)
    g_iso = _oracle_isotropic_g_tensor(P, adj, gaps[0], gaps[2])
    g_ani = _oracle_anisotropic_g_tensor(P, adj, gaps[0], gaps[1])
    g = _oracle_effective_g_tensor(g_iso, g_ani)
    return float(g[2, 2] - g[0, 0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 0.417, 0.390, 0.905, 6.0, 1.8, 3.6\ne = np.array([[0.020, 0.006, 0.004], [0.006, 0.015, 0.008], [0.004, 0.008, -0.018]])\n",
            "call": "strain_g_anisotropy(delta_g, delta_soff, P, a, b, d_cv, e)",
            "gold_call": "_oracle_strain_g_anisotropy(delta_g, delta_soff, P, a, b, d_cv, e)",
        },
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 0.417, 0.390, 0.905, 6.0, 1.8, 3.6\ne = np.zeros((3, 3))\n",
            "call": "strain_g_anisotropy(delta_g, delta_soff, P, a, b, d_cv, e)",
            "gold_call": "_oracle_strain_g_anisotropy(delta_g, delta_soff, P, a, b, d_cv, e)",
        },
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 1.519, 0.341, 1.047, 8.33, 2.0, 4.8\ne = np.diag([0.01, 0.01, -0.009])\n",
            "call": "strain_g_anisotropy(delta_g, delta_soff, P, a, b, d_cv, e)",
            "gold_call": "_oracle_strain_g_anisotropy(delta_g, delta_soff, P, a, b, d_cv, e)",
        },
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 0.417, 0.390, 0.905, 6.0, 1.8, 3.6\ne = np.zeros((2, 2))\ndef run_model():\n    try:\n        strain_g_anisotropy(delta_g, delta_soff, P, a, b, d_cv, e)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_strain_g_anisotropy(delta_g, delta_soff, P, a, b, d_cv, e)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
