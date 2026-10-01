"""
Return the three strained gaps between the conduction band and, in order, the light-hole, heavy-hole and split-off bands, using exactly the three expressions given in the problem statement, with delta_g and delta_soff the unstrained gaps, a and b the deformation potentials, d_cv the strain correction to the interband coupling, e the strain tensor and e_s its shear-strain vector. Raise ValueError if either unstrained gap is not positive, if d_cv is negative, if the shapes are wrong, or if any strained gap is not positive.

Strain shifts the valence-band edges by different amounts, so the three gaps that control the conduction-band parameters open or close by different amounts. The effective model is only meaningful while all three stay open.

Returns
-------
numpy.ndarray of float64 with shape (3,): [gap to LHB, gap to HHB, gap to SOB] in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def strained_band_gaps(delta_g: float, delta_soff: float, a: float, b: float, d_cv: float,
                               e: "np.ndarray", e_s: "np.ndarray") -> "np.ndarray":
    """Return the three strained gaps between the conduction band and, in order, the light-hole, heavy-hole and split-off bands, using exactly the three expressions given in the problem statement, with delta_g and delta_soff the unstrained gaps, a and b the deformation potentials, d_cv the strain correction to the interband coupling, e the strain tensor and e_s its shear-strain vector. Raise ValueError if either unstrained gap is not positive, if d_cv is negative, if the shapes are wrong, or if any strained gap is not positive.

    Parameters
    ----------
    delta_g : float
        Unstrained band gap Delta_g in eV, positive.
    delta_soff : float
        Unstrained split-off gap Delta_soff in eV, positive.
    a : float
        Hydrostatic deformation potential a in eV.
    b : float
        Uniaxial (shear) deformation potential b in eV.
    d_cv : float
        Strain correction to the interband coupling d_cv in eV, nonnegative.
    e : numpy.ndarray
        Symmetric strain tensor of shape (3, 3) in the crystal frame (x, y, z along the cubic axes, z the (001) growth axis), dimensionless.
    e_s : numpy.ndarray
        Shear-strain vector of shape (3,) from step 01.

    Returns
    -------
    gaps : numpy.ndarray
        Array of shape (3,): [gap to LHB, gap to HHB, gap to SOB] in eV (float64).

    Raises
    ------
    ValueError
        If delta_g or delta_soff is not positive, d_cv is negative, e is not (3, 3), e_s is not (3,), or any strained gap is not positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_strained_band_gaps(delta_g: float, delta_soff: float, a: float, b: float, d_cv: float,
                               e: "np.ndarray", e_s: "np.ndarray") -> "np.ndarray":
    """The three strained gaps CB-LHB, CB-HHB, CB-SOB, as DECLARED in the prompt:
        D_LHB = D_g - (a + b/2) tr e - (3b/2) e_zz - d_cv |e_s|
        D_HHB = D_g - (a - b/2) tr e + (3b/2) e_zz + d_cv |e_s|
        D_SOB = D_g + D_soff - a tr e
    Returns array([D_LHB, D_HHB, D_SOB]).
    """
    if delta_g <= 0.0 or delta_soff <= 0.0:
        raise ValueError("unstrained gaps must be positive")
    if d_cv < 0.0:
        raise ValueError("d_cv must be nonnegative")
    e = np.asarray(e, dtype=np.float64)
    e_s = np.asarray(e_s, dtype=np.float64)
    if e.shape != (3, 3) or e_s.shape != (3,):
        raise ValueError("e must be 3x3 and e_s must have 3 components")
    tr = float(np.trace(e))
    ezz = float(e[2, 2])
    sn = float(np.linalg.norm(e_s))
    d_lhb = delta_g - (a + 0.5 * b) * tr - 1.5 * b * ezz - d_cv * sn
    d_hhb = delta_g - (a - 0.5 * b) * tr + 1.5 * b * ezz + d_cv * sn
    d_sob = delta_g + delta_soff - a * tr
    gaps = np.array([d_lhb, d_hhb, d_sob], dtype=np.float64)
    if np.any(gaps <= 0.0):
        raise ValueError("a strained gap closed; the effective model is not valid here")
    return gaps

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 0.417, 0.390, 0.905, 6.0, 1.8, 3.6\ne = np.array([[0.020, 0.006, 0.004], [0.006, 0.015, 0.008], [0.004, 0.008, -0.018]])\ne_s = _oracle_shear_strain_vector(e)\n",
            "call": "strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)",
            "gold_call": "_oracle_strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)",
        },
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 0.417, 0.390, 0.905, 6.0, 1.8, 3.6\ne = np.zeros((3, 3))\ne_s = _oracle_shear_strain_vector(e)\n",
            "call": "strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)",
            "gold_call": "_oracle_strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)",
        },
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, P, a, b, d_cv = 1.519, 0.341, 1.047, 8.33, 2.0, 4.8\ne = np.array([[0.0, 0.004, 0.0], [0.004, 0.0, 0.006], [0.0, 0.006, 0.0]])\ne_s = _oracle_shear_strain_vector(e)\n",
            "call": "strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)",
            "gold_call": "_oracle_strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)",
        },
        {
            "setup": "import numpy as np\ndelta_g, delta_soff, a, b, d_cv = 0.417, 0.390, 6.0, 1.8, 3.6\ne = 0.05 * np.eye(3)\ne_s = np.zeros(3)\ndef run_model():\n    try:\n        strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
