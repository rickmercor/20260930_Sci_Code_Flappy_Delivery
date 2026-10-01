"""
Combine the two spin channels' effective hole normalisations into the effective exchange-correlation hole normalisation of the same-spin channel, pointwise. Write a for the same-spin normalisation and b for the other-spin one. Take f as the least of three numbers: (1-a)/b, (1-b)/a, and one; treat a ratio whose denominator is zero as larger than the other two so that it never wins. Return a + f*b. The function is not symmetric in its two arguments, so calling it for the other channel means exchanging them.

The same-spin hole is deepened by as much of the opposite-spin hole as can be added without pushing either channel past one whole electron, and without adding more than the opposite-spin hole actually holds. Those three ceilings are exactly the three numbers compared. Whenever the first ceiling is the binding one the result collapses to one, which is why closed-shell regions show no effect at all and only regions where a channel is short of an electron behave differently.

Returns
-------
ndarray of shape (len(same_spin_norm),): the effective exchange-correlation hole normalisation of the same-spin channel, dimensionless.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def xc_hole_normalisation(same_spin_norm: "np.ndarray", other_spin_norm: "np.ndarray") -> "np.ndarray":
    '''Combine the two spin channels' effective hole normalisations into the effective exchange-correlation hole normalisation of the same-spin channel, pointwise. Write a for the same-spin normalisation and b for the other-spin one. Take f as the least of three numbers: (1-a)/b, (1-b)/a, and one; treat a ratio whose denominator is zero as larger than the other two so that it never wins. Return a + f*b. The function is not symmetric in its two arguments, so calling it for the other channel means exchanging them.

    Parameters
    ----------
    same_spin_norm : np.ndarray
        Effective hole normalisation of the channel being combined, each between zero and one.
    other_spin_norm : np.ndarray
        Effective hole normalisation of the opposite channel, same length, each between zero and one.

    Returns
    -------
    normalisation : np.ndarray
        ndarray of shape (len(same_spin_norm),): the effective exchange-correlation hole normalisation of the same-spin channel, dimensionless.

    Raises
    ------
    ValueError
        if the two arrays have different lengths, or if any value lies outside zero to one.
    '''
    return normalisation  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_xc_hole_normalisation(same_spin_norm: "np.ndarray", other_spin_norm: "np.ndarray") -> "np.ndarray":
    """Effective exchange-correlation hole normalisation of the same-spin channel."""
    a = np.asarray(same_spin_norm, dtype=float).ravel()
    b = np.asarray(other_spin_norm, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("both normalisation arrays must have the same length")
    if np.any(a < 0.0) or np.any(a > 1.0) or np.any(b < 0.0) or np.any(b > 1.0):
        raise ValueError("every hole normalisation must lie between zero and one")
    big = np.finfo(float).max
    first = np.where(b > 0.0, (1.0 - a) / np.where(b > 0.0, b, 1.0), big)
    second = np.where(a > 0.0, (1.0 - b) / np.where(a > 0.0, a, 1.0), big)
    f = np.minimum(np.minimum(first, second), 1.0)
    return a + f * b

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)\nNA=_oracle_effective_hole_normalisation(FA[0],FA[4],XA); NB=_oracle_effective_hole_normalisation(FB[0],FB[4],XB)",
         "call": "xc_hole_normalisation(NA, NB)",
         "gold_call": "_oracle_xc_hole_normalisation(NA, NB)"},   # normal
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)\nNA=_oracle_effective_hole_normalisation(FA[0],FA[4],XA); NB=_oracle_effective_hole_normalisation(FB[0],FB[4],XB)",
         "call": "xc_hole_normalisation(NB, NA)",
         "gold_call": "_oracle_xc_hole_normalisation(NB, NA)"},   # boundary
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)\nNA=_oracle_effective_hole_normalisation(FA[0],FA[4],XA); NB=_oracle_effective_hole_normalisation(FB[0],FB[4],XB)",
         "call": "xc_hole_normalisation(NA, np.zeros(RG.size))",
         "gold_call": "_oracle_xc_hole_normalisation(NA, np.zeros(RG.size))"},   # edge
        {"setup": "import numpy as np",
         "call": "xc_hole_normalisation(np.array([0.2,0.45,0.7,0.95,0.3,0.6]), np.array([0.9,0.45,0.2,0.05,0.3,0.99]))",
         "gold_call": "_oracle_xc_hole_normalisation(np.array([0.2,0.45,0.7,0.95,0.3,0.6]), np.array([0.9,0.45,0.2,0.05,0.3,0.99]))"},   # edge, both branches exercised
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)\nNA=_oracle_effective_hole_normalisation(FA[0],FA[4],XA); NB=_oracle_effective_hole_normalisation(FB[0],FB[4],XB)\ndef _exc(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_exc(lambda: xc_hole_normalisation(NA, 1.5*np.ones(RG.size)))",
         "gold_call": "_exc(lambda: _oracle_xc_hole_normalisation(NA, 1.5*np.ones(RG.size)))"},   # invalid input
    ]
