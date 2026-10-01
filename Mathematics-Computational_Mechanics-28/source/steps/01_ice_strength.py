"""
Return the source's empirical ice strength from the ice thickness and concentration. It is proportional to the thickness and depends exponentially on how far the concentration falls short of full cover, with the two material constants supplied as arguments. Recover the exact relation from the paper. Raise ValueError on a negative thickness, on a concentration outside [0, 1], or on a non-positive material constant.

Sea ice is modelled as a two-dimensional continuum whose resistance to deformation grows with how thick and how tightly packed the ice cover is; that resistance sets the scale of every internal stress in the model.

Returns
-------
return array like H (float64): the source's ice strength
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ice_strength(H, A, Pstar, C):
    """H: ice thickness; A: ice concentration in [0, 1]; Pstar, C: material
    constants. Returns a float64 array shaped like H."""
    return np.zeros_like(np.asarray(H, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: ice_strength."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def _oracle_ice_strength(H, A, Pstar, C):
    """Eq 8."""
    H = np.asarray(H, dtype=np.float64); A = np.asarray(A, dtype=np.float64)
    if np.any(H < 0) or np.any(A < 0) or np.any(A > 1):
        raise ValueError("need H >= 0 and A in [0, 1]")
    if Pstar <= 0 or C <= 0:
        raise ValueError("need Pstar > 0 and C > 0")
    return Pstar * H * np.exp(-C * (1.0 - A))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nH = np.array([0.3, 1.2, 2.0])\nA = np.array([1.0, 0.95, 0.8])\nPstar = 27.5e3\nC = 20.0', "call": 'ice_strength(H, A, Pstar, C)', "gold_call": '_oracle_ice_strength(H, A, Pstar, C)', "tol": 1e-08},
        {"setup": 'import numpy as np\nH = np.array([[0.5, 0.9], [1.5, 0.2]])\nA = np.array([[0.99, 0.9], [0.85, 1.0]])\nPstar = 27.5e3\nC = 20.0', "call": 'ice_strength(H, A, Pstar, C)', "gold_call": '_oracle_ice_strength(H, A, Pstar, C)', "tol": 1e-08},
        {"setup": 'import numpy as np\nH = np.array([1.0])\nA = np.array([0.5])\nPstar = 1.0e4\nC = 15.0', "call": 'ice_strength(H, A, Pstar, C)', "gold_call": '_oracle_ice_strength(H, A, Pstar, C)', "tol": 1e-08},
    ]
