"""
Return the source's two nonlinear viscosities, stacked with the bulk one first and the shear one second. The bulk viscosity is the ice strength divided by the effective deformation rate times a numerical factor, and the shear viscosity follows from it through the yield-curve eccentricity. Both the numerical factor and the power of the eccentricity, including whether it multiplies or divides, are the source's conventions; recover them from the paper. Raise ValueError on a non-positive effective deformation rate or eccentricity.

The elliptical yield curve of this rheology is encoded entirely in the ratio between the two viscosities, so that ratio decides how much more readily the ice fails in shear than in compression.

Returns
-------
return (2, ...) float64: the bulk viscosity first, the shear viscosity second
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def vp_viscosities(P, Delta, e):
    """P: ice strength; Delta: effective deformation rate; e: eccentricity.
    Returns (2, ...) float64 with the bulk viscosity first."""
    return np.zeros((2,) + np.shape(np.asarray(P, dtype=np.float64)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: vp_viscosities."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def _oracle_vp_viscosities(P, Delta, e):
    """Eq 6."""
    P = np.asarray(P, dtype=np.float64); Delta = np.asarray(Delta, dtype=np.float64)
    if np.any(Delta <= 0) or e <= 0:
        raise ValueError("need Delta > 0 and e > 0")
    zeta = P / (2.0 * Delta)
    return np.stack([zeta, zeta / e ** 2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nP = np.array([2.0e4, 1.1e4])\nDelta = np.array([1.2e-6, 5.0e-7])\ne = 2.0', "call": 'vp_viscosities(P, Delta, e)', "gold_call": '_oracle_vp_viscosities(P, Delta, e)', "tol": 1e-06},
        {"setup": 'import numpy as np\nP = np.array([[3.0e4, 9.0e3], [1.5e4, 2.2e4]])\nDelta = np.array([[2e-6, 8e-7], [3e-7, 1e-6]])\ne = 2.0', "call": 'vp_viscosities(P, Delta, e)', "gold_call": '_oracle_vp_viscosities(P, Delta, e)', "tol": 1e-06},
        {"setup": 'import numpy as np\nP = np.array([5.0e3])\nDelta = np.array([2e-9])\ne = 1.7', "call": 'vp_viscosities(P, Delta, e)', "gold_call": '_oracle_vp_viscosities(P, Delta, e)', "tol": 0.01},
    ]
