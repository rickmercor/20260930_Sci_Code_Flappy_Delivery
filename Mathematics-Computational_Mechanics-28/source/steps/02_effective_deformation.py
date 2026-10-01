"""
Return the source's effective deformation rate for a stack of two-dimensional strain-rate tensors. It combines a regularisation floor, a term built from the TRACE-FREE part of the strain rate weighted by the yield-curve eccentricity, and a term built from the trace, all under a square root. The numerical factor on the trace-free term, the power of the eccentricity that multiplies it, and the fact that this term uses the trace-free part rather than the full tensor are the source's conventions; recover them from the paper. Note that the medium is two-dimensional, so the trace-free part is formed accordingly. Raise ValueError unless the input is a stack of 2x2 tensors and the floor and eccentricity are positive.

The effective deformation rate is what tells the rheology whether the ice is deforming fast enough to be treated as plastic or slowly enough to be treated as very viscous; the regularisation floor keeps the viscosities bounded in the near-rigid limit.

Returns
-------
return array (float64): the source's effective deformation rate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def effective_deformation(eps, dmin, e):
    """eps: (..., 2, 2) strain-rate tensors; dmin: regularisation floor;
    e: yield-curve eccentricity. Returns a float64 array of shape eps.shape[:-2]."""
    return np.zeros(np.asarray(eps, dtype=np.float64).shape[:-2])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: effective_deformation."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def _oracle_effective_deformation(eps, dmin, e):
    """Eq 7, with the two-dimensional trace-free part."""
    eps = np.asarray(eps, dtype=np.float64)
    if eps.shape[-2:] != (2, 2):
        raise ValueError("eps must be a stack of 2x2 tensors")
    if dmin <= 0 or e <= 0:
        raise ValueError("need dmin > 0 and e > 0")
    tr = eps[..., 0, 0] + eps[..., 1, 1]
    d = eps.copy()
    d[..., 0, 0] = d[..., 0, 0] - 0.5 * tr
    d[..., 1, 1] = d[..., 1, 1] - 0.5 * tr
    return np.sqrt(dmin ** 2 + 2.0 * e ** -2 * np.einsum('...ij,...ij->...', d, d) + tr ** 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\neps = np.array([[1.2e-6, -3.1e-7], [-3.1e-7, -5.4e-7]])\neps2 = np.array([[[2.0e-7, 1.1e-7], [1.1e-7, 4.5e-7]], [[-8.0e-7, 2.2e-8], [2.2e-8, 9.0e-8]]])\ndmin = 2e-9\ne = 2.0\n', "call": 'effective_deformation(eps, dmin, e)', "gold_call": '_oracle_effective_deformation(eps, dmin, e)', "tol": 1e-14},
        {"setup": 'import numpy as np\neps = np.array([[1.2e-6, -3.1e-7], [-3.1e-7, -5.4e-7]])\neps2 = np.array([[[2.0e-7, 1.1e-7], [1.1e-7, 4.5e-7]], [[-8.0e-7, 2.2e-8], [2.2e-8, 9.0e-8]]])\ndmin = 2e-9\ne = 2.0\n', "call": 'effective_deformation(eps2, dmin, e)', "gold_call": '_oracle_effective_deformation(eps2, dmin, e)', "tol": 1e-14},
        {"setup": 'import numpy as np\neps = np.array([[1.2e-6, -3.1e-7], [-3.1e-7, -5.4e-7]])\neps2 = np.array([[[2.0e-7, 1.1e-7], [1.1e-7, 4.5e-7]], [[-8.0e-7, 2.2e-8], [2.2e-8, 9.0e-8]]])\ndmin = 2e-9\ne = 2.0\neps3 = np.array([[3.3e-7, 0.0], [0.0, 3.3e-7]])\n', "call": 'effective_deformation(eps3, dmin, 1.5)', "gold_call": '_oracle_effective_deformation(eps3, dmin, 1.5)', "tol": 1e-14},
    ]
