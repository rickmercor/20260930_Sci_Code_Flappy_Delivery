"""
Return the source's viscous-plastic stress for a stack of strain-rate tensors. It is built from the two viscosities of the previous sub-problem acting on the strain rate and on its trace, together with an isotropic pressure contribution proportional to the ice strength. Which viscosity multiplies the strain rate directly, which combination multiplies the trace, and the numerical factor carried by the pressure term are the source's conventions; recover them from the paper. Assemble by calling the earlier sub-problem functions.

This closure is what makes the model nonlinear: the stress depends on the strain rate both directly and through the viscosities, which themselves depend on the strain rate.

Returns
-------
return (..., 2, 2) float64: the source's viscous-plastic stress tensor
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def vp_stress(eps, P, dmin, e):
    """eps: (..., 2, 2) strain rates; P: ice strength; dmin, e as before.
    Returns a float64 array of the same shape as eps."""
    return np.zeros_like(np.asarray(eps, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: vp_stress."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def _oracle_vp_stress(eps, P, dmin, e):
    """Eq 5."""
    eps = np.asarray(eps, dtype=np.float64)
    if eps.shape[-2:] != (2, 2):
        raise ValueError("eps must be a stack of 2x2 tensors")
    if np.any(np.asarray(P, dtype=np.float64) < 0):
        raise ValueError("the ice strength cannot be negative")
    D = _oracle_effective_deformation(eps, dmin, e)
    zeta, eta = _oracle_vp_viscosities(P, D, e)
    tr = eps[..., 0, 0] + eps[..., 1, 1]
    return (2.0 * eta[..., None, None] * eps
            + ((zeta - eta) * tr - 0.5 * np.asarray(P, dtype=np.float64))[..., None, None] * np.eye(2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\neps = np.array([[1.2e-6, -3.1e-7], [-3.1e-7, -5.4e-7]])\neps2 = np.array([[[2.0e-7, 1.1e-7], [1.1e-7, 4.5e-7]], [[-8.0e-7, 2.2e-8], [2.2e-8, 9.0e-8]]])\ndmin = 2e-9\ne = 2.0\nP = 2.0e4\n', "call": 'vp_stress(eps, P, dmin, e)', "gold_call": '_oracle_vp_stress(eps, P, dmin, e)', "tol": 1e-06},
        {"setup": 'import numpy as np\neps = np.array([[1.2e-6, -3.1e-7], [-3.1e-7, -5.4e-7]])\neps2 = np.array([[[2.0e-7, 1.1e-7], [1.1e-7, 4.5e-7]], [[-8.0e-7, 2.2e-8], [2.2e-8, 9.0e-8]]])\ndmin = 2e-9\ne = 2.0\nP = np.array([1.8e4, 9.0e3])\n', "call": 'vp_stress(eps2, P, dmin, e)', "gold_call": '_oracle_vp_stress(eps2, P, dmin, e)', "tol": 1e-06},
        {"setup": 'import numpy as np\neps = np.array([[1.2e-6, -3.1e-7], [-3.1e-7, -5.4e-7]])\neps2 = np.array([[[2.0e-7, 1.1e-7], [1.1e-7, 4.5e-7]], [[-8.0e-7, 2.2e-8], [2.2e-8, 9.0e-8]]])\ndmin = 2e-9\ne = 2.0\nP = 3.3e4\n', "call": 'vp_stress(eps, P, 1e-8, 1.6)', "gold_call": '_oracle_vp_stress(eps, P, 1e-8, 1.6)', "tol": 1e-06},
    ]
