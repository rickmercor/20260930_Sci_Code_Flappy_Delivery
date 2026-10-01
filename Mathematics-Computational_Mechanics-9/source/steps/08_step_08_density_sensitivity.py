"""
Contract the adjoint field with the design derivative of the residual to give

the derivative of the compliance with respect to the pseudo-density of every

material point.

Once the adjoint field is known, the design derivative of the compliance is an

inner product between that field and the partial derivative of the discrete

residual with respect to each design variable, taken at fixed displacement. The

external force is prescribed on the material points and does not depend on the

design, so only the internal force contributes. The pseudo-density of a material

point enters the internal force through its Lame parameters alone, and the SIMP

rule scales both of them by the same power gamma^q, so at fixed strain the

Kirchhoff stress, the Cauchy stress and hence the entire internal force

contribution of that point are proportional to gamma^q. Differentiating a

quantity proportional to gamma^q gives q / gamma times the quantity itself, so

the residual derivative for a point is its own internal force contribution

rescaled by that factor, and no separate constitutive differentiation is needed.

Only the point whose density is perturbed contributes, because the design

variables are carried independently by the points, which makes the sensitivity

field local in the design variable and global only through the adjoint. The

overall sign convention follows from the implicit function theorem, which

carries a minus sign in front of the inverse tangent.

Returns
-------
np.ndarray of shape (n_points,), the compliance sensitivity as a float64 array
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def density_sensitivity(adjoint: np.ndarray, cauchy_stress: np.ndarray,
                        deformation_gradients: np.ndarray, volumes: np.ndarray,
                        shape_gradients: np.ndarray, densities: np.ndarray,
                        penalty: float) -> np.ndarray:
    """Compute the compliance sensitivity with respect to every pseudo-density.

    Parameters
    ----------
    adjoint : np.ndarray
        Adjoint field, shape (n_nodes, 2).
    cauchy_stress : np.ndarray
        Converged Cauchy stress, shape (n_points, 2, 2).
    deformation_gradients : np.ndarray
        Converged deformation gradients, shape (n_points, 2, 2).
    volumes : np.ndarray
        Reference volumes, shape (n_points,).
    shape_gradients : np.ndarray
        Reference basis gradients, shape (n_points, n_nodes, 2).
    densities : np.ndarray
        Pseudo-densities, shape (n_points,).
    penalty : float
        SIMP exponent q.

    Returns
    -------
    sensitivity : np.ndarray
        Compliance sensitivity, shape (n_points,).

    Raises
    ------
    ValueError
        If `shape_gradients` is not of shape (n_points, n_nodes, 2), if
        `adjoint` is not of shape (n_nodes, 2), if the stress or deformation
        gradient array is not of shape (n_points, 2, 2), if `volumes` or
        `densities` is not of shape (n_points,), if any density is not strictly
        positive, if `penalty` is less than 1, or if any deformation gradient has
        a non-positive determinant.
    """
    return sensitivity

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_density_sensitivity(adjoint: np.ndarray, cauchy_stress: np.ndarray,
                                deformation_gradients: np.ndarray, volumes: np.ndarray,
                                shape_gradients: np.ndarray, densities: np.ndarray,
                                penalty: float) -> np.ndarray:
    """Reference implementation."""
    adjoint = np.asarray(adjoint, dtype=float)
    stress = np.asarray(cauchy_stress, dtype=float)
    gradients = np.asarray(deformation_gradients, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    reference_gradients = np.asarray(shape_gradients, dtype=float)
    densities = np.asarray(densities, dtype=float)
    exponent = float(penalty)
    if reference_gradients.ndim != 3 or reference_gradients.shape[2] != 2:
        raise ValueError("shape_gradients must have shape (n_points, n_nodes, 2)")
    n_points, n_nodes = reference_gradients.shape[0], reference_gradients.shape[1]
    if adjoint.shape != (n_nodes, 2):
        raise ValueError("adjoint must have shape (n_nodes, 2)")
    if stress.shape != (n_points, 2, 2) or gradients.shape != (n_points, 2, 2):
        raise ValueError("stress and deformation gradient arrays must have shape (n_points, 2, 2)")
    if volumes.shape != (n_points,) or densities.shape != (n_points,):
        raise ValueError("volumes and densities must have shape (n_points,)")
    if np.any(densities <= 0.0):
        raise ValueError("densities must be strictly positive")
    if exponent < 1.0:
        raise ValueError("penalty must be >= 1")
    sensitivity = np.empty(n_points)
    for p in range(n_points):
        determinant = float(np.linalg.det(gradients[p]))
        if determinant <= 0.0:
            raise ValueError("every deformation gradient must have a positive determinant")
        spatial_gradients = reference_gradients[p] @ np.linalg.inv(gradients[p])
        contribution = (determinant * volumes[p]) * (spatial_gradients @ stress[p].T)
        sensitivity[p] = -(exponent / densities[p]) * float(np.sum(adjoint * contribution))
    return sensitivity

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

_SHAPE_GRADIENTS = np.array([[[-1.0, -1.2], [1.0, -0.8], [-1.0, 1.2], [1.0, 0.8]],
                             [[-1.0, -0.8], [1.0, -1.2], [-1.0, 0.8], [1.0, 1.2]]])
_STRESS = np.array([[[0.051469407641, -0.019953851179],
                     [-0.019953851179, 0.014577031322]],
                    [[-0.032517446129, 0.008914552037],
                     [0.008914552037, -0.005271993844]]])
_GRADIENTS = np.array([[[1.182159787136, 0.035865387354],
                        [-0.207004690418, 0.984255344907]],
                       [[0.964121773395, -0.111403618763],
                        [0.089720145522, 1.043118293701]]])


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: two points against a four node adjoint field ---
        {
            "setup": """import numpy as np
adjoint = np.array([[0.0, 0.0], [-0.041, -0.187], [0.012, -0.203], [-0.055, -0.331]])
volumes = np.array([0.04, 0.04])
densities = np.array([0.7, 0.5])
""",
            "call": "np.round(density_sensitivity(adjoint, _STRESS, _GRADIENTS, volumes,"
                    " _SHAPE_GRADIENTS, densities, 3.0), 10).tolist()",
            "gold_call": "np.round(_oracle_density_sensitivity(adjoint, _STRESS, _GRADIENTS, volumes,"
                         " _SHAPE_GRADIENTS, densities, 3.0), 10).tolist()",
        },
        # --- Boundary case: a vanishing adjoint field and unit density ---
        {
            "setup": """import numpy as np
adjoint = np.zeros((4, 2))
volumes = np.array([0.04, 0.04])
densities = np.array([1.0, 1.0])
""",
            "call": "np.round(density_sensitivity(adjoint, _STRESS, _GRADIENTS, volumes,"
                    " _SHAPE_GRADIENTS, densities, 1.0), 10).tolist()",
            "gold_call": "np.round(_oracle_density_sensitivity(adjoint, _STRESS, _GRADIENTS, volumes,"
                         " _SHAPE_GRADIENTS, densities, 1.0), 10).tolist()",
        },
        # --- Edge case: a zero pseudo-density has no defined sensitivity ---
        {
            "setup": """import numpy as np
adjoint = np.zeros((4, 2))
volumes = np.array([0.04, 0.04])
densities = np.array([0.7, 0.0])
def run_model():
    try:
        density_sensitivity(adjoint, _STRESS, _GRADIENTS, volumes, _SHAPE_GRADIENTS, densities, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_density_sensitivity(adjoint, _STRESS, _GRADIENTS, volumes, _SHAPE_GRADIENTS, densities, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
