"""
Evaluate the Hencky hyperelastic constitutive law under plane stress at every

material point, returning the Cauchy stress and the Jacobian of the deformation

gradient.

Hencky hyperelasticity extends the isotropic linear elastic relation to finite

strain by replacing the infinitesimal strain with the logarithmic strain, which

is one half of the matrix logarithm of the left Cauchy-Green tensor B = F F^T.

The matrix logarithm is evaluated by spectral mapping: B is symmetric positive

definite, so B = Q diag(b) Q^T and ln(B) = Q diag(ln b) Q^T. The Kirchhoff

stress is the isotropic linear map of that logarithmic strain through the Lame

parameters of the point, and the Cauchy stress follows from it by

dividing by the Jacobian J = det(F), since Kirchhoff and Cauchy measures differ

by exactly that volume ratio. For a body that is thin compared with its in-plane

dimensions the out-of-plane normal stress vanishes, which does not make the

out-of-plane strain vanish: the plane-stress condition instead determines the

out-of-plane logarithmic strain from the in-plane one, and that additional

component enters the trace that multiplies the first Lame parameter. Using the

in-plane trace alone corresponds to plane strain and gives a different, stiffer

response.

Returns
-------
tuple (cauchy_stress, jacobian) of np.ndarray with shapes (n_points, 2, 2) and (n_points,), both float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hencky_cauchy_stress(deformation_gradients: np.ndarray, lame: np.ndarray):
    """Evaluate the plane-stress Hencky constitutive law at every material point.

    Parameters
    ----------
    deformation_gradients : np.ndarray
        Deformation gradients, shape (n_points, 2, 2).
    lame : np.ndarray
        Lame parameters, shape (n_points, 2), with lambda_p in column 0 and
        mu_p in column 1.

    Returns
    -------
    stress_state : tuple of np.ndarray
        The pair (cauchy_stress, jacobian), holding the in-plane Cauchy stress
        of shape (n_points, 2, 2) and the determinant of each deformation
        gradient, of shape (n_points,).

    Raises
    ------
    ValueError
        If `deformation_gradients` is not of shape (n_points, 2, 2), if `lame`
        is not of shape (n_points, 2), if any point has mu <= 0 or
        lambda + 2 mu <= 0, or if any deformation gradient has a non-positive
        determinant.
    """
    return stress_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _logarithmic_strain(deformation_gradient):
    """Half the matrix logarithm of the left Cauchy-Green tensor."""
    eigenvalues, eigenvectors = np.linalg.eigh(deformation_gradient @ deformation_gradient.T)
    return 0.5 * (eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T)


def _oracle_hencky_cauchy_stress(deformation_gradients: np.ndarray, lame: np.ndarray):
    """Reference implementation."""
    gradients = np.asarray(deformation_gradients, dtype=float)
    lame = np.asarray(lame, dtype=float)
    if gradients.ndim != 3 or gradients.shape[1:] != (2, 2):
        raise ValueError("deformation_gradients must have shape (n_points, 2, 2)")
    if lame.shape != (gradients.shape[0], 2):
        raise ValueError("lame must have shape (n_points, 2)")
    if np.any(lame[:, 1] <= 0.0) or np.any(lame[:, 0] + 2.0 * lame[:, 1] <= 0.0):
        raise ValueError("lame parameters must satisfy mu > 0 and lambda + 2 mu > 0")
    n_points = gradients.shape[0]
    cauchy_stress = np.empty((n_points, 2, 2))
    jacobian = np.empty(n_points)
    identity = np.eye(2)
    for p in range(n_points):
        determinant = float(np.linalg.det(gradients[p]))
        if determinant <= 0.0:
            raise ValueError("every deformation gradient must have a positive determinant")
        lambda_p = float(lame[p, 0])
        mu_p = float(lame[p, 1])
        strain = _logarithmic_strain(gradients[p])
        in_plane_trace = strain[0, 0] + strain[1, 1]
        strain_zz = -lambda_p / (lambda_p + 2.0 * mu_p) * in_plane_trace
        kirchhoff = lambda_p * (in_plane_trace + strain_zz) * identity + 2.0 * mu_p * strain
        cauchy_stress[p] = kirchhoff / determinant
        jacobian[p] = determinant
    return cauchy_stress, jacobian

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

_GRADIENTS = np.array([[[1.182159787136, 0.035865387354],
                        [-0.207004690418, 0.984255344907]],
                       [[0.964121773395, -0.111403618763],
                        [0.089720145522, 1.043118293701]]])
_LAME = np.array([[0.197884615385, 0.131923076923],
                  [0.072115384615, 0.048076923077]])


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: converged deformation gradients of two points ---
        {
            "setup": """import numpy as np
deformation_gradients = _GRADIENTS.copy()
lame = _LAME.copy()
""",
            "call": "[np.round(hencky_cauchy_stress(deformation_gradients, lame)[0], 10).tolist(),"
                    " np.round(hencky_cauchy_stress(deformation_gradients, lame)[1], 10).tolist()]",
            "gold_call": "[np.round(_oracle_hencky_cauchy_stress(deformation_gradients, lame)[0], 10).tolist(),"
                         " np.round(_oracle_hencky_cauchy_stress(deformation_gradients, lame)[1], 10).tolist()]",
        },
        # --- Boundary case: the undeformed state, where the left Cauchy-Green
        # --- tensor has a repeated eigenvalue and the stress must vanish
        {
            "setup": """import numpy as np
deformation_gradients = np.array([np.eye(2), 1.05 * np.eye(2)])
lame = _LAME.copy()
""",
            "call": "[np.round(hencky_cauchy_stress(deformation_gradients, lame)[0], 10).tolist(),"
                    " np.round(hencky_cauchy_stress(deformation_gradients, lame)[1], 10).tolist()]",
            "gold_call": "[np.round(_oracle_hencky_cauchy_stress(deformation_gradients, lame)[0], 10).tolist(),"
                         " np.round(_oracle_hencky_cauchy_stress(deformation_gradients, lame)[1], 10).tolist()]",
        },
        # --- Edge case: an inverted deformation gradient must raise ---
        {
            "setup": """import numpy as np
deformation_gradients = np.array([[[1.0, 0.0], [0.0, -1.0]]])
lame = _LAME[:1].copy()
def run_model():
    try:
        hencky_cauchy_stress(deformation_gradients, lame)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_hencky_cauchy_stress(deformation_gradients, lame)
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
