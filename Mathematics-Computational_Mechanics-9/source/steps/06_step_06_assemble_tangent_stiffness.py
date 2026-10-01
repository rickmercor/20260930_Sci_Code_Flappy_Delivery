"""
Assemble the consistent tangent stiffness matrix of the discrete force balance

and impose the Dirichlet constraints on it, returning the matrix used both by

the Newton correction and by the adjoint solve.

The tangent of the discrete residual is the derivative of the internal nodal

force with respect to the nodal displacement, and because the internal force is

built from the Cauchy stress acting on spatial basis gradients it has two

contributions: the derivative of the stress itself, and the derivative of the

push-forward that maps the reference basis gradient to the current

configuration. Both are collected by differentiating the first Piola-Kirchhoff

stress P = tau F^-T with respect to the deformation gradient, since the internal

force can be written as the reference volume times P acting on the reference

basis gradient. Differentiating the Hencky law is the delicate part, because the

logarithmic strain contains a matrix logarithm and the derivative of a matrix

function of a symmetric tensor is not the scalar derivative applied

eigenvalue by eigenvalue. The Frechet derivative in a direction E is obtained by

the Daleckii-Krein formula, which conjugates E into the eigenbasis of the left

Cauchy-Green tensor, multiplies elementwise by a symmetric matrix of divided

differences of the logarithm built from the eigenvalues, and conjugates back.

The off-diagonal divided difference is a difference quotient that becomes

indeterminate as two eigenvalues approach one another, and its limit is the

ordinary derivative of the logarithm at the repeated eigenvalue; the undeformed

state, at which every Newton solve starts, has a left Cauchy-Green tensor equal

to the identity and therefore hits that case exactly, so the limiting value must

be used rather than the difference quotient. The plane-stress condition applies

to the derivative exactly as it does to the stress, and the constrained degrees

of freedom are removed by replacing their rows and columns with the identity so

that the constrained corrections and adjoint components are zero.

Returns
-------
np.ndarray of shape (2 n_nodes, 2 n_nodes), the constrained tangent stiffness as a float64 array
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_tangent_stiffness(deformation_gradients: np.ndarray, lame: np.ndarray,
                               volumes: np.ndarray, shape_gradients: np.ndarray,
                               fixed_nodes: np.ndarray) -> np.ndarray:
    """Assemble the constrained consistent tangent stiffness matrix.

    Parameters
    ----------
    deformation_gradients : np.ndarray
        Deformation gradients, shape (n_points, 2, 2).
    lame : np.ndarray
        Lame parameters, shape (n_points, 2).
    volumes : np.ndarray
        Reference volumes, shape (n_points,).
    shape_gradients : np.ndarray
        Reference basis gradients, shape (n_points, n_nodes, 2).
    fixed_nodes : np.ndarray
        Boolean mask of constrained nodes, shape (n_nodes,).

    Returns
    -------
    stiffness : np.ndarray
        Tangent stiffness with constrained rows and columns replaced by the
        identity, shape (2 n_nodes, 2 n_nodes), with degree of freedom
        2 v + i belonging to component i of node v.

    Raises
    ------
    ValueError
        If `shape_gradients` is not of shape (n_points, n_nodes, 2), if
        `deformation_gradients`, `lame` or `volumes` is not sized by n_points, if
        `fixed_nodes` is not of shape (n_nodes,), or if any deformation gradient
        has a non-positive determinant.
    """
    return stiffness

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_EIGENVALUE_TOLERANCE = 1e-12


def _divided_difference_matrix(eigenvalues):
    """Divided differences of the logarithm, with the repeated-eigenvalue limit."""
    size = eigenvalues.shape[0]
    matrix = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            if abs(eigenvalues[i] - eigenvalues[j]) < _EIGENVALUE_TOLERANCE:
                matrix[i, j] = 1.0 / eigenvalues[i]
            else:
                matrix[i, j] = ((np.log(eigenvalues[i]) - np.log(eigenvalues[j]))
                                / (eigenvalues[i] - eigenvalues[j]))
    return matrix


def _first_piola_tangent(deformation_gradient, lambda_p, mu_p):
    """Derivative of the first Piola-Kirchhoff stress with respect to F."""
    left_cauchy_green = deformation_gradient @ deformation_gradient.T
    eigenvalues, eigenvectors = np.linalg.eigh(left_cauchy_green)
    divided = _divided_difference_matrix(eigenvalues)
    strain = 0.5 * (eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T)
    identity = np.eye(2)
    in_plane_trace = strain[0, 0] + strain[1, 1]
    strain_zz = -lambda_p / (lambda_p + 2.0 * mu_p) * in_plane_trace
    kirchhoff = lambda_p * (in_plane_trace + strain_zz) * identity + 2.0 * mu_p * strain
    lambda_plane_stress = 2.0 * lambda_p * mu_p / (lambda_p + 2.0 * mu_p)
    inverse = np.linalg.inv(deformation_gradient)
    tangent = np.empty((2, 2, 2, 2))
    for a in range(2):
        for b in range(2):
            direction = np.zeros((2, 2))
            direction[a, b] = 1.0
            perturbation = direction @ deformation_gradient.T + deformation_gradient @ direction.T
            strain_rate = 0.5 * (eigenvectors
                                 @ (divided * (eigenvectors.T @ perturbation @ eigenvectors))
                                 @ eigenvectors.T)
            kirchhoff_rate = (lambda_plane_stress * (strain_rate[0, 0] + strain_rate[1, 1]) * identity
                              + 2.0 * mu_p * strain_rate)
            tangent[:, :, a, b] = (kirchhoff_rate @ inverse.T
                                   - np.outer(kirchhoff @ inverse[b, :], inverse[:, a]))
    return tangent


def _oracle_assemble_tangent_stiffness(deformation_gradients: np.ndarray, lame: np.ndarray,
                                       volumes: np.ndarray, shape_gradients: np.ndarray,
                                       fixed_nodes: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    gradients = np.asarray(deformation_gradients, dtype=float)
    lame = np.asarray(lame, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    reference_gradients = np.asarray(shape_gradients, dtype=float)
    fixed = np.asarray(fixed_nodes, dtype=bool)
    if reference_gradients.ndim != 3 or reference_gradients.shape[2] != 2:
        raise ValueError("shape_gradients must have shape (n_points, n_nodes, 2)")
    n_points, n_nodes = reference_gradients.shape[0], reference_gradients.shape[1]
    if gradients.shape != (n_points, 2, 2) or lame.shape != (n_points, 2):
        raise ValueError("deformation_gradients and lame must be sized by n_points")
    if volumes.shape != (n_points,):
        raise ValueError("volumes must have shape (n_points,)")
    if fixed.shape != (n_nodes,):
        raise ValueError("fixed_nodes must have shape (n_nodes,)")
    stiffness = np.zeros((2 * n_nodes, 2 * n_nodes))
    for p in range(n_points):
        if float(np.linalg.det(gradients[p])) <= 0.0:
            raise ValueError("every deformation gradient must have a positive determinant")
        tangent = _first_piola_tangent(gradients[p], float(lame[p, 0]), float(lame[p, 1]))
        stiffness += volumes[p] * np.einsum(
            'ijab,wb,vj->viwa', tangent, reference_gradients[p], reference_gradients[p]
        ).reshape(2 * n_nodes, 2 * n_nodes)
    constrained = np.repeat(fixed, 2)
    stiffness[constrained, :] = 0.0
    stiffness[:, constrained] = 0.0
    stiffness[constrained, constrained] = 1.0
    return stiffness

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

_SHAPE_GRADIENTS = np.array([[[-1.0, -1.2], [1.0, -0.8], [-1.0, 1.2], [1.0, 0.8]],
                             [[-1.0, -0.8], [1.0, -1.2], [-1.0, 0.8], [1.0, 1.2]]])
_LAME = np.array([[0.197884615385, 0.131923076923],
                  [0.072115384615, 0.048076923077]])
_GRADIENTS = np.array([[[1.182159787136, 0.035865387354],
                        [-0.207004690418, 0.984255344907]],
                       [[0.964121773395, -0.111403618763],
                        [0.089720145522, 1.043118293701]]])


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: a deformed state with two nodes constrained ---
        {
            "setup": """import numpy as np
volumes = np.array([0.04, 0.04])
fixed_nodes = np.array([True, False, True, False])
""",
            "call": "np.round(assemble_tangent_stiffness(_GRADIENTS, _LAME, volumes,"
                    " _SHAPE_GRADIENTS, fixed_nodes), 10).tolist()",
            "gold_call": "np.round(_oracle_assemble_tangent_stiffness(_GRADIENTS, _LAME, volumes,"
                         " _SHAPE_GRADIENTS, fixed_nodes), 10).tolist()",
        },
        # --- Boundary case: the undeformed state, where the left Cauchy-Green
        # --- tensor is the identity and its eigenvalues coincide exactly
        {
            "setup": """import numpy as np
gradients = np.array([np.eye(2), np.eye(2)])
volumes = np.array([0.04, 0.04])
fixed_nodes = np.zeros(4, dtype=bool)
""",
            "call": "np.round(assemble_tangent_stiffness(gradients, _LAME, volumes,"
                    " _SHAPE_GRADIENTS, fixed_nodes), 10).tolist()",
            "gold_call": "np.round(_oracle_assemble_tangent_stiffness(gradients, _LAME, volumes,"
                         " _SHAPE_GRADIENTS, fixed_nodes), 10).tolist()",
        },
        # --- Edge case: a constraint mask of the wrong length must raise ---
        {
            "setup": """import numpy as np
volumes = np.array([0.04, 0.04])
bad_mask = np.zeros(3, dtype=bool)
def run_model():
    try:
        assemble_tangent_stiffness(_GRADIENTS, _LAME, volumes, _SHAPE_GRADIENTS, bad_mask)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_assemble_tangent_stiffness(_GRADIENTS, _LAME, volumes, _SHAPE_GRADIENTS, bad_mask)
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
