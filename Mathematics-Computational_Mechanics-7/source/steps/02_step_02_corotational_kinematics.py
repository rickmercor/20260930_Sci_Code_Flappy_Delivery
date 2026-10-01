"""
Recover polar kinematics, equivalent strain, and the exact Mandel-strain Fréchet derivative for each element.

For reference and deformed edge matrices $D_m$ and $D_s$, the element deformation gradient is



$$

F=D_sD_m^{-1}.

$$



Only orientation-preserving states with $\\det F>0$ are admitted. The polar decomposition separates rigid motion from stretch:



$$

F=RS,

$$



where $R$ is a proper rotation and $S$ is symmetric positive definite. The strain used by the constitutive update is the corotational stretch strain



$$

\\epsilon=S-I,

$$



with deviatoric component and scalar equivalent strain



$$

e=\\epsilon-\\dfrac{1}{3}\\operatorname{tr}(\\epsilon)I,

\\qquad

q=\\sqrt{\\dfrac{2}{3}(e:e)}.

$$



Removing $R$ prevents a superposed rigid rotation from producing material strain. Such a rotation changes $F$ and $R$, but leaves $S$, $e$, and $q$ unchanged. The scalar $q$ drives activation and history evolution, while the tensor $e$ retains the current deviatoric direction.



For a perturbation $\\delta F$, define $A=R^T\\delta F$ and the skew spin $\\Omega=R^T\\delta R$. Differentiating $F=RS$ while enforcing symmetric $\\delta S$ gives



$$

\\Omega S+S\\Omega=A-A^T,

\\qquad

\\delta S=A-\\Omega S.

$$



The returned Fréchet derivative maps row-major components of $\\delta F$ to Mandel components $(\\delta S_{xx},\\delta S_{yy},\\delta S_{zz},\\sqrt2\\delta S_{yz},\\sqrt2\\delta S_{xz},\\sqrt2\\delta S_{xy})$. It must remain well defined for repeated stretch eigenvalues; differentiating singular vectors individually is not a valid shortcut there.

Returns
-------
Return F, R, S, strain, deviator, equivalent strain, and the Mandel-strain Fréchet derivative with shapes (n_elements, 3, 3), (n_elements,), and (n_elements, 6, 9).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def corotational_kinematics(
    reference: np.ndarray,
    deformed: np.ndarray,
    cells: np.ndarray,
) -> tuple:
    r"""Recover the corotational strain and equivalent deviatoric strain from nodal positions.

    Reference coordinates must be finite with shape (n_nodes, 3), n_nodes >= 4.
    Connectivity must be a nonempty integer (n_elements, 4) array with distinct, in-
    range vertices per cell. A cell is rejected if its absolute edge determinant is at
    most 1e-12 times the product of its edge lengths. Deformed coordinates must match
    reference shape and be finite; every deformation determinant must be positive.
    Equivalent strain below 1e-12 is treated as exactly zero, together with its
    deviator, so that a rigid motion produces an exact branch tie in later steps.
    Invalid inputs raise ValueError.

    Parameters
    ----------
    reference : np.ndarray
        Reference coordinates, shape (n_nodes, 3), in mm.
    deformed : np.ndarray
        Deformed coordinates, same shape and units as reference.
    cells : np.ndarray
        Zero-based connectivity, shape (n_elements, 4).

    Raises
    ------
    ValueError
        If either coordinate array is nonfinite or malformed; connectivity is empty,
        nonintegral, repeated, or out of range; a reference cell is degenerate; or a
        deformation reverses orientation or has zero determinant.

    Returns
    -------
    tuple
        (F, R, S, strain, deviator, q, strain_jacobian): five dimensionless arrays
        of shape (n_elements, 3, 3), one of shape (n_elements,), and the Fréchet
        derivative of Mandel strain with respect to row-major F of shape
        (n_elements, 6, 9).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def _mesh_arrays(reference, cells):
    points = _finite_array(reference)
    cells = np.asarray(cells)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("reference must have shape (n_nodes, 3), n_nodes >= 4")
    if cells.ndim != 2 or cells.shape[1] != 4 or len(cells) == 0:
        raise ValueError("cells must have shape (n_elements, 4)")
    if not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("cell indices must be integers")
    if np.any(cells < 0) or np.any(cells >= len(points)):
        raise ValueError("cell index out of range")
    if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
        raise ValueError("each cell needs four distinct vertices")
    edges = (points[cells[:, 1:]] - points[cells[:, :1]]).transpose(0, 2, 1)
    lengths = np.linalg.norm(edges, axis=1)
    scale = np.prod(lengths, axis=1)
    determinant = np.linalg.det(edges)
    if np.any(scale == 0) or np.any(np.abs(determinant) <= 1e-12 * scale):
        raise ValueError(
            "reference tetrahedra are degenerate at relative tolerance 1e-12"
        )
    return points, cells.astype(int), edges


def _strain_mandel_vector(tensor):
    root_two = np.sqrt(2.0)
    return np.array(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            root_two * tensor[1, 2],
            root_two * tensor[0, 2],
            root_two * tensor[0, 1],
        ]
    )


def _oracle_corotational_kinematics(
    reference: np.ndarray,
    deformed: np.ndarray,
    cells: np.ndarray,
) -> tuple:
    points, indices, edges = _mesh_arrays(reference, cells)
    current = _finite_array(deformed)
    if current.shape != points.shape:
        raise ValueError("deformed coordinates must match reference shape")
    spatial_edges = (current[indices[:, 1:]] - current[indices[:, :1]]).transpose(
        0, 2, 1
    )
    deform = spatial_edges @ np.linalg.inv(edges)
    if np.any(np.linalg.det(deform) <= 0):
        raise ValueError("deformation must preserve orientation")
    left, singular, right = np.linalg.svd(deform)
    rotation = left @ right
    stretch = (right.swapaxes(1, 2) * singular[:, None, :]) @ right
    strain = stretch - np.eye(3)
    deviator = (
        strain - np.trace(strain, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3.0
    )
    equivalent = np.sqrt(2.0 / 3.0 * np.sum(deviator**2, axis=(1, 2)))
    tiny = equivalent < 1e-12
    equivalent[tiny] = 0.0
    deviator[tiny] = 0.0
    strain_jacobian = np.empty((len(deform), 6, 9))
    for element in range(len(deform)):
        eigenvalues, eigenvectors = np.linalg.eigh(stretch[element])
        for component in range(9):
            perturbation = np.zeros((3, 3))
            perturbation.flat[component] = 1.0
            local = rotation[element].T @ perturbation
            local_eigen = eigenvectors.T @ local @ eigenvectors
            spin_eigen = (local_eigen - local_eigen.T) / (
                eigenvalues[:, None] + eigenvalues[None, :]
            )
            np.fill_diagonal(spin_eigen, 0.0)
            spin = eigenvectors @ spin_eigen @ eigenvectors.T
            stretch_increment = local - spin @ stretch[element]
            stretch_increment = 0.5 * (stretch_increment + stretch_increment.T)
            strain_jacobian[element, :, component] = _strain_mandel_vector(
                stretch_increment
            )
    return deform, rotation, stretch, strain, deviator, equivalent, strain_jacobian

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """
import numpy as np
X = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])
T = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
Y = X @ np.array([[1.12, .06, 0.], [0., .96, .03], [.01, 0., 1.04]]).T
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in corotational_kinematics(X, Y, T)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_corotational_kinematics(X, Y, T)])",
        },
        {
            "setup": """
import numpy as np
X = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])
T = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
Y = X @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).T
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in corotational_kinematics(X, Y, T)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_corotational_kinematics(X, Y, T)])",
        },
        {
            "setup": """
import numpy as np
X = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])
T = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
Y = 1.08 * X
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in corotational_kinematics(X, Y, T)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_corotational_kinematics(X, Y, T)])",
        },
        {
            "setup": """
import numpy as np
X = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])
T = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
Y = X * np.array([-1., 1., 1.])
def _invalid_status(function):
    try:
        function(X, Y, T)
    except ValueError:
        return 1.0
    except Exception:
        return 2.0
    return 0.0
""",
            "call": "_invalid_status(corotational_kinematics)",
            "gold_call": "_invalid_status(_oracle_corotational_kinematics)",
        },
    ]
