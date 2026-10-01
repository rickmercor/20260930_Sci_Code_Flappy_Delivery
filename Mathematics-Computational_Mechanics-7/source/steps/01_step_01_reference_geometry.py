"""
Compute the reference volume and barycentric shape-function gradients for every tetrahedral element.

For a linear tetrahedral element, the reference edge matrix is formed from the three edges leaving local vertex 0:



$$

D_m=[X_1-X_0,X_2-X_0,X_3-X_0].

$$



Its signed determinant records local orientation, while the physical reference volume is



$$

V=\\dfrac{|\\det D_m|}{6}.

$$



Writing the barycentric coordinates as $\\xi=D_m^{-1}(X-X_0)$ shows that the gradients for local vertices 1–3 are the rows of $D_m^{-1}$:



$$

\\nabla N_j=(D_m^{-1})_{j-1,:},\\qquad j=1,2,3,

$$



and partition of unity gives



$$

\\nabla N_0=-\\sum_{j=1}^{3}\\nabla N_j.

$$



The row convention is load-bearing on a skew mesh because it preserves the affine identities $\\sum_i\\nabla N_i=0$ and $\\sum_iX_i\\otimes\\nabla N_i=I$. Volumes have units $\\mathrm{mm}^3$ and gradients have units $\\mathrm{mm}^{-1}$. The original local vertex ordering is retained even when its orientation is negative.

Returns
-------
Return a tuple containing the reference volumes in mm cubed as an array of shape (n_elements,) and the reference gradients in inverse mm as an array of shape (n_elements, 4, 3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reference_geometry(
    reference: np.ndarray,
    cells: np.ndarray,
) -> tuple:
    r"""Construct reference volumes and barycentric gradients for a tetrahedral patch.

    Reference coordinates must be finite with shape (n_nodes, 3), n_nodes >= 4.
    Connectivity must be a nonempty integer (n_elements, 4) array with distinct, in-
    range vertices per cell. A cell is rejected if its absolute edge determinant is at
    most 1e-12 times the product of its edge lengths. Invalid inputs raise ValueError.

    Parameters
    ----------
    reference : np.ndarray
        Reference coordinates, shape (n_nodes, 3), in mm.
    cells : np.ndarray
        Zero-based cell indices, shape (n_elements, 4), in local vertex order.

    Raises
    ------
    ValueError
        If coordinates are nonfinite or have the wrong shape; connectivity is empty,
        nonintegral, repeated, or out of range; or any reference tetrahedron is
        degenerate at the stated relative tolerance.

    Returns
    -------
    tuple
        (volumes, gradients): float arrays of shapes (n_elements,) and (n_elements, 4,
        3), in mm cubed and inverse mm.
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


def _oracle_reference_geometry(
    reference: np.ndarray,
    cells: np.ndarray,
) -> tuple:
    _, _, edges = _mesh_arrays(reference, cells)
    inverse = np.linalg.inv(edges)
    volumes = np.abs(np.linalg.det(edges)) / 6.0
    gradients = np.concatenate((-inverse.sum(axis=1, keepdims=True), inverse), axis=1)
    if not np.all(np.isfinite(volumes)) or not np.all(np.isfinite(gradients)):
        raise ValueError("reference geometry exceeds finite arithmetic range")
    return volumes, gradients

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
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in reference_geometry(X, T)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_reference_geometry(X, T)])",
        },
        {
            "setup": """
import numpy as np
X = np.vstack((np.zeros(3), np.eye(3)))
T = np.array([[0, 1, 2, 3]])
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in reference_geometry(X, T)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_reference_geometry(X, T)])",
        },
        {
            "setup": """
import numpy as np
X = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])
T = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
X *= 2
T = T[:, [0, 2, 1, 3]]
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in reference_geometry(X, T)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_reference_geometry(X, T)])",
        },
        {
            "setup": """
import numpy as np
X = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])
T = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
T[0, 1] = 0
def _invalid_status(function):
    try:
        function(X, T)
    except ValueError:
        return 1.0
    except Exception:
        return 2.0
    return 0.0
""",
            "call": "_invalid_status(reference_geometry)",
            "gold_call": "_invalid_status(_oracle_reference_geometry)",
        },
    ]
