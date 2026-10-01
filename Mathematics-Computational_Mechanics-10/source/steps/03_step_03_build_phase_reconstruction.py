"""
Build the local constant-phase affine reconstruction and jump stabilization.

For one cell value followed by four face values, the affine phase gradient is



$$G_T\boldsymbol{\phi}_T =|T|^{-1}\sum_{F\in\mathcal{F}_T}|F|\phi_F\mathbf{n}_{TF}.$$



The jump residual on a face is the face-average difference between the affine reconstruction and the face value. The returned packed operator stores $G_T$ in its first two rows and the $5\times5$ jump matrix $J_T$ in its remaining rows.

Returns
-------
np.ndarray of shape (7, 5), with gradient rows first and jump matrix last
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_phase_reconstruction(
    cell_bounds: np.ndarray,
    face_segments: np.ndarray,
    outward_normals: np.ndarray,
) -> np.ndarray:
    r"""Return the packed phase gradient and stabilization operators.

    Parameters
    ----------
    cell_bounds : np.ndarray, shape (4,)
        Axis-aligned rectangle bounds.
    face_segments : np.ndarray, shape (4, 2, 2)
        Four local face segments.
    outward_normals : np.ndarray, shape (4, 2)
        Corresponding outward unit normals.

    Returns
    -------
    np.ndarray, shape (7, 5)
        Rows $0{:}2$ are $G_T$ and rows $2{:}7$ are the symmetric jump
        matrix $J_T$.

    Raises
    ------
    ValueError
        If shapes, finiteness, bounds, or face geometry are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_phase_reconstruction(cell_bounds, face_segments, outward_normals):
    """Reference affine phase gradient and face-average jump matrix."""
    

    bounds = np.asarray(cell_bounds, dtype=float)
    faces = np.asarray(face_segments, dtype=float)
    normals = np.asarray(outward_normals, dtype=float)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("cell_bounds must be finite with shape (4,)")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("cell_bounds must have positive side lengths")
    if faces.shape != (4, 2, 2) or not np.all(np.isfinite(faces)):
        raise ValueError("face_segments must be finite with shape (4, 2, 2)")
    if normals.shape != (4, 2) or not np.all(np.isfinite(normals)):
        raise ValueError("outward_normals must be finite with shape (4, 2)")

    x0, x1, y0, y1 = bounds
    centroid = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    area = (x1 - x0) * (y1 - y0)
    diameter = np.hypot(x1 - x0, y1 - y0)
    gradient = np.zeros((2, 5))
    lengths = np.empty(4)
    for local_face, (face, normal) in enumerate(zip(faces, normals)):
        tangent = face[1] - face[0]
        lengths[local_face] = np.linalg.norm(tangent)
        midpoint = 0.5 * (face[0] + face[1])
        if lengths[local_face] <= 0.0:
            raise ValueError("face segments must be nondegenerate")
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("outward normals must be unit vectors")
        if not np.isclose(tangent @ normal, 0.0, atol=1e-12, rtol=0.0):
            raise ValueError("normals must be perpendicular to faces")
        if (midpoint - centroid) @ normal <= 0.0:
            raise ValueError("normals must point outward")
        gradient[:, 1 + local_face] = lengths[local_face] / area * normal

    jump = np.zeros((5, 5))
    for local_face, face in enumerate(faces):
        midpoint = 0.5 * (face[0] + face[1])
        residual = np.zeros(5)
        residual[0] = 1.0
        residual += (midpoint - centroid) @ gradient
        residual[1 + local_face] -= 1.0
        jump += lengths[local_face] / diameter * np.outer(residual, residual)
    return np.vstack([gradient, 0.5 * (jump + jump.T)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return two rectangular cells and one invalid-face case."""
    return [
        {
            "setup": """import numpy as np
bounds = np.array([0.0, 0.5, 0.0, 1.0])
faces = np.array([[[0.0,0.0],[0.5,0.0]], [[0.5,0.0],[0.5,1.0]], [[0.5,1.0],[0.0,1.0]], [[0.0,1.0],[0.0,0.0]]])
normals = np.array([[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]])
""",
            "call": "build_phase_reconstruction(bounds, faces, normals)",
            "gold_call": "_oracle_build_phase_reconstruction(bounds, faces, normals)",
        },
        {
            "setup": """import numpy as np
bounds = np.array([-1.0, 2.0, -2.0, 2.0])
faces = np.array([[[-1.0,-2.0],[2.0,-2.0]], [[2.0,-2.0],[2.0,2.0]], [[2.0,2.0],[-1.0,2.0]], [[-1.0,2.0],[-1.0,-2.0]]])
normals = np.array([[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]])
""",
            "call": "build_phase_reconstruction(bounds, faces, normals)",
            "gold_call": "_oracle_build_phase_reconstruction(bounds, faces, normals)",
        },
        {
            "setup": """import numpy as np
bounds = np.array([0.0, 0.5, 0.0, 1.0])
faces = np.zeros((3, 2, 2))
normals = np.zeros((3, 2))
def run_model():
    try:
        build_phase_reconstruction(bounds, faces, normals)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_build_phase_reconstruction(bounds, faces, normals)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
