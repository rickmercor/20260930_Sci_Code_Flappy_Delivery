"""
Build the local affine symmetric-strain reconstruction.

For a degree-$1$ cell/face displacement vector $\mathbf{v}_T$, the reconstructed strain $E_T^1\mathbf{v}_T$ is the affine symmetric tensor satisfying



$$\int_T E_T^1\mathbf{v}_T:\boldsymbol{\tau} =-\int_T \mathbf{v}_T\cdot\nabla\cdot\boldsymbol{\tau} +\sum_{F\in\mathcal{F}_T}\int_F \mathbf{v}_F\cdot(\boldsymbol{\tau}\mathbf{n}_{TF}).$$



for every affine symmetric test tensor. The local ordering is six cell coefficients followed by four coefficients for each oriented face. The output uses strain-component order $(xx,yy,zz,xy)$ and four tensor-product Gauss nodes; plane strain makes the $zz$ row identically zero.

Returns
-------
np.ndarray of shape (4, 4, 22) in (xx, yy, zz, xy) component order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_affine_strain_reconstruction(
    cell_bounds: np.ndarray,
    face_segments: np.ndarray,
    outward_normals: np.ndarray,
) -> np.ndarray:
    r"""Return the quadrature-point strain reconstruction matrices.

    Parameters
    ----------
    cell_bounds : np.ndarray, shape (4,)
        $[x_{\min},x_{\max},y_{\min},y_{\max}]$ for an axis-aligned rectangle.
    face_segments : np.ndarray, shape (4, 2, 2)
        Four nondegenerate oriented face segments in local order.
    outward_normals : np.ndarray, shape (4, 2)
        Corresponding finite outward unit normals.

    Returns
    -------
    np.ndarray, shape (4, 4, 22)
        Strain matrices at the four cell Gauss nodes.

    Raises
    ------
    ValueError
        If shapes, finiteness, rectangle bounds, face geometry, or outward
        unit-normal conditions are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_affine_strain_reconstruction(
    cell_bounds, face_segments, outward_normals
):
    """Reference affine reconstruction from discrete integration by parts."""
    

    bounds = np.asarray(cell_bounds, dtype=float)
    faces = np.asarray(face_segments, dtype=float)
    normals = np.asarray(outward_normals, dtype=float)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("cell_bounds must be a finite array of shape (4,)")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("cell_bounds must have positive side lengths")
    if faces.shape != (4, 2, 2) or not np.all(np.isfinite(faces)):
        raise ValueError("face_segments must have shape (4, 2, 2)")
    if normals.shape != (4, 2) or not np.all(np.isfinite(normals)):
        raise ValueError("outward_normals must have shape (4, 2)")

    x0, x1, y0, y1 = bounds
    centroid = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    side_x = x1 - x0
    side_y = y1 - y0
    area = side_x * side_y
    diameter = np.hypot(side_x, side_y)
    root = 1.0 / np.sqrt(3.0)
    gauss = np.array([-root, root])

    for face, normal in zip(faces, normals):
        tangent = face[1] - face[0]
        length = np.linalg.norm(tangent)
        midpoint = 0.5 * (face[0] + face[1])
        if length <= 0.0:
            raise ValueError("face segments must be nondegenerate")
        if not np.isclose(np.linalg.norm(normal), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("outward normals must be unit vectors")
        if not np.isclose(tangent @ normal, 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("each normal must be perpendicular to its face")
        if (midpoint - centroid) @ normal <= 0.0:
            raise ValueError("normals must point outward")
        on_vertical = np.isclose(midpoint[0], [x0, x1], atol=1e-12).any()
        on_horizontal = np.isclose(midpoint[1], [y0, y1], atol=1e-12).any()
        if not (on_vertical or on_horizontal):
            raise ValueError("each segment must lie on the rectangle boundary")

    points = []
    for xi in gauss:
        for eta in gauss:
            points.append(
                [
                    centroid[0] + 0.5 * side_x * xi,
                    centroid[1] + 0.5 * side_y * eta,
                ]
            )
    points = np.asarray(points)
    weights = np.full(4, area / 4.0)

    def basis(coordinates):
        coordinates = np.asarray(coordinates)
        return np.column_stack(
            [
                np.ones(coordinates.shape[0]),
                (coordinates[:, 0] - centroid[0]) / diameter,
                (coordinates[:, 1] - centroid[1]) / diameter,
            ]
        )

    cell_basis = basis(points)
    scalar_mass = cell_basis.T @ (weights[:, None] * cell_basis)
    strain_mass = np.zeros((9, 9))
    strain_mass[0:3, 0:3] = scalar_mass
    strain_mass[3:6, 3:6] = scalar_mass
    strain_mass[6:9, 6:9] = 2.0 * scalar_mass
    right_hand_side = np.zeros((9, 22))

    integrated_cell_basis = weights @ cell_basis
    for polynomial in range(3):
        derivative_x = 1.0 / diameter if polynomial == 1 else 0.0
        derivative_y = 1.0 / diameter if polynomial == 2 else 0.0
        right_hand_side[polynomial, 0:3] -= derivative_x * integrated_cell_basis
        right_hand_side[3 + polynomial, 3:6] -= derivative_y * integrated_cell_basis
        right_hand_side[6 + polynomial, 0:3] -= derivative_y * integrated_cell_basis
        right_hand_side[6 + polynomial, 3:6] -= derivative_x * integrated_cell_basis

    for local_face, (face, normal) in enumerate(zip(faces, normals)):
        length = np.linalg.norm(face[1] - face[0])
        midpoint = 0.5 * (face[0] + face[1])
        half_tangent = 0.5 * (face[1] - face[0])
        face_points = midpoint + gauss[:, None] * half_tangent
        face_weights = np.full(2, length / 2.0)
        face_basis = np.column_stack([np.ones(2), gauss])
        moments = basis(face_points).T @ (face_weights[:, None] * face_basis)
        offset = 6 + 4 * local_face
        normal_x, normal_y = normal
        right_hand_side[0:3, offset : offset + 2] += normal_x * moments
        right_hand_side[3:6, offset + 2 : offset + 4] += normal_y * moments
        right_hand_side[6:9, offset : offset + 2] += normal_y * moments
        right_hand_side[6:9, offset + 2 : offset + 4] += normal_x * moments

    coefficients = np.linalg.solve(strain_mass, right_hand_side)
    result = np.zeros((4, 4, 22))
    for node, values in enumerate(cell_basis):
        result[node, 0] = values @ coefficients[0:3]
        result[node, 1] = values @ coefficients[3:6]
        result[node, 3] = values @ coefficients[6:9]
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return canonical, scaled, and invalid-normal reconstruction cases."""
    return [
        {
            "setup": """import numpy as np
bounds = np.array([0.0, 0.5, 0.0, 1.0])
faces = np.array([[[0.0,0.0],[0.5,0.0]], [[0.5,0.0],[0.5,1.0]], [[0.5,1.0],[0.0,1.0]], [[0.0,1.0],[0.0,0.0]]])
normals = np.array([[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]])
""",
            "call": "build_affine_strain_reconstruction(bounds, faces, normals)",
            "gold_call": "_oracle_build_affine_strain_reconstruction(bounds, faces, normals)",
        },
        {
            "setup": """import numpy as np
bounds = np.array([-1.0, 2.0, 0.5, 2.5])
faces = np.array([[[-1.0,0.5],[2.0,0.5]], [[2.0,0.5],[2.0,2.5]], [[2.0,2.5],[-1.0,2.5]], [[-1.0,2.5],[-1.0,0.5]]])
normals = np.array([[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]])
""",
            "call": "build_affine_strain_reconstruction(bounds, faces, normals)",
            "gold_call": "_oracle_build_affine_strain_reconstruction(bounds, faces, normals)",
        },
        {
            "setup": """import numpy as np
bounds = np.array([0.0, 0.5, 0.0, 1.0])
faces = np.array([[[0.0,0.0],[0.5,0.0]], [[0.5,0.0],[0.5,1.0]], [[0.5,1.0],[0.0,1.0]], [[0.0,1.0],[0.0,0.0]]])
normals = np.array([[0.0,1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]])
def run_model():
    try:
        build_affine_strain_reconstruction(bounds, faces, normals)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_build_affine_strain_reconstruction(bounds, faces, normals)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
