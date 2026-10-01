"""
Build the local quadratic-reconstruction stabilization matrix.

The reconstructed quadratic displacement has its symmetric gradient equal to the best $L^2$ approximation of the affine reconstructed strain. Cell means and mean skew gradients fix its three rigid-body modes. Projecting the difference between that field and the cell/face unknowns back to degree $1$ gives the second local least-squares stabilization, which vanishes for every interpolated quadratic displacement.

Returns
-------
symmetric np.ndarray of shape (22, 22) with the quadratic consistency nullspace
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_quadratic_stabilization(
    cell_bounds: np.ndarray,
    face_segments: np.ndarray,
    outward_normals: np.ndarray,
    strain_reconstruction: np.ndarray,
) -> np.ndarray:
    """Return the quadratic-exact local stabilization matrix.

    Parameters
    ----------
    cell_bounds : np.ndarray, shape (4,)
        Axis-aligned rectangle bounds.
    face_segments : np.ndarray, shape (4, 2, 2)
        Four oriented local face segments.
    outward_normals : np.ndarray, shape (4, 2)
        Corresponding outward unit normals.
    strain_reconstruction : np.ndarray, shape (4, 4, 22)
        Affine strain matrices at the four cell Gauss nodes.

    Returns
    -------
    np.ndarray, shape (22, 22)
        Symmetric positive-semidefinite stabilization matrix.

    Raises
    ------
    ValueError
        If any input has an invalid shape, nonfinite value, or degenerate
        geometry.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_quadratic_stabilization(
    cell_bounds, face_segments, outward_normals, strain_reconstruction
):
    """Reference constrained quadratic reconstruction and stabilization."""
    

    bounds = np.asarray(cell_bounds, dtype=float)
    faces = np.asarray(face_segments, dtype=float)
    normals = np.asarray(outward_normals, dtype=float)
    strain = np.asarray(strain_reconstruction, dtype=float)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("cell_bounds must be finite with shape (4,)")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("cell_bounds must have positive side lengths")
    if faces.shape != (4, 2, 2) or not np.all(np.isfinite(faces)):
        raise ValueError("face_segments must be finite with shape (4, 2, 2)")
    if normals.shape != (4, 2) or not np.all(np.isfinite(normals)):
        raise ValueError("outward_normals must be finite with shape (4, 2)")
    if strain.shape != (4, 4, 22) or not np.all(np.isfinite(strain)):
        raise ValueError("strain_reconstruction must have shape (4, 4, 22)")

    x0, x1, y0, y1 = bounds
    centroid = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    side_x = x1 - x0
    side_y = y1 - y0
    area = side_x * side_y
    diameter = np.hypot(side_x, side_y)
    root = 1.0 / np.sqrt(3.0)
    gauss = np.array([-root, root])
    points = np.array(
        [
            [centroid[0] + 0.5 * side_x * xi, centroid[1] + 0.5 * side_y * eta]
            for xi in gauss
            for eta in gauss
        ]
    )
    weights = np.full(4, area / 4.0)

    for face, normal in zip(faces, normals):
        tangent = face[1] - face[0]
        if np.linalg.norm(tangent) <= 0.0:
            raise ValueError("face segments must be nondegenerate")
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("outward normals must be unit vectors")
        if not np.isclose(tangent @ normal, 0.0, atol=1e-12, rtol=0.0):
            raise ValueError("normals must be perpendicular to faces")
        if (0.5 * (face[0] + face[1]) - centroid) @ normal <= 0.0:
            raise ValueError("normals must point outward")

    def p1_basis(coordinates):
        coordinates = np.asarray(coordinates)
        return np.column_stack(
            [
                np.ones(coordinates.shape[0]),
                (coordinates[:, 0] - centroid[0]) / diameter,
                (coordinates[:, 1] - centroid[1]) / diameter,
            ]
        )

    def p2_basis(coordinates):
        coordinates = np.asarray(coordinates)
        scaled_x = (coordinates[:, 0] - centroid[0]) / diameter
        scaled_y = (coordinates[:, 1] - centroid[1]) / diameter
        return np.column_stack(
            [
                np.ones(coordinates.shape[0]),
                scaled_x,
                scaled_y,
                scaled_x**2,
                scaled_x * scaled_y,
                scaled_y**2,
            ]
        )

    def p2_gradient(coordinates):
        coordinates = np.asarray(coordinates)
        scaled_x = (coordinates[:, 0] - centroid[0]) / diameter
        scaled_y = (coordinates[:, 1] - centroid[1]) / diameter
        derivative_x = np.column_stack(
            [
                np.zeros(coordinates.shape[0]),
                np.ones(coordinates.shape[0]) / diameter,
                np.zeros(coordinates.shape[0]),
                2.0 * scaled_x / diameter,
                scaled_y / diameter,
                np.zeros(coordinates.shape[0]),
            ]
        )
        derivative_y = np.column_stack(
            [
                np.zeros(coordinates.shape[0]),
                np.zeros(coordinates.shape[0]),
                np.ones(coordinates.shape[0]) / diameter,
                np.zeros(coordinates.shape[0]),
                scaled_x / diameter,
                2.0 * scaled_y / diameter,
            ]
        )
        return derivative_x, derivative_y

    p1_values = p1_basis(points)
    p2_values = p2_basis(points)
    derivative_x, derivative_y = p2_gradient(points)
    metric = np.diag([1.0, 1.0, 2.0])
    energy = np.zeros((12, 12))
    forcing = np.zeros((12, 22))
    for node, weight in enumerate(weights):
        symmetric_gradient = np.zeros((3, 12))
        symmetric_gradient[0, 0:6] = derivative_x[node]
        symmetric_gradient[1, 6:12] = derivative_y[node]
        symmetric_gradient[2, 0:6] = 0.5 * derivative_y[node]
        symmetric_gradient[2, 6:12] = 0.5 * derivative_x[node]
        reconstructed = strain[node, [0, 1, 3]]
        energy += weight * (symmetric_gradient.T @ metric @ symmetric_gradient)
        forcing += weight * (symmetric_gradient.T @ metric @ reconstructed)

    constraints = np.zeros((3, 12))
    constraint_data = np.zeros((3, 22))
    integrated_p2 = weights @ p2_values
    integrated_p1 = weights @ p1_values
    constraints[0, 0:6] = integrated_p2
    constraints[1, 6:12] = integrated_p2
    constraint_data[0, 0:3] = integrated_p1
    constraint_data[1, 3:6] = integrated_p1
    constraints[2, 0:6] = -0.5 * (weights @ derivative_y)
    constraints[2, 6:12] = 0.5 * (weights @ derivative_x)

    for local_face, (face, normal) in enumerate(zip(faces, normals)):
        length = np.linalg.norm(face[1] - face[0])
        face_basis = np.column_stack([np.ones(2), gauss])
        integrated_face_basis = (length / 2.0) * np.sum(face_basis, axis=0)
        offset = 6 + 4 * local_face
        constraint_data[2, offset : offset + 2] += (
            -0.5 * normal[1] * integrated_face_basis
        )
        constraint_data[2, offset + 2 : offset + 4] += (
            0.5 * normal[0] * integrated_face_basis
        )

    saddle = np.block([[energy, constraints.T], [constraints, np.zeros((3, 3))]])
    reconstruction = np.linalg.solve(saddle, np.vstack([forcing, constraint_data]))[
        0:12
    ]

    cell_mass = p1_values.T @ (weights[:, None] * p1_values)
    cell_cross = p1_values.T @ (weights[:, None] * p2_values)
    cell_projection = np.linalg.solve(cell_mass, cell_cross)
    result = np.zeros((22, 22))
    for component in range(2):
        difference = (
            cell_projection @ reconstruction[6 * component : 6 * (component + 1)]
        )
        difference[:, 3 * component : 3 * (component + 1)] -= np.eye(3)
        result += diameter**-2 * (difference.T @ cell_mass @ difference)

    for local_face, face in enumerate(faces):
        length = np.linalg.norm(face[1] - face[0])
        midpoint = 0.5 * (face[0] + face[1])
        half_tangent = 0.5 * (face[1] - face[0])
        face_points = midpoint + gauss[:, None] * half_tangent
        face_basis = np.column_stack([np.ones(2), gauss])
        face_p2 = p2_basis(face_points)
        face_weights = np.full(2, length / 2.0)
        face_mass = face_basis.T @ (face_weights[:, None] * face_basis)
        face_cross = face_basis.T @ (face_weights[:, None] * face_p2)
        face_projection = np.linalg.solve(face_mass, face_cross)
        offset = 6 + 4 * local_face
        for component in range(2):
            difference = (
                face_projection @ reconstruction[6 * component : 6 * (component + 1)]
            )
            start = offset + 2 * component
            difference[:, start : start + 2] -= np.eye(2)
            result += length**-1 * (difference.T @ face_mass @ difference)
    return 0.5 * (result + result.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return canonical, scaled, and invalid reconstruction cases."""
    return [
        {
            "setup": """import numpy as np
bounds = np.array([0.0, 0.5, 0.0, 1.0])
faces = np.array([[[0.0,0.0],[0.5,0.0]], [[0.5,0.0],[0.5,1.0]], [[0.5,1.0],[0.0,1.0]], [[0.0,1.0],[0.0,0.0]]])
normals = np.array([[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]])
B = _oracle_build_affine_strain_reconstruction(bounds, faces, normals)
""",
            "call": "build_quadratic_stabilization(bounds, faces, normals, B)",
            "gold_call": "_oracle_build_quadratic_stabilization(bounds, faces, normals, B)",
        },
        {
            "setup": """import numpy as np
bounds = np.array([0.5, 1.0, 0.0, 1.0])
faces = np.array([[[0.5,0.0],[1.0,0.0]], [[1.0,0.0],[1.0,1.0]], [[1.0,1.0],[0.5,1.0]], [[0.5,1.0],[0.5,0.0]]])
normals = np.array([[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]])
B = _oracle_build_affine_strain_reconstruction(bounds, faces, normals)
""",
            "call": "build_quadratic_stabilization(bounds, faces, normals, B)",
            "gold_call": "_oracle_build_quadratic_stabilization(bounds, faces, normals, B)",
        },
        {
            "setup": """import numpy as np
bounds = np.array([0.0, 0.5, 0.0, 1.0])
faces = np.zeros((4, 2, 2))
normals = np.zeros((4, 2))
B = np.zeros((4, 4, 21))
def run_model():
    try:
        build_quadratic_stabilization(bounds, faces, normals, B)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_build_quadratic_stabilization(bounds, faces, normals, B)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
