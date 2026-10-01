"""
Evaluate the finite-deformation kinematics of a 10-node isoparametric tetrahedron at one natural-coordinate point.



Use the reference nodal coordinates to form the geometric Jacobian and transform the natural shape-function derivatives to derivatives with respect to the reference Cartesian coordinates.



Using the current nodal positions X + u, construct the deformation gradient, the engineering-Voigt Green-Lagrange strain vector, and the corresponding nonlinear strain-displacement matrix.



Return the reference Jacobian determinant, reference-coordinate shape gradients, deformation gradient, strain vector, and nonlinear B matrix.



Reject an integration point with a nonpositive reference Jacobian determinant. Do not round intermediate values.

Finite-strain solid elements distinguish between the reference and current configurations. Shape-function derivatives are first mapped from natural coordinates to the reference configuration.

The deformation gradient measures the local mapping from reference to current coordinates. Green-Lagrange strain is objective under rigid rotation, while the associated nonlinear strain-displacement matrix supplies the displacement linearization required by the element tangent.

Returns
-------
A tuple (detJ, gradN, F, E_green, B), where gradN has shape (10,3), F has shape (3,3), E_green has shape (6,), and B has shape (6,30).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def tet10_kinematics(
    coords: np.ndarray,
    disp: np.ndarray,
    xi: np.ndarray,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return finite-deformation T10 kinematic quantities."""

    return (
        0.0,
        np.zeros((10, 3), dtype=float),
        np.zeros((3, 3), dtype=float),
        np.zeros(6, dtype=float),
        np.zeros((6, 30), dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_tet10_kinematics(
    coords: np.ndarray,
    disp: np.ndarray,
    xi: np.ndarray,
):

    coords = np.asarray(
        coords,
        dtype=float,
    )

    disp = np.asarray(
        disp,
        dtype=float,
    )

    if (
        coords.shape != (10, 3)
        or disp.shape != (10, 3)
    ):
        raise ValueError(
            "coords and disp must have shape (10,3)"
        )

    N, dN_nat = (
        _oracle_tet10_shape_data(
            xi
        )
    )

    J = (
        coords.T
        @ dN_nat
    )

    detJ = float(
        np.linalg.det(J)
    )

    if (
        not np.isfinite(detJ)
        or detJ <= 0.0
    ):
        raise ValueError(
            "nonpositive reference Jacobian"
        )

    gradN = (
        dN_nat
        @ np.linalg.inv(J)
    )

    current = (
        coords
        + disp
    )

    F = (
        current.T
        @ gradN
    )

    E_tensor = (
        0.5
        * (
            F.T @ F
            - np.eye(3)
        )
    )

    E_green = np.array(
        [
            E_tensor[0, 0],
            E_tensor[1, 1],
            E_tensor[2, 2],
            2.0 * E_tensor[0, 1],
            2.0 * E_tensor[0, 2],
            2.0 * E_tensor[1, 2],
        ],
        dtype=float,
    )

    B = np.zeros(
        (6, 30),
        dtype=float,
    )

    x1 = F[:, 0]
    x2 = F[:, 1]
    x3 = F[:, 2]

    for i in range(10):

        g1, g2, g3 = (
            gradN[i]
        )

        block = np.vstack(
            (
                g1 * x1,
                g2 * x2,
                g3 * x3,
                g1 * x2
                + g2 * x1,
                g1 * x3
                + g3 * x1,
                g2 * x3
                + g3 * x2,
            )
        )

        B[
            :,
            3 * i : 3 * i + 3,
        ] = block

    return (
        detJ,
        gradN,
        F,
        E_green,
        B,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
coords = np.array([
    [ 0.00, 0.00, 0.00],
    [ 1.20, 0.10, 0.00],
    [ 0.10, 1.10, 0.05],
    [ 0.00, 0.15, 0.95],
    [ 0.62, 0.02, 0.03],
    [ 0.68, 0.62,-0.02],
    [ 0.02, 0.58, 0.04],
    [-0.03, 0.08, 0.50],
    [ 0.58, 0.15, 0.50],
    [ 0.06, 0.62, 0.52],
], dtype=float)

X = coords[:, 0]
Y = coords[:, 1]
Z = coords[:, 2]

disp = np.column_stack([
    0.08*X + 0.025*Y + 0.015*X*Z,
   -0.035*Y + 0.020*Z + 0.010*X*Y,
    0.060*Z - 0.018*X + 0.012*Y*Z,
])

xi = np.array([0.25, 0.25, 0.25], dtype=float)
""",
            "call": """
tet10_kinematics(coords, disp, xi)
""",
            "gold_call": """
_oracle_tet10_kinematics(coords, disp, xi)
""",
            "tol": 5e-12,
        },
        {
            "setup": """
coords = np.array([
    [ 0.00, 0.00, 0.00],
    [ 1.20, 0.10, 0.00],
    [ 0.10, 1.10, 0.05],
    [ 0.00, 0.15, 0.95],
    [ 0.62, 0.02, 0.03],
    [ 0.68, 0.62,-0.02],
    [ 0.02, 0.58, 0.04],
    [-0.03, 0.08, 0.50],
    [ 0.58, 0.15, 0.50],
    [ 0.06, 0.62, 0.52],
], dtype=float)

X = coords[:, 0]
Y = coords[:, 1]
Z = coords[:, 2]

disp = np.column_stack([
    0.08*X + 0.025*Y + 0.015*X*Z,
   -0.035*Y + 0.020*Z + 0.010*X*Y,
    0.060*Z - 0.018*X + 0.012*Y*Z,
])

xi = np.array([0.0, 0.0, 0.0], dtype=float)
""",
            "call": """
tet10_kinematics(coords, disp, xi)
""",
            "gold_call": """
_oracle_tet10_kinematics(coords, disp, xi)
""",
            "tol": 5e-12,
        },
        {
            "setup": """
coords = np.array([
    [ 0.00, 0.00, 0.00],
    [ 1.20, 0.10, 0.00],
    [ 0.10, 1.10, 0.05],
    [ 0.00, 0.15, 0.95],
    [ 0.62, 0.02, 0.03],
    [ 0.68, 0.62,-0.02],
    [ 0.02, 0.58, 0.04],
    [-0.03, 0.08, 0.50],
    [ 0.58, 0.15, 0.50],
    [ 0.06, 0.62, 0.52],
], dtype=float)

X = coords[:, 0]
Y = coords[:, 1]
Z = coords[:, 2]

disp = np.column_stack([
    0.08*X + 0.025*Y + 0.015*X*Z,
   -0.035*Y + 0.020*Z + 0.010*X*Y,
    0.060*Z - 0.018*X + 0.012*Y*Z,
])

xi = np.array([0.5, 0.5, 0.0], dtype=float)
""",
            "call": """
tet10_kinematics(coords, disp, xi)
""",
            "gold_call": """
_oracle_tet10_kinematics(coords, disp, xi)
""",
            "tol": 5e-12,
        },
        {
            "setup": """
coords = np.array([
    [ 0.00, 0.00, 0.00],
    [ 1.20, 0.10, 0.00],
    [ 0.10, 1.10, 0.05],
    [ 0.00, 0.15, 0.95],
    [ 0.62, 0.02, 0.03],
    [ 0.68, 0.62,-0.02],
    [ 0.02, 0.58, 0.04],
    [-0.03, 0.08, 0.50],
    [ 0.58, 0.15, 0.50],
    [ 0.06, 0.62, 0.52],
], dtype=float)

X = coords[:, 0]
Y = coords[:, 1]
Z = coords[:, 2]

disp = np.column_stack([
    0.08*X + 0.025*Y + 0.015*X*Z,
   -0.035*Y + 0.020*Z + 0.010*X*Y,
    0.060*Z - 0.018*X + 0.012*Y*Z,
])

xi = np.array([0.18, 0.22, 0.17], dtype=float)
""",
            "call": """
tet10_kinematics(coords, disp, xi)
""",
            "gold_call": """
_oracle_tet10_kinematics(coords, disp, xi)
""",
            "tol": 5e-12,
        },
    ]
