"""
Evaluate the standard quadratic 10-node tetrahedral displacement interpolation for the task's prescribed natural-coordinate node ordering.

Use the four barycentric coordinates associated with the tetrahedron vertices and preserve the node order specified in the task. Return both the ten shape-function values and their derivatives with respect to xi, eta, and zeta.

The shape functions must satisfy partition of unity, and the sum of their natural derivatives must vanish. Do not round intermediate values.

A quadratic tetrahedral finite element has four vertex nodes and six edge nodes. Its interpolation is conveniently expressed using tetrahedral barycentric coordinates.

The same displacement shape functions are used for the reference geometry and for the displacement field in an isoparametric ten-node tetrahedron. Their natural derivatives provide the starting point for the element Jacobian and finite-strain kinematics.

Returns
-------
A tuple (N, dN_dxi), where N has shape (10,) and dN_dxi has shape (10, 3), with columns corresponding to derivatives with respect to xi, eta, and zeta.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def tet10_shape_data(
    xi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return T10 shape functions and natural derivatives."""

    return (
        np.zeros(10, dtype=float),
        np.zeros((10, 3), dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_tet10_shape_data(
    xi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    xi = np.asarray(
        xi,
        dtype=float,
    )

    if xi.shape != (3,):
        raise ValueError(
            "xi must have shape (3,)"
        )

    x, y, z = xi

    L = np.array(
        [
            1.0 - x - y - z,
            x,
            y,
            z,
        ],
        dtype=float,
    )

    if (
        np.min(L) < -1e-14
        or np.max(L) > 1.0 + 1e-14
    ):
        raise ValueError(
            "point lies outside the reference tetrahedron"
        )

    dL = np.array(
        [
            [-1.0, -1.0, -1.0],
            [ 1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0,  1.0],
        ],
        dtype=float,
    )

    N = np.empty(
        10,
        dtype=float,
    )

    dN = np.empty(
        (10, 3),
        dtype=float,
    )

    for a in range(4):

        N[a] = (
            L[a]
            * (
                2.0 * L[a]
                - 1.0
            )
        )

        dN[a] = (
            (
                4.0 * L[a]
                - 1.0
            )
            * dL[a]
        )

    edges = (
        (0, 1),
        (1, 2),
        (2, 0),
        (0, 3),
        (1, 3),
        (2, 3),
    )

    for k, (a, b) in enumerate(
        edges,
        start=4,
    ):

        N[k] = (
            4.0
            * L[a]
            * L[b]
        )

        dN[k] = (
            4.0
            * (
                dL[a] * L[b]
                + L[a] * dL[b]
            )
        )

    return N, dN

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
xi = np.array([0.25, 0.25, 0.25], dtype=float)
""",
            "call": """
tet10_shape_data(xi)
""",
            "gold_call": """
_oracle_tet10_shape_data(xi)
""",
            "tol": 1e-12,
        },
        {
            "setup": """
xi = np.array([1.0, 0.0, 0.0], dtype=float)
""",
            "call": """
tet10_shape_data(xi)
""",
            "gold_call": """
_oracle_tet10_shape_data(xi)
""",
            "tol": 1e-12,
        },
        {
            "setup": """
xi = np.array([0.20, 0.30, 0.10], dtype=float)
""",
            "call": """
tet10_shape_data(xi)
""",
            "gold_call": """
_oracle_tet10_shape_data(xi)
""",
            "tol": 1e-12,
        },
        {
            "setup": """
xi = np.array([0.05, 0.15, 0.60], dtype=float)
""",
            "call": """
tet10_shape_data(xi)
""",
            "gold_call": """
_oracle_tet10_shape_data(xi)
""",
            "tol": 1e-12,
        },
    ]
