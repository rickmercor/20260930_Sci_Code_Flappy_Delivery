"""
Assemble the dual-mortar coupling matrices for one rectangular non-mortar Q4 surface coupled to the supplied regular mortar-side grid.

Use the source interface quadrature order from the preceding step.

At every tensor-product Gauss point evaluate the source primal and dual non-mortar functions, map the point to the physical interface, evaluate the mortar-side trial interpolation, and integrate the two source coupling matrices.

Construct the mortar projection by solving the resulting dual coupling system.

Return D, A, P, and the one-dimensional Gauss order.

Mortar matrices relate the constraint interpolation to the displacement interpolation on the two sides of an interface.

A dual basis chosen to be bi-orthogonal to the non-mortar primal basis produces a particularly simple first coupling matrix. The resulting projection operator transfers the retained mortar-side field to the eliminated non-mortar interface field.

Returns
-------
A tuple (D, A, P, n_gauss).  D has shape (4,4).  A has shape (4,(n_cells_x+1)*(n_cells_y+1)).  P has the same shape as A.  n_gauss is the source one-dimensional interface quadrature order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_dual_mortar_projection(
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return D, A, dual-mortar projection P, and Gauss order."""

    n2 = (
        (n_cells_x + 1)
        * (n_cells_y + 1)
    )

    return (
        np.zeros((4, 4), dtype=float),
        np.zeros((4, n2), dtype=float),
        np.zeros((4, n2), dtype=float),
        1,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_dual_mortar_projection(
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
):

    Lx = float(length_x)
    Ly = float(length_y)

    nx = int(n_cells_x)
    ny = int(n_cells_y)

    spacing = np.array(
        [
            Lx / nx,
            Ly / ny,
        ],
        dtype=float,
    )

    (
        n_gauss,
        points,
        weights,
    ) = _oracle_coupling_gauss_rule(
        np.array(
            [Lx, Ly],
            dtype=float,
        ),
        spacing,
    )

    n2 = (
        (nx + 1)
        * (ny + 1)
    )

    D = np.zeros(
        (4, 4),
        dtype=float,
    )

    A = np.zeros(
        (4, n2),
        dtype=float,
    )

    jacobian = (
        Lx
        * Ly
        / 4.0
    )

    for i in range(
        n_gauss
    ):

        xi = points[i]

        for j in range(
            n_gauss
        ):

            eta = points[j]

            Nfe, Phi = (
                _oracle_dual_mortar_q4_shapes(
                    xi,
                    eta,
                )
            )

            x = (
                0.5
                * Lx
                * xi
            )

            y = (
                0.5
                * Ly
                * eta
            )

            Nmpm = (
                _oracle_regular_grid_interface_shapes(
                    x,
                    y,
                    Lx,
                    Ly,
                    nx,
                    ny,
                )
            )

            weight = (
                weights[i]
                * weights[j]
                * jacobian
            )

            D += (
                weight
                * np.outer(
                    Phi,
                    Nfe,
                )
            )

            A += (
                weight
                * np.outer(
                    Phi,
                    Nmpm,
                )
            )

    P = np.linalg.solve(
        D,
        A,
    )

    return (
        D,
        A,
        P,
        n_gauss,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
length_x=2.4
length_y=1.6
n_cells_x=3
n_cells_y=2
""",
            "call": """
(lambda z: np.concatenate((
z[0].ravel(),z[1].ravel(),z[2].ravel(),[float(z[3])]
)))(assemble_dual_mortar_projection(
length_x,length_y,n_cells_x,n_cells_y))
""",
            "gold_call": """
(lambda z: np.concatenate((
z[0].ravel(),z[1].ravel(),z[2].ravel(),[float(z[3])]
)))(_oracle_assemble_dual_mortar_projection(
length_x,length_y,n_cells_x,n_cells_y))
""",
            "tol": 5e-11,
        },
        {
            "setup": """
length_x=2.
length_y=1.5
n_cells_x=2
n_cells_y=3
""",
            "call": """
(lambda z: np.concatenate((
z[0].ravel(),z[1].ravel(),z[2].ravel(),[float(z[3])]
)))(assemble_dual_mortar_projection(
length_x,length_y,n_cells_x,n_cells_y))
""",
            "gold_call": """
(lambda z: np.concatenate((
z[0].ravel(),z[1].ravel(),z[2].ravel(),[float(z[3])]
)))(_oracle_assemble_dual_mortar_projection(
length_x,length_y,n_cells_x,n_cells_y))
""",
            "tol": 5e-11,
        },
        {
            "setup": """
length_x=3.
length_y=1.8
n_cells_x=4
n_cells_y=3
""",
            "call": """
(lambda z: np.concatenate((
z[0].ravel(),z[1].ravel(),z[2].ravel(),[float(z[3])]
)))(assemble_dual_mortar_projection(
length_x,length_y,n_cells_x,n_cells_y))
""",
            "gold_call": """
(lambda z: np.concatenate((
z[0].ravel(),z[1].ravel(),z[2].ravel(),[float(z[3])]
)))(_oracle_assemble_dual_mortar_projection(
length_x,length_y,n_cells_x,n_cells_y))
""",
            "tol": 5e-10,
        },
        {
            "setup": """
length_x=1.8
length_y=1.2
n_cells_x=3
n_cells_y=3
""",
            "call": """
(lambda z: np.concatenate((
z[0].ravel(),z[1].ravel(),z[2].ravel(),[float(z[3])]
)))(assemble_dual_mortar_projection(
length_x,length_y,n_cells_x,n_cells_y))
""",
            "gold_call": """
(lambda z: np.concatenate((
z[0].ravel(),z[1].ravel(),z[2].ravel(),[float(z[3])]
)))(_oracle_assemble_dual_mortar_projection(
length_x,length_y,n_cells_x,n_cells_y))
""",
            "tol": 5e-10,
        },
    ]
