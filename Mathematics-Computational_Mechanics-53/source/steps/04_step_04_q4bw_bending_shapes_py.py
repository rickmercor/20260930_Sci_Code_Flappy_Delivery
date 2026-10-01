"""
Construct the source Q4BW bending interpolation at one point of a rectangular element centered at the origin.



Use the preceding polynomial derivative table, the supplied laminate bending and transverse-shear stiffness matrices, and the equilibrium-reduced rotation interpolation of the cited method.

Build the twelve-by-twelve nodal basis using the four element nodes in the prescribed counter-clockwise order.

Return the transverse displacement interpolation, its x and y derivatives, the two rotation interpolations, the curvature operator, and the transverse-shear operator.

Use linear solves for the nodal interpolation transformation. Do not round intermediate values.

In a plate-bending finite element, the interpolation of transverse displacement determines how bending deformation is represented inside an element.

Equilibrium-based formulations can reconstruct rotational and transverse-shear quantities from the bending field and laminate stiffness rather than relying on a conventional independent low-order rotation interpolation. The resulting interpolation must still reproduce the prescribed nodal bending parameters exactly.

Returns
-------
A tuple (Nw, Nwx, Nwy, Nphiy, Nphix, Bkappa, Bgamma). The first five arrays have shape (12,), Bkappa has shape (3,12), and Bgamma has shape (2,12).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def q4bw_bending_shapes(
    a: float,
    b: float,
    D: np.ndarray,
    As: np.ndarray,
    x: float,
    y: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return Q4BW displacement, rotation, curvature, and shear operators."""

    return (
        np.zeros(12, dtype=float),
        np.zeros(12, dtype=float),
        np.zeros(12, dtype=float),
        np.zeros(12, dtype=float),
        np.zeros(12, dtype=float),
        np.zeros((3, 12), dtype=float),
        np.zeros((2, 12), dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_q4bw_bending_shapes(
    a: float,
    b: float,
    D: np.ndarray,
    As: np.ndarray,
    x: float,
    y: float,
):

    D = np.asarray(
        D,
        dtype=float,
    )

    As = np.asarray(
        As,
        dtype=float,
    )

    As_inv = np.linalg.solve(
        As,
        np.eye(2),
    )

    def reduced_rows(
        data: np.ndarray,
    ):

        (
            H,
            Hx,
            Hy,
            Hxx,
            Hxy,
            Hyy,
            Hxxx,
            Hxxy,
            Hxyy,
            Hyyy,
            Hxxxx,
            Hxxxy,
            Hxxyy,
            Hxyyy,
            Hyyyy,
        ) = data

        M1 = (
            D[0]
            @ np.vstack(
                (
                    Hxxx,
                    Hxyy,
                    2.0 * Hxxy,
                )
            )
            +
            D[2]
            @ np.vstack(
                (
                    Hxxy,
                    Hyyy,
                    2.0 * Hxyy,
                )
            )
        )

        M2 = (
            D[1]
            @ np.vstack(
                (
                    Hxxy,
                    Hyyy,
                    2.0 * Hxyy,
                )
            )
            +
            D[2]
            @ np.vstack(
                (
                    Hxxx,
                    Hxyy,
                    2.0 * Hxxy,
                )
            )
        )

        M1x = (
            D[0]
            @ np.vstack(
                (
                    Hxxxx,
                    Hxxyy,
                    2.0 * Hxxxy,
                )
            )
            +
            D[2]
            @ np.vstack(
                (
                    Hxxxy,
                    Hxyyy,
                    2.0 * Hxxyy,
                )
            )
        )

        M2x = (
            D[1]
            @ np.vstack(
                (
                    Hxxxy,
                    Hxyyy,
                    2.0 * Hxxyy,
                )
            )
            +
            D[2]
            @ np.vstack(
                (
                    Hxxxx,
                    Hxxyy,
                    2.0 * Hxxxy,
                )
            )
        )

        M1y = (
            D[0]
            @ np.vstack(
                (
                    Hxxxy,
                    Hxyyy,
                    2.0 * Hxxyy,
                )
            )
            +
            D[2]
            @ np.vstack(
                (
                    Hxxyy,
                    Hyyyy,
                    2.0 * Hxyyy,
                )
            )
        )

        M2y = (
            D[1]
            @ np.vstack(
                (
                    Hxxyy,
                    Hyyyy,
                    2.0 * Hxyyy,
                )
            )
            +
            D[2]
            @ np.vstack(
                (
                    Hxxxy,
                    Hxyyy,
                    2.0 * Hxxyy,
                )
            )
        )

        Hphiy = (
            -Hx
            - As_inv[0, 0] * M1
            - As_inv[0, 1] * M2
        )

        Hphix = (
            Hy
            + As_inv[1, 0] * M1
            + As_inv[1, 1] * M2
        )

        Hphiy_x = (
            -Hxx
            - As_inv[0, 0] * M1x
            - As_inv[0, 1] * M2x
        )

        Hphiy_y = (
            -Hxy
            - As_inv[0, 0] * M1y
            - As_inv[0, 1] * M2y
        )

        Hphix_x = (
            Hxy
            + As_inv[1, 0] * M1x
            + As_inv[1, 1] * M2x
        )

        Hphix_y = (
            Hyy
            + As_inv[1, 0] * M1y
            + As_inv[1, 1] * M2y
        )

        return (
            Hphiy,
            Hphix,
            Hphiy_x,
            Hphiy_y,
            Hphix_x,
            Hphix_y,
        )

    nodes = np.array(
        [
            [-0.5*a, -0.5*b],
            [ 0.5*a, -0.5*b],
            [ 0.5*a,  0.5*b],
            [-0.5*a,  0.5*b],
        ],
        dtype=float,
    )

    H_i = np.zeros(
        (12, 12),
        dtype=float,
    )

    for node, (
        xn,
        yn,
    ) in enumerate(nodes):

        data_n = (
            _oracle_bw_basis_derivatives(
                xn,
                yn,
            )
        )

        (
            Hphiy_n,
            Hphix_n,
            _,
            _,
            _,
            _,
        ) = reduced_rows(
            data_n
        )

        H_i[
            3*node
        ] = data_n[0]

        H_i[
            3*node + 1
        ] = Hphiy_n

        H_i[
            3*node + 2
        ] = Hphix_n

    data = (
        _oracle_bw_basis_derivatives(
            x,
            y,
        )
    )

    (
        Hphiy,
        Hphix,
        Hphiy_x,
        Hphiy_y,
        Hphix_x,
        Hphix_y,
    ) = reduced_rows(data)

    def interpolate(
        row: np.ndarray,
    ) -> np.ndarray:

        return np.linalg.solve(
            H_i.T,
            row,
        )

    Nw = interpolate(
        data[0]
    )

    Nwx = interpolate(
        data[1]
    )

    Nwy = interpolate(
        data[2]
    )

    Nphiy = interpolate(
        Hphiy
    )

    Nphix = interpolate(
        Hphix
    )

    Nphiy_x = interpolate(
        Hphiy_x
    )

    Nphiy_y = interpolate(
        Hphiy_y
    )

    Nphix_x = interpolate(
        Hphix_x
    )

    Nphix_y = interpolate(
        Hphix_y
    )

    Bkappa = np.vstack(
        (
            Nphiy_x,
            -Nphix_y,
            Nphiy_y
            - Nphix_x,
        )
    )

    Bgamma = np.vstack(
        (
            Nwx + Nphiy,
            Nwy - Nphix,
        )
    )

    return (
        Nw,
        Nwx,
        Nwy,
        Nphiy,
        Nphix,
        Bkappa,
        Bgamma,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
a=2.4
b=1.6
D=np.array([
[3481.6813210379605,1065.6058331546217,1292.1589332648687],
[1065.6058331546217,779.5582243191081,462.9215011685319],
[1292.1589332648687,462.9215011685319,1170.8542426192012],
])
As=np.array([[2833.333333333334,0.0],[0.0,2500.0]])
x=0.0
y=0.0
""",
            "call": """
q4bw_bending_shapes(a,b,D,As,x,y)
""",
            "gold_call": """
_oracle_q4bw_bending_shapes(a,b,D,As,x,y)
""",
            "tol": 5e-11,
        },
        {
            "setup": """
a=1.8
b=1.1
D=np.array([[200.,40.,15.],[40.,130.,-8.],[15.,-8.,60.]])
As=np.array([[1200.,90.],[90.,950.]])
x=-0.9
y=-0.55
""",
            "call": """
q4bw_bending_shapes(a,b,D,As,x,y)
""",
            "gold_call": """
_oracle_q4bw_bending_shapes(a,b,D,As,x,y)
""",
            "tol": 5e-11,
        },
        {
            "setup": """
a=2.0
b=1.4
D=np.array([[350.,75.,-20.],[75.,220.,35.],[-20.,35.,100.]])
As=np.array([[800.,-50.],[-50.,1050.]])
x=1.0
y=0.0
""",
            "call": """
q4bw_bending_shapes(a,b,D,As,x,y)
""",
            "gold_call": """
_oracle_q4bw_bending_shapes(a,b,D,As,x,y)
""",
            "tol": 5e-11,
        },
        {
            "setup": """
a=1.5
b=2.2
D=np.array([[180.,30.,10.],[30.,260.,-25.],[10.,-25.,85.]])
As=np.array([[1500.,120.],[120.,1100.]])
x=-0.75
y=1.10
""",
            "call": """
q4bw_bending_shapes(a,b,D,As,x,y)
""",
            "gold_call": """
_oracle_q4bw_bending_shapes(a,b,D,As,x,y)
""",
            "tol": 5e-11,
        },
    ]
