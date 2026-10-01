"""
Assemble the nonlinear condensed tangent stiffness of the cited HWT10 element at a prescribed displacement state.



Use the previously determined independent strain and stress fields together with the source-defined element equations and integration rule.



Construct the local mixed-field tangent consistently and eliminate the element-internal variables according to the cited formulation.



Return the complete condensed displacement tangent together with its material-condensation and geometric contributions. Do not round intermediate values.

The tangent matrix of a finite-deformation mixed element contains both constitutive and geometric effects.

The internal Hu-Washizu stress and strain variables are element-local. Condensing their increments produces a tangent involving only nodal displacement degrees of freedom, so the resulting element can be assembled into a conventional global structural system.

Returns
-------
A tuple (K_tangent, K_material, K_geometric), each with shape (30,30).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_hwt10_condensed_tangent(
    coords: np.ndarray,
    disp: np.ndarray,
    young: float,
    poisson: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return condensed, material, and geometric HWT10 tangents."""

    return (
        np.zeros((30, 30), dtype=float),
        np.zeros((30, 30), dtype=float),
        np.zeros((30, 30), dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_hwt10_condensed_tangent(
    coords: np.ndarray,
    disp: np.ndarray,
    young: float,
    poisson: float,
):

    (
        L,
        epsilon_hat,
        sigma_hat,
    ) = _oracle_project_hwt10_internal_fields(
        coords,
        disp,
        young,
        poisson,
    )

    q = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.25,0.25,0.25],
        ],
        dtype=float,
    )

    w = np.array(
        [
            1/360,1/360,1/360,1/360,
            4/360,4/360,4/360,
            4/360,4/360,4/360,
            32/360,
        ],
        dtype=float,
    )

    G = np.zeros(
        (24,30),
        dtype=float,
    )

    H = np.zeros(
        (24,24),
        dtype=float,
    )

    K_geometric = np.zeros(
        (30,30),
        dtype=float,
    )

    for xi, wi in zip(q, w):

        (
            detJ,
            gradN,
            F,
            E_green,
            B,
        ) = _oracle_tet10_kinematics(
            coords,
            disp,
            xi,
        )

        (
            N_sigma,
            N_epsilon,
        ) = _oracle_hwt10_mixed_interpolation(
            xi
        )

        physical_strain = (
            N_epsilon
            @ epsilon_hat
        )

        (
            W,
            stress,
            D,
        ) = _oracle_neo_hooke_green_response(
            physical_strain,
            young,
            poisson,
        )

        dv = (
            wi * detJ
        )

        G += (
            dv
            * (
                N_sigma.T
                @ B
            )
        )

        H += (
            dv
            * (
                N_epsilon.T
                @ D
                @ N_epsilon
            )
        )

        stress_h = (
            N_sigma
            @ sigma_hat
        )

        for i in range(10):

            gi = gradN[i]

            for k in range(10):

                gk = gradN[k]

                N_ik = np.array(
                    [
                        gi[0]*gk[0],
                        gi[1]*gk[1],
                        gi[2]*gk[2],
                        gi[0]*gk[1]
                        + gi[1]*gk[0],
                        gi[0]*gk[2]
                        + gi[2]*gk[0],
                        gi[1]*gk[2]
                        + gi[2]*gk[1],
                    ],
                    dtype=float,
                )

                scalar = float(
                    N_ik
                    @ stress_h
                )

                K_geometric[
                    3*i:3*i+3,
                    3*k:3*k+3,
                ] += (
                    dv
                    * scalar
                    * np.eye(3)
                )

    left = np.linalg.solve(
        L,
        H,
    )

    H_hat = np.linalg.solve(
        L,
        left.T,
    ).T

    K_material = (
        G.T
        @ H_hat
        @ G
    )

    K_tangent = (
        K_material
        + K_geometric
    )

    return (
        K_tangent,
        K_material,
        K_geometric,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
coords = np.array([
    [0,0,0],
    [1.2,.1,0],
    [.1,1.1,.05],
    [0,.15,.95],
    [.62,.02,.03],
    [.68,.62,-.02],
    [.02,.58,.04],
    [-.03,.08,.50],
    [.58,.15,.50],
    [.06,.62,.52],
], dtype=float)

X = coords[:,0]
Y = coords[:,1]
Z = coords[:,2]

disp = np.column_stack([
    .08*X + .025*Y + .015*X*Z,
   -.035*Y + .020*Z + .010*X*Y,
    .060*Z - .018*X + .012*Y*Z,
])

young = 1000.0
poisson = 0.30
""",
            "call": """
assemble_hwt10_condensed_tangent(
    coords,
    disp,
    young,
    poisson,
)
""",
            "gold_call": """
_oracle_assemble_hwt10_condensed_tangent(
    coords,
    disp,
    young,
    poisson,
)
""",
            "tol": 5e-9,
        },
        {
            "setup": """
coords = np.array([
    [0,0,0],
    [1,.08,.02],
    [.04,.92,.10],
    [.06,.02,1.05],
    [.5,.04,.01],
    [.52,.5,.06],
    [.02,.46,.05],
    [.03,.01,.525],
    [.53,.05,.535],
    [.05,.47,.575],
], dtype=float)

X = coords[:,0]
Y = coords[:,1]
Z = coords[:,2]

disp = 0.55*np.column_stack([
    .08*X + .025*Y + .015*X*Z,
   -.035*Y + .020*Z + .010*X*Y,
    .060*Z - .018*X + .012*Y*Z,
])

young = 800.0
poisson = 0.25
""",
            "call": """
assemble_hwt10_condensed_tangent(
    coords,
    disp,
    young,
    poisson,
)
""",
            "gold_call": """
_oracle_assemble_hwt10_condensed_tangent(
    coords,
    disp,
    young,
    poisson,
)
""",
            "tol": 5e-9,
        },
        {
            "setup": """
coords = np.array([
    [0,0,0],
    [.85,-.05,.03],
    [.12,1.15,-.02],
    [-.04,.1,.88],
    [.445,-.04,.025],
    [.475,.57,-.01],
    [.075,.585,-.005],
    [-.03,.065,.46],
    [.425,.015,.45],
    [.025,.63,.445],
], dtype=float)

X = coords[:,0]
Y = coords[:,1]
Z = coords[:,2]

disp = 0.70*np.column_stack([
    .08*X + .025*Y + .015*X*Z,
   -.035*Y + .020*Z + .010*X*Y,
    .060*Z - .018*X + .012*Y*Z,
])

young = 1500.0
poisson = 0.32
""",
            "call": """
assemble_hwt10_condensed_tangent(
    coords,
    disp,
    young,
    poisson,
)
""",
            "gold_call": """
_oracle_assemble_hwt10_condensed_tangent(
    coords,
    disp,
    young,
    poisson,
)
""",
            "tol": 5e-9,
        },
        {
            "setup": """
coords = np.array([
    [0,0,0],
    [1.2,.1,0],
    [.1,1.1,.05],
    [0,.15,.95],
    [.62,.02,.03],
    [.68,.62,-.02],
    [.02,.58,.04],
    [-.03,.08,.50],
    [.58,.15,.50],
    [.06,.62,.52],
], dtype=float)

disp = np.zeros_like(coords)

young = 600.0
poisson = 0.45
""",
            "call": """
assemble_hwt10_condensed_tangent(
    coords,
    disp,
    young,
    poisson,
)
""",
            "gold_call": """
_oracle_assemble_hwt10_condensed_tangent(
    coords,
    disp,
    young,
    poisson,
)
""",
            "tol": 5e-9,
        },
    ]
