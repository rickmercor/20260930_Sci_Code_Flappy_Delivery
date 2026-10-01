"""
Determine the independent strain and stress parameter vectors associated with a prescribed HWT10 displacement state.



Use the source-defined element integration rule, mixed interpolation, constitutive response, and local stationarity equations of the cited formulation.



The returned internal fields must satisfy both local Hu-Washizu stationarity conditions consistently at the prescribed displacement state.



Return the source coupling matrix together with the complete independent strain and stress parameter vectors. Do not round intermediate values.

The stress and strain variables of a Hu-Washizu element are local element variables. For a prescribed displacement state, stationarity with respect to these fields determines their element-level parameter values.



Because these internal variables are discontinuous between elements, their equations can be solved locally. This local projection is the first stage of static condensation and avoids introducing additional global degrees of freedom.

Returns
-------
A tuple (L, epsilon_hat, sigma_hat), where L has shape (24,24) and each parameter vector has shape (24,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def project_hwt10_internal_fields(
    coords: np.ndarray,
    disp: np.ndarray,
    young: float,
    poisson: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return L, independent strain parameters, and stress parameters."""

    return (
        np.zeros((24, 24), dtype=float),
        np.zeros(24, dtype=float),
        np.zeros(24, dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_project_hwt10_internal_fields(
    coords: np.ndarray,
    disp: np.ndarray,
    young: float,
    poisson: float,
):

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
            1.0/360.0,
            1.0/360.0,
            1.0/360.0,
            1.0/360.0,
            4.0/360.0,
            4.0/360.0,
            4.0/360.0,
            4.0/360.0,
            4.0/360.0,
            4.0/360.0,
            32.0/360.0,
        ],
        dtype=float,
    )

    L = np.zeros(
        (24,24),
        dtype=float,
    )

    a = np.zeros(
        24,
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

        dv = (
            wi * detJ
        )

        L += (
            -dv
            * (
                N_epsilon.T
                @ N_sigma
            )
        )

        a += (
            dv
            * (
                N_sigma.T
                @ E_green
            )
        )

    epsilon_hat = np.linalg.solve(
        L.T,
        -a,
    )

    b = np.zeros(
        24,
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

        b += (
            dv
            * (
                N_epsilon.T
                @ stress
            )
        )

    sigma_hat = np.linalg.solve(
        L,
        -b,
    )

    return (
        L,
        epsilon_hat,
        sigma_hat,
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
project_hwt10_internal_fields(
    coords,
    disp,
    young,
    poisson,
)
""",
            "gold_call": """
_oracle_project_hwt10_internal_fields(
    coords,
    disp,
    young,
    poisson,
)
""",
            "tol": 5e-10,
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
project_hwt10_internal_fields(
    coords,
    disp,
    young,
    poisson,
)
""",
            "gold_call": """
_oracle_project_hwt10_internal_fields(
    coords,
    disp,
    young,
    poisson,
)
""",
            "tol": 5e-10,
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
project_hwt10_internal_fields(
    coords,
    disp,
    young,
    poisson,
)
""",
            "gold_call": """
_oracle_project_hwt10_internal_fields(
    coords,
    disp,
    young,
    poisson,
)
""",
            "tol": 5e-10,
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
project_hwt10_internal_fields(
    coords,
    disp,
    young,
    poisson,
)
""",
            "gold_call": """
_oracle_project_hwt10_internal_fields(
    coords,
    disp,
    young,
    poisson,
)
""",
            "tol": 5e-10,
        },
    ]
