"""
Evaluate the compressible Neo-Hookean constitutive model used in the nonlinear patch-test discussion of the cited paper.



The strain input uses engineering Voigt order



[E11, E22, E33, 2E12, 2E13, 2E23].



Construct the right Cauchy-Green tensor from the Green-Lagrange strain and evaluate the source strain-energy density, Second Piola-Kirchhoff stress, and exact consistent material tangent.



The returned 6x6 tangent must map an increment of the engineering Green-Lagrange strain vector to the corresponding increment of the stress vector



[S11, S22, S33, S12, S13, S23].

Do not approximate the tangent by finite differences.

Hyperelastic material models derive stress and tangent response from a strain-energy function. The Second Piola-Kirchhoff stress is work-conjugate to Green-Lagrange strain in a total-Lagrangian formulation.



A consistent material tangent is required by Newton-type nonlinear finite-element methods. Its Voigt representation must use a shear convention consistent with the strain-displacement matrix.

Returns
-------
A tuple (W, stress, D), where W is a native Python float, stress has shape (6,), and D has shape (6,6).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def neo_hooke_green_response(
    E_green: np.ndarray,
    young: float,
    poisson: float,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
]:
    """Return energy, Second Piola stress, and consistent tangent."""

    return (
        0.0,
        np.zeros(6, dtype=float),
        np.zeros((6, 6), dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_neo_hooke_green_response(
    E_green: np.ndarray,
    young: float,
    poisson: float,
):

    E_green = np.asarray(
        E_green,
        dtype=float,
    )

    if E_green.shape != (6,):
        raise ValueError(
            "E_green must have shape (6,)"
        )

    young = float(young)
    poisson = float(poisson)

    if (
        not np.isfinite(young)
        or young <= 0.0
        or not np.isfinite(poisson)
        or not (-1.0 < poisson < 0.5)
    ):
        raise ValueError(
            "invalid elastic constants"
        )

    mu = (
        young
        / (
            2.0
            * (
                1.0 + poisson
            )
        )
    )

    lame = (
        young
        * poisson
        / (
            (
                1.0 + poisson
            )
            * (
                1.0
                - 2.0 * poisson
            )
        )
    )

    E = np.array(
        [
            [
                E_green[0],
                0.5 * E_green[3],
                0.5 * E_green[4],
            ],
            [
                0.5 * E_green[3],
                E_green[1],
                0.5 * E_green[5],
            ],
            [
                0.5 * E_green[4],
                0.5 * E_green[5],
                E_green[2],
            ],
        ],
        dtype=float,
    )

    C = (
        np.eye(3)
        + 2.0 * E
    )

    detC = float(
        np.linalg.det(C)
    )

    if (
        not np.isfinite(detC)
        or detC <= 0.0
    ):
        raise ValueError(
            "C must be positive definite"
        )

    Cinv = np.linalg.inv(C)

    W = (
        0.5
        * mu
        * (
            np.trace(C)
            - 3.0
            - np.log(detC)
        )
        + 0.25
        * lame
        * (
            detC
            - 1.0
            - np.log(detC)
        )
    )

    alpha = (
        -mu
        + 0.5
        * lame
        * (
            detC
            - 1.0
        )
    )

    S = (
        mu * np.eye(3)
        + alpha * Cinv
    )

    stress = np.array(
        [
            S[0, 0],
            S[1, 1],
            S[2, 2],
            S[0, 1],
            S[0, 2],
            S[1, 2],
        ],
        dtype=float,
    )

    D = np.zeros(
        (6, 6),
        dtype=float,
    )

    for j in range(6):

        dC = np.zeros(
            (3, 3),
            dtype=float,
        )

        if j == 0:
            dC[0, 0] = 2.0

        elif j == 1:
            dC[1, 1] = 2.0

        elif j == 2:
            dC[2, 2] = 2.0

        elif j == 3:
            dC[0, 1] = 1.0
            dC[1, 0] = 1.0

        elif j == 4:
            dC[0, 2] = 1.0
            dC[2, 0] = 1.0

        else:
            dC[1, 2] = 1.0
            dC[2, 1] = 1.0

        ddetC = (
            detC
            * np.trace(
                Cinv @ dC
            )
        )

        dCinv = (
            -Cinv
            @ dC
            @ Cinv
        )

        dS = (
            0.5
            * lame
            * ddetC
            * Cinv
            + alpha
            * dCinv
        )

        D[:, j] = np.array(
            [
                dS[0, 0],
                dS[1, 1],
                dS[2, 2],
                dS[0, 1],
                dS[0, 2],
                dS[1, 2],
            ],
            dtype=float,
        )

    return (
        float(W),
        stress,
        D,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
E_green = np.array(
    [0.05,-0.02,0.03,0.01,-0.015,0.02],
    dtype=float,
)
young = 1000.0
poisson = 0.30
""",
            "call": """
neo_hooke_green_response(E_green, young, poisson)
""",
            "gold_call": """
_oracle_neo_hooke_green_response(E_green, young, poisson)
""",
            "tol": 5e-11,
        },
        {
            "setup": """
E_green = np.zeros(6, dtype=float)
young = 800.0
poisson = 0.25
""",
            "call": """
neo_hooke_green_response(E_green, young, poisson)
""",
            "gold_call": """
_oracle_neo_hooke_green_response(E_green, young, poisson)
""",
            "tol": 5e-11,
        },
        {
            "setup": """
E_green = np.array(
    [0.15,0.05,-0.02,0.04,0.01,-0.03],
    dtype=float,
)
young = 1200.0
poisson = 0.35
""",
            "call": """
neo_hooke_green_response(E_green, young, poisson)
""",
            "gold_call": """
_oracle_neo_hooke_green_response(E_green, young, poisson)
""",
            "tol": 5e-11,
        },
        {
            "setup": """
E_green = np.array(
    [-0.08,0.12,0.07,0.02,-0.01,0.04],
    dtype=float,
)
young = 600.0
poisson = 0.45
""",
            "call": """
neo_hooke_green_response(E_green, young, poisson)
""",
            "gold_call": """
_oracle_neo_hooke_green_response(E_green, young, poisson)
""",
            "tol": 5e-11,
        },
    ]
