"""
Transform the reduced orthotropic stiffness and thermal-expansion data of one lamina from its material axes to the global plate axes.

Use the engineering-shear convention for the in-plane strain vector and preserve the sign of the angle-dependent coupling terms.

Return the transformed in-plane reduced stiffness, transformed transverse-shear stiffness, and transformed thermal-expansion vector.

Angles are supplied in degrees. Do not round intermediate values.

The elastic response of an orthotropic lamina is simplest when expressed in its material principal directions. A laminate generally contains plies rotated relative to the structural axes, so each lamina contribution must first be transformed to a common coordinate system.

The transformed constitutive data determine both the direct stiffness components and the coupling between normal and shear response. Thermal expansion must be transformed using the same material orientation.

Returns
-------
A tuple (Qbar_b, Qbar_s, alpha_bar), where Qbar_b has shape (3,3), Qbar_s has shape (2,2), and alpha_bar has shape (3,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def transform_orthotropic_ply(
    theta_deg: float,
    E1: float,
    E2: float,
    nu12: float,
    G12: float,
    G13: float,
    G23: float,
    alpha1: float,
    alpha2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return transformed in-plane stiffness, shear stiffness, and CTE."""

    return (
        np.zeros((3, 3), dtype=float),
        np.zeros((2, 2), dtype=float),
        np.zeros(3, dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_transform_orthotropic_ply(
    theta_deg: float,
    E1: float,
    E2: float,
    nu12: float,
    G12: float,
    G13: float,
    G23: float,
    alpha1: float,
    alpha2: float,
):

    nu21 = nu12 * E2 / E1

    den = (
        1.0
        - nu12 * nu21
    )

    Q11 = E1 / den
    Q22 = E2 / den
    Q12 = nu12 * E2 / den
    Q66 = G12

    theta = np.deg2rad(
        float(theta_deg)
    )

    m = np.cos(theta)
    n = np.sin(theta)

    m2 = m * m
    n2 = n * n

    m4 = m2 * m2
    n4 = n2 * n2

    Qbar_b = np.zeros(
        (3, 3),
        dtype=float,
    )

    Qbar_b[0, 0] = (
        Q11 * m4
        + Q22 * n4
        + 2.0
        * (
            Q12
            + 2.0 * Q66
        )
        * m2 * n2
    )

    Qbar_b[1, 1] = (
        Q11 * n4
        + Q22 * m4
        + 2.0
        * (
            Q12
            + 2.0 * Q66
        )
        * m2 * n2
    )

    Qbar_b[0, 1] = (
        (
            Q11
            + Q22
            - 4.0 * Q66
        )
        * m2 * n2
        + Q12
        * (
            m4 + n4
        )
    )

    Qbar_b[1, 0] = (
        Qbar_b[0, 1]
    )

    Qbar_b[0, 2] = (
        (
            Q11
            - Q12
            - 2.0 * Q66
        )
        * m**3 * n
        -
        (
            Q22
            - Q12
            - 2.0 * Q66
        )
        * m * n**3
    )

    Qbar_b[2, 0] = (
        Qbar_b[0, 2]
    )

    Qbar_b[1, 2] = (
        (
            Q11
            - Q12
            - 2.0 * Q66
        )
        * m * n**3
        -
        (
            Q22
            - Q12
            - 2.0 * Q66
        )
        * m**3 * n
    )

    Qbar_b[2, 1] = (
        Qbar_b[1, 2]
    )

    Qbar_b[2, 2] = (
        (
            Q11
            + Q22
            - 2.0 * Q12
            - 2.0 * Q66
        )
        * m2 * n2
        + Q66
        * (
            m4 + n4
        )
    )

    Qbar_s = np.array(
        [
            [
                G13 * m2
                + G23 * n2,
                (
                    G13 - G23
                )
                * m * n,
            ],
            [
                (
                    G13 - G23
                )
                * m * n,
                G13 * n2
                + G23 * m2,
            ],
        ],
        dtype=float,
    )

    alpha_bar = np.array(
        [
            alpha1 * m2
            + alpha2 * n2,
            alpha1 * n2
            + alpha2 * m2,
            2.0
            * (
                alpha1 - alpha2
            )
            * m * n,
        ],
        dtype=float,
    )

    return (
        Qbar_b,
        Qbar_s,
        alpha_bar,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
theta_deg = 30.0
E1 = 135000.0
E2 = 9000.0
nu12 = 0.28
G12 = 5000.0
G13 = 4500.0
G23 = 3500.0
alpha1 = -0.2e-6
alpha2 = 28.0e-6
""",
            "call": """
transform_orthotropic_ply(
    theta_deg,E1,E2,nu12,G12,G13,G23,alpha1,alpha2
)
""",
            "gold_call": """
_oracle_transform_orthotropic_ply(
    theta_deg,E1,E2,nu12,G12,G13,G23,alpha1,alpha2
)
""",
            "tol": 5e-11,
        },
        {
            "setup": """
theta_deg = 0.0
E1 = 120000.0
E2 = 8000.0
nu12 = 0.27
G12 = 4200.0
G13 = 3800.0
G23 = 3200.0
alpha1 = -0.1e-6
alpha2 = 25.0e-6
""",
            "call": """
transform_orthotropic_ply(
    theta_deg,E1,E2,nu12,G12,G13,G23,alpha1,alpha2
)
""",
            "gold_call": """
_oracle_transform_orthotropic_ply(
    theta_deg,E1,E2,nu12,G12,G13,G23,alpha1,alpha2
)
""",
            "tol": 5e-11,
        },
        {
            "setup": """
theta_deg = 90.0
E1 = 120000.0
E2 = 8000.0
nu12 = 0.27
G12 = 4200.0
G13 = 3800.0
G23 = 3200.0
alpha1 = -0.1e-6
alpha2 = 25.0e-6
""",
            "call": """
transform_orthotropic_ply(
    theta_deg,E1,E2,nu12,G12,G13,G23,alpha1,alpha2
)
""",
            "gold_call": """
_oracle_transform_orthotropic_ply(
    theta_deg,E1,E2,nu12,G12,G13,G23,alpha1,alpha2
)
""",
            "tol": 5e-11,
        },
        {
            "setup": """
theta_deg = -45.0
E1 = 145000.0
E2 = 10000.0
nu12 = 0.30
G12 = 5200.0
G13 = 4700.0
G23 = 3600.0
alpha1 = -0.3e-6
alpha2 = 27.0e-6
""",
            "call": """
transform_orthotropic_ply(
    theta_deg,E1,E2,nu12,G12,G13,G23,alpha1,alpha2
)
""",
            "gold_call": """
_oracle_transform_orthotropic_ply(
    theta_deg,E1,E2,nu12,G12,G13,G23,alpha1,alpha2
)
""",
            "tol": 5e-11,
        },
    ]
