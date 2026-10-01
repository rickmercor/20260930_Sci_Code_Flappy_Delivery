"""
Construct the laminate stiffness and thermal-resultant data for an equal-thickness symmetric stacking sequence.

Transform each orthotropic ply to the global plate axes using the preceding step and integrate its contribution through the laminate thickness.

Return the extensional, extension-bending, bending, and transverse-shear stiffness matrices together with the thermal membrane and thermal moment resultants.

The stacking sequence is supplied from the bottom surface to the top surface. Do not round intermediate values.

Equivalent-single-layer laminate theory collects the contributions of individual plies into stiffness matrices defined over the plate mid-surface.

Different powers of the thickness coordinate determine extensional, coupling, and bending response. A symmetric stacking sequence eliminates extension-bending coupling in exact arithmetic. Thermal expansion produces analogous force and moment resultants.

Returns
-------
A tuple (A, B, D, As, NT, MT), with A, B, and D of shape (3,3), As of shape (2,2), and NT and MT of shape (3,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_symmetric_laminate(
    angles_deg: np.ndarray,
    ply_thickness: float,
    E1: float,
    E2: float,
    nu12: float,
    G12: float,
    G13: float,
    G23: float,
    alpha1: float,
    alpha2: float,
    delta_T: float,
    shear_correction: float = 5.0 / 6.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return A, B, D, As, NT, and MT."""

    return (
        np.zeros((3, 3), dtype=float),
        np.zeros((3, 3), dtype=float),
        np.zeros((3, 3), dtype=float),
        np.zeros((2, 2), dtype=float),
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_symmetric_laminate(
    angles_deg: np.ndarray,
    ply_thickness: float,
    E1: float,
    E2: float,
    nu12: float,
    G12: float,
    G13: float,
    G23: float,
    alpha1: float,
    alpha2: float,
    delta_T: float,
    shear_correction: float = 5.0 / 6.0,
):

    angles_deg = np.asarray(
        angles_deg,
        dtype=float,
    )

    n_ply = angles_deg.size

    if (
        angles_deg.ndim != 1
        or n_ply == 0
    ):
        raise ValueError(
            "angles_deg must be a nonempty vector"
        )

    if not np.array_equal(
        angles_deg,
        angles_deg[::-1],
    ):
        raise ValueError(
            "stacking sequence must be symmetric"
        )

    h = (
        n_ply
        * float(ply_thickness)
    )

    z = np.linspace(
        -0.5 * h,
        0.5 * h,
        n_ply + 1,
    )

    A = np.zeros(
        (3, 3),
        dtype=float,
    )

    B = np.zeros(
        (3, 3),
        dtype=float,
    )

    D = np.zeros(
        (3, 3),
        dtype=float,
    )

    As = np.zeros(
        (2, 2),
        dtype=float,
    )

    NT = np.zeros(
        3,
        dtype=float,
    )

    MT = np.zeros(
        3,
        dtype=float,
    )

    for k, angle in enumerate(
        angles_deg
    ):

        (
            Qbar_b,
            Qbar_s,
            alpha_bar,
        ) = _oracle_transform_orthotropic_ply(
            angle,
            E1,
            E2,
            nu12,
            G12,
            G13,
            G23,
            alpha1,
            alpha2,
        )

        z0 = z[k]
        z1 = z[k + 1]

        dz = z1 - z0

        A += (
            Qbar_b
            * dz
        )

        B += (
            0.5
            * Qbar_b
            * (
                z1**2
                - z0**2
            )
        )

        D += (
            Qbar_b
            * (
                z1**3
                - z0**3
            )
            / 3.0
        )

        As += (
            shear_correction
            * Qbar_s
            * dz
        )

        thermal = (
            Qbar_b
            @ alpha_bar
            * delta_T
        )

        NT += (
            thermal
            * dz
        )

        MT += (
            0.5
            * thermal
            * (
                z1**2
                - z0**2
            )
        )

    return (
        A,
        B,
        D,
        As,
        NT,
        MT,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
angles_deg=np.array([30.0,-30.0,-30.0,30.0])
ply_thickness=0.2
E1=135000.0
E2=9000.0
nu12=0.28
G12=5000.0
G13=4500.0
G23=3500.0
alpha1=-0.2e-6
alpha2=28e-6
delta_T=45.0
shear_correction=5.0/6.0
""",
            "call": """
build_symmetric_laminate(
angles_deg,ply_thickness,E1,E2,nu12,G12,G13,G23,
alpha1,alpha2,delta_T,shear_correction)
""",
            "gold_call": """
_oracle_build_symmetric_laminate(
angles_deg,ply_thickness,E1,E2,nu12,G12,G13,G23,
alpha1,alpha2,delta_T,shear_correction)
""",
            "tol": 5e-10,
        },
        {
            "setup": """
angles_deg=np.array([0.0,90.0,90.0,0.0])
ply_thickness=0.15
E1=120000.0
E2=8000.0
nu12=0.27
G12=4200.0
G13=3800.0
G23=3200.0
alpha1=-0.1e-6
alpha2=25e-6
delta_T=0.0
shear_correction=5.0/6.0
""",
            "call": """
build_symmetric_laminate(
angles_deg,ply_thickness,E1,E2,nu12,G12,G13,G23,
alpha1,alpha2,delta_T,shear_correction)
""",
            "gold_call": """
_oracle_build_symmetric_laminate(
angles_deg,ply_thickness,E1,E2,nu12,G12,G13,G23,
alpha1,alpha2,delta_T,shear_correction)
""",
            "tol": 5e-10,
        },
        {
            "setup": """
angles_deg=np.array([45.0,-45.0,-45.0,45.0])
ply_thickness=0.18
E1=145000.0
E2=10000.0
nu12=0.30
G12=5200.0
G13=4700.0
G23=3600.0
alpha1=-0.3e-6
alpha2=27e-6
delta_T=30.0
shear_correction=5.0/6.0
""",
            "call": """
build_symmetric_laminate(
angles_deg,ply_thickness,E1,E2,nu12,G12,G13,G23,
alpha1,alpha2,delta_T,shear_correction)
""",
            "gold_call": """
_oracle_build_symmetric_laminate(
angles_deg,ply_thickness,E1,E2,nu12,G12,G13,G23,
alpha1,alpha2,delta_T,shear_correction)
""",
            "tol": 5e-10,
        },
        {
            "setup": """
angles_deg=np.array([20.0,0.0,0.0,20.0])
ply_thickness=0.12
E1=95000.0
E2=7500.0
nu12=0.26
G12=3900.0
G13=3600.0
G23=3100.0
alpha1=0.1e-6
alpha2=22e-6
delta_T=-25.0
shear_correction=0.82
""",
            "call": """
build_symmetric_laminate(
angles_deg,ply_thickness,E1,E2,nu12,G12,G13,G23,
alpha1,alpha2,delta_T,shear_correction)
""",
            "gold_call": """
_oracle_build_symmetric_laminate(
angles_deg,ply_thickness,E1,E2,nu12,G12,G13,G23,
alpha1,alpha2,delta_T,shear_correction)
""",
            "tol": 5e-10,
        },
    ]
