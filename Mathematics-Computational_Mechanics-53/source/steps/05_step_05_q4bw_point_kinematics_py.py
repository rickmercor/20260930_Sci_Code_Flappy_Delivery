"""
Evaluate the nonlinear plate kinematics at one point of the Q4BW element.



Use the source bending interpolation from the preceding step and standard bilinear interpolation for the in-plane nodal displacements.

The generalized displacement vector is ordered as twelve bending parameters followed by the eight node-wise in-plane parameters.

Evaluate the von Kármán membrane strain, bending curvature, and transverse-shear strain together with their first-variation operators with respect to the complete twenty-component generalized displacement vector.

Do not round intermediate quantities.

Geometrically nonlinear plate kinematics couple transverse displacement gradients to membrane deformation.

The first variation of the strain field is required for both the internal residual and the consistent tangent. Bending and shear terms remain linear in the generalized bending variables for a fixed interpolation, whereas the membrane strain contains nonlinear displacement-gradient contributions.

Returns
-------
A tuple (epsilon0, kappa, gamma, Bepsilon, Bkappa_full, Bgamma_full).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def q4bw_point_kinematics(
    a: float,
    b: float,
    D: np.ndarray,
    As: np.ndarray,
    q: np.ndarray,
    x: float,
    y: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return strains and first-variation operators."""

    return (
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
        np.zeros(2, dtype=float),
        np.zeros((3, 20), dtype=float),
        np.zeros((3, 20), dtype=float),
        np.zeros((2, 20), dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_q4bw_point_kinematics(
    a: float,
    b: float,
    D: np.ndarray,
    As: np.ndarray,
    q: np.ndarray,
    x: float,
    y: float,
):

    q = np.asarray(
        q,
        dtype=float,
    )

    wb = q[:12]
    wm = q[12:]

    (
        Nw,
        Nwx,
        Nwy,
        Nphiy,
        Nphix,
        Bkappa,
        Bgamma,
    ) = _oracle_q4bw_bending_shapes(
        a,
        b,
        D,
        As,
        x,
        y,
    )

    xi = (
        2.0 * x / a
    )

    eta = (
        2.0 * y / b
    )

    dN_dxi = np.array(
        [
            -0.25*(1.0-eta),
             0.25*(1.0-eta),
             0.25*(1.0+eta),
            -0.25*(1.0+eta),
        ],
        dtype=float,
    )

    dN_deta = np.array(
        [
            -0.25*(1.0-xi),
            -0.25*(1.0+xi),
             0.25*(1.0+xi),
             0.25*(1.0-xi),
        ],
        dtype=float,
    )

    dNdx = (
        2.0
        * dN_dxi
        / a
    )

    dNdy = (
        2.0
        * dN_deta
        / b
    )

    Bm = np.zeros(
        (3, 8),
        dtype=float,
    )

    for node in range(4):

        Bm[
            0,
            2*node,
        ] = dNdx[node]

        Bm[
            1,
            2*node + 1,
        ] = dNdy[node]

        Bm[
            2,
            2*node,
        ] = dNdy[node]

        Bm[
            2,
            2*node + 1,
        ] = dNdx[node]

    wx = float(
        Nwx @ wb
    )

    wy = float(
        Nwy @ wb
    )

    epsilon_linear = (
        Bm @ wm
    )

    epsilon_nonlinear = np.array(
        [
            0.5 * wx * wx,
            0.5 * wy * wy,
            wx * wy,
        ],
        dtype=float,
    )

    epsilon0 = (
        epsilon_linear
        + epsilon_nonlinear
    )

    kappa = (
        Bkappa @ wb
    )

    gamma = (
        Bgamma @ wb
    )

    Bepsilon = np.zeros(
        (3, 20),
        dtype=float,
    )

    Bepsilon[
        :,
        12:,
    ] = Bm

    Bepsilon[
        0,
        :12,
    ] = wx * Nwx

    Bepsilon[
        1,
        :12,
    ] = wy * Nwy

    Bepsilon[
        2,
        :12,
    ] = (
        wy * Nwx
        + wx * Nwy
    )

    Bkappa_full = np.zeros(
        (3, 20),
        dtype=float,
    )

    Bkappa_full[
        :,
        :12,
    ] = Bkappa

    Bgamma_full = np.zeros(
        (2, 20),
        dtype=float,
    )

    Bgamma_full[
        :,
        :12,
    ] = Bgamma

    return (
        epsilon0,
        kappa,
        gamma,
        Bepsilon,
        Bkappa_full,
        Bgamma_full,
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
[1292.1589332648687,462.9215011685319,1170.8542426192012]])
As=np.array([[2833.333333333334,0.0],[0.0,2500.0]])
q=np.array([
.018,.010,-.006,.027,.006,-.004,.041,-.003,.009,.022,-.005,.007,
.0015,-.0008,.0024,-.0011,.0032,.0017,.0011,.0013])
x=0.0
y=0.0
""",
            "call": """
q4bw_point_kinematics(a,b,D,As,q,x,y)
""",
            "gold_call": """
_oracle_q4bw_point_kinematics(a,b,D,As,q,x,y)
""",
            "tol": 5e-11,
        },
        {
            "setup": """
a=2.4
b=1.6
D=np.array([
[3481.6813210379605,1065.6058331546217,1292.1589332648687],
[1065.6058331546217,779.5582243191081,462.9215011685319],
[1292.1589332648687,462.9215011685319,1170.8542426192012]])
As=np.array([[2833.333333333334,0.0],[0.0,2500.0]])
q=np.zeros(20)
x=-1.2
y=-0.8
""",
            "call": """
q4bw_point_kinematics(a,b,D,As,q,x,y)
""",
            "gold_call": """
_oracle_q4bw_point_kinematics(a,b,D,As,q,x,y)
""",
            "tol": 5e-11,
        },
        {
            "setup": """
a=2.4
b=1.6
D=np.array([
[3481.6813210379605,1065.6058331546217,1292.1589332648687],
[1065.6058331546217,779.5582243191081,462.9215011685319],
[1292.1589332648687,462.9215011685319,1170.8542426192012]])
As=np.array([[2833.333333333334,0.0],[0.0,2500.0]])
q=np.array([
.010,.004,-.003,.014,.003,-.002,.022,-.002,.005,.012,-.003,.004,
.0008,-.0004,.0012,-.0006,.0018,.0009,.0007,.0008])
x=1.2
y=0.0
""",
            "call": """
q4bw_point_kinematics(a,b,D,As,q,x,y)
""",
            "gold_call": """
_oracle_q4bw_point_kinematics(a,b,D,As,q,x,y)
""",
            "tol": 5e-11,
        },
        {
            "setup": """
a=2.4
b=1.6
D=np.array([
[3481.6813210379605,1065.6058331546217,1292.1589332648687],
[1065.6058331546217,779.5582243191081,462.9215011685319],
[1292.1589332648687,462.9215011685319,1170.8542426192012]])
As=np.array([[2833.333333333334,0.0],[0.0,2500.0]])
q=np.array([
.013,.006,-.004,.020,.004,-.003,.031,-.003,.007,.016,-.004,.006,
.0010,-.0005,.0017,-.0008,.0023,.0012,.0009,.0010])
x=-0.43
y=0.31
""",
            "call": """
q4bw_point_kinematics(a,b,D,As,q,x,y)
""",
            "gold_call": """
_oracle_q4bw_point_kinematics(a,b,D,As,q,x,y)
""",
            "tol": 5e-11,
        },
    ]
