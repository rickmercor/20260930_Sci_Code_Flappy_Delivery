"""
Evaluate the transverse-deflection polynomial basis used by the cited Bergan-Wang element together with the derivatives required by the equilibrium-reduced interpolation.



Return one array whose rows are ordered as



H,

H_x,

H_y,

H_xx,

H_xy,

H_yy,

H_xxx,

H_xxy,

H_xyy,

H_yyy,

H_xxxx,

H_xxxy,

H_xxyy,

H_xyyy,

H_yyyy.



Each row must preserve the source ordering of the twelve polynomial coefficients. Evaluate all derivatives analytically.

Higher-order polynomial interpolation can reproduce bending fields that cannot be represented by a simple bilinear displacement approximation.

Equilibrium-based plate formulations may require higher spatial derivatives of the transverse displacement. Evaluating these derivatives directly from a common polynomial basis avoids numerical differentiation and preserves exact polynomial consistency.

Returns
-------
A NumPy array with shape (15,12), using the derivative-row ordering specified in the step description.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bw_basis_derivatives(
    x: float,
    y: float,
) -> np.ndarray:
    """Return the ordered Bergan-Wang polynomial derivative table."""

    return np.zeros(
        (15, 12),
        dtype=float,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bw_basis_derivatives(
    x: float,
    y: float,
) -> np.ndarray:

    exponents = (
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
        (3, 1),
        (1, 3),
    )

    derivative_orders = (
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
        (4, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 4),
    )

    out = np.zeros(
        (15, 12),
        dtype=float,
    )

    for row, (dx, dy) in enumerate(
        derivative_orders
    ):

        for col, (px, py) in enumerate(
            exponents
        ):

            if (
                px < dx
                or py < dy
            ):
                continue

            coeff = 1.0

            for k in range(dx):
                coeff *= (
                    px - k
                )

            for k in range(dy):
                coeff *= (
                    py - k
                )

            out[row, col] = (
                coeff
                * x**(px - dx)
                * y**(py - dy)
            )

    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
x=0.0
y=0.0
""",
            "call": """
bw_basis_derivatives(x,y)
""",
            "gold_call": """
_oracle_bw_basis_derivatives(x,y)
""",
            "tol": 1e-12,
        },
        {
            "setup": """
x=1.2
y=-0.8
""",
            "call": """
bw_basis_derivatives(x,y)
""",
            "gold_call": """
_oracle_bw_basis_derivatives(x,y)
""",
            "tol": 1e-12,
        },
        {
            "setup": """
x=1.2
y=0.0
""",
            "call": """
bw_basis_derivatives(x,y)
""",
            "gold_call": """
_oracle_bw_basis_derivatives(x,y)
""",
            "tol": 1e-12,
        },
        {
            "setup": """
x=-0.37
y=0.29
""",
            "call": """
bw_basis_derivatives(x,y)
""",
            "gold_call": """
_oracle_bw_basis_derivatives(x,y)
""",
            "tol": 1e-12,
        },
    ]
