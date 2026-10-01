"""
Construct the independent stress and strain interpolation matrices of the cited HWT10 element at one natural-coordinate point.



Follow the exact parameterization used by the paper, including the constant and varying parts of each six-component field.



Preserve the source ordering of the 24 parameters. Return the complete stress and strain interpolation matrices without rounding intermediate values.

Mixed Hu-Washizu elements use independent stress and strain fields in addition to the displacement interpolation.

The choice of interpolation for these internal fields is central to the behavior of the element. The parameterization determines which stress and strain modes can be represented and is therefore a defining part of a particular mixed element formulation.

Returns
-------
A tuple (N_sigma, N_epsilon), each with shape (6,24), using the parameter ordering of the cited HWT10 formulation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hwt10_mixed_interpolation(
    xi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return HWT10 stress and strain interpolation matrices."""

    return (
        np.zeros((6, 24), dtype=float),
        np.zeros((6, 24), dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_hwt10_mixed_interpolation(
    xi: np.ndarray,
):

    xi = np.asarray(
        xi,
        dtype=float,
    )

    if xi.shape != (3,):
        raise ValueError(
            "xi must have shape (3,)"
        )

    x, y, z = xi

    M = np.zeros(
        (6, 18),
        dtype=float,
    )

    values = np.array(
        [x, y, z],
        dtype=float,
    )

    for i in range(6):

        M[
            i,
            3 * i : 3 * i + 3,
        ] = values

    N_sigma = np.concatenate(
        (
            np.eye(6),
            M,
        ),
        axis=1,
    )

    N_epsilon = (
        N_sigma.copy()
    )

    return (
        N_sigma,
        N_epsilon,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
xi = np.array([0.25,0.25,0.25], dtype=float)
""",
            "call": """
hwt10_mixed_interpolation(xi)
""",
            "gold_call": """
_oracle_hwt10_mixed_interpolation(xi)
""",
            "tol": 1e-12,
        },
        {
            "setup": """
xi = np.array([0.0,0.0,0.0], dtype=float)
""",
            "call": """
hwt10_mixed_interpolation(xi)
""",
            "gold_call": """
_oracle_hwt10_mixed_interpolation(xi)
""",
            "tol": 1e-12,
        },
        {
            "setup": """
xi = np.array([0.5,0.5,0.0], dtype=float)
""",
            "call": """
hwt10_mixed_interpolation(xi)
""",
            "gold_call": """
_oracle_hwt10_mixed_interpolation(xi)
""",
            "tol": 1e-12,
        },
        {
            "setup": """
xi = np.array([0.13,0.27,0.31], dtype=float)
""",
            "call": """
hwt10_mixed_interpolation(xi)
""",
            "gold_call": """
_oracle_hwt10_mixed_interpolation(xi)
""",
            "tol": 1e-12,
        },
    ]
