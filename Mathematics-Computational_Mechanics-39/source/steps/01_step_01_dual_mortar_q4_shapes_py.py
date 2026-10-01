"""
Evaluate the primal and source dual interpolation functions on one bilinear quadrilateral non-mortar interface element.

Use the standard four-node bilinear primal basis in the supplied natural coordinates and construct the four dual multiplier interpolation values exactly as defined for the three-dimensional interface formulation in the cited paper.

Preserve the node ordering 1,2,3,4 corresponding to bottom-left, bottom-right, top-right, top-left.

Return the primal and dual interpolation rows.

Mortar coupling introduces an independent interpolation for the interface constraint field.

Dual interpolation functions can be constructed to be bi-orthogonal to a primal finite-element basis. This property can simplify the interface mass-like coupling matrix and facilitate elimination of constrained degrees of freedom.

Returns
-------
A tuple (N, Phi).  N has shape (4,) and contains the primal bilinear Q4 shape-function values.  Phi has shape (4,) and contains the corresponding source dual multiplier shape-function values in the same node ordering.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def dual_mortar_q4_shapes(
    xi: float,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return primal Q4 and source dual interface interpolation rows."""

    return (
        np.zeros(4, dtype=float),
        np.zeros(4, dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_dual_mortar_q4_shapes(
    xi: float,
    eta: float,
):

    x = float(xi)
    y = float(eta)

    N = 0.25 * np.array(
        [
            (1.0 - x) * (1.0 - y),
            (1.0 + x) * (1.0 - y),
            (1.0 + x) * (1.0 + y),
            (1.0 - x) * (1.0 + y),
        ],
        dtype=float,
    )

    Phi = np.array(
        [
            4.0 * N[0]
            - 2.0 * N[1]
            + N[2]
            - 2.0 * N[3],

            -2.0 * N[0]
            + 4.0 * N[1]
            - 2.0 * N[2]
            + N[3],

            N[0]
            - 2.0 * N[1]
            + 4.0 * N[2]
            - 2.0 * N[3],

            -2.0 * N[0]
            + N[1]
            - 2.0 * N[2]
            + 4.0 * N[3],
        ],
        dtype=float,
    )

    return N, Phi

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
xi=0.0
eta=0.0
""",
            "call": """
(lambda z: np.concatenate((z[0],z[1])))(
dual_mortar_q4_shapes(xi,eta))
""",
            "gold_call": """
(lambda z: np.concatenate((z[0],z[1])))(
_oracle_dual_mortar_q4_shapes(xi,eta))
""",
            "tol": 1e-12,
        },
        {
            "setup": """
import numpy as np
xi=.23
eta=-.37
""",
            "call": """
(lambda z: np.concatenate((z[0],z[1])))(
dual_mortar_q4_shapes(xi,eta))
""",
            "gold_call": """
(lambda z: np.concatenate((z[0],z[1])))(
_oracle_dual_mortar_q4_shapes(xi,eta))
""",
            "tol": 1e-12,
        },
        {
            "setup": """
import numpy as np
xi=-1.0
eta=-1.0
""",
            "call": """
(lambda z: np.concatenate((z[0],z[1])))(
dual_mortar_q4_shapes(xi,eta))
""",
            "gold_call": """
(lambda z: np.concatenate((z[0],z[1])))(
_oracle_dual_mortar_q4_shapes(xi,eta))
""",
            "tol": 1e-12,
        },
        {
            "setup": """
import numpy as np
xi=.71
eta=.44
""",
            "call": """
(lambda z: np.concatenate((z[0],z[1])))(
dual_mortar_q4_shapes(xi,eta))
""",
            "gold_call": """
(lambda z: np.concatenate((z[0],z[1])))(
_oracle_dual_mortar_q4_shapes(xi,eta))
""",
            "tol": 1e-12,
        },
    ]
