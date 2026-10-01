"""
Evaluate the mortar-side background-grid trial interpolation used by this benchmark specialization of the dual-mortar coupling algorithm.

The mortar side is a regular Cartesian MPM background grid covering the rectangular interface. For this benchmark, use standard bilinear background-grid trial functions in the cell containing the supplied physical point.

Order all global background-grid nodes row-major from the bottom row to the top row and from left to right within each row.

Return the complete global interpolation row, with zeros for inactive grid nodes.

The mortar coupling procedure requires evaluation of the displacement trial interpolation on the opposing side of the interface at every integration point.

For a regular Cartesian background grid with bilinear trial functions, only the four grid nodes of the cell containing the evaluation point are active. Their interpolation weights form a partition of unity.

Returns
-------
A NumPy array with shape ((n_cells_x+1)*(n_cells_y+1),).  The entries are the global bilinear mortar-side background-grid trial-function values in row-major grid-node ordering.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def regular_grid_interface_shapes(
    x: float,
    y: float,
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
) -> np.ndarray:
    """Return global bilinear background-grid interpolation row."""

    return np.zeros(
        (n_cells_x + 1)
        * (n_cells_y + 1),
        dtype=float,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_regular_grid_interface_shapes(
    x: float,
    y: float,
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
):

    nx = int(n_cells_x)
    ny = int(n_cells_y)

    xs = np.linspace(
        -0.5 * length_x,
        0.5 * length_x,
        nx + 1,
    )

    ys = np.linspace(
        -0.5 * length_y,
        0.5 * length_y,
        ny + 1,
    )

    i = (
        np.searchsorted(
            xs,
            float(x),
            side="right",
        )
        - 1
    )

    j = (
        np.searchsorted(
            ys,
            float(y),
            side="right",
        )
        - 1
    )

    i = min(
        max(i, 0),
        nx - 1,
    )

    j = min(
        max(j, 0),
        ny - 1,
    )

    tx = (
        (float(x) - xs[i])
        / (xs[i + 1] - xs[i])
    )

    ty = (
        (float(y) - ys[j])
        / (ys[j + 1] - ys[j])
    )

    local = np.array(
        [
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            tx * ty,
            (1.0 - tx) * ty,
        ],
        dtype=float,
    )

    active = np.array(
        [
            j * (nx + 1) + i,
            j * (nx + 1) + i + 1,
            (j + 1) * (nx + 1) + i + 1,
            (j + 1) * (nx + 1) + i,
        ],
        dtype=int,
    )

    N = np.zeros(
        (nx + 1) * (ny + 1),
        dtype=float,
    )

    N[active] = local

    return N

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
x=.276
y=-.296
length_x=2.4
length_y=1.6
n_cells_x=3
n_cells_y=2
""",
            "call": """
regular_grid_interface_shapes(
x,y,length_x,length_y,n_cells_x,n_cells_y)
""",
            "gold_call": """
_oracle_regular_grid_interface_shapes(
x,y,length_x,length_y,n_cells_x,n_cells_y)
""",
            "tol": 1e-12,
        },
        {
            "setup": """
x=0.
y=0.
length_x=2.4
length_y=1.6
n_cells_x=3
n_cells_y=2
""",
            "call": """
regular_grid_interface_shapes(
x,y,length_x,length_y,n_cells_x,n_cells_y)
""",
            "gold_call": """
_oracle_regular_grid_interface_shapes(
x,y,length_x,length_y,n_cells_x,n_cells_y)
""",
            "tol": 1e-12,
        },
        {
            "setup": """
x=-1.
y=-.75
length_x=2.
length_y=1.5
n_cells_x=2
n_cells_y=3
""",
            "call": """
regular_grid_interface_shapes(
x,y,length_x,length_y,n_cells_x,n_cells_y)
""",
            "gold_call": """
_oracle_regular_grid_interface_shapes(
x,y,length_x,length_y,n_cells_x,n_cells_y)
""",
            "tol": 1e-12,
        },
        {
            "setup": """
x=1.5
y=.6
length_x=3.
length_y=1.2
n_cells_x=5
n_cells_y=1
""",
            "call": """
regular_grid_interface_shapes(
x,y,length_x,length_y,n_cells_x,n_cells_y)
""",
            "gold_call": """
_oracle_regular_grid_interface_shapes(
x,y,length_x,length_y,n_cells_x,n_cells_y)
""",
            "tol": 1e-12,
        },
    ]
