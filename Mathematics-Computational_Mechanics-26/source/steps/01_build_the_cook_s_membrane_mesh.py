"""
Produce the nodal coordinates of an n x n bilinear quadrilateral mesh of Cook's membrane. The domain is the quadrilateral with corners (0,0), (480,440), (480,600) and (0,440), and the mesh is its bilinear image: node (i, j) sits at parameter values xi = i/n and eta = j/n, and carries the id j*(n+1)+i. Everything downstream indexes into this array, so the node ordering is part of the contract rather than an implementation detail. The clamped boundary is identified later by the nodes whose first coordinate vanishes, and the loaded edge by the last node in each row, both of which only work if the ids follow the stated ordering.

Cook's membrane is the standard bending-dominated benchmark for nonlinear solid elements. It is a tapered cantilever: clamped along its short left edge, loaded on the right, and skewed enough that a coarse mesh of low-order elements is visibly too stiff. That sensitivity is exactly why it is used to compare formulations, and why the source paper runs it in both two and three dimensions.

The geometry is not a rectangle, so the mesh cannot be built by an outer product of two coordinate vectors. Instead the unit square is mapped bilinearly onto the quadrilateral, which keeps element edges straight and the element Jacobians well conditioned. Writing the map out, the horizontal coordinate is a pure scaling and the vertical one interpolates between the lower edge, which rises linearly from 0 to 440, and the upper edge, which rises from 440 to 600. The 2 x 2 mesh used in the task keeps the whole problem small enough to march by hand while remaining a genuine finite-deformation calculation.

Returns
-------
np.ndarray of shape (2*(n+1)**2,) holding the node coordinates flattened as [x_0, y_0, x_1, y_1, ...] in ascending node id. For n = 1 the array is exactly [0, 0, 480, 440, 0, 440, 480, 600]; for n = 2 the centre node, id 4, sits at (240, 370).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_cooks_mesh(n):
    """Nodal coordinates of an n x n bilinear (Q1) mesh of Cook's membrane.

    Cook's membrane has corners P1=(0,0), P2=(480,440), P3=(480,600),
    P4=(0,440), all in mm.  Node (i, j), i, j = 0..n, carries the id
    j*(n+1)+i and sits at the bilinear image of (xi, eta) = (i/n, j/n).

    Args:
        n (int): number of elements per side.

    Raises:
        ZeroDivisionError: if n == 0.

    Expected return:
        np.ndarray of shape (2*(n+1)**2,), the node coordinates flattened as
        [x_0, y_0, x_1, y_1, ...] in ascending node id.
    """
    return np.zeros(2 * (n + 1) ** 2)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_cooks_mesh(n):
    out = np.zeros(2 * (n + 1) ** 2)
    for j in range(n + 1):
        for i in range(n + 1):
            xi, eta = i / n, j / n
            nd = j * (n + 1) + i
            out[2 * nd] = 480.0 * xi
            out[2 * nd + 1] = 440.0 * xi + eta * (440.0 - 280.0 * xi)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "m = build_cooks_mesh(1)",
            "call": "m",
            "gold_call": "np.array([0.0, 0.0, 480.0, 440.0, 0.0, 440.0, 480.0, 600.0])",
        },
        {
            "setup": "m = build_cooks_mesh(2).reshape(-1, 2)",
            "call": "m[4]",
            "gold_call": "np.array([240.0, 370.0])",
        },
        {
            "setup": "m = build_cooks_mesh(4).reshape(-1, 2)",
            "call": "float(np.linalg.norm(m[-1] - np.array([480.0, 600.0])))",
            "gold_call": "0.0",
        },
        {
            "setup": "m = build_cooks_mesh(3).reshape(-1, 2)",
            "call": "int(m.shape[0])",
            "gold_call": "16",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        build_cooks_mesh(0)\n"
                "        return 0\n"
                "    except ZeroDivisionError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_build_cooks_mesh(0)\n"
                "        return 0\n"
                "    except ZeroDivisionError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
