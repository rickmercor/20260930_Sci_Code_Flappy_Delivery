"""
Given the four node coordinates of one element and a point in the parent square, return the determinant of the isoparametric Jacobian together with the gradients of the four shape functions taken with respect to the reference coordinates. Node ordering is counter-clockwise, matching the parent corners (-1,-1), (1,-1), (1,1) and (-1,1), which is the same order the connectivity produced by the previous step uses. The determinant is returned alongside the gradients because both are needed at every quadrature point and computing them together avoids inverting the Jacobian twice.

In an isoparametric element the same bilinear functions interpolate the geometry and the unknown field. Derivatives with respect to physical coordinates are therefore not available directly: one differentiates with respect to the parent coordinates and converts using the Jacobian of the map, which for a general quadrilateral varies from point to point.

Two properties are worth keeping in mind because they make good sanity checks. The shape functions form a partition of unity, so their gradients sum to zero at every point in the element; a set of gradients that does not sum to zero cannot represent a rigid translation and will produce spurious forces. And the Jacobian determinant must stay positive throughout: it is the local area scaling, so a sign change means the element has turned inside out. On the skewed Cook geometry the determinant varies noticeably across each element, which is part of what makes the benchmark discriminating.

Returns
-------
np.ndarray of shape (9,) packed as [detJ, dN1/dX, dN1/dY, dN2/dX, dN2/dY, dN3/dX, dN3/dY, dN4/dX, dN4/dY]. On the unit square scaled to side 2 the determinant is 1 and the gradients at the centre are +-0.25.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def q1_shape_gradients(xe, xi, eta):
    """Reference-configuration gradients of the Q1 shape functions.

    Nodes are ordered counter-clockwise at the parent corners
    (-1,-1), (1,-1), (1,1), (-1,1).

    Args:
        xe: (4, 2) array of element node coordinates.
        xi, eta (float): parent coordinates in [-1, 1].

    Expected return:
        np.ndarray of shape (9,) packed as
        [detJ, dN1/dX, dN1/dY, dN2/dX, dN2/dY, dN3/dX, dN3/dY, dN4/dX, dN4/dY].
    """
    return np.zeros(9)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_q1_shape_gradients(xe, xi, eta):
    dN = 0.25 * np.array([[-(1 - eta), -(1 - xi)],
                          [(1 - eta), -(1 + xi)],
                          [(1 + eta), (1 + xi)],
                          [-(1 + eta), (1 - xi)]])
    J = np.asarray(xe, dtype=float).T @ dN
    detJ = float(np.linalg.det(J))
    dNdX = dN @ np.linalg.inv(J)
    return np.concatenate(([detJ], dNdX.reshape(-1)))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "xe = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])",
            "call": "float(q1_shape_gradients(xe, 0.0, 0.0)[0])",
            "gold_call": "1.0",
        },
        {
            "setup": "xe = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])",
            "call": "q1_shape_gradients(xe, 0.0, 0.0)[1:]",
            "gold_call": "np.array([-0.25, -0.25, 0.25, -0.25, 0.25, 0.25, -0.25, 0.25])",
        },
        {
            "setup": (
                "xe = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])\n"
                "g = q1_shape_gradients(xe, 0.3, -0.7)[1:].reshape(4, 2)"
            ),
            "call": "g.sum(axis=0)",
            "gold_call": "np.zeros(2)",
        },
        {
            "setup": (
                "xe = np.array([[0.0, 0.0], [480.0, 440.0], [480.0, 600.0], [0.0, 440.0]])\n"
                "d = float(q1_shape_gradients(xe, -1.0, -1.0)[0])"
            ),
            "call": "d > 0.0",
            "gold_call": "True",
        },
        {
            "setup": (
                "xe = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])\n"
                "def run_model():\n"
                "    try:\n"
                "        v = q1_shape_gradients(xe, 0.0, 0.0)\n"
                "        return 0 if np.all(np.isfinite(v)) else 1\n"
                "    except Exception:\n"
                "        return 1\n"
                "def run_gold():\n"
                "    try:\n"
                "        v = _oracle_q1_shape_gradients(xe, 0.0, 0.0)\n"
                "        return 0 if np.all(np.isfinite(v)) else 1\n"
                "    except Exception:\n"
                "        return 1"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
