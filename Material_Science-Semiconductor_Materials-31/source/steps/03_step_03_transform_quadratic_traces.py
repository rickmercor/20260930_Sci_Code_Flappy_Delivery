"""
Express the nine quadratic nodal basis functions in oriented face coordinates.

On a primary rectangle $[x_0,x_1]\times[y_0,y_1]$, the one-dimensional nodal functions are $L_0(r)=2r^2-3r+1$, $L_1(r)=4r-4r^2$, and $L_2(r)=2r^2-r$.

Use $N_{3j+i}=L_i((x-x_0)/h_x)L_j((y-y_0)/h_y)$.

For axis zero, $(x,y)=(c_x+s,c_y+t)$; for axis one, $(x,y)=(c_x-t,c_y+s)$.

Both choices give a right-handed normal-tangent frame.

Return coefficients $C_{k,a,b}$ in $N_k=*\sum_*{a,b=0}^2C_{k,a,b}s^at^b$.

These coefficients retain transverse curvature and all mixed tensor-product terms.

Returns
-------
A float tensor of nine basis polynomials in ascending normal and tangential powers.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def transform_quadratic_traces(
    bounds: np.ndarray, center: np.ndarray, axis: int
) -> np.ndarray:
    r"""Express the nine quadratic nodal basis functions in oriented face coordinates.

    Parameters
    ----------
    bounds : np.ndarray
        Shape $(4,)$, ordered as $[x_0,x_1,y_0,y_1]$ with positive widths.
    center : np.ndarray
        Shape $(2,)$; finite face midpoint within the closed rectangle.
    axis : int
        Zero for a positive horizontal normal, one for a positive vertical normal.

    Returns
    -------
    np.ndarray
        Shape $(9,3,3)$; local node, ascending normal power, ascending tangent power.

    Raises
    ------
    ValueError
        If shapes, finite values, rectangle widths, midpoint location, or axis are invalid.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _rectangle(bounds):
    bounds = _finite_array(bounds, "bounds", (4,))
    if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
        raise ValueError("rectangle widths must be positive")
    return bounds


def _oriented_frame(axis):
    if not isinstance(axis, (int, np.integer)) or axis not in (0, 1):
        raise ValueError("axis must be zero or one")
    return (
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        if axis == 0
        else (np.array([0.0, 1.0]), np.array([-1.0, 0.0]))
    )


def _mapped_lagrange(offset, slope):
    base = np.array([[1.0, -3.0, 2.0], [0.0, 4.0, -4.0], [0.0, -1.0, 2.0]])
    return np.column_stack(
        (
            base[:, 0] + base[:, 1] * offset + base[:, 2] * offset**2,
            slope * (base[:, 1] + 2 * base[:, 2] * offset),
            base[:, 2] * slope**2,
        )
    )


def _oracle_transform_quadratic_traces(
    bounds: np.ndarray, center: np.ndarray, axis: int
) -> np.ndarray:
    """Evaluate the reference numerical operation."""
    bounds = _rectangle(bounds)
    center = _finite_array(center, "center", (2,))
    _oriented_frame(axis)
    x0, x1, y0, y1 = bounds
    if not x0 <= center[0] <= x1 or not y0 <= center[1] <= y1:
        raise ValueError("center must lie in the rectangle")
    lx = _mapped_lagrange(
        (center[0] - x0) / (x1 - x0), (1 if axis == 0 else -1) / (x1 - x0)
    )
    ly = _mapped_lagrange((center[1] - y0) / (y1 - y0), 1 / (y1 - y0))
    result = np.empty((9, 3, 3))
    for j in range(3):
        for i in range(3):
            result[3 * j + i] = (
                np.outer(lx[i], ly[j]) if axis == 0 else np.outer(ly[j], lx[i])
            )
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nbounds = np.array([0., .22, .31, .64])\ncenter = np.array([.0594, .42])\naxis = 0\n",
            "call": "transform_quadratic_traces(bounds.copy(), center.copy(), axis)",
            "gold_call": "_oracle_transform_quadratic_traces(bounds.copy(), center.copy(), axis)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nbounds = np.array([0., 1., 0., 1.])\ncenter = np.array([.5, .5])\naxis = 1\n",
            "call": "transform_quadratic_traces(bounds.copy(), center.copy(), axis)",
            "gold_call": "_oracle_transform_quadratic_traces(bounds.copy(), center.copy(), axis)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nbounds = np.array([-1., -.999, 2., 5.])\ncenter = np.array([-1., 5.])\naxis = 0\n",
            "call": "transform_quadratic_traces(bounds.copy(), center.copy(), axis)",
            "gold_call": "_oracle_transform_quadratic_traces(bounds.copy(), center.copy(), axis)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nbounds = np.array([0., 1., 0., 1.])\ncenter = np.array([.5, .5])\naxis = 2\ndef _exception_code(function):\n    try:\n        function(bounds.copy(), center.copy(), axis)\n    except ValueError:\n        return 1\n    return 0\n\n",
            "call": "_exception_code(transform_quadratic_traces)",
            "gold_call": "_exception_code(_oracle_transform_quadratic_traces)",
        },
    ]
