"""
Integrate the prescribed generation term against the signed normal-flux kernel.

Let the generation profile be $f(x,y)=f_0+f_xx+f_yy+f_{xy}xy$.

The face source correction is

$$

J^f=*\int_*{-H/2}^{H/2}*\int_*{-\ell}^{\ell}Q(s)f(\mathbf c+s\mathbf n+t\mathbf t)\,ds\,dt.

$$

The axis-aligned right-handed frames yield

$J^f=H[f(c_x,c_y)I_0+(\nabla f(\mathbf c)\cdot\mathbf n)I_1]$.

The balance $-\sum(J^h+J^t)=\int_V f+\sum J^f$ fixes the side on which this contribution appears.

Returns
-------
A finite float giving the source contribution to the positive-normal face flux.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_source_face(
    source: np.ndarray,
    center: np.ndarray,
    axis: int,
    face_length: float,
    moments: np.ndarray,
) -> float:
    r"""Integrate the prescribed generation term against the signed normal-flux kernel.

    Parameters
    ----------
    source : np.ndarray
        Shape $(4,)$, ordered as $[f_0,f_x,f_y,f_{xy}]$.
    center : np.ndarray
        Shape $(2,)$; finite face midpoint.
    axis : int
        Zero for a positive horizontal normal or one for a positive vertical normal.
    face_length : float
        Positive tangent length $H$.
    moments : np.ndarray
        Shape $(3,)$, signed moments $I_0,I_1,I_2$.

    Returns
    -------
    float
        Integrated positive-normal source flux $J^f$.

    Raises
    ------
    ValueError
        If shapes, finite values or axis are invalid, or face length is non-positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_source_face(
    source: np.ndarray,
    center: np.ndarray,
    axis: int,
    face_length: float,
    moments: np.ndarray,
) -> float:
    """Evaluate the reference numerical operation."""
    source = _finite_array(source, "source", (4,))
    center = _finite_array(center, "center", (2,))
    normal, _ = _oriented_frame(axis)
    length = _positive_scalar(face_length, "face_length")
    moments = _finite_array(moments, "moments", (3,))
    f0, fx, fy, fxy = source
    x, y = center
    value = f0 + fx * x + fy * y + fxy * x * y
    derivative = np.dot(normal, [fx + fxy * y, fy + fxy * x])
    return float(length * (value * moments[0] + derivative * moments[1]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nsource = np.array([1.,1.,2.,3.]); center = np.array([.2,.4])\naxis, length = 0, .3\nmoments = np.array([-.03,.001,-.0002])\n",
            "call": "compute_source_face(source.copy(), center.copy(), axis, length, moments.copy())",
            "gold_call": "_oracle_compute_source_face(source.copy(), center.copy(), axis, length, moments.copy())",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nsource = np.array([1.,1.,2.,3.]); center = np.array([.2,.4])\naxis, length = 1, .3\nmoments = np.array([0.,.002,0.])\n",
            "call": "compute_source_face(source.copy(), center.copy(), axis, length, moments.copy())",
            "gold_call": "_oracle_compute_source_face(source.copy(), center.copy(), axis, length, moments.copy())",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nsource = np.zeros(4); center = np.array([0.,1.])\naxis, length = 1, .6\nmoments = np.array([.03,.001,.0002])\n",
            "call": "compute_source_face(source.copy(), center.copy(), axis, length, moments.copy())",
            "gold_call": "_oracle_compute_source_face(source.copy(), center.copy(), axis, length, moments.copy())",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nsource = np.ones(3); center = np.array([.2,.4])\naxis, length = 0, .3\nmoments = np.ones(3)\ndef _exception_code(function):\n    try:\n        function(source.copy(), center.copy(), axis, length, moments.copy())\n    except ValueError:\n        return 1\n    return 0\n\n",
            "call": "_exception_code(compute_source_face)",
            "gold_call": "_exception_code(_oracle_compute_source_face)",
        },
    ]
