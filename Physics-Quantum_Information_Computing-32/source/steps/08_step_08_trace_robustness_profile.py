"""
Recover every maximal affine segment of the reduced robustness along a mixture.

On an affine correlation path $y(t)=(1-t)y_0+ty_1$, the reduced robustness is a continuous convex piecewise-affine function. Each optimal dual witness gives a globally supporting affine line. Endpoint values alone can hide several changes of optimal face. Recover all maximal open affine pieces on $[0,1]$, using supporting lines and primal/dual values to distinguish a genuine adjacent intersection from one that lies below another active line. Interior kinks use the two limiting slopes; a flat interval with $R=1$ is a legitimate piece. Return global intercepts, not values at segment starts. Adjacent collinear fragments represent one piece. The input families have nonzero piece widths and slope jumps exceeding $10^{-6}$, and results are compared within $10^{-7}$.

Returns
-------
Real array $(K,4)$ with rows $(\ell,r,a,s)$ for $R(t)=a+st$ on $[\ell,r]$; rows cover $[0,1]$ in ascending order and are maximal affine pieces.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def trace_robustness_profile(
    vertices: "np.ndarray", start: "np.ndarray", end: "np.ndarray"
) -> "np.ndarray":
    r"""Recover every maximal affine segment of the reduced robustness along a mixture.

    Parameters
    ----------
    vertices : np.ndarray
        Finite real $(m,N)$ projected vertices spanning all of $\mathbb R^m$.
    start : np.ndarray
        Finite real initial correlation vector $(m,)$.
    end : np.ndarray
        Finite real final correlation vector $(m,)$, using the same measured coordinates.

    Returns
    -------
    segments : np.ndarray
        Real array $(K,4)$ with rows $(\ell,r,a,s)$ for $R(t)=a+st$ on $[\ell,r]$; rows
        cover $[0,1]$ in ascending order and are maximal affine pieces.

    Raises
    ------
    ValueError
        If vertex or vector data violate the directional-support contract.
    RuntimeError
        If support optimization, convexity certification, or profile refinement fails.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import linprog


def _oracle_trace_robustness_profile(
    vertices: "np.ndarray", start: "np.ndarray", end: "np.ndarray"
) -> "np.ndarray":
    vertices, start, end, _ = _checked_profile_data(vertices, start, end)
    direction = end - start
    records = []

    def _probe(t):
        return _oracle_certify_directional_support(vertices, start, direction, t)

    def _refine(left, right, left_support, right_support, depth):
        if depth > 80:
            raise RuntimeError("profile refinement did not converge")
        left_slope, right_slope = left_support[2], right_support[1]
        left_offset = left_support[0] - left_slope * left
        right_offset = right_support[0] - right_slope * right
        scale = max(
            1.0, abs(left_offset), abs(right_offset), abs(left_slope), abs(right_slope)
        )
        if (
            max(abs(left_slope - right_slope), abs(left_offset - right_offset))
            <= 1e-8 * scale
        ):
            records.append([left, right, left_offset, left_slope])
            return
        if right_slope <= left_slope:
            raise RuntimeError("support slopes violate convexity")
        crossing = (left_offset - right_offset) / (right_slope - left_slope)
        if not left + 1e-12 < crossing < right - 1e-12:
            raise RuntimeError(
                "support intersection is outside the unresolved interval"
            )
        middle_support = _probe(crossing)
        envelope = left_offset + left_slope * crossing
        if middle_support[0] - envelope <= 1e-8 * scale:
            records.extend(
                [
                    [left, crossing, left_offset, left_slope],
                    [crossing, right, right_offset, right_slope],
                ]
            )
            return
        _refine(left, crossing, left_support, middle_support, depth + 1)
        _refine(crossing, right, middle_support, right_support, depth + 1)

    _refine(0.0, 1.0, _probe(0.0), _probe(1.0), 0)
    merged = []
    for record in records:
        if merged and np.allclose(merged[-1][2:], record[2:], rtol=1e-8, atol=1e-8):
            merged[-1][1] = record[1]
        else:
            merged.append(record)
    return np.array(merged)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent scientific cases for this numerical contract."""
    return [
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
vertices = np.array([[1.,-1.,0.,0.],[0.,0.,1.,-1.]])
start = np.array([1.2,0.])
end = np.array([-1.2,1.2])
""",
            "call": "trace_robustness_profile(vertices.copy(), start.copy(), end.copy())",
            "gold_call": "_oracle_trace_robustness_profile(vertices.copy(), start.copy(), end.copy())",
            "tol": 1e-07,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
vertices = np.array([[-1.,-1.,1.,1.],[-1.,1.,-1.,1.]])
start = np.array([.2,-.4])
end = np.array([.7,.3])
""",
            "call": "trace_robustness_profile(vertices.copy(), start.copy(), end.copy())",
            "gold_call": "_oracle_trace_robustness_profile(vertices.copy(), start.copy(), end.copy())",
            "tol": 1e-07,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
vertices = np.array([[1.,-1.,0.,0.],[0.,0.,1.,-1.]])
start = np.array([-1.2,1.2])
end = np.array([1.2,0.])
""",
            "call": "trace_robustness_profile(vertices.copy(), start.copy(), end.copy())",
            "gold_call": "_oracle_trace_robustness_profile(vertices.copy(), start.copy(), end.copy())",
            "tol": 1e-07,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
vertices = np.array([[-1.,-1.,1.,1.],[-1.,1.,-1.,1.],[-1.,1.,1.,-1.]])
start = np.array([.4,.5,.6])
end = np.array([-.8,-.5,-.4])
""",
            "call": "trace_robustness_profile(vertices.copy(), start.copy(), end.copy())",
            "gold_call": "_oracle_trace_robustness_profile(vertices.copy(), start.copy(), end.copy())",
            "tol": 1e-07,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
vertices = np.array([[1.,1.]])
start = np.array([0.])
end = np.array([1.])

def _reject(fn):
    try:
        fn(vertices.copy(), start.copy(), end.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_reject(trace_robustness_profile)",
            "gold_call": "_reject(_oracle_trace_robustness_profile)",
            "tol": 0.0,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
vertices = np.array([[-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [-1, 0, 0, 0, 0, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, -1, -1, 0, 0, 0, 0, 1, 1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, -1, 1, -1, 1, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
start = np.array([0.4881061706360769, -0.5047267608512109, -0.6343511876714107, 0.48806898086536976, -0.4794438102616684, -0.6260487176417672, 0.5528668038500252, -0.545317248949569, -0.6719335565556124])
end = np.array([-0.3198007756697726, 0.26269550416478615, -0.6891614976078312, -0.5430477601301613, 0.5425475579091171, -0.8137856938132001, 0.4244863139386151, -0.46982037759272927, -0.7575288789739564])
""",
            "call": "trace_robustness_profile(vertices.copy(), start.copy(), end.copy())",
            "gold_call": "_oracle_trace_robustness_profile(vertices.copy(), start.copy(), end.copy())",
            "tol": 1e-7,
        },
    ]
