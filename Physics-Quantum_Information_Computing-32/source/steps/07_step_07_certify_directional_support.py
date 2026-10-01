"""
Compute robustness and both limiting slopes from the entire optimal dual face.

For projected vertices $V$, let $y(t)=y_0+t\,d$ and append a row of ones to $V$ to obtain $A$. The affine robustness is the support function of $D=\{z:|A^Tz|\le1\}$ in direction $q(t)=(y(t)^T,1)^T$. The optimal face is $F(t)=\{z\in D:q(t)^Tz=R(t)\}$. Return $R(t)$ and the left and right derivatives of the unrestricted real-parameter profile at $t$, including at the requested interval endpoints. These derivatives are the extreme directional pairings on $F(t)$, not the slope of an arbitrary optimal witness. An exposed face can contain different slopes even when the optimal value is unique. The constant affine coordinate has zero directional derivative. Use primal/dual feasibility and objective agreement to certify the value; values and slopes are compared with relative and absolute tolerance $10^{-6}$.

Returns
-------
Real array $(3,)$ containing $[R(t),R^{\prime}_-(t),R^{\prime}_+(t)]$, in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def certify_directional_support(
    vertices: "np.ndarray",
    base: "np.ndarray",
    direction: "np.ndarray",
    parameter: float,
) -> "np.ndarray":
    r"""Compute robustness and both limiting slopes from the entire optimal dual face.

    Parameters
    ----------
    vertices : np.ndarray
        Finite real $(m,N)$ vertex matrix whose affine span is all of $\mathbb R^m$.
    base : np.ndarray
        Finite real $(m,)$ correlation vector at $t=0$.
    direction : np.ndarray
        Finite real $(m,)$ change per unit mixing parameter;
        $y(t)=\mathrm{base}+t\,\mathrm{direction}$.
    parameter : float
        Finite real evaluation parameter $t$; slopes refer to two-sided continuation on its
        affine line.

    Returns
    -------
    support : np.ndarray
        Real array $(3,)$ containing $[R(t),R^{\prime}_-(t),R^{\prime}_+(t)]$, in that order.

    Raises
    ------
    ValueError
        If shapes, realness, finiteness, or full affine span fail.
    RuntimeError
        If a linear optimization or its feasibility/optimality certificate fails.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import linprog


def _affine_robustness_certificate(vertices, expectations):
    raw_v, raw_y = np.asarray(vertices), np.asarray(expectations)
    if raw_v.ndim != 2 or min(raw_v.shape) < 1 or raw_y.shape != (raw_v.shape[0],):
        raise ValueError("vertices and expectations have incompatible shapes")
    if np.iscomplexobj(raw_v) or np.iscomplexobj(raw_y):
        raise ValueError("linear program data must be real")
    vertices, expectations = raw_v.astype(float), raw_y.astype(float)
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(expectations)):
        raise ValueError("linear program data must be finite")
    count = vertices.shape[1]
    affine = np.vstack((vertices, np.ones(count)))
    target = np.append(expectations, 1.0)
    split = np.column_stack((affine, -affine))
    # Rescale coefficients, not the feasible correlations. This resolves small
    # positive weights near a face transition above the solver's absolute tolerance.
    scale = 65536.0 / max(1.0, float(np.max(np.abs(target))))
    options = {
        "primal_feasibility_tolerance": 1e-10,
        "dual_feasibility_tolerance": 1e-10,
    }
    primal = linprog(
        np.ones(2 * count),
        A_eq=split,
        b_eq=scale * target,
        bounds=(0, None),
        method="highs",
        options=options,
    )
    if primal.status == 2:
        raise ValueError("expectations lie outside the affine span")
    if not primal.success:
        raise RuntimeError("primal optimization failed")
    dual = linprog(
        -scale * target,
        A_ub=np.vstack((affine.T, -affine.T)),
        b_ub=np.ones(2 * count),
        bounds=[(None, None)] * len(target),
        method="highs",
        options=options,
    )
    if not dual.success:
        raise RuntimeError("dual optimization failed")
    coefficients = primal.x / scale
    optimum = float(primal.fun / scale)
    dual_value = float(-dual.fun / scale)
    residual = float(np.max(np.abs(split @ coefficients - target)))
    violation = float(max(0.0, np.max(np.abs(affine.T @ dual.x)) - 1.0))
    gap = abs(optimum - dual_value)
    tolerance = 1e-8 * max(1.0, abs(optimum))
    if (
        max(residual, violation, gap, -float(np.min(coefficients)), 1.0 - optimum)
        > tolerance
    ):
        raise RuntimeError("affine robustness certificate failed")
    return (
        optimum,
        dual_value,
        residual,
        violation,
        coefficients[:count] - coefficients[count:],
        dual.x,
    )


def _checked_profile_data(vertices, base, direction):
    raw_v, raw_b, raw_d = map(np.asarray, (vertices, base, direction))
    if raw_v.ndim != 2 or min(raw_v.shape) < 1:
        raise ValueError("vertices must be a nonempty matrix")
    if raw_b.shape != (raw_v.shape[0],) or raw_d.shape != raw_b.shape:
        raise ValueError("correlation vectors have incompatible dimensions")
    if any(
        np.iscomplexobj(x) or not np.all(np.isfinite(x)) for x in (raw_v, raw_b, raw_d)
    ):
        raise ValueError("profile data must be finite and real")
    vertices, base, direction = (x.astype(float) for x in (raw_v, raw_b, raw_d))
    affine = np.vstack((vertices, np.ones(vertices.shape[1])))
    if np.linalg.matrix_rank(affine) != len(base) + 1:
        raise ValueError("vertices must span the complete affine measurement space")
    return vertices, base, direction, affine


def _oracle_certify_directional_support(
    vertices: "np.ndarray",
    base: "np.ndarray",
    direction: "np.ndarray",
    parameter: float,
) -> "np.ndarray":
    vertices, base, direction, affine = _checked_profile_data(vertices, base, direction)
    if not np.isfinite(parameter):
        raise ValueError("parameter must be finite")
    target = np.append(base + parameter * direction, 1.0)
    tangent = np.append(direction, 0.0)
    _, optimum, _, _, coefficients, _ = _affine_robustness_certificate(
        vertices, target[:-1]
    )
    # Complementary slackness gives the same entire optimal face. Its equalities
    # use vertex normals and remain well-conditioned as a positive weight tends
    # to zero, unlike imposing the almost-parallel objective equality directly.
    cutoff = 16.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(coefficients))))
    support = np.abs(coefficients) > cutoff
    face_matrix = affine[:, support].T
    face_values = np.sign(coefficients[support])
    inequalities = np.vstack((affine.T, -affine.T))
    limits = np.ones(len(inequalities))
    options = {
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
    }
    extrema = []
    for sense in (1.0, -1.0):
        result = linprog(
            sense * tangent,
            A_ub=inequalities,
            b_ub=limits,
            A_eq=face_matrix,
            b_eq=face_values,
            bounds=[(None, None)] * len(target),
            method="highs",
            options=options,
        )
        if not result.success:
            raise RuntimeError("optimal-face slope optimization failed")
        error = max(
            abs(target @ result.x - optimum),
            float(np.max(np.abs(face_matrix @ result.x - face_values))),
            float(np.max(inequalities @ result.x - limits)),
        )
        if error > 1e-8 * max(1.0, abs(optimum)):
            raise RuntimeError("optimal-face feasibility certificate failed")
        extrema.append(float(tangent @ result.x))
    if extrema[0] > extrema[1] + 1e-8 * max(1.0, *map(abs, extrema)):
        raise RuntimeError("directional derivative ordering failed")
    return np.array([optimum, extrema[0], extrema[1]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent scientific cases for this numerical contract."""
    cases = [
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
vertices = np.array([[1.,-1.,0.,0.],[0.,0.,1.,-1.]])
base = np.array([1.2,0.])
direction = np.array([-2.4,1.2])
t = 0.1
""",
            "call": "certify_directional_support(vertices.copy(), base.copy(), direction.copy(), t)",
            "gold_call": "_oracle_certify_directional_support(vertices.copy(), base.copy(), direction.copy(), t)",
            "tol": 1e-06,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
vertices = np.array([[1.,-1.,0.,0.],[0.,0.,1.,-1.]])
base = np.array([1.2,0.])
direction = np.array([-2.4,1.2])
t = 1.0/6.0
""",
            "call": "certify_directional_support(vertices.copy(), base.copy(), direction.copy(), t)",
            "gold_call": "_oracle_certify_directional_support(vertices.copy(), base.copy(), direction.copy(), t)",
            "tol": 1e-06,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
vertices = np.array([[1.,-1.,0.,0.],[0.,0.,1.,-1.]])
base = np.array([1.2,0.])
direction = np.array([-2.4,1.2])
t = 11.0/18.0
""",
            "call": "certify_directional_support(vertices.copy(), base.copy(), direction.copy(), t)",
            "gold_call": "_oracle_certify_directional_support(vertices.copy(), base.copy(), direction.copy(), t)",
            "tol": 1e-06,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
vertices = np.array([[-1.,-1.,1.,1.],[-1.,1.,-1.,1.]])
base = np.array([.2,-.4])
direction = np.zeros(2)
t = 0.5
""",
            "call": "certify_directional_support(vertices.copy(), base.copy(), direction.copy(), t)",
            "gold_call": "_oracle_certify_directional_support(vertices.copy(), base.copy(), direction.copy(), t)",
            "tol": 1e-06,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
vertices = np.array([[1.,1.]])
base = np.array([0.])
direction = np.array([1.])

def _reject(fn):
    try:
        fn(vertices.copy(), base.copy(), direction.copy(), 0.0)
    except ValueError:
        return 1
    return 0
""",
            "call": "_reject(certify_directional_support)",
            "gold_call": "_reject(_oracle_certify_directional_support)",
            "tol": 0.0,
        },
    ]

    for parameter in (
        1.0 / 6.0 - 1e-12,
        1.0 / 6.0 + 1e-12,
        1.0 / 6.0 - 1e-9,
        1.0 / 6.0 + 1e-9,
        11.0 / 18.0 - 1e-9,
        11.0 / 18.0 + 1e-9,
    ):
        case = cases[0].copy()
        case["setup"] = case["setup"].replace("t = 0.1", f"t = {parameter!r}")
        cases.append(case)
    fixture = "import numpy as np\nfrom scipy.optimize import linprog\nvertices = np.array([[-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [-1, 0, 0, 0, 0, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, -1, -1, 0, 0, 0, 0, 1, 1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, -1, 1, -1, 1, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])\nstart = np.array([0.4881061706360769, -0.5047267608512109, -0.6343511876714107, 0.48806898086536976, -0.4794438102616684, -0.6260487176417672, 0.5528668038500252, -0.545317248949569, -0.6719335565556124])\nend = np.array([-0.3198007756697726, 0.26269550416478615, -0.6891614976078312, -0.5430477601301613, 0.5425475579091171, -0.8137856938132001, 0.4244863139386151, -0.46982037759272927, -0.7575288789739564])\nbase = start\ndirection = end - start\n"
    for parameter in (
        0.0822818082246778 - 1e-9,
        0.0822818082246778 + 1e-9,
        0.620964723942486 - 1e-10,
        0.620964723942486 + 1e-10,
    ):
        cases.append(
            {
                "setup": fixture + f"t = {parameter!r}\n",
                "call": "certify_directional_support(vertices.copy(), base.copy(), direction.copy(), t)",
                "gold_call": "_oracle_certify_directional_support(vertices.copy(), base.copy(), direction.copy(), t)",
                "tol": 1e-6,
            }
        )
    return cases
