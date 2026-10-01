"""
This step evaluates the modified enrichment and its gradient at a supplied set of points of one element, from the nodal level set values and the constant gradients of the four linear shape functions.

Both quantities are returned because the enriched strain operator needs both: the value multiplies the shape function gradient and the gradient multiplies the shape function, and the two terms are of comparable size on a barely cut element. The gradient is evaluated phasewise, from the sign of the interpolated level set at each point, rather than by differentiating a smoothed surrogate, so the jump across the interface is represented exactly and not spread over a layer of points.

An absolute-value enrichment built directly on the interpolated level set, $rho^a = |\sum_i N_i L_i|$, reproduces the kink in the displacement at an interface but is nearly linearly dependent on the standard basis, which ruins the conditioning. Subtracting it from its own linear interpolant gives the modified enrichment $rho^m(x) = \sum_i N_i(x) |L_i| - |\sum_i N_i(x) L_i|$, which vanishes identically on every element whose nodal level set values share a sign, and therefore vanishes on the boundary of the cut region. The enriched basis is then supported only on cut elements and the linear dependence is removed.

Within one element $rho^m$ is piecewise linear, with the gradient $\operatorname{grad}(rho^m) = \sum_i |L_i| \operatorname{grad}(N_i) - \operatorname{sign}(L_h) \sum_i L_i \operatorname{grad}(N_i)$, where $L_h = \sum_i N_i L_i$. The gradient is constant on each side of the interface and jumps across it, which is exactly the weak discontinuity the enrichment is meant to carry. The enriched scalar shape function of node $i$ is $N_i rho^m$, so its gradient is $rho^m \operatorname{grad}(N_i) + N_i \operatorname{grad}(rho^m)$ and is linear in space on each side.

Returns
-------
dict, the enrichment values at the quadrature points and their gradients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_modified_abs_enrichment(barycentric, levels, gradients) -> dict:
    """Evaluate the modified absolute enrichment and its gradient at given points.

    Parameters
    ----------
    barycentric : array_like
        Array of shape (q, 4) of barycentric coordinates.
    levels : array_like
        Array of shape (4,) of nodal level set values.
    gradients : array_like
        Array of shape (4, 3) of linear shape function gradients.

    Returns
    -------
    dict
        Keys rho of shape (q,) and grad_rho of shape (q, 3).

    Raises
    ------
    ValueError
        If barycentric does not have shape (q, 4), or if levels does not have shape (4,) or gradients does not have shape (4, 3).
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _enrichment(barycentric, levels, gradients):
    """Modified absolute enrichment and its phasewise gradient at the quadrature points."""
    levels = np.asarray(levels, dtype=float)
    interpolated = barycentric @ levels
    sign = np.where(interpolated >= 0.0, 1.0, -1.0)
    rho = barycentric @ np.abs(levels) - np.abs(interpolated)
    grad_rho = ((np.abs(levels) @ gradients)[None, :]
                - sign[:, None] * (levels @ gradients)[None, :])
    return rho, grad_rho


def _oracle_evaluate_modified_abs_enrichment(barycentric, levels, gradients) -> dict:
    """Reference implementation."""
    barycentric = np.asarray(barycentric, dtype=float)
    levels = np.asarray(levels, dtype=float)
    gradients = np.asarray(gradients, dtype=float)
    if barycentric.ndim != 2 or barycentric.shape[1] != 4:
        raise ValueError("barycentric must have shape (q, 4)")
    if levels.shape != (4,) or gradients.shape != (4, 3):
        raise ValueError("levels must have shape (4,) and gradients shape (4, 3)")
    rho, grad_rho = _enrichment(barycentric, levels, gradients)
    return {"rho": rho, "grad_rho": grad_rho}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
G = np.linalg.inv(np.column_stack([np.ones(4), V]))[1:, :].T
B = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.3, 0.5], [0.25, 0.25, 0.25, 0.25]])
L = np.array([-0.4, 0.3, 0.6, 0.9])
def summarize(d):
    return (tuple(round(float(x), 12) for x in d["rho"]),
            tuple(round(float(x), 12) for x in d["grad_rho"].ravel()))
""",
            "call": "summarize(evaluate_modified_abs_enrichment(B, L, G))",
            "gold_call": "summarize(_oracle_evaluate_modified_abs_enrichment(B, L, G))",
        },
        {
            "setup": """import numpy as np
V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
G = np.linalg.inv(np.column_stack([np.ones(4), V]))[1:, :].T
B = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.3, 0.5]])
def uncut(fn):
    d = fn(B, np.array([0.4, 0.3, 0.6, 0.9]), G)
    e = fn(B, np.array([-0.4, -0.3, -0.6, -0.9]), G)
    return (round(float(np.abs(d["rho"]).max()), 13),
            round(float(np.abs(d["grad_rho"]).max()), 13),
            round(float(np.abs(e["rho"]).max()), 13),
            round(float(np.abs(e["grad_rho"]).max()), 13))
""",
            "call": "uncut(evaluate_modified_abs_enrichment)",
            "gold_call": "uncut(_oracle_evaluate_modified_abs_enrichment)",
        },
        {
            "setup": """import numpy as np
V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
G = np.linalg.inv(np.column_stack([np.ones(4), V]))[1:, :].T
L = np.array([-0.4, 0.3, 0.6, 0.9])
def consistency(fn):
    # finite difference of rho along a segment that stays on one side of the interface
    a = np.array([0.05, 0.35, 0.30, 0.30])
    step = np.array([0.0, 1.0, -1.0, 0.0]) * 1e-6
    d = fn(np.array([a, a + step]), L, G)
    direction = (step @ V)
    predicted = float(d["grad_rho"][0] @ direction)
    observed = float(d["rho"][1] - d["rho"][0])
    return round(abs(predicted - observed), 14)
""",
            "call": "consistency(evaluate_modified_abs_enrichment)",
            "gold_call": "consistency(_oracle_evaluate_modified_abs_enrichment)",
        },
    ]
