"""
Modified absolute enrichment and its physical gradient.




For linear shape functions N_i and nodal signed distances L_i, the modified absolute enrichment subtracts the absolute value of the interpolated level set from the interpolation of the nodal absolute values. Its product with each N_i is the local enriched scalar shape function. The gradient is evaluated separately on each side of the planar interface, where the sign of the interpolated level set is constant.




Inputs

------

barycentric : np.ndarray of shape (q, 4)

levels : np.ndarray of shape (4,)

shape_gradients : np.ndarray of shape (4, 3)




Returns

-------

enrichment : dict

    Enrichment values, gradients, and gradients of all four enriched shapes.

Returns
-------
dict with float64 arrays rho (q,), grad_rho (q, 3), and grad_enriched_shapes (q, 4, 3), where q is the number of quadrature points.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_modified_abs_enrichment(
    barycentric: np.ndarray,
    levels: np.ndarray,
    shape_gradients: np.ndarray,
) -> dict:
    """Evaluate the modified absolute enrichment at quadrature points.

    Parameters
    ----------
    barycentric : np.ndarray
        Original-tetrahedron barycentric points with shape (q, 4).
    levels : np.ndarray
        Nodal signed-distance values with shape (4,).
    shape_gradients : np.ndarray
        Physical gradients of the four P1 shape functions with shape (4, 3).

    Returns
    -------
    enrichment : dict
        Arrays rho, grad_rho, and grad_enriched_shapes.
    Raises
    ------
    ValueError
        If barycentric is not two-dimensional with four columns, if levels is
        not shape (4,) or shape_gradients is not shape (4, 3), if any
        barycentric row does not sum to one (atol 1e-12) or has an entry below
        -1e-12, if the four shape gradients do not sum to zero (atol 1e-12),
        or if any quadrature point lies on the interface (|interpolated
        level| <= 1e-14).
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_evaluate_modified_abs_enrichment(
    barycentric: np.ndarray,
    levels: np.ndarray,
    shape_gradients: np.ndarray,
) -> dict:
    """Reference implementation."""
    barycentric = np.asarray(barycentric, dtype=float)
    levels = np.asarray(levels, dtype=float)
    shape_gradients = np.asarray(shape_gradients, dtype=float)
    if barycentric.ndim != 2 or barycentric.shape[1] != 4:
        raise ValueError("barycentric must have shape (q, 4)")
    if levels.shape != (4,) or shape_gradients.shape != (4, 3):
        raise ValueError("levels and shape_gradients have invalid shapes")
    if not np.allclose(np.sum(barycentric, axis=1), 1.0, atol=1e-12):
        raise ValueError("every barycentric row must sum to one")
    if np.any(barycentric < -1e-12):
        raise ValueError("barycentric coordinates must be nonnegative")
    if not np.allclose(np.sum(shape_gradients, axis=0), 0.0, atol=1e-12):
        raise ValueError("P1 shape gradients must sum to zero")

    interpolated_level = barycentric @ levels
    if np.any(np.abs(interpolated_level) <= 1e-14):
        raise ValueError("quadrature points may not lie on the interface")
    signs = np.sign(interpolated_level)
    rho = barycentric @ np.abs(levels) - np.abs(interpolated_level)
    interpolated_abs_gradient = np.abs(levels) @ shape_gradients
    level_gradient = levels @ shape_gradients
    grad_rho = interpolated_abs_gradient[None, :] - signs[:, None] * level_gradient
    grad_enriched_shapes = (
        rho[:, None, None] * shape_gradients[None, :, :]
        + barycentric[:, :, None] * grad_rho[:, None, :]
    )
    return {
        "rho": rho,
        "grad_rho": grad_rho,
        "grad_enriched_shapes": grad_enriched_shapes,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
barycentric = np.array([[0.4, 0.2, 0.2, 0.2], [0.1, 0.3, 0.3, 0.3]])
levels = np.array([-0.35, 0.65, 0.45, 0.25])
shape_gradients = np.array([[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
def summarize(result):
    return (
        np.round(result["rho"], 12).tolist(),
        np.round(result["grad_rho"], 12).tolist(),
        np.round(result["grad_enriched_shapes"], 12).tolist(),
    )
""",
            "call": "summarize(evaluate_modified_abs_enrichment(barycentric, levels, shape_gradients))",
            "gold_call": "summarize(_oracle_evaluate_modified_abs_enrichment(barycentric, levels, shape_gradients))",
        },
        {
            "setup": """import numpy as np
barycentric = np.array([[0.25, 0.25, 0.25, 0.25]])
levels = np.ones(4)
shape_gradients = np.array([[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
def summarize(result):
    return (
        result["rho"].tolist(),
        result["grad_rho"].tolist(),
        result["grad_enriched_shapes"].tolist(),
    )
""",
            "call": "summarize(evaluate_modified_abs_enrichment(barycentric, levels, shape_gradients))",
            "gold_call": "summarize(_oracle_evaluate_modified_abs_enrichment(barycentric, levels, shape_gradients))",
        },
        {
            "setup": """import numpy as np
barycentric = np.array([[0.5, 0.5, 0.0]])
levels = np.ones(4)
shape_gradients = np.zeros((4, 3))
def run_model():
    try:
        evaluate_modified_abs_enrichment(barycentric, levels, shape_gradients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_evaluate_modified_abs_enrichment(barycentric, levels, shape_gradients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
