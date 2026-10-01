"""
Turn one boundary element and one source point into the quadrature points, kernel weights and outward normals that carry the boundary integral of the representation formula.

The interior representation formula of the steady operator pairs the fundamental solution of the two-dimensional Laplace operator, normalised so that its Laplacian is minus the Dirac distribution at the source point, with the normal derivative of the field, and the normal derivative of that fundamental solution with the field itself, integrated over the boundary. Discretising one element with a quadrature rule in its local coordinate reduces each of those pairings to a weighted sum of field values at the quadrature points, with the arc length factor of the element folded into the weights.

Returns
-------
np.ndarray, float, shape (m, 6): the quadrature point coordinates, the two kernel weights and the outward unit normal components.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def element_boundary_terms(source_point: "np.ndarray", element_nodes: "np.ndarray",
                           rule: "np.ndarray") -> "np.ndarray":
    """Assemble the boundary quadrature data of one element for one source point.

    The element is interpolated with the three quadratic shape functions of its
    local coordinate, whose nodes lie at ``-1``, ``0`` and ``1``, and its
    outward normal follows from the anticlockwise traversal of the layer
    outline.

    Parameters
    ----------
    source_point : np.ndarray
        Float array of shape ``(2,)`` holding the ``(x, y)`` coordinates of the
        source point, which must not lie on the element.
    element_nodes : np.ndarray
        Float array of shape ``(3, 2)`` holding the first end node, the middle
        node and the second end node of the element.
    rule : np.ndarray
        Float array of shape ``(m, 2)`` holding the element local coordinates
        of the quadrature points and their weights.

    Returns
    -------
    terms : np.ndarray
        Float array of shape ``(m, 6)`` whose columns are, in order, the x and
        y coordinates of the quadrature points, the weight that multiplies the
        normal derivative of the field there, the weight that multiplies the
        field there, and the two components of the outward unit normal. Summing
        the third column against the normal derivative minus the fourth column
        against the field gives this element's contribution to the boundary
        integral of the representation formula.

    Raises
    ------
    ValueError
        If ``source_point`` is not a finite real array of shape ``(2,)``, if
        ``element_nodes`` is not a finite real array of shape ``(3, 2)``, if
        ``rule`` is not a finite real array of shape ``(m, 2)`` with
        ``m >= 1``, if the element has a vanishing tangent at a quadrature
        point, or if the source point coincides with a quadrature point.

    """
    return terms  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_element_boundary_terms(source_point: "np.ndarray", element_nodes: "np.ndarray",
                                   rule: "np.ndarray") -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    point = np.asarray(source_point, dtype=float)
    nodes = np.asarray(element_nodes, dtype=float)
    table = np.asarray(rule, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("source_point must be a finite array of shape (2,)")
    if nodes.shape != (3, 2) or not np.all(np.isfinite(nodes)):
        raise ValueError("element_nodes must be a finite array of shape (3, 2)")
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] != 2:
        raise ValueError("rule must have shape (m, 2) with m >= 1")
    if not np.all(np.isfinite(table)):
        raise ValueError("rule must be finite")

    local = table[:, 0]
    weight = table[:, 1]

    # Quadratic shape functions with nodes at -1, 0 and 1, and their slopes.
    shape = np.column_stack([0.5 * local * (local - 1.0),
                             1.0 - local ** 2,
                             0.5 * local * (local + 1.0)])
    slope = np.column_stack([local - 0.5, -2.0 * local, local + 0.5])

    coords = shape @ nodes
    tangent = slope @ nodes
    jacobian = np.hypot(tangent[:, 0], tangent[:, 1])
    if np.any(jacobian <= 0.0):
        raise ValueError("the element tangent vanishes at a quadrature point")

    # Anticlockwise traversal puts the outward normal to the right of the
    # tangent.
    normal_x = tangent[:, 1] / jacobian
    normal_y = -tangent[:, 0] / jacobian

    offset_x = coords[:, 0] - point[0]
    offset_y = coords[:, 1] - point[1]
    distance2 = offset_x ** 2 + offset_y ** 2
    if np.any(distance2 <= 0.0):
        raise ValueError("source_point coincides with a quadrature point")

    # Fundamental solution of the Laplace operator and its normal derivative.
    fundamental = -0.25 / np.pi * np.log(distance2)
    flux_kernel = -0.5 / np.pi * (offset_x * normal_x + offset_y * normal_y) / distance2

    return np.column_stack([coords[:, 0], coords[:, 1],
                            weight * jacobian * fundamental,
                            weight * jacobian * flux_kernel,
                            normal_x, normal_y])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Four separate square-face cases retain the full outline coverage
        # while invoking the candidate and oracle exactly once per case. ---
        {
            "setup": """import numpy as np
nodes = np.array([[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]])
gx, gw = np.polynomial.legendre.leggauss(60)
rule = np.column_stack([gx, gw])
src = np.array([0.2, -0.3])
def summarize(terms):
    return float(np.sum(terms[:, 3]) + 10.0 * np.mean(terms[:, 4])
                 + 100.0 * np.mean(terms[:, 5]))
""",
            "call": "summarize(element_boundary_terms(src.copy(), nodes.copy(), rule.copy()))",
            "gold_call": "summarize(_oracle_element_boundary_terms(src.copy(), nodes.copy(), rule.copy()))",
        },
        {
            "setup": """import numpy as np
nodes = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
gx, gw = np.polynomial.legendre.leggauss(60)
rule = np.column_stack([gx, gw])
src = np.array([0.2, -0.3])
def summarize(terms):
    return float(np.sum(terms[:, 3]) + 10.0 * np.mean(terms[:, 4])
                 + 100.0 * np.mean(terms[:, 5]))
""",
            "call": "summarize(element_boundary_terms(src.copy(), nodes.copy(), rule.copy()))",
            "gold_call": "summarize(_oracle_element_boundary_terms(src.copy(), nodes.copy(), rule.copy()))",
        },
        {
            "setup": """import numpy as np
nodes = np.array([[1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]])
gx, gw = np.polynomial.legendre.leggauss(60)
rule = np.column_stack([gx, gw])
src = np.array([0.2, -0.3])
def summarize(terms):
    return float(np.sum(terms[:, 3]) + 10.0 * np.mean(terms[:, 4])
                 + 100.0 * np.mean(terms[:, 5]))
""",
            "call": "summarize(element_boundary_terms(src.copy(), nodes.copy(), rule.copy()))",
            "gold_call": "summarize(_oracle_element_boundary_terms(src.copy(), nodes.copy(), rule.copy()))",
        },
        {
            "setup": """import numpy as np
nodes = np.array([[-1.0, 1.0], [-1.0, 0.0], [-1.0, -1.0]])
gx, gw = np.polynomial.legendre.leggauss(60)
rule = np.column_stack([gx, gw])
src = np.array([0.2, -0.3])
def summarize(terms):
    return float(np.sum(terms[:, 3]) + 10.0 * np.mean(terms[:, 4])
                 + 100.0 * np.mean(terms[:, 5]))
""",
            "call": "summarize(element_boundary_terms(src.copy(), nodes.copy(), rule.copy()))",
            "gold_call": "summarize(_oracle_element_boundary_terms(src.copy(), nodes.copy(), rule.copy()))",
        },
        # --- The quadrature points of a straight element are the linear image
        # of the local coordinates. ---
        {
            "setup": """import numpy as np
nodes = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]])
rule = np.array([[-0.5, 1.0], [0.25, 1.0]])
src = np.array([0.2, 0.01])
def summarize(terms):
    return float(np.sum(terms[:, 0]) + 1000.0 * np.sum(terms[:, 1]))
""",
            "call": "summarize(element_boundary_terms(src.copy(), nodes.copy(), rule.copy()))",
            "gold_call": "summarize(_oracle_element_boundary_terms(src.copy(), nodes.copy(), rule.copy()))",
        },
        # --- Valid: an ultra-close source point over a long coating element,
        # integrated with the hyperbolic-sine rule ---
        {
            "setup": """import numpy as np
def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    rank = np.arange(1.0, flat.size + 1.0)
    logs = np.log(np.abs(flat) + 1.0)
    scale = np.exp(np.mean(logs))
    order = np.sum(rank * np.sign(flat) * logs) / np.sum(rank)
    spread = np.sum(np.sqrt(rank) * logs ** 2) / np.sum(np.sqrt(rank))
    return float(flat.size * (1.0 + scale + spread + 0.5 * order))
nodes = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]])
src = np.array([0.2, 2.5e-5])
xi0 = 0.0
d = 2.5e-5 / 0.2
a = np.arcsinh((1.0 + xi0) / d)
b = np.arcsinh((1.0 - xi0) / d)
k1 = 0.5 * (a + b)
k2 = 0.5 * (a - b)
gx, gw = np.polynomial.legendre.leggauss(40)
s = k1 * gx - k2
rule = np.column_stack([xi0 + d * np.sinh(s), gw * d * k1 * np.cosh(s)])
""",
            "call": "float(pin(element_boundary_terms(src.copy(), nodes.copy(), rule.copy())))",
            "gold_call": "float(pin(_oracle_element_boundary_terms(src.copy(), nodes.copy(), rule.copy())))",
        },
        # --- Valid: a vertical element of the substrate seen from its interior ---
        {
            "setup": """import numpy as np
def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    rank = np.arange(1.0, flat.size + 1.0)
    logs = np.log(np.abs(flat) + 1.0)
    scale = np.exp(np.mean(logs))
    order = np.sum(rank * np.sign(flat) * logs) / np.sum(rank)
    spread = np.sum(np.sqrt(rank) * logs ** 2) / np.sum(np.sqrt(rank))
    return float(flat.size * (1.0 + scale + spread + 0.5 * order))
nodes = np.array([[4.0, -1.0], [4.0, -2.0 / 3.0], [4.0, -1.0 / 3.0]])
gx, gw = np.polynomial.legendre.leggauss(16)
rule = np.column_stack([gx, gw])
""",
            "call": "float(pin(element_boundary_terms(np.array([1.75, -0.5]), nodes.copy(), rule.copy())))",
            "gold_call": "float(pin(_oracle_element_boundary_terms(np.array([1.75, -0.5]), nodes.copy(), rule.copy())))",
        },
        # --- Boundary: a single quadrature point, which isolates one kernel
        # evaluation ---
        {
            "setup": """import numpy as np
def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    rank = np.arange(1.0, flat.size + 1.0)
    logs = np.log(np.abs(flat) + 1.0)
    scale = np.exp(np.mean(logs))
    order = np.sum(rank * np.sign(flat) * logs) / np.sum(rank)
    spread = np.sum(np.sqrt(rank) * logs ** 2) / np.sum(np.sqrt(rank))
    return float(flat.size * (1.0 + scale + spread + 0.5 * order))
nodes = np.array([[4.0, 1.0e-4], [2.0, 1.0e-4], [0.0, 1.0e-4]])
rule = np.array([[0.3, 1.7]])
""",
            "call": "float(pin(element_boundary_terms(np.array([1.0, 7.5e-5]), nodes.copy(), rule.copy())))",
            "gold_call": "float(pin(_oracle_element_boundary_terms(np.array([1.0, 7.5e-5]), nodes.copy(), rule.copy())))",
        },
        # --- Edge: a curved element, where the arc length factor varies along
        # the local coordinate ---
        {
            "setup": """import numpy as np
def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    rank = np.arange(1.0, flat.size + 1.0)
    logs = np.log(np.abs(flat) + 1.0)
    scale = np.exp(np.mean(logs))
    order = np.sum(rank * np.sign(flat) * logs) / np.sum(rank)
    spread = np.sum(np.sqrt(rank) * logs ** 2) / np.sum(np.sqrt(rank))
    return float(flat.size * (1.0 + scale + spread + 0.5 * order))
nodes = np.array([[0.0, 0.0], [0.5, 0.4], [1.0, 0.0]])
gx, gw = np.polynomial.legendre.leggauss(10)
rule = np.column_stack([gx, gw])
""",
            "call": "float(pin(element_boundary_terms(np.array([0.5, -0.6]), nodes.copy(), rule.copy())))",
            "gold_call": "float(pin(_oracle_element_boundary_terms(np.array([0.5, -0.6]), nodes.copy(), rule.copy())))",
        },
        # --- Invalid: the source point sits on a quadrature point ---
        {
            "setup": """import numpy as np
nodes = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]])
rule = np.array([[0.0, 2.0]])
def run_model():
    try:
        element_boundary_terms(np.array([0.2, 0.0]), nodes.copy(), rule.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_element_boundary_terms(np.array([0.2, 0.0]), nodes.copy(), rule.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a rule with the wrong number of columns ---
        {
            "setup": """import numpy as np
nodes = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]])
def run_model():
    try:
        element_boundary_terms(np.array([0.2, 0.01]), nodes.copy(), np.zeros((4, 3)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_element_boundary_terms(np.array([0.2, 0.01]), nodes.copy(), np.zeros((4, 3)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
