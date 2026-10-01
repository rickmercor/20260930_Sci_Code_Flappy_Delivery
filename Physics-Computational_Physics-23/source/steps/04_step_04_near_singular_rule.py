"""
Build the quadrature rule used to integrate one boundary element as seen from a source point that may lie very close to it.

When a source point sits a distance far smaller than an element length away from that element, the kernels of the boundary integral equation vary over a region of the element much shorter than the spacing of a plain Gauss-Legendre rule, and the rule loses all accuracy. A hyperbolic-sine change of the element coordinate spreads that region out over the whole reference interval, so that the same number of Gauss points resolves it.

Returns
-------
np.ndarray, float, shape (num_gauss, 2): the element local coordinates of the quadrature points and their weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def near_singular_rule(source_point: "np.ndarray", element_nodes: "np.ndarray",
                       num_gauss: int, use_sinh: bool) -> "np.ndarray":
    """Return the quadrature nodes and weights for one element and source point.

    The rule integrates a function of the element local coordinate over
    ``[-1, 1]``. With ``use_sinh`` set, the local coordinate is written as
    ``xi(t) = xi0 + d * sinh(k1 * t - k2)`` for a new variable ``t`` running
    over ``[-1, 1]``, where ``xi0`` is the local coordinate of the foot of the
    perpendicular dropped from the source point onto the straight line carrying
    the element, ``d`` is the distance from the source point to that foot
    divided by half the element length, and the two constants are fixed by
    requiring that ``t = -1`` and ``t = 1`` map onto the two ends of the
    element. The Gauss-Legendre rule is then applied in ``t``.

    Parameters
    ----------
    source_point : np.ndarray
        Float array of shape ``(2,)`` holding the ``(x, y)`` coordinates of the
        source point.
    element_nodes : np.ndarray
        Float array of shape ``(3, 2)`` holding the first end node, the middle
        node and the second end node of a straight boundary element whose
        middle node is its midpoint.
    num_gauss : int
        Number of Gauss-Legendre points of the rule (``num_gauss >= 1``).
    use_sinh : bool
        If true, apply the hyperbolic-sine change of variable; if false, return
        the plain Gauss-Legendre rule of the element local coordinate.

    Returns
    -------
    rule : np.ndarray
        Float array of shape ``(num_gauss, 2)`` whose first column holds the
        element local coordinates of the quadrature points and whose second
        column holds the matching weights, already carrying the derivative of
        the change of variable, so that the integral of a function over the
        element local coordinate is the weighted sum of its values.

    Raises
    ------
    ValueError
        If ``source_point`` is not a finite real array of shape ``(2,)``, if
        ``element_nodes`` is not a finite real array of shape ``(3, 2)``, if
        the element has zero length or its middle node is not its midpoint, if
        the source point lies on the straight line carrying the element, if
        ``num_gauss`` is not an integer >= 1, or if ``use_sinh`` is not a
        boolean.

    """
    return rule  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_near_singular_rule(source_point: "np.ndarray", element_nodes: "np.ndarray",
                               num_gauss: int, use_sinh: bool) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    point = np.asarray(source_point, dtype=float)
    nodes = np.asarray(element_nodes, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("source_point must be a finite array of shape (2,)")
    if nodes.shape != (3, 2) or not np.all(np.isfinite(nodes)):
        raise ValueError("element_nodes must be a finite array of shape (3, 2)")
    if isinstance(num_gauss, bool) or not isinstance(num_gauss, (int, np.integer)):
        raise ValueError("num_gauss must be an integer")
    if int(num_gauss) < 1:
        raise ValueError("num_gauss must be an integer >= 1")
    if not isinstance(use_sinh, (bool, np.bool_)):
        raise ValueError("use_sinh must be a boolean")

    half = 0.5 * (nodes[2] - nodes[0])
    half_length = float(np.hypot(half[0], half[1]))
    if half_length <= 0.0:
        raise ValueError("element_nodes must describe an element of non-zero length")
    if float(np.hypot(*(nodes[1] - 0.5 * (nodes[0] + nodes[2])))) > 1.0e-9 * half_length:
        raise ValueError("the middle node must be the midpoint of the element")

    points, weights = np.polynomial.legendre.leggauss(int(num_gauss))
    if not bool(use_sinh):
        return np.column_stack([np.asarray(points, dtype=float),
                                np.asarray(weights, dtype=float)])

    # Foot of the perpendicular from the source point, expressed in the element
    # local coordinate, and the distance to it in units of the half length.
    foot_local = float(np.dot(point - nodes[1], half) / np.dot(half, half))
    foot = nodes[1] + foot_local * half
    offset = float(np.hypot(*(point - foot))) / half_length
    if offset <= 0.0:
        raise ValueError("source_point must not lie on the line carrying the element")

    # Requiring t = -1 and t = 1 to land on the two element ends fixes the two
    # constants of the change of variable.
    to_far = float(np.arcsinh((1.0 + foot_local) / offset))
    to_near = float(np.arcsinh((1.0 - foot_local) / offset))
    stretch = 0.5 * (to_far + to_near)
    shift = 0.5 * (to_far - to_near)

    argument = stretch * points - shift
    local = foot_local + offset * np.sinh(argument)
    scaled = weights * offset * stretch * np.cosh(argument)
    return np.column_stack([np.asarray(local, dtype=float), np.asarray(scaled, dtype=float)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: whatever the change of variable, the rule must reproduce
        # the length of the reference interval and integrate a linear function
        # over it exactly. Both are closed forms independent of the oracle.
        {
            "setup": """import numpy as np
nodes = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]])
src = np.array([0.2, 2.5e-5])
def summarize(rule):
    return float(np.sum(rule[:, 1]) + 100.0 * np.sum(rule[:, 1] * rule[:, 0])
                 + 10000.0 * np.sum(rule[:, 1] * rule[:, 0] ** 2))
""",
            "call": "summarize(near_singular_rule(src.copy(), nodes.copy(), 40, True))",
            "gold_call": "summarize(_oracle_near_singular_rule(src.copy(), nodes.copy(), 40, True))",
        },
        # --- Pinned: with the change of variable switched off the rule is the
        # plain Gauss-Legendre rule of the reference interval.
        {
            "setup": """import numpy as np
nodes = np.array([[1.0, -1.0], [1.0, -0.5], [1.0, 0.0]])
src = np.array([0.5, -0.5])
def summarize(rule):
    return float(np.sum(np.arange(1.0, 7.0) * rule[:, 0])
                 + 50.0 * np.sum(np.sqrt(np.arange(1.0, 7.0)) * rule[:, 1]))
""",
            "call": "summarize(near_singular_rule(src.copy(), nodes.copy(), 6, False))",
            "gold_call": "summarize(_oracle_near_singular_rule(src.copy(), nodes.copy(), 6, False))",
        },
        # --- Valid: an ultra-close source point above the middle of a long
        # element, the situation the coating creates ---
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
""",
            "call": "float(pin(near_singular_rule(src.copy(), nodes.copy(), 40, True)))",
            "gold_call": "float(pin(_oracle_near_singular_rule(src.copy(), nodes.copy(), 40, True)))",
        },
        # --- Valid: a source point whose perpendicular foot falls far outside
        # the element, as happens for the far end of the same face ---
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
nodes = np.array([[3.6, 0.0], [3.8, 0.0], [4.0, 0.0]])
src = np.array([0.2, 2.5e-5])
""",
            "call": "float(pin(near_singular_rule(src.copy(), nodes.copy(), 40, True)))",
            "gold_call": "float(pin(_oracle_near_singular_rule(src.copy(), nodes.copy(), 40, True)))",
        },
        # --- Valid: a well separated source point, where the change of variable
        # is close to harmless ---
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
nodes = np.array([[0.0, -1.0], [0.2, -1.0], [0.4, -1.0]])
src = np.array([2.0, -0.5])
""",
            "call": "float(pin(near_singular_rule(src.copy(), nodes.copy(), 12, True)))",
            "gold_call": "float(pin(_oracle_near_singular_rule(src.copy(), nodes.copy(), 12, True)))",
        },
        # --- Boundary: an inclined element with a source point almost on it,
        # which drives the stretch constant to its largest values ---
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
nodes = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
src = np.array([0.3, 0.3 + 1.0e-8])
""",
            "call": "float(pin(near_singular_rule(src.copy(), nodes.copy(), 24, True)))",
            "gold_call": "float(pin(_oracle_near_singular_rule(src.copy(), nodes.copy(), 24, True)))",
        },
        # --- Invalid: the source point sits exactly on the element line ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        near_singular_rule(np.array([0.3, 0.0]), np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]), 8, True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_near_singular_rule(np.array([0.3, 0.0]), np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]), 8, True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a middle node that is not the midpoint ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        near_singular_rule(np.array([0.2, 0.1]), np.array([[0.0, 0.0], [0.3, 0.0], [0.4, 0.0]]), 8, True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_near_singular_rule(np.array([0.2, 0.1]), np.array([[0.0, 0.0], [0.3, 0.0], [0.4, 0.0]]), 8, True)
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
