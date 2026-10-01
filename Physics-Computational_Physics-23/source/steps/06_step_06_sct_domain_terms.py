"""
Convert the part of one layer swept by rays from a source point through one boundary element into quadrature points and weights for the domain integral of the fundamental solution against a known density.

A domain integral over a region seen from an interior source point can be written in a scaled coordinate system whose radial variable runs from the source point out to the boundary and whose circumferential variable runs along the boundary, leaving an integration that needs only the boundary discretisation. That change of variables is what keeps a transient model boundary-only, and it also tames the singularity of the two-dimensional fundamental solution at the source point.

Returns
-------
np.ndarray, float, shape (num_radial * m, 3): the domain quadrature point coordinates and the weight of the fundamental solution there.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sct_domain_terms(source_point: "np.ndarray", element_nodes: "np.ndarray",
                     rule: "np.ndarray", num_radial: int) -> "np.ndarray":
    """Build the domain quadrature of one angular sector of a layer.

    The sector is the set of points swept by the segments joining the source
    point to the boundary element. Its points are written as the source point
    plus a radial fraction, running over ``[0, 1]``, of the vector from the
    source point to the boundary, and the element is interpolated with the
    three quadratic shape functions of its local coordinate, whose nodes lie at
    ``-1``, ``0`` and ``1``. The radial direction is integrated with a
    Gauss-Legendre rule of ``num_radial`` points mapped onto ``[0, 1]``.

    Parameters
    ----------
    source_point : np.ndarray
        Float array of shape ``(2,)`` holding the ``(x, y)`` coordinates of the
        source point, which must lie strictly inside the layer.
    element_nodes : np.ndarray
        Float array of shape ``(3, 2)`` holding the first end node, the middle
        node and the second end node of the element, in the anticlockwise
        traversal of the layer outline.
    rule : np.ndarray
        Float array of shape ``(m, 2)`` holding the element local coordinates
        of the circumferential quadrature points and their weights.
    num_radial : int
        Number of Gauss-Legendre points in the radial direction
        (``num_radial >= 1``).

    Returns
    -------
    terms : np.ndarray
        Float array of shape ``(num_radial * m, 3)`` whose columns are the x
        and y coordinates of the quadrature points and the weight to be applied
        there, so that the weighted sum of a density over all sectors of the
        layer is the domain integral of the fundamental solution of the
        two-dimensional Laplace operator, normalised so that its Laplacian is
        minus the Dirac distribution at the source point, against that density.
        The rows are ordered with the radial index varying slowest.

    Raises
    ------
    ValueError
        If ``source_point`` is not a finite real array of shape ``(2,)``, if
        ``element_nodes`` is not a finite real array of shape ``(3, 2)``, if
        ``rule`` is not a finite real array of shape ``(m, 2)`` with
        ``m >= 1``, if ``num_radial`` is not an integer >= 1, or if the source
        point coincides with a circumferential quadrature point.

    """
    return terms  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_sct_domain_terms(source_point: "np.ndarray", element_nodes: "np.ndarray",
                             rule: "np.ndarray", num_radial: int) -> "np.ndarray":
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
    if isinstance(num_radial, bool) or not isinstance(num_radial, (int, np.integer)):
        raise ValueError("num_radial must be an integer")
    if int(num_radial) < 1:
        raise ValueError("num_radial must be an integer >= 1")

    local = table[:, 0]
    circ_weight = table[:, 1]

    shape = np.column_stack([0.5 * local * (local - 1.0),
                             1.0 - local ** 2,
                             0.5 * local * (local + 1.0)])
    slope = np.column_stack([local - 0.5, -2.0 * local, local + 0.5])

    # Boundary points and tangents measured from the source point.
    spoke = shape @ nodes - point
    tangent = slope @ nodes

    # The area swept per unit radial fraction and per unit local coordinate.
    sweep = spoke[:, 0] * tangent[:, 1] - spoke[:, 1] * tangent[:, 0]
    radius = np.hypot(spoke[:, 0], spoke[:, 1])
    if np.any(radius <= 0.0):
        raise ValueError("source_point coincides with a circumferential quadrature point")

    fraction, radial_weight = np.polynomial.legendre.leggauss(int(num_radial))
    fraction = 0.5 * (1.0 + fraction)
    radial_weight = 0.5 * radial_weight

    coords_x = fraction[:, None] * spoke[None, :, 0] + point[0]
    coords_y = fraction[:, None] * spoke[None, :, 1] + point[1]

    # The radial factor of the area element cancels the logarithm's singularity
    # at the source point, leaving a bounded integrand.
    log_distance = np.log(fraction[:, None] * radius[None, :])
    weight = (-0.5 / np.pi) * log_distance * fraction[:, None] * sweep[None, :] \
        * radial_weight[:, None] * circ_weight[None, :]

    return np.column_stack([coords_x.ravel(), coords_y.ravel(), weight.ravel()])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- First-area-moment checks for the four rectangle sectors. Each case
        # invokes the candidate and oracle exactly once. ---
        {
            "setup": """import numpy as np
nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
gx, gw = np.polynomial.legendre.leggauss(12)
rule = np.column_stack([gx, gw])
src = np.array([0.4, 0.3])
def summarize(t):
    r = np.hypot(t[:, 0] - src[0], t[:, 1] - src[1])
    w = t[:, 2] / (-0.5 / np.pi * np.log(r))
    return float(np.sum(w) + 10.0 * np.sum(w * t[:, 0]) + 100.0 * np.sum(w * t[:, 1]))
""",
            "call": "summarize(sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
            "gold_call": "summarize(_oracle_sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
        },
        {
            "setup": """import numpy as np
nodes = np.array([[2.0, 0.0], [2.0, 0.5], [2.0, 1.0]])
gx, gw = np.polynomial.legendre.leggauss(12)
rule = np.column_stack([gx, gw])
src = np.array([0.4, 0.3])
def summarize(t):
    r = np.hypot(t[:, 0] - src[0], t[:, 1] - src[1])
    w = t[:, 2] / (-0.5 / np.pi * np.log(r))
    return float(np.sum(w) + 10.0 * np.sum(w * t[:, 0]) + 100.0 * np.sum(w * t[:, 1]))
""",
            "call": "summarize(sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
            "gold_call": "summarize(_oracle_sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
        },
        {
            "setup": """import numpy as np
nodes = np.array([[2.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
gx, gw = np.polynomial.legendre.leggauss(12)
rule = np.column_stack([gx, gw])
src = np.array([0.4, 0.3])
def summarize(t):
    r = np.hypot(t[:, 0] - src[0], t[:, 1] - src[1])
    w = t[:, 2] / (-0.5 / np.pi * np.log(r))
    return float(np.sum(w) + 10.0 * np.sum(w * t[:, 0]) + 100.0 * np.sum(w * t[:, 1]))
""",
            "call": "summarize(sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
            "gold_call": "summarize(_oracle_sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
        },
        {
            "setup": """import numpy as np
nodes = np.array([[0.0, 1.0], [0.0, 0.5], [0.0, 0.0]])
gx, gw = np.polynomial.legendre.leggauss(12)
rule = np.column_stack([gx, gw])
src = np.array([0.4, 0.3])
def summarize(t):
    r = np.hypot(t[:, 0] - src[0], t[:, 1] - src[1])
    w = t[:, 2] / (-0.5 / np.pi * np.log(r))
    return float(np.sum(w) + 10.0 * np.sum(w * t[:, 0]) + 100.0 * np.sum(w * t[:, 1]))
""",
            "call": "summarize(sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
            "gold_call": "summarize(_oracle_sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
        },
        # --- Second-area-moment checks for the four rectangle sectors. ---
        {
            "setup": """import numpy as np
nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
gx, gw = np.polynomial.legendre.leggauss(12)
rule = np.column_stack([gx, gw])
src = np.array([1.3, 0.8])
def summarize(t):
    r = np.hypot(t[:, 0] - src[0], t[:, 1] - src[1])
    w = t[:, 2] / (-0.5 / np.pi * np.log(r))
    return float(np.sum(w * t[:, 0] ** 2) + 25.0 * np.sum(w * t[:, 1] ** 2))
""",
            "call": "summarize(sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
            "gold_call": "summarize(_oracle_sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
        },
        {
            "setup": """import numpy as np
nodes = np.array([[2.0, 0.0], [2.0, 0.5], [2.0, 1.0]])
gx, gw = np.polynomial.legendre.leggauss(12)
rule = np.column_stack([gx, gw])
src = np.array([1.3, 0.8])
def summarize(t):
    r = np.hypot(t[:, 0] - src[0], t[:, 1] - src[1])
    w = t[:, 2] / (-0.5 / np.pi * np.log(r))
    return float(np.sum(w * t[:, 0] ** 2) + 25.0 * np.sum(w * t[:, 1] ** 2))
""",
            "call": "summarize(sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
            "gold_call": "summarize(_oracle_sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
        },
        {
            "setup": """import numpy as np
nodes = np.array([[2.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
gx, gw = np.polynomial.legendre.leggauss(12)
rule = np.column_stack([gx, gw])
src = np.array([1.3, 0.8])
def summarize(t):
    r = np.hypot(t[:, 0] - src[0], t[:, 1] - src[1])
    w = t[:, 2] / (-0.5 / np.pi * np.log(r))
    return float(np.sum(w * t[:, 0] ** 2) + 25.0 * np.sum(w * t[:, 1] ** 2))
""",
            "call": "summarize(sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
            "gold_call": "summarize(_oracle_sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
        },
        {
            "setup": """import numpy as np
nodes = np.array([[0.0, 1.0], [0.0, 0.5], [0.0, 0.0]])
gx, gw = np.polynomial.legendre.leggauss(12)
rule = np.column_stack([gx, gw])
src = np.array([1.3, 0.8])
def summarize(t):
    r = np.hypot(t[:, 0] - src[0], t[:, 1] - src[1])
    w = t[:, 2] / (-0.5 / np.pi * np.log(r))
    return float(np.sum(w * t[:, 0] ** 2) + 25.0 * np.sum(w * t[:, 1] ** 2))
""",
            "call": "summarize(sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
            "gold_call": "summarize(_oracle_sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 6))",
        },
        # --- Valid: a sector of the substrate, integrated with a plain rule ---
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
gx, gw = np.polynomial.legendre.leggauss(8)
rule = np.column_stack([gx, gw])
""",
            "call": "float(pin(sct_domain_terms(np.array([1.75, -0.5]), nodes.copy(), rule.copy(), 6)))",
            "gold_call": "float(pin(_oracle_sct_domain_terms(np.array([1.75, -0.5]), nodes.copy(), rule.copy(), 6)))",
        },
        # --- Valid: a sector of the ultra-thin coating, where the sweep factor
        # is tiny and the circumferential rule is strongly clustered ---
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
d = 2.5e-5 / 0.2
a = np.arcsinh(1.0 / d)
gx, gw = np.polynomial.legendre.leggauss(40)
s = a * gx
rule = np.column_stack([d * np.sinh(s), gw * d * a * np.cosh(s)])
""",
            "call": "float(pin(sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 16)))",
            "gold_call": "float(pin(_oracle_sct_domain_terms(src.copy(), nodes.copy(), rule.copy(), 16)))",
        },
        # --- Boundary: one radial point and one circumferential point, which
        # isolates a single weight. The element is the right face of the
        # substrate outline, traversed anticlockwise as the docstring requires. ---
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
nodes = np.array([[4.0, -1.0], [4.0, -0.5], [4.0, 0.0]])
""",
            "call": "float(pin(sct_domain_terms(np.array([2.0, -0.5]), nodes.copy(), np.array([[0.4, 1.3]]), 1)))",
            "gold_call": ("float(pin(_oracle_sct_domain_terms(np.array([2.0, -0.5]), nodes.copy(),"
                          " np.array([[0.4, 1.3]]), 1)))"),
        },
        # --- Edge: a curved element, where the sweep factor varies along the
        # circumferential coordinate ---
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
nodes = np.array([[1.0, 0.0], [0.6, 0.6], [0.0, 1.0]])
gx, gw = np.polynomial.legendre.leggauss(10)
rule = np.column_stack([gx, gw])
""",
            "call": "float(pin(sct_domain_terms(np.array([0.1, 0.05]), nodes.copy(), rule.copy(), 8)))",
            "gold_call": "float(pin(_oracle_sct_domain_terms(np.array([0.1, 0.05]), nodes.copy(), rule.copy(), 8)))",
        },
        # --- Invalid: a zero radial point count ---
        {
            "setup": """import numpy as np
nodes = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]])
rule = np.array([[0.0, 2.0]])
def run_model():
    try:
        sct_domain_terms(np.array([0.2, 0.1]), nodes.copy(), rule.copy(), 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sct_domain_terms(np.array([0.2, 0.1]), nodes.copy(), rule.copy(), 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: the source point lands on the boundary quadrature point ---
        {
            "setup": """import numpy as np
nodes = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]])
rule = np.array([[0.0, 2.0]])
def run_model():
    try:
        sct_domain_terms(np.array([0.2, 0.0]), nodes.copy(), rule.copy(), 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sct_domain_terms(np.array([0.2, 0.0]), nodes.copy(), rule.copy(), 4)
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
