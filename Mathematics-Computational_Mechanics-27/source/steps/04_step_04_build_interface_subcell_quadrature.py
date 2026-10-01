"""
This step returns the quadrature of one tetrahedron: the barycentric coordinates of its points, their weights, and the side of the discrete interface on which each point lies. The points are reported in barycentric rather than Cartesian coordinates because the shape functions and the enrichment are both evaluated from them, and because the caller already holds the vertices.

The side label is the reason the rule is built at all. It is what allows the later assembly to give a point the coating stiffness or the positive-side stiffness without ever asking for the true spherical radius, so the material data seen by the quadrature is consistent with the linearised interface that the subdivision has just resolved.

A vanishing nodal level set value is rejected. It leaves a sub-tetrahedron of zero volume and a crossing point that coincides with a vertex, so the decomposition is degenerate rather than merely awkward.

The enrichment used for weak discontinuities makes the strain of the enriched basis functions jump across the discrete interface, so the element integrand is discontinuous and a rule applied to the whole tetrahedron is wrong. Splitting the tetrahedron into sub-tetrahedra that resolve the plane $L_h = 0$ restores a smooth integrand on each piece, and the element integrals are then exact because the integrand is quadratic there.

The decomposition is the minimal one. When one vertex is separated from the other three the tetrahedron splits into a corner tetrahedron and a prism, that is one plus three sub-tetrahedra. When two vertices are separated from the other two it splits into two prisms, that is six sub-tetrahedra. A prism with matched triangular faces $(t_0, t_1, t_2)$ and $(u_0, u_1, u_2)$ is split as $(t_0, t_1, t_2, u_0)$, $(t_1, t_2, u_0, u_1)$ and $(t_2, u_0, u_1, u_2)$. An uncut tetrahedron is treated as a single sub-tetrahedron, so every element receives a rule of the same construction.

Each sub-tetrahedron carries the symmetric four-point rule that is exact for polynomials of degree two: the four barycentric permutations of $(a, b, b, b)$ with $a = (5 + 3 \sqrt{5}) / 20$ and $b = (5 - \sqrt{5}) / 20$, each weighted by one quarter of the sub-tetrahedron volume.

Returns
-------
dict, the interface subcell quadrature points, their weights and the interface side of each.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_interface_subcell_quadrature(vertices, levels) -> dict:
    """Build the interface-resolving quadrature of one linear tetrahedron.

    Parameters
    ----------
    vertices : array_like
        Array of shape (4, 3) holding the tetrahedron vertices.
    levels : array_like
        Array of shape (4,) holding the nodal level set values.

    Returns
    -------
    dict
        Keys barycentric of shape (q, 4), weights of shape (q,) and side of shape (q,).

    Raises
    ------
    ValueError
        If vertices does not have shape (4, 3) or levels does not have shape (4,), or if a nodal level set value vanishes, which leaves the interface unresolved.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_RULE_A = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
_RULE_B = (5.0 - np.sqrt(5.0)) / 20.0
RULE = np.array([[_RULE_A, _RULE_B, _RULE_B, _RULE_B],
                 [_RULE_B, _RULE_A, _RULE_B, _RULE_B],
                 [_RULE_B, _RULE_B, _RULE_A, _RULE_B],
                 [_RULE_B, _RULE_B, _RULE_B, _RULE_A]])

def _split_prism(lower, upper):
    """Three tetrahedra covering the prism with matched triangles lower and upper."""
    return [(lower[0], lower[1], lower[2], upper[0]),
            (lower[1], lower[2], upper[0], upper[1]),
            (lower[2], upper[0], upper[1], upper[2])]

def _minimal_subcells(levels):
    """Minimal barycentric decomposition of a tetrahedron cut by the plane L_h = 0.

    Returns a list of pairs (barycentric vertices of a sub-tetrahedron, side), where side
    is -1 where the interpolated level set is negative and +1 where it is positive. An
    uncut tetrahedron is returned as one sub-tetrahedron.
    """
    levels = np.asarray(levels, dtype=float)
    negative = [i for i in range(4) if levels[i] < 0.0]
    positive = [i for i in range(4) if levels[i] > 0.0]
    if len(negative) + len(positive) != 4:
        raise ValueError("a nodal level set value vanishes; the interface is not resolved")
    if not negative or not positive:
        return [(np.eye(4), -1 if not positive else 1)]

    def _vertex(index):
        point = np.zeros(4)
        point[index] = 1.0
        return point

    def _crossing(i, j):
        weight = levels[i] / (levels[i] - levels[j])
        point = np.zeros(4)
        point[i] = 1.0 - weight
        point[j] = weight
        return point

    cells = []
    if len(negative) == 1 or len(positive) == 1:
        lone = negative[0] if len(negative) == 1 else positive[0]
        others = positive if len(negative) == 1 else negative
        lone_side = -1 if len(negative) == 1 else 1
        cuts = [_crossing(lone, other) for other in others]
        cells.append((np.array([_vertex(lone), cuts[0], cuts[1], cuts[2]]), lone_side))
        for cell in _split_prism(cuts, [_vertex(other) for other in others]):
            cells.append((np.array(cell), -lone_side))
    else:
        first, second = negative
        third, fourth = positive
        cut_13, cut_14 = _crossing(first, third), _crossing(first, fourth)
        cut_23, cut_24 = _crossing(second, third), _crossing(second, fourth)
        for cell in _split_prism([_vertex(first), cut_13, cut_14],
                                 [_vertex(second), cut_23, cut_24]):
            cells.append((np.array(cell), -1))
        for cell in _split_prism([_vertex(third), cut_13, cut_23],
                                 [_vertex(fourth), cut_14, cut_24]):
            cells.append((np.array(cell), 1))
    return cells

def _element_quadrature(vertices, levels):
    """Barycentric quadrature points, weights and interface sides for one tetrahedron."""
    barycentric, weights, sides = [], [], []
    for cell, side in _minimal_subcells(levels):
        physical = cell @ vertices
        volume = abs(np.linalg.det((physical[1:] - physical[0]).T)) / 6.0
        barycentric.append(RULE @ cell)
        weights.extend([volume / 4.0] * 4)
        sides.extend([side] * 4)
    return np.vstack(barycentric), np.array(weights), np.array(sides, dtype=int)


def _oracle_build_interface_subcell_quadrature(vertices, levels) -> dict:
    """Reference implementation."""
    vertices = np.asarray(vertices, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if vertices.shape != (4, 3) or levels.shape != (4,):
        raise ValueError("vertices must have shape (4, 3) and levels shape (4,)")
    barycentric, weights, sides = _element_quadrature(vertices, levels)
    return {"barycentric": barycentric, "weights": weights, "side": sides}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
def summarize(data, v):
    w = data["weights"]
    centroid = data["barycentric"] @ v
    return (len(w), round(float(w.sum()), 12),
            round(float(w[data["side"] < 0].sum()), 12),
            round(float((w[:, None] * centroid).sum()), 12))
""",
            "call": "summarize(build_interface_subcell_quadrature(V, np.array([-0.4, 0.3, 0.6, 0.9])), V)",
            "gold_call": "summarize(_oracle_build_interface_subcell_quadrature(V, np.array([-0.4, 0.3, 0.6, 0.9])), V)",
        },
        {
            "setup": """import numpy as np
V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
def summarize(data, v):
    w = data["weights"]
    centroid = data["barycentric"] @ v
    return (len(w), round(float(w.sum()), 12),
            round(float(w[data["side"] < 0].sum()), 12),
            round(float((w[:, None] * centroid).sum()), 12))
""",
            "call": "(summarize(build_interface_subcell_quadrature(V, np.array([-0.4, -0.2, 0.6, 0.9])), V), summarize(build_interface_subcell_quadrature(V, np.array([0.4, 0.2, 0.6, 0.9])), V))",
            "gold_call": "(summarize(_oracle_build_interface_subcell_quadrature(V, np.array([-0.4, -0.2, 0.6, 0.9])), V), summarize(_oracle_build_interface_subcell_quadrature(V, np.array([0.4, 0.2, 0.6, 0.9])), V))",
        },
        {
            "setup": """import numpy as np
V = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]])
def exactness(fn):
    # the rule is exact for quadratic integrands on each sub-tetrahedron, so the two
    # decompositions of the same tetrahedron must integrate x*y + z**2 identically
    out = []
    for levels in [np.array([1.0, 1.0, 1.0, 1.0]), np.array([-1.0, 0.5, 0.5, 0.5])]:
        d = fn(V, levels)
        p = d["barycentric"] @ V
        out.append(round(float(np.sum(d["weights"] * (p[:, 0] * p[:, 1] + p[:, 2] ** 2))), 10))
    return tuple(out)
""",
            "call": "exactness(build_interface_subcell_quadrature)",
            "gold_call": "exactness(_oracle_build_interface_subcell_quadrature)",
        },
    ]
