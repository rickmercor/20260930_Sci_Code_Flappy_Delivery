"""
Build the node table of the discrete duality mesh, holding the coordinate, the control volume measure, the equation class and the contact tag of every unknown.

A discrete duality finite volume scheme carries one unknown per primal cell, one per boundary edge and one per primal vertex, and each of those unknowns satisfies a different equation depending on where it sits. The node table records, for every unknown, where it is, how large its control volume is, which equation governs it, and whether it belongs to a contact.




The control volume measures come from two partitions of the same domain. The primal cells are the triangles, so their measures are the triangle areas. The dual cells are less obvious, because a dual cell at an interior vertex is the polygon through the barycentres of the surrounding triangles, while a dual cell at a boundary vertex must also be closed off by the vertex itself and by the midpoints of its two boundary edges, otherwise the dual mesh would not tile the domain. Both cases are obtained without any polygon construction by observing that the diagonal of a diamond joining the two primal barycentres separates its two dual corners, so each diamond contributes one triangle to each of the two dual cells it touches. Summing those contributions over all diamonds recovers the dual measures exactly, and the primal and the dual measures then both sum to the domain area, which is a useful check that the diamond table and the node table agree.




The equation classes follow the structure of the mixed boundary condition. Interior primal cells and interior dual cells carry the ordinary balance equations. Primal and dual nodes lying on an Ohmic contact are Dirichlet nodes and carry prescribed values instead. Boundary edges that are not on a contact are contact-free, and the equation attached to them is the homogeneous Neumann condition, that is, the vanishing of the normal flux on the single boundary diamond they belong to; the dual nodes on such edges remain genuine unknowns and keep their balance equations. A vertex lying where a contact meets a contact-free edge belongs to both boundary portions, and the convention that removes the ambiguity is that the Dirichlet set wins, so a corner vertex is a Dirichlet node.




The classes are encoded as small integers so that the table stays a plain numeric array: 0 for an interior primal cell, 1 for a primal cell on a Dirichlet contact, 2 for a contact-free boundary edge, 3 for an interior dual cell, 4 for a dual cell on a Dirichlet contact, and 5 for a dual cell on a contact-free part of the boundary. The contact tag is 0 away from any contact, 1 on the grounded cathode along y equal to zero, and 2 on the biased anode along y equal to one. Because the boundary vertices of the primal mesh are never displaced, the coordinates on the four sides of the square are exact, so the classification by coordinate comparison is unambiguous rather than a matter of tolerance.

Returns
-------
np.ndarray of shape (n_nodes, 5), float: coordinate, control volume measure, equation class code and contact tag of every unknown.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def build_ddfv_node_table(nx: int, ny: int, distortion: float) -> np.ndarray:
    """Build the node table of the discrete duality mesh.

    The global node ordering matches build_ddfv_mesh. The vertex at column i
    and row j carries index j * (nx + 1) + i, triangles are numbered rectangle
    by rectangle in that same row-major order with the triangle below the
    splitting diagonal first, and the node indices then run over primal cells
    in triangle order, then boundary primal cells in ascending vertex-pair
    order, then dual cells in vertex order.

    Parameters
    ----------
    nx : int
        Number of rectangle columns across the unit square (nx >= 2).
    ny : int
        Number of rectangle rows up the unit square (ny >= 2).
    distortion : float
        Interior-vertex displacement as a fraction of the rectangle side
        lengths, 0 <= distortion < 0.5.

    Returns
    -------
    nodes : np.ndarray
        Array of shape (n_nodes, 5). Columns 0 and 1 hold the node
        coordinate, column 2 the control volume measure (zero for a boundary
        edge, whose cell is degenerate), column 3 the equation class code in
        0 to 5, and column 4 the contact tag, 0 away from a contact, 1 on the
        cathode at y equal to zero and 2 on the anode at y equal to one.

    Raises
    ------
    ValueError
        If nx or ny is not an integer at least 2, or distortion is not finite
        or lies outside [0, 0.5).
    """
    return nodes  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_ddfv_node_table(
    nx: int,
    ny: int,
    distortion: float,
) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (
        isinstance(nx, (int, np.integer))
        and not isinstance(nx, bool)
        and int(nx) >= 2
    ):
        raise ValueError("nx must be an integer >= 2")

    if not (
        isinstance(ny, (int, np.integer))
        and not isinstance(ny, bool)
        and int(ny) >= 2
    ):
        raise ValueError("ny must be an integer >= 2")

    if not (
        isinstance(distortion, (int, float))
        and np.isfinite(distortion)
        and 0.0 <= float(distortion) < 0.5
    ):
        raise ValueError(
            "distortion must be a finite number in [0, 0.5)"
        )

    nx = int(nx)
    ny = int(ny)
    distortion = float(distortion)

    def _primal_mesh():
        """Vertex coordinates and triangle table of the distorted mesh."""
        dx = 1.0 / nx
        dy = 1.0 / ny
        coords = np.zeros(
            ((nx + 1) * (ny + 1), 2),
            dtype=float,
        )

        for j in range(ny + 1):
            for i in range(nx + 1):
                x = i * dx
                y = j * dy

                if 0 < i < nx and 0 < j < ny:
                    sign = 1.0 if (i + j) % 2 == 0 else -1.0
                    x += distortion * dx * sign
                    y += distortion * dy * sign

                coords[j * (nx + 1) + i] = (x, y)

        cells = []
        for j in range(ny):
            for i in range(nx):
                v00 = j * (nx + 1) + i
                v01 = (j + 1) * (nx + 1) + i
                cells.append((v00, v00 + 1, v01 + 1))
                cells.append((v00, v01 + 1, v01))

        return coords, np.asarray(cells, dtype=int)

    def _edge_table(cells):
        """Map every vertex pair to the triangles carrying it."""
        table = {}

        for t, (a, b, c) in enumerate(cells):
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                table.setdefault(key, []).append(t)

        return table

    def _triangle_area(p, q, r):
        """Return the unsigned area of a triangle."""
        return 0.5 * abs(
            (q[0] - p[0]) * (r[1] - p[1])
            - (q[1] - p[1]) * (r[0] - p[0])
        )

    xy, tris = _primal_mesh()
    edges = _edge_table(tris)

    interior = sorted(
        edge for edge, touching in edges.items()
        if len(touching) == 2
    )
    boundary = sorted(
        edge for edge, touching in edges.items()
        if len(touching) == 1
    )

    n_tri = len(tris)
    n_bnd = len(boundary)
    n_vert = len(xy)
    n_nodes = n_tri + n_bnd + n_vert

    bary = xy[tris].mean(axis=1)
    bnd_index = {
        edge: index
        for index, edge in enumerate(boundary)
    }

    nodes = np.zeros((n_nodes, 5), dtype=float)
    nodes[:n_tri, :2] = bary

    for edge, index in bnd_index.items():
        nodes[n_tri + index, :2] = (
            0.5 * (xy[edge[0]] + xy[edge[1]])
        )

    nodes[n_tri + n_bnd:, :2] = xy

    # Primal measures are triangle areas.
    for t, (a, b, c) in enumerate(tris):
        nodes[t, 2] = _triangle_area(
            xy[a],
            xy[b],
            xy[c],
        )

    # Dual measures accumulate one triangle per incident diamond.
    for edge in interior + boundary:
        ks, ls = edge
        touching = edges[edge]
        x_k = bary[touching[0]]

        if len(touching) == 2:
            x_l = bary[touching[1]]
        else:
            x_l = 0.5 * (xy[ks] + xy[ls])

        nodes[n_tri + n_bnd + ks, 2] += _triangle_area(
            x_k,
            xy[ks],
            x_l,
        )
        nodes[n_tri + n_bnd + ls, 2] += _triangle_area(
            x_k,
            x_l,
            xy[ls],
        )

    # Equation classes and contact tags.
    for i in range(n_tri, n_tri + n_bnd):
        x, y = nodes[i, 0], nodes[i, 1]

        if y == 0.0:
            nodes[i, 3] = 1.0
            nodes[i, 4] = 1.0
        elif y == 1.0:
            nodes[i, 3] = 1.0
            nodes[i, 4] = 2.0
        else:
            nodes[i, 3] = 2.0

    for i in range(n_tri + n_bnd, n_nodes):
        x, y = nodes[i, 0], nodes[i, 1]

        if y == 0.0:
            nodes[i, 3] = 4.0
            nodes[i, 4] = 1.0
        elif y == 1.0:
            nodes[i, 3] = 4.0
            nodes[i, 4] = 2.0
        elif x == 0.0 or x == 1.0:
            nodes[i, 3] = 5.0
        else:
            nodes[i, 3] = 3.0

    return nodes

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark mesh (normal scenario) ---
        {
            "setup": """import numpy as np
nx, ny, distortion = 6, 12, 0.40
""",
            "call": "float(1.0e12 * (_a := np.ravel(build_ddfv_node_table(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_build_ddfv_node_table(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Valid: anisotropic mesh, so primal and dual measures differ strongly ---
        {
            "setup": """import numpy as np
nx, ny, distortion = 3, 8, 0.30
""",
            "call": "float(1.0e12 * (_a := np.ravel(build_ddfv_node_table(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_build_ddfv_node_table(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Boundary: undistorted mesh, where every interior vertex keeps its grid position ---
        {
            "setup": """import numpy as np
nx, ny, distortion = 4, 4, 0.0
""",
            "call": "float(1.0e12 * (_a := np.ravel(build_ddfv_node_table(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_build_ddfv_node_table(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Edge: even row count places vertices exactly on the metallurgical junction ---
        {
            "setup": """import numpy as np
nx, ny, distortion = 2, 2, 0.45
""",
            "call": "float(1.0e12 * (_a := np.ravel(build_ddfv_node_table(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_build_ddfv_node_table(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Invalid: distortion outside the admissible interval ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_ddfv_node_table(4, 4, 0.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_ddfv_node_table(4, 4, 0.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-integer row count ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_ddfv_node_table(4, 4.5, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_ddfv_node_table(4, 4.5, 0.4)
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
