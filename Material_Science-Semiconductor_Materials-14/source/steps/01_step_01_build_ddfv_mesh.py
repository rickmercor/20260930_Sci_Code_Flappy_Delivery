"""
Build the diamond-cell geometry table of the discrete duality finite volume mesh for the distorted triangulation of the unit square.

A discrete duality finite volume discretisation never works on a single mesh. It starts from a primal mesh of polygonal cells, attaches to every primal vertex a dual cell built from the barycentres of the primal cells that surround it, and pairs each primal edge with the dual edge that crosses it. That pair spans a quadrilateral called a diamond cell, whose four corners are the barycentres of the two primal cells sharing the primal edge and the two vertices that are its endpoints. All differential operators of the method are reconstructed on these diamond cells, so the diamond geometry is the only mesh information the rest of the pipeline needs.




Write the primal edge as sigma with unit normal n_KL pointing from cell K to cell L, and the dual edge as sigma_star with unit normal n_KsLs pointing from dual cell K_star to dual cell L_star. The diamond area is half the magnitude of the cross product of its two diagonals, the segment joining the two primal barycentres and the segment joining the two vertices; that form needs no trigonometry and stays exact when the diamond is degenerate. The scalar product of the two normals is stored alongside, because every cross-coupling coefficient assembled later is proportional to it and it vanishes precisely on the orthogonal meshes where the classical scheme already applies.




Boundary edges are treated as degenerate primal cells, so a boundary edge carries its own unknown and its own index, and its barycentre is the edge midpoint. The diamond attached to one collapses from a quadrilateral to a triangle, and the cross-product formula returns the triangle area, so no special case is needed. The orientation convention is that the ordered quadruple of corners must be positively oriented, which is enforced by swapping the two vertices whenever the cross product of the two diagonals is negative.




The primal mesh is a structured triangulation of the unit square that is then deliberately ruined. The square is cut into nx columns and ny rows of equal rectangles, each rectangle is split along the diagonal rising from its lower left to its upper right corner, and every vertex strictly interior to the square is displaced diagonally by a fixed fraction of the rectangle side lengths with a sign that alternates like a checkerboard in the sum of the vertex indices. The boundary vertices are left where they are, so the domain remains exactly the unit square, but the interior becomes strongly obtuse and loses the Delaunay property.

Returns
-------
np.ndarray of shape (n_diamonds, 9), float: the diamond-cell geometry table described above.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def build_ddfv_mesh(nx: int, ny: int, distortion: float) -> np.ndarray:
    """Build the diamond-cell geometry table of the discrete duality mesh.

    The vertex at column i and row j carries index j * (nx + 1) + i. Triangles
    are numbered rectangle by rectangle in that same row-major order, the
    triangle below the splitting diagonal preceding the one above it. Global
    node indices run over primal cells first (one per triangle, in triangle
    order), then boundary primal cells (one per boundary edge, in ascending
    vertex-pair order), then dual cells (one per vertex, in vertex order).
    Diamonds are listed with the interior primal edges first and the boundary
    primal edges last, each group in ascending vertex-pair order, and for an
    interior primal edge K is the lower-numbered and L the higher-numbered of
    the two triangles sharing it.

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
    diamonds : np.ndarray
        Array of shape (n_diamonds, 9). Column 0 is the global index of the
        primal cell K, column 1 that of the primal cell L, columns 2 and 3
        those of the dual cells K_star and L_star, column 4 the diamond area,
        column 5 the primal edge length, column 6 the dual edge length,
        column 7 the scalar product of the two unit normals, and column 8 the
        flag 1.0 when the primal edge lies on the boundary and 0.0 otherwise.

    Raises
    ------
    ValueError
        If nx or ny is not an integer at least 2, distortion is not finite or
        lies outside [0, 0.5), or the requested mesh contains a degenerate
        diamond.
    """
    return diamonds  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_ddfv_mesh(nx: int, ny: int, distortion: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(nx, (int, np.integer)) and not isinstance(nx, bool)
            and int(nx) >= 2):
        raise ValueError("nx must be an integer >= 2")
    if not (isinstance(ny, (int, np.integer)) and not isinstance(ny, bool)
            and int(ny) >= 2):
        raise ValueError("ny must be an integer >= 2")
    if not (isinstance(distortion, (int, float)) and np.isfinite(distortion)
            and 0.0 <= float(distortion) < 0.5):
        raise ValueError("distortion must be a finite number in [0, 0.5)")

    nx, ny, distortion = int(nx), int(ny), float(distortion)

    def _primal_mesh():
        """Vertex coordinates and triangle vertex table of the distorted mesh.

        The unit square is cut into nx by ny equal rectangles, each rectangle
        is split along its lower-left to upper-right diagonal, and every
        strictly interior vertex is displaced by a checkerboard-signed
        diagonal offset.
        """
        dx, dy = 1.0 / nx, 1.0 / ny
        coords = np.zeros(((nx + 1) * (ny + 1), 2), dtype=float)
        for j in range(ny + 1):
            for i in range(nx + 1):
                x, y = i * dx, j * dy
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
        """Map every unordered vertex pair to the triangles carrying it."""
        table = {}
        for t, (a, b, c) in enumerate(cells):
            for u, v in ((a, b), (b, c), (c, a)):
                table.setdefault((min(u, v), max(u, v)), []).append(t)
        return table

    xy, tris = _primal_mesh()
    edges = _edge_table(tris)

    interior = sorted(e for e, ts in edges.items() if len(ts) == 2)
    boundary = sorted(e for e, ts in edges.items() if len(ts) == 1)
    n_tri, n_bnd = len(tris), len(boundary)
    bnd_index = {e: k for k, e in enumerate(boundary)}
    bary = xy[tris].mean(axis=1)

    rows = []
    for edge in interior + boundary:
        ks, ls = edge
        touching = edges[edge]
        cell_k = touching[0]
        x_k = bary[cell_k]

        if len(touching) == 2:
            cell_l = touching[1]
            x_l = bary[touching[1]]
            on_boundary = 0.0
        else:
            cell_l = n_tri + bnd_index[edge]
            x_l = 0.5 * (xy[ks] + xy[ls])
            on_boundary = 1.0

        diag_star = x_l - x_k
        diag = xy[ls] - xy[ks]
        cross = diag_star[0] * diag[1] - diag_star[1] * diag[0]

        if cross < 0.0:
            ks, ls = ls, ks
            diag, cross = -diag, -cross
        if cross == 0.0:
            raise ValueError("degenerate diamond cell with zero area")

        area = 0.5 * abs(cross)
        len_sigma = float(np.hypot(diag[0], diag[1]))
        len_sigma_star = float(np.hypot(diag_star[0], diag_star[1]))

        normal_kl = np.array([-diag[1], diag[0]]) / len_sigma
        if normal_kl @ diag_star < 0.0:
            normal_kl = -normal_kl

        normal_ks = np.array(
            [-diag_star[1], diag_star[0]]
        ) / len_sigma_star
        if normal_ks @ diag < 0.0:
            normal_ks = -normal_ks

        rows.append([
            float(cell_k),
            float(cell_l),
            float(n_tri + n_bnd + ks),
            float(n_tri + n_bnd + ls),
            area,
            len_sigma,
            len_sigma_star,
            float(normal_kl @ normal_ks),
            on_boundary,
        ])

    return np.asarray(rows, dtype=float)

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
            "call": "float(1.0e12 * (_a := np.ravel(build_ddfv_mesh(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_build_ddfv_mesh(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Valid: small anisotropic mesh with a different distortion ---
        {
            "setup": """import numpy as np
nx, ny, distortion = 3, 5, 0.25
""",
            "call": "float(1.0e12 * (_a := np.ravel(build_ddfv_mesh(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_build_ddfv_mesh(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Boundary: no distortion, so every diamond comes from the regular grid ---
        {
            "setup": """import numpy as np
nx, ny, distortion = 4, 4, 0.0
""",
            "call": "float(1.0e12 * (_a := np.ravel(build_ddfv_mesh(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_build_ddfv_mesh(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Edge: smallest admissible mesh at a near-maximal distortion ---
        {
            "setup": """import numpy as np
nx, ny, distortion = 2, 2, 0.49
""",
            "call": "float(1.0e12 * (_a := np.ravel(build_ddfv_mesh(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_build_ddfv_mesh(nx, ny, distortion)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Invalid: distortion at the tangling threshold ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_ddfv_mesh(4, 4, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_ddfv_mesh(4, 4, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: mesh too coarse to have an interior vertex ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_ddfv_mesh(1, 4, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_ddfv_mesh(1, 4, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative distortion ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_ddfv_mesh(4, 4, -0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_ddfv_mesh(4, 4, -0.1)
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
