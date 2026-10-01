"""
Construct the deterministic triangulation used by the benchmark. The mesh starts from a uniform Cartesian partition of the unit square. Each cell is split according to the benchmark-specific diagonal rule, followed by one uniform red refinement of every triangle. The implementation is deterministic and returns consistently oriented triangles together with the vertex coordinates.

The Mini finite element method is defined on a triangulation of the domain, with the discrete displacement and pressure spaces constructed over that triangulation. The benchmark replaces the paper's generic quasi-uniform mesh by a deterministic, explicitly defined refinement pattern so that the numerical instance is reproducible without reproducing one of the paper's tabulated experiments. The source paper establishes the use of a triangulation $\mathcal{T}_h$ and corresponding Mini spaces.

Returns
-------
tuple[np.ndarray, np.ndarray], where vertices has shape (N,2) and triangles has shape (T,3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_mesh(nx: int, ny: int, refine: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the deterministic benchmark triangulation.

    The initial mesh is the Cartesian partition of (0,1)^2 into nx by ny
    square cells. Initial vertices use the deterministic numbering

        vid(i, j) = j * (nx + 1) + i,

    with i increasing from left to right and j increasing from bottom to top.
    Initial square cells are visited in increasing j, then increasing i.

    For cell (i, j), use the SW-NE diagonal when

        (i + 3*j) % 5 in {0, 1},

    and use the opposite diagonal otherwise. All triangles are stored with
    counter-clockwise vertex order.

    Each uniform refinement level is a red refinement: every triangle is
    divided into four child triangles by inserting the three edge midpoints.
    Existing vertex indices are preserved. New midpoint vertices are appended
    in the first-encountered order while traversing the triangles, so the
    refinement is deterministic. The four children of each parent are emitted
    in the fixed local order induced by the parent triangle and its three
    edge midpoints, and each child is stored counter-clockwise.

    Parameters
    ----------
    nx : int
        Number of Cartesian cells in the x direction.
    ny : int
        Number of Cartesian cells in the y direction.
    refine : int
        Number of uniform red-refinement levels.

    Returns
    -------
    vertices : np.ndarray
        Array of shape (N, 2) containing vertex coordinates. Vertices are
        indexed according to the deterministic numbering convention above,
        with newly created refinement midpoints appended in first-encounter
        order.
    triangles : np.ndarray
        Integer array of shape (T, 3) containing counter-clockwise triangle
        vertex indices.

    Raises
    ------
    ValueError
        If nx <= 0, ny <= 0, or refine < 0.
    """
    return vertices, triangles

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_mesh(
    nx: int,
    ny: int,
    refine: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation."""
    if not isinstance(nx, (int, np.integer)) or nx < 1:
        raise ValueError("nx must be a positive integer")

    if not isinstance(ny, (int, np.integer)) or ny < 1:
        raise ValueError("ny must be a positive integer")

    if not isinstance(refine, (int, np.integer)) or refine < 0:
        raise ValueError(
            "refine must be a nonnegative integer"
        )

    xs = np.linspace(
        0.0,
        1.0,
        nx + 1,
        dtype=float,
    )

    ys = np.linspace(
        0.0,
        1.0,
        ny + 1,
        dtype=float,
    )

    # Vertex indexing convention:
    # vid(i, j) = j * (nx + 1) + i
    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    # Build vertices in exactly the same ordering as vid().
    xx, yy = np.meshgrid(
        xs,
        ys,
        indexing="xy",
    )

    vertices = np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
        ]
    ).astype(float)

    triangles = []

    # Deterministic diagonal pattern.
    for j in range(ny):
        for i in range(nx):
            sw = vid(i, j)
            se = vid(i + 1, j)
            ne = vid(i + 1, j + 1)
            nw = vid(i, j + 1)

            if (i + 3 * j) % 5 in (0, 1):
                triangles.append(
                    [sw, se, ne]
                )
                triangles.append(
                    [sw, ne, nw]
                )
            else:
                triangles.append(
                    [sw, se, nw]
                )
                triangles.append(
                    [se, ne, nw]
                )

    triangles = np.asarray(
        triangles,
        dtype=np.int64,
    )

    # Uniform red refinement.
    for _ in range(refine):
        edge_mid = {}
        new_triangles = []

        verts = vertices.tolist()

        def midpoint(a: int, b: int) -> int:
            key = (
                min(a, b),
                max(a, b),
            )

            if key not in edge_mid:
                pa = vertices[key[0]]
                pb = vertices[key[1]]

                edge_mid[key] = len(verts)

                verts.append(
                    (
                        0.5 * (pa + pb)
                    ).tolist()
                )

            return edge_mid[key]

        for a, b, c in triangles:
            a = int(a)
            b = int(b)
            c = int(c)

            mab = midpoint(a, b)
            mbc = midpoint(b, c)
            mca = midpoint(c, a)

            new_triangles.extend(
                [
                    [a, mab, mca],
                    [mab, b, mbc],
                    [mca, mbc, c],
                    [mab, mbc, mca],
                ]
            )

        vertices = np.asarray(
            verts,
            dtype=float,
        )

        triangles = np.asarray(
            new_triangles,
            dtype=np.int64,
        )

    return vertices, triangles

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np

def pack_mesh(mesh):
    vertices, triangles = mesh
    return np.concatenate([
        np.array([vertices.shape[0], triangles.shape[0]], dtype=float),
        np.asarray(vertices, dtype=float).ravel(),
        np.asarray(triangles, dtype=float).ravel(),
    ])
""",
            "call": "pack_mesh(build_mesh(2, 2, refine=1))",
            "gold_call": "pack_mesh(_oracle_build_mesh(2, 2, refine=1))",
        },
        {
            "setup": """import numpy as np

def pack_mesh(mesh):
    vertices, triangles = mesh
    return np.concatenate([
        np.array([vertices.shape[0], triangles.shape[0]], dtype=float),
        np.asarray(vertices, dtype=float).ravel(),
        np.asarray(triangles, dtype=float).ravel(),
    ])
""",
            "call": "pack_mesh(build_mesh(1, 1, refine=0))",
            "gold_call": "pack_mesh(_oracle_build_mesh(1, 1, refine=0))",
        },
        {
            "setup": """import numpy as np

def pack_mesh(mesh):
    vertices, triangles = mesh
    return np.concatenate([
        np.array([vertices.shape[0], triangles.shape[0]], dtype=float),
        np.asarray(vertices, dtype=float).ravel(),
        np.asarray(triangles, dtype=float).ravel(),
    ])
""",
            "call": "pack_mesh(build_mesh(1, 2, refine=2))",
            "gold_call": "pack_mesh(_oracle_build_mesh(1, 2, refine=2))",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_mesh(0, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_mesh(0, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
