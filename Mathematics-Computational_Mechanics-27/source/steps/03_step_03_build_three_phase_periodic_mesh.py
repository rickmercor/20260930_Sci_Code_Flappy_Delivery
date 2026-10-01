"""
This step builds the periodic tetrahedral mesh, evaluates the nodal level set on it, and derives from those two the bookkeeping that every later step reads: which elements are cut, which phase lies on the positive side of each cut, which nodes carry enrichment and how many degrees of freedom the enriched system therefore has.

The single-interface assumption is enforced rather than assumed. An element whose nodes include both a matrix node and an inclusion node would need two interfaces inside one tetrahedron, which the subdivision of the next step cannot represent, so such a configuration is rejected as invalid. The smallest absolute nodal level set value is returned as well, because an interface that passes very close to a node is what degrades the conditioning that the internal scaling is later designed to repair.

The cell is divided into equal cubic voxels and each voxel into six linear tetrahedra that share one body diagonal. Node indices are identified across opposite cell faces, so the mesh carries $n^3$ periodic nodes and $6 n^3$ elements. The identification is what makes the standard block of the later preconditioner a circulant convolution, so the mesh and the preconditioner are not independent choices.

Two concentric spherical interfaces separate three phases. A single scalar level set encodes both: it is the signed distance to whichever interface is nearer, taken negative in the coating. With $r = \|x - c\|$ and $r_{mid} = (r_i + r_c) / 2$, it is $L(x) = r - r_c$ where $r \ge r_{mid}$ and $L(x) = r_i - r$ where $r < r_{mid}$, so $L$ is negative in the coating and positive in both the matrix and the inclusion. The discrete interface is the zero set of the linear interpolant of the nodal values of $L$, and an element is cut when its four nodal values do not share a sign.

Because a tetrahedron has diameter $\sqrt{3} h$ while the coating is thicker than that, no element can meet both interfaces. Each cut element therefore carries exactly one interface, and the phase on the positive side of that interface is the matrix if any node is matrix and the inclusion otherwise. A node is enriched when it belongs to at least one cut element.

Returns
-------
dict, the periodic three-phase mesh with its level set, cut flags, enrichment numbering and counts.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_three_phase_periodic_mesh(
    n_voxels: int,
    cell_size: float,
    centre: tuple,
    radius_inclusion: float,
    radius_coating: float,
) -> dict:
    """Build the periodic tetrahedral mesh and the level set data of a coated sphere.

    Parameters
    ----------
    n_voxels : int
        Number of voxels along each cell edge, at least two.
    cell_size : float
        Edge length of the cubic periodic cell, strictly positive.
    centre : tuple
        Three coordinates of the centre of the coated sphere.
    radius_inclusion : float
        Radius of the inner interface.
    radius_coating : float
        Radius of the outer interface.

    Returns
    -------
    dict
        Keys element_vertices, element_nodes, element_levels, is_cut, element_outer_phase,
        enriched_index, n_nodes, n_elements, n_enriched, n_dof and smallest_absolute_level.

    Raises
    ------
    ValueError
        If n_voxels is below two, if cell_size is not strictly positive, if the radii do not satisfy 0 < radius_inclusion < radius_coating, or if any element touches both interfaces, which makes the configuration invalid for single-interface subdivision.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
TETRAHEDRA = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
              (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
MATRIX, COATING, INCLUSION = 0, 1, 2

def _build_mesh(n_voxels, cell_size, centre, radius_inclusion, radius_coating):
    """Periodic tetrahedral mesh with two concentric spherical interfaces."""
    n_voxels = int(n_voxels)
    spacing = float(cell_size) / n_voxels
    centre = np.asarray(centre, dtype=float)
    middle = 0.5 * (radius_inclusion + radius_coating)
    n_nodes = n_voxels ** 3
    index = np.arange(n_voxels)
    grid = np.stack(np.meshgrid(index, index, index, indexing="ij"), axis=-1).reshape(-1, 3)
    origins = grid * spacing
    corner_positions = origins[:, None, :] + spacing * CORNERS[None, :, :]
    corner_ids = ((grid[:, None, :] + CORNERS[None, :, :].astype(int)) % n_voxels)
    corner_ids = (corner_ids[..., 0] * n_voxels * n_voxels
                  + corner_ids[..., 1] * n_voxels + corner_ids[..., 2])
    radius = np.linalg.norm(corner_positions - centre, axis=-1)
    corner_levels = np.where(radius >= middle, radius - radius_coating, radius_inclusion - radius)
    corner_phase = np.where(radius >= radius_coating, MATRIX,
                            np.where(radius >= radius_inclusion, COATING, INCLUSION))
    order = np.array(TETRAHEDRA)
    vertices = corner_positions[:, order, :].reshape(-1, 4, 3)
    nodes = corner_ids[:, order].reshape(-1, 4)
    levels = corner_levels[:, order].reshape(-1, 4)
    phases = corner_phase[:, order].reshape(-1, 4)
    is_cut = (levels.min(axis=1) < 0.0) & (levels.max(axis=1) > 0.0)
    outer = np.where((phases == MATRIX).any(axis=1), MATRIX,
                     np.where((phases == INCLUSION).any(axis=1), INCLUSION, COATING))
    if np.any((phases == MATRIX).any(axis=1) & (phases == INCLUSION).any(axis=1)):
        raise ValueError("an element touches both interfaces; the configuration is invalid")
    enriched_nodes = np.unique(nodes[is_cut])
    enriched_index = -np.ones(n_nodes, dtype=int)
    enriched_index[enriched_nodes] = np.arange(enriched_nodes.size)
    return {
        "element_vertices": vertices,
        "element_nodes": nodes,
        "element_levels": levels,
        "is_cut": is_cut,
        "element_outer_phase": outer,
        "enriched_index": enriched_index,
        "n_nodes": n_nodes,
        "n_elements": vertices.shape[0],
        "n_enriched": int(enriched_nodes.size),
        "n_dof": 3 * n_nodes + 3 * int(enriched_nodes.size),
        "smallest_absolute_level": float(np.abs(levels).min()),
    }


def _oracle_build_three_phase_periodic_mesh(
    n_voxels: int,
    cell_size: float,
    centre: tuple,
    radius_inclusion: float,
    radius_coating: float,
) -> dict:
    """Reference implementation."""
    if int(n_voxels) < 2:
        raise ValueError("n_voxels must be at least two")
    if float(cell_size) <= 0.0:
        raise ValueError("cell_size must be strictly positive")
    if not 0.0 < float(radius_inclusion) < float(radius_coating):
        raise ValueError("the radii must satisfy 0 < radius_inclusion < radius_coating")
    return _build_mesh(int(n_voxels), float(cell_size), centre,
                       float(radius_inclusion), float(radius_coating))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
RC = 2*np.pi
RI = RC * 0.1399225665006822 ** (1.0/3.0)
def summarize(mesh):
    return (mesh["n_elements"], mesh["n_nodes"], int(mesh["is_cut"].sum()),
            mesh["n_enriched"], mesh["n_dof"],
            round(mesh["smallest_absolute_level"], 10),
            int((mesh["element_outer_phase"] == 2).sum()))
""",
            "call": "summarize(build_three_phase_periodic_mesh(16, 16.0, (8.0, 8.0, 8.0), RI, RC))",
            "gold_call": "summarize(_oracle_build_three_phase_periodic_mesh(16, 16.0, (8.0, 8.0, 8.0), RI, RC))",
        },
        {
            "setup": """import numpy as np
def volume(mesh):
    v = mesh["element_vertices"]
    det = np.linalg.det(np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], axis=1))
    return (round(float(np.abs(det).sum() / 6.0), 9),
            int(mesh["element_nodes"].max()), int(mesh["element_nodes"].min()))
""",
            "call": "volume(build_three_phase_periodic_mesh(6, 3.0, (1.5, 1.5, 1.5), 0.5, 1.1))",
            "gold_call": "volume(_oracle_build_three_phase_periodic_mesh(6, 3.0, (1.5, 1.5, 1.5), 0.5, 1.1))",
        },
        {
            "setup": """def run(fn):
    codes = []
    for args in [(1, 1.0, (0.5, 0.5, 0.5), 0.1, 0.2), (6, 3.0, (1.5, 1.5, 1.5), 1.1, 0.5),
                 (4, 3.0, (1.5, 1.5, 1.5), 0.2, 1.2)]:
        try:
            fn(*args)
            codes.append(0)
        except ValueError:
            codes.append(1)
        except Exception:
            codes.append(2)
    return tuple(codes)
""",
            "call": "run(build_three_phase_periodic_mesh)",
            "gold_call": "run(_oracle_build_three_phase_periodic_mesh)",
        },
    ]
