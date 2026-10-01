"""
Periodic voxel geometry and interface data for the X-FEM discretization.




A cuboid periodic cell is partitioned into voxels and each voxel into six P1 tetrahedra around the body diagonal. The level set at every physical voxel corner is the periodic signed distance to a sphere. A tetrahedron is enriched exactly when its four nodal level-set values contain both signs, and all global periodic nodes incident to those tetrahedra carry enriched displacement degrees of freedom.




Inputs

------

n_voxels : int

    Number of voxels on every cell edge.

radius : float

    Sphere radius in a unit periodic cell.

center : array-like of shape (3,)

    Sphere center in cell coordinates.




Returns

-------

mesh : dict

    Keys nodes, vertices, node_ids, levels, cut, enriched_nodes, n_voxels, radius, and center, with shapes and dtypes specified below.

Returns
-------
dict with float64 arrays nodes (n_voxels**3, 3), vertices (6*n_voxels**3, 4, 3), levels (6*n_voxels**3, 4), and center (3,); integer arrays node_ids (6*n_voxels**3, 4) and enriched_nodes (k,); Boolean array cut (6*n_voxels**3,); native int n_voxels; and native float radius, where k is the number of enriched nodes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_periodic_sphere_mesh(
    n_voxels: int,
    radius: float,
    center: np.ndarray,
) -> dict:
    """Build the periodic tetrahedral mesh and signed-distance interface data.

    Parameters
    ----------
    n_voxels : int
        Number of equal voxels along each axis, at least two.
    radius : float
        Sphere radius, strictly between zero and one half.
    center : np.ndarray
        Sphere center with three coordinates strictly inside the unit cell.

    Returns
    -------
    mesh : dict
        Keys nodes (n_voxels**3, 3) float64, vertices (6*n_voxels**3, 4, 3)
        float64, node_ids (6*n_voxels**3, 4) int, levels (6*n_voxels**3, 4)
        float64, cut (6*n_voxels**3,) bool, enriched_nodes (k,) int, and the
        native n_voxels, radius and center.
    Raises
    ------
    ValueError
        If n_voxels is not an integer of at least two, if radius is not finite
        or not strictly between 0 and 0.5, if center is not a finite array of
        shape (3,), or if any center coordinate lies on or outside the open
        unit cell (0, 1).
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_CUBE_VERTICES = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=int,
)
_CUBE_TETS = np.array(
    [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ],
    dtype=int,
)


def _oracle_build_periodic_sphere_mesh(
    n_voxels: int,
    radius: float,
    center: np.ndarray,
) -> dict:
    """Reference implementation."""
    if not isinstance(n_voxels, (int, np.integer)) or int(n_voxels) < 2:
        raise ValueError("n_voxels must be an integer of at least two")
    n_voxels = int(n_voxels)
    if not np.isfinite(radius) or not 0.0 < float(radius) < 0.5:
        raise ValueError("radius must lie strictly between zero and one half")
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center must be a finite array with shape (3,)")
    if np.any(center <= 0.0) or np.any(center >= 1.0):
        raise ValueError("center must lie strictly inside the unit cell")

    h = 1.0 / n_voxels
    nodes = np.array(
        [
            [i * h, j * h, k * h]
            for i in range(n_voxels)
            for j in range(n_voxels)
            for k in range(n_voxels)
        ],
        dtype=float,
    )
    element_vertices = []
    element_node_ids = []
    element_levels = []
    for i in range(n_voxels):
        for j in range(n_voxels):
            for k in range(n_voxels):
                origin = np.array([i, j, k], dtype=float) * h
                cube_vertices = origin + h * _CUBE_VERTICES
                cube_node_ids = np.array(
                    [
                        np.ravel_multi_index(
                            (
                                (i + dx) % n_voxels,
                                (j + dy) % n_voxels,
                                (k + dz) % n_voxels,
                            ),
                            (n_voxels, n_voxels, n_voxels),
                        )
                        for dx, dy, dz in _CUBE_VERTICES
                    ],
                    dtype=int,
                )
                distances = np.abs(cube_vertices - center)
                distances = np.minimum(distances, 1.0 - distances)
                cube_levels = np.linalg.norm(distances, axis=1) - float(radius)
                for tetrahedron in _CUBE_TETS:
                    element_vertices.append(cube_vertices[tetrahedron])
                    element_node_ids.append(cube_node_ids[tetrahedron])
                    element_levels.append(cube_levels[tetrahedron])

    element_vertices = np.asarray(element_vertices, dtype=float)
    element_node_ids = np.asarray(element_node_ids, dtype=int)
    element_levels = np.asarray(element_levels, dtype=float)
    cut = (np.min(element_levels, axis=1) < 0.0) & (
        np.max(element_levels, axis=1) > 0.0
    )
    enriched_nodes = np.unique(element_node_ids[cut])
    return {
        "nodes": nodes,
        "vertices": element_vertices,
        "node_ids": element_node_ids,
        "levels": element_levels,
        "cut": cut,
        "enriched_nodes": enriched_nodes,
        "n_voxels": n_voxels,
        "radius": float(radius),
        "center": center,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
n_voxels = 3
radius = 0.31
center = np.array([0.43, 0.37, 0.52], dtype=float)
def summarize(result):
    return (
        result["nodes"].shape,
        result["vertices"].shape,
        int(np.sum(result["cut"])),
        result["enriched_nodes"].tolist(),
        round(float(np.min(result["levels"])), 12),
        round(float(np.max(result["levels"])), 12),
    )
""",
            "call": "summarize(build_periodic_sphere_mesh(n_voxels, radius, center))",
            "gold_call": "summarize(_oracle_build_periodic_sphere_mesh(n_voxels, radius, center))",
        },
        {
            "setup": """import numpy as np
n_voxels = 2
radius = 0.25
center = np.array([0.5, 0.5, 0.5], dtype=float)
def summarize(result):
    return (result["nodes"].shape, result["vertices"].shape, int(np.sum(result["cut"])))
""",
            "call": "summarize(build_periodic_sphere_mesh(n_voxels, radius, center))",
            "gold_call": "summarize(_oracle_build_periodic_sphere_mesh(n_voxels, radius, center))",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_periodic_sphere_mesh(1, 0.31, np.array([0.5, 0.5, 0.5]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_build_periodic_sphere_mesh(1, 0.31, np.array([0.5, 0.5, 0.5]))
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
