"""
Construct the global degree-of-freedom maps for the Mini mixed element. The displacement contains continuous P1 vertex degrees of freedom plus one element-local cubic bubble per triangle for each displacement component. The pressure space contains continuous P1 vertex degrees of freedom.

The paper defines the Mini displacement space as a conforming linear finite-element space enriched by the cubic bubble space $B_3$, while the pressure space is continuous piecewise linear.

Returns
-------
dict[str, np.ndarray], containing deterministic scalar/vector DOF maps
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_mini_space(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build Mini mixed finite-element degree-of-freedom maps.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates of shape (N, 2).
    triangles : np.ndarray
        Triangle indices of shape (T, 3).

    Returns
    -------
    maps : dict
        Deterministic global degree-of-freedom and element-connectivity maps
        with the following required keys:

        - ``vertex_count`` : np.ndarray, shape (1,)
            Number of mesh vertices.
        - ``triangle_count`` : np.ndarray, shape (1,)
            Number of refined mesh triangles.
        - ``scalar_displacement_count`` : np.ndarray, shape (1,)
            Number of enriched scalar displacement degrees of freedom, that
            is the continuous P1 vertex DOFs together with the one bubble
            DOF per triangle.
        - ``displacement_count`` : np.ndarray, shape (1,)
            Total number of displacement degrees of freedom for the two
            components, equal to ``2 * scalar_displacement_count``.
        - ``pressure_count`` : np.ndarray, shape (1,)
            Number of continuous scalar P1 pressure degrees of freedom.
        - ``element_scalar`` : np.ndarray, shape (T, 4)
            Local scalar Mini-element DOF map for each triangle, containing
            the three vertex P1 DOFs followed by the triangle bubble DOF.
        - ``element_x`` : np.ndarray, shape (T, 4)
            Global x-component displacement DOF map corresponding to
            ``element_scalar``.
        - ``element_y`` : np.ndarray, shape (T, 4)
            Global y-component displacement DOF map corresponding to
            ``element_scalar``.
    """
    return maps

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_mini_space(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reference implementation."""
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (N,2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (T,3)")
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("mesh must be nonempty")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("triangle indices out of range")

    n_vertices = len(vertices)
    n_triangles = len(triangles)
    scalar_disp = n_vertices + n_triangles

    vx = np.arange(n_vertices, dtype=np.int64)
    bx = n_vertices + np.arange(n_triangles, dtype=np.int64)

    x_dofs = np.concatenate([vx, bx])
    y_dofs = scalar_disp + x_dofs

    element_scalar = np.column_stack([
        triangles,
        n_vertices + np.arange(n_triangles),
    ])

    element_x = element_scalar
    element_y = scalar_disp + element_scalar

    return {
        "vertex_count": np.asarray([n_vertices], dtype=np.int64),
        "triangle_count": np.asarray([n_triangles], dtype=np.int64),
        "scalar_displacement_count": np.asarray([scalar_disp], dtype=np.int64),
        "displacement_count": np.asarray([2 * scalar_disp], dtype=np.int64),
        "pressure_count": np.asarray([n_vertices], dtype=np.int64),
        "element_scalar": element_scalar,
        "element_x": element_x,
        "element_y": element_y,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
v = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
t = np.array([[0,1,2],[0,2,3]], dtype=int)

def pack_space(maps):
    return np.concatenate([
        np.asarray(maps["vertex_count"], dtype=float).ravel(),
        np.asarray(maps["triangle_count"], dtype=float).ravel(),
        np.asarray(maps["scalar_displacement_count"], dtype=float).ravel(),
        np.asarray(maps["displacement_count"], dtype=float).ravel(),
        np.asarray(maps["pressure_count"], dtype=float).ravel(),
        np.asarray(maps["element_scalar"], dtype=float).ravel(),
        np.asarray(maps["element_x"], dtype=float).ravel(),
        np.asarray(maps["element_y"], dtype=float).ravel(),
    ])
""",
            "call": "pack_space(build_mini_space(v, t))",
            "gold_call": "pack_space(_oracle_build_mini_space(v, t))",
        },
        {
            "setup": """import numpy as np
v = np.array([[0.,0.],[1.,0.],[0.,1.]])
t = np.array([[0,1,2]], dtype=int)

def pack_space(maps):
    return np.concatenate([
        np.asarray(maps["vertex_count"], dtype=float).ravel(),
        np.asarray(maps["triangle_count"], dtype=float).ravel(),
        np.asarray(maps["scalar_displacement_count"], dtype=float).ravel(),
        np.asarray(maps["displacement_count"], dtype=float).ravel(),
        np.asarray(maps["pressure_count"], dtype=float).ravel(),
        np.asarray(maps["element_scalar"], dtype=float).ravel(),
        np.asarray(maps["element_x"], dtype=float).ravel(),
        np.asarray(maps["element_y"], dtype=float).ravel(),
    ])
""",
            "call": "pack_space(build_mini_space(v, t))",
            "gold_call": "pack_space(_oracle_build_mini_space(v, t))",
        },
        {
            "setup": """import numpy as np
v = np.array([[0.,0.],[1.,0.],[0.,1.],
              [1.,1.],[2.,0.]], dtype=float)
t = np.array([[0,1,2],[1,3,2],[1,4,3]], dtype=int)

def pack_space(maps):
    return np.concatenate([
        np.asarray(maps["vertex_count"], dtype=float).ravel(),
        np.asarray(maps["triangle_count"], dtype=float).ravel(),
        np.asarray(maps["scalar_displacement_count"], dtype=float).ravel(),
        np.asarray(maps["displacement_count"], dtype=float).ravel(),
        np.asarray(maps["pressure_count"], dtype=float).ravel(),
        np.asarray(maps["element_scalar"], dtype=float).ravel(),
        np.asarray(maps["element_x"], dtype=float).ravel(),
        np.asarray(maps["element_y"], dtype=float).ravel(),
    ])
""",
            "call": "pack_space(build_mini_space(v, t))",
            "gold_call": "pack_space(_oracle_build_mini_space(v, t))",
        },
    ]
