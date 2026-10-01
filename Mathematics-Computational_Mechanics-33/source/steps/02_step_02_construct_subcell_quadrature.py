"""
Interface-resolving subcell quadrature for a P1 tetrahedron.




The nodal level-set interpolant is planar inside a P1 tetrahedron. Clipping the tetrahedron by its zero plane yields one convex polyhedron per material phase. Each phase polyhedron is tetrahedralized from an interior centroid, and the resulting sub-tetrahedra use the symmetric four-point degree-two rule needed to integrate the quadratic enriched stiffness integrand exactly.




Inputs

------

vertices : np.ndarray of shape (4, 3)

levels : np.ndarray of shape (4,)




Returns

-------

quadrature : dict

    Original-tetrahedron barycentric points, physical weights, and phase signs.

Returns
-------
dict with float64 arrays barycentric (q, 4) and weights (q,), and integer array phases (q,), where q is the number of quadrature points.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_subcell_quadrature(
    vertices: np.ndarray,
    levels: np.ndarray,
) -> dict:
    """Construct exact degree-two quadrature on the two phase subdomains.

    Parameters
    ----------
    vertices : np.ndarray
        Physical tetrahedron vertices with shape (4, 3).
    levels : np.ndarray
        Nodal values of the linear level-set interpolant with shape (4,).

    Returns
    -------
    quadrature : dict
        Arrays named barycentric, weights, and phases.
    Raises
    ------
    ValueError
        If vertices is not shape (4, 3) or levels is not shape (4,), if either
        contains a non-finite value, if the tetrahedron volume is at most
        1e-14 (degenerate), or if any nodal level satisfies |level| <= 1e-12
        (an interface node, excluded by the benchmark).
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_TET_FACES = ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2))

def _unique_rows(rows, tolerance=1e-12):
    unique = []
    for row in rows:
        if not any(np.linalg.norm(row - other) <= tolerance for other in unique):
            unique.append(row)
    return unique


def _clip_face(face, levels, phase, tolerance=1e-13):
    clipped = []
    for index, current in enumerate(face):
        previous = face[index - 1]
        level_previous = float(levels @ previous)
        level_current = float(levels @ current)
        previous_inside = phase * level_previous >= -tolerance
        current_inside = phase * level_current >= -tolerance
        if current_inside != previous_inside:
            fraction = level_previous / (level_previous - level_current)
            clipped.append(previous + fraction * (current - previous))
        if current_inside:
            clipped.append(current)
    return _unique_rows(clipped)


def _interface_polygon(levels):
    points = []
    for first in range(4):
        if abs(levels[first]) <= 1e-13:
            point = np.zeros(4)
            point[first] = 1.0
            points.append(point)
        for second in range(first + 1, 4):
            if levels[first] * levels[second] < 0.0:
                fraction = levels[first] / (levels[first] - levels[second])
                point = np.zeros(4)
                point[first] = 1.0 - fraction
                point[second] = fraction
                points.append(point)
    return _unique_rows(points)


def _order_polygon(barycentric_points, vertices):
    coordinates = np.asarray(barycentric_points) @ vertices
    center = np.mean(coordinates, axis=0)
    first_axis = coordinates[0] - center
    first_axis /= np.linalg.norm(first_axis)
    normal = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
    normal /= np.linalg.norm(normal)
    second_axis = np.cross(normal, first_axis)
    angles = np.arctan2(
        (coordinates - center) @ second_axis,
        (coordinates - center) @ first_axis,
    )
    return [barycentric_points[index] for index in np.argsort(angles)]


def _phase_subtets(vertices, levels, phase):
    boundary_polygons = []
    for face_ids in _TET_FACES:
        face = []
        for node in face_ids:
            point = np.zeros(4)
            point[node] = 1.0
            face.append(point)
        clipped = _clip_face(face, levels, phase)
        if len(clipped) >= 3:
            boundary_polygons.append(clipped)
    interface = _interface_polygon(levels)
    if len(interface) >= 3:
        boundary_polygons.append(_order_polygon(interface, vertices))
    polyhedron_vertices = _unique_rows(
        [point for polygon in boundary_polygons for point in polygon]
    )
    if len(polyhedron_vertices) < 4:
        return []
    center = np.mean(polyhedron_vertices, axis=0)
    subtetrahedra = []
    for polygon in boundary_polygons:
        for index in range(1, len(polygon) - 1):
            barycentric_tet = np.array(
                [center, polygon[0], polygon[index], polygon[index + 1]]
            )
            physical_tet = barycentric_tet @ vertices
            volume = abs(np.linalg.det((physical_tet[1:] - physical_tet[0]).T)) / 6.0
            if volume > 1e-15:
                subtetrahedra.append((barycentric_tet, volume, phase))
    return subtetrahedra


def _oracle_construct_subcell_quadrature(
    vertices: np.ndarray,
    levels: np.ndarray,
) -> dict:
    """Reference implementation."""
    vertices = np.asarray(vertices, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if vertices.shape != (4, 3) or levels.shape != (4,):
        raise ValueError("vertices and levels must have shapes (4, 3) and (4,)")
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(levels)):
        raise ValueError("vertices and levels must be finite")
    volume = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / 6.0
    if volume <= 1e-14:
        raise ValueError("vertices must define a nondegenerate tetrahedron")
    if np.any(np.abs(levels) <= 1e-12):
        raise ValueError("the benchmark convention excludes interface nodes")

    high = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    low = (5.0 - np.sqrt(5.0)) / 20.0
    rule = np.array(
        [
            [high, low, low, low],
            [low, high, low, low],
            [low, low, high, low],
            [low, low, low, high],
        ]
    )
    if np.all(levels > 0.0) or np.all(levels < 0.0):
        subtetrahedra = [(np.eye(4), volume, 1 if np.mean(levels) > 0.0 else -1)]
    else:
        subtetrahedra = _phase_subtets(vertices, levels, -1)
        subtetrahedra += _phase_subtets(vertices, levels, 1)

    barycentric = []
    weights = []
    phases = []
    for barycentric_tet, subvolume, phase in subtetrahedra:
        barycentric.extend(rule @ barycentric_tet)
        weights.extend([subvolume / 4.0] * 4)
        phases.extend([phase] * 4)
    return {
        "barycentric": np.asarray(barycentric, dtype=float),
        "weights": np.asarray(weights, dtype=float),
        "phases": np.asarray(phases, dtype=int),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
levels = np.array([-0.35, 0.65, 0.45, 0.25])
def summarize(result):
    return (
        result["barycentric"].shape,
        round(float(np.sum(result["weights"])), 14),
        round(float(np.sum(result["weights"][result["phases"] < 0])), 14),
        bool(np.allclose(np.sum(result["barycentric"], axis=1), 1.0)),
    )
""",
            "call": "summarize(construct_subcell_quadrature(vertices, levels))",
            "gold_call": "summarize(_oracle_construct_subcell_quadrature(vertices, levels))",
        },
        {
            "setup": """import numpy as np
vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
levels = np.ones(4)
def summarize(result):
    return (result["barycentric"].shape, result["phases"].tolist(), round(float(np.sum(result["weights"])), 14))
""",
            "call": "summarize(construct_subcell_quadrature(vertices, levels))",
            "gold_call": "summarize(_oracle_construct_subcell_quadrature(vertices, levels))",
        },
        {
            "setup": """import numpy as np
vertices = np.zeros((4, 3))
levels = np.array([-1., 1., 1., 1.])
def run_model():
    try:
        construct_subcell_quadrature(vertices, levels)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_construct_subcell_quadrature(vertices, levels)
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
