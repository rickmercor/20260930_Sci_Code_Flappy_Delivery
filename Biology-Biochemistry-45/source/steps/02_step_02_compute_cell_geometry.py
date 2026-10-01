"""
Evaluate the areas, perimeters and edge lengths of every cell of a periodic vertex mesh and flag the edges that lie on a free boundary of the tissue.

In a vertex model each cell is a polygon of shared vertices, and a junction bordered by a single cell is part of the tissue's free edge.

Returns
-------
tuple: (areas (C,), perimeters (C,), edge_lengths (C, 6), free_edges (C, 6) 0/1 int) of the periodic mesh.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_cell_geometry(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
) -> tuple:
    """Return the areas, perimeters, edge lengths and free-edge flags of the cells.

    The mesh follows ``build_stress_free_stripe``: local vertex ``j`` of cell
    ``i`` is at ``vertices[cells[i, j]] + (shifts[i, j] * period, 0)``, and
    local edge ``j`` of cell ``i`` runs from its local vertex ``j`` to local
    vertex ``(j + 1) % 6``. Areas are signed, positive for counter-clockwise
    vertex order. Two cell edges are the same junction when they join the
    same two vertex indices with the same difference of periodic shifts
    between their ends. An edge is free (flag 1) when that junction belongs
    to exactly one cell, and interior (flag 0) when it belongs to two.

    Parameters
    ----------
    vertices : np.ndarray
        Finite float array of shape ``(V, 2)``.
    cells : np.ndarray
        Integer array of shape ``(C, 6)`` with entries in ``[0, V)``.
    shifts : np.ndarray
        Integer array of the same shape as ``cells``.
    period : float
        Positive period along ``x``.

    Returns
    -------
    tuple
        ``(areas, perimeters, edge_lengths, free_edges)``: float arrays of
        shapes ``(C,)``, ``(C,)`` and ``(C, 6)``, and an integer ``(C, 6)``
        array of 0/1 flags.

    Raises
    ------
    ValueError
        If ``vertices`` is not a nonempty finite ``(V, 2)`` array, ``cells``
        is not a nonempty integer ``(C, 6)`` array with entries in
        ``[0, V)``, ``shifts`` is not an integer array of the shape of
        ``cells``, ``period`` is not a finite positive real number, or a
        junction belongs to more than two cells.
    """
    return areas, perimeters, edge_lengths, free_edges

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_cell_geometry(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
) -> tuple:
    """Reference implementation (unwrapped polygons, junction keys by relative shift)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    v = np.asarray(vertices)
    c = np.asarray(cells)
    s = np.asarray(shifts)
    if not (v.ndim == 2 and v.shape[0] >= 1 and v.shape[1] == 2
            and np.issubdtype(v.dtype, np.number) and np.all(np.isfinite(v))):
        raise ValueError("vertices must be a nonempty finite (V, 2) array")
    if not (c.ndim == 2 and c.shape[0] >= 1 and c.shape[1] == 6 and np.issubdtype(c.dtype, np.integer)):
        raise ValueError("cells must be a nonempty integer (C, 6) array")
    if c.min() < 0 or c.max() >= v.shape[0]:
        raise ValueError("cells must index existing vertices")
    if not (s.shape == c.shape and np.issubdtype(s.dtype, np.integer)):
        raise ValueError("shifts must be an integer array shaped like cells")
    if not (_is_number(period) and period > 0.0):
        raise ValueError("period must be a finite positive number")
    v = v.astype(float)
    c = c.astype(int)
    s = s.astype(int)

    polygon = v[c]
    polygon[..., 0] += s * float(period)
    following = np.roll(polygon, -1, axis=1)
    edges = following - polygon
    lengths = np.sqrt(np.sum(edges * edges, axis=2))
    areas = 0.5 * np.sum(polygon[..., 0] * following[..., 1] - following[..., 0] * polygon[..., 1], axis=1)
    perimeters = np.sum(lengths, axis=1)

    # A junction is keyed by its two (vertex, shift relative to the smaller shift)
    # ends in sorted order, so the same junction seen from either cell matches.
    next_c, next_s = np.roll(c, -1, axis=1), np.roll(s, -1, axis=1)
    base = np.minimum(s, next_s)
    span = int(np.max(np.abs(s - next_s))) + 1
    first = c.astype(np.int64) * span + (s - base)
    second = next_c.astype(np.int64) * span + (next_s - base)
    low, high = np.minimum(first, second), np.maximum(first, second)
    keys = (low * (v.shape[0] * span) + high).ravel()
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("a junction belongs to more than two cells")
    free = (counts[inverse] == 1).astype(int).reshape(c.shape)
    return areas, perimeters, lengths, free

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    mesh = (
        "import numpy as np\n"
        "def _mesh(rows, columns, edge, stretch=1.0, jitter=0.0, seed=0):\n"
        "    root3 = np.sqrt(3.0)\n"
        "    period = columns * root3 * edge\n"
        "    offsets = ((0.5 * root3, -0.5), (0.5 * root3, 0.5), (0.0, 1.0),\n"
        "               (-0.5 * root3, 0.5), (-0.5 * root3, -0.5), (0.0, -1.0))\n"
        "    index, vertices, cells, shifts = {}, [], [], []\n"
        "    for row in range(rows):\n"
        "        for column in range(columns):\n"
        "            cell, shift = [], []\n"
        "            for dx, dy in offsets:\n"
        "                x = (column + 0.5 * (row % 2)) * root3 * edge + dx * edge\n"
        "                y = 1.5 * edge * row + dy * edge\n"
        "                s = int(np.floor(x / period + 1e-9))\n"
        "                key = (int(round((x - s * period) / (0.5 * root3 * edge))) % (2 * columns),\n"
        "                       int(round(y / (0.5 * edge))))\n"
        "                if key not in index:\n"
        "                    index[key] = len(vertices)\n"
        "                    vertices.append((x - s * period, y))\n"
        "                cell.append(index[key])\n"
        "                shift.append(s)\n"
        "            cells.append(cell)\n"
        "            shifts.append(shift)\n"
        "    v = np.array(vertices, dtype=float)\n"
        "    v[:, 0] *= stretch\n"
        "    if jitter:\n"
        "        v = v + jitter * np.random.default_rng(seed).standard_normal(v.shape)\n"
        "    return v, np.array(cells, dtype=int), np.array(shifts, dtype=int), float(period * stretch)\n"
        "def _signature(out):\n"
        "    areas, perimeters, lengths, free = (np.asarray(item) for item in out)\n"
        "    if lengths.ndim != 2 or lengths.shape[1] != 6 or free.shape != lengths.shape:\n"
        "        return -1.0\n"
        "    q = np.cos(0.37 * np.arange(lengths.size)).reshape(lengths.shape)\n"
        "    w = np.cos(1.1 * np.arange(areas.size))\n"
        "    return float(w @ areas + 0.3 * (w @ perimeters) + np.sum(q * lengths) + 0.1 * np.sum(q * free))\n"
    )
    status = mesh + (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "v, c, s, L = _mesh(2, 2, 0.6)\n"
    )
    return [
        {
            "setup": mesh,
            "call": "_signature(compute_cell_geometry(*_mesh(3, 2, 0.62)))",
            "gold_call": "_signature(_oracle_compute_cell_geometry(*_mesh(3, 2, 0.62)))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_cell_geometry(*_mesh(4, 3, 0.6, 1.4, 0.03, 3)))",
            "gold_call": "_signature(_oracle_compute_cell_geometry(*_mesh(4, 3, 0.6, 1.4, 0.03, 3)))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_cell_geometry(*_mesh(1, 2, 0.5, 0.8, 0.02, 7)))",
            "gold_call": "_signature(_oracle_compute_cell_geometry(*_mesh(1, 2, 0.5, 0.8, 0.02, 7)))",
        },
        {
            "setup": mesh,
            "call": "float(np.sum(compute_cell_geometry(*_mesh(2, 2, 0.7, 1.2, 0.05, 11))[3]))",
            "gold_call": "float(np.sum(_oracle_compute_cell_geometry(*_mesh(2, 2, 0.7, 1.2, 0.05, 11))[3]))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_cell_geometry(*_mesh(5, 2, 0.58, 1.7, 0.01, 5)))",
            "gold_call": "_signature(_oracle_compute_cell_geometry(*_mesh(5, 2, 0.58, 1.7, 0.01, 5)))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_cell_geometry(v, c, s[:, :5], L))",
            "gold_call": "_status(lambda: _oracle_compute_cell_geometry(v, c, s[:, :5], L))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_cell_geometry(v, c, s, 0.0))",
            "gold_call": "_status(lambda: _oracle_compute_cell_geometry(v, c, s, 0.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_cell_geometry(v, c.astype(float), s, L))",
            "gold_call": "_status(lambda: _oracle_compute_cell_geometry(v, c.astype(float), s, L))",
        },
    ]
