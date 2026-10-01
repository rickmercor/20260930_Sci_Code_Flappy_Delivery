"""
Evaluate the Virial stress tensor of every cell of the stripe from the vertex forces of the cell's own energy, including the line tension on its free edges.

Treating each cell as a small continuum element, its stress follows from the moments of the point forces acting on its vertices divided by its area.

Returns
-------
np.ndarray: (C, 3) per-cell Virial stresses [sigma_xx, sigma_yy, sigma_xy] including free-edge line tension.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_virial_stresses(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
    kappa: float,
    chi: float,
    line_tension: float,
) -> "np.ndarray":
    """Return the Virial stress components of every cell.

    Cell ``i`` owns the energy ``E_i = (a_i - 1)**2 / 2 + kappa * (p_i - chi)**2
    / 2`` plus ``line_tension`` times the length of its own free edges (the
    free-edge flags of ``compute_cell_geometry``). With ``r_b`` the unwrapped
    positions of its six vertices, its Virial stress is
    ``sigma_i = (1 / a_i) * sum_b r_b (outer) dE_i/dr_b``, where the outer
    product takes the position component first; ``E_i`` does not change
    when the cell is translated, so the origin of ``r_b`` is immaterial.
    Return ``[sigma_xx, sigma_yy, sigma_xy]`` for every cell.

    Parameters
    ----------
    vertices, cells, shifts, period
        Periodic mesh as in ``compute_cell_geometry``.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.
    line_tension : float
        Non-negative line tension of the free edges.

    Returns
    -------
    np.ndarray
        Float array of shape ``(C, 3)``.

    Raises
    ------
    ValueError
        If ``kappa`` or ``chi`` is not a finite positive real number,
        ``line_tension`` is not a finite non-negative real number (booleans
        are rejected), any cell area is not positive, any edge has zero
        length, or ``compute_cell_geometry`` rejects the mesh.
    """
    return stresses

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_virial_stresses(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
    kappa: float,
    chi: float,
    line_tension: float,
) -> "np.ndarray":
    """Reference implementation (pressure plus edge-tension dyads of the cell)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    areas, perimeters, lengths, free = _oracle_compute_cell_geometry(vertices, cells, shifts, period)
    if not (_is_number(kappa) and kappa > 0.0):
        raise ValueError("kappa must be a finite positive number")
    if not (_is_number(chi) and chi > 0.0):
        raise ValueError("chi must be a finite positive number")
    if not (_is_number(line_tension) and line_tension >= 0.0):
        raise ValueError("line_tension must be a finite non-negative number")
    if np.any(areas <= 0.0):
        raise ValueError("every cell must have a positive area")
    if np.any(lengths <= 0.0):
        raise ValueError("every edge must have a positive length")
    polygon = np.asarray(vertices, dtype=float)[np.asarray(cells, dtype=int)]
    polygon[..., 0] += np.asarray(shifts, dtype=int) * float(period)
    edges = np.roll(polygon, -1, axis=1) - polygon

    # The area term contributes (a - 1) a times the identity to sum r (outer) dE/dr,
    # and an edge of tension t contributes t times its dyad divided by its length.
    tension = float(kappa) * (perimeters - float(chi))[:, None] + float(line_tension) * free
    weight = tension / lengths
    xx = np.sum(weight * edges[..., 0] ** 2, axis=1)
    yy = np.sum(weight * edges[..., 1] ** 2, axis=1)
    xy = np.sum(weight * edges[..., 0] * edges[..., 1], axis=1)
    pressure = areas - 1.0
    return np.column_stack([pressure + xx / areas, pressure + yy / areas, xy / areas])

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
        "    a = np.asarray(out, dtype=float)\n"
        "    if a.ndim != 2 or a.shape[1] != 3:\n"
        "        return -1.0\n"
        "    w = np.cos(0.5 * np.arange(a.shape[0]))\n"
        "    return float(w @ a[:, 0] + 0.8 * (w @ a[:, 1]) + 1.3 * (w @ a[:, 2]))\n"
    )
    status = mesh + (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "v, c, s, L = _mesh(2, 2, 0.6)\n"
        "mirrored = v.copy()\n"
        "mirrored[:, 1] *= -1.0\n"
    )
    return [
        {
            "setup": mesh,
            "call": "_signature(compute_virial_stresses(*_mesh(3, 2, 0.6, 1.3, 0.02, 1), 0.16, 3.5, 0.03))",
            "gold_call": "_signature(_oracle_compute_virial_stresses(*_mesh(3, 2, 0.6, 1.3, 0.02, 1), 0.16, 3.5, 0.03))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_virial_stresses(*_mesh(2, 3, 0.58, 1.0, 0.04, 2), 0.35, 2.45, 0.0))",
            "gold_call": "_signature(_oracle_compute_virial_stresses(*_mesh(2, 3, 0.58, 1.0, 0.04, 2), 0.35, 2.45, 0.0))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_virial_stresses(*_mesh(4, 2, 0.6069024253629453), 0.16, 3.5, 0.0))",
            "gold_call": "_signature(_oracle_compute_virial_stresses(*_mesh(4, 2, 0.6069024253629453), 0.16, 3.5, 0.0))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_virial_stresses(*_mesh(1, 2, 0.62, 1.4, 0.03, 9), 0.2, 3.0, 0.1))",
            "gold_call": "_signature(_oracle_compute_virial_stresses(*_mesh(1, 2, 0.62, 1.4, 0.03, 9), 0.2, 3.0, 0.1))",
        },
        {
            "setup": mesh,
            "call": "float(np.sum(compute_virial_stresses(*_mesh(5, 2, 0.6, 1.5), 0.25, 3.3, 0.05)[:, 0]))",
            "gold_call": "float(np.sum(_oracle_compute_virial_stresses(*_mesh(5, 2, 0.6, 1.5), 0.25, 3.3, 0.05)[:, 0]))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_virial_stresses(mirrored, c, s, L, 0.16, 3.5, 0.03))",
            "gold_call": "_status(lambda: _oracle_compute_virial_stresses(mirrored, c, s, L, 0.16, 3.5, 0.03))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_virial_stresses(v, c, s, L, 0.16, float('nan'), 0.03))",
            "gold_call": "_status(lambda: _oracle_compute_virial_stresses(v, c, s, L, 0.16, float('nan'), 0.03))",
        },
    ]
