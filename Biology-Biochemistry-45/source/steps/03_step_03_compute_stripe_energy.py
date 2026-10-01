"""
Evaluate the total vertex-model energy of a periodic stripe whose free edges carry a constant line tension, together with its gradient with respect to every vertex position.

Vertex forces follow from the area-perimeter energy of every cell, and a tissue boundary under line tension adds an energy proportional to the length of its free edge.

Returns
-------
tuple: (energy float, gradient (V, 2) float) of the line-tension vertex-model energy.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_stripe_energy(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
    kappa: float,
    chi: float,
    line_tension: float,
) -> tuple:
    """Return the total energy of the stripe and its gradient.

    With the mesh, areas, perimeters, edge lengths and free-edge flags of
    ``compute_cell_geometry``, the total energy is the sum over cells of
    ``(a - 1)**2 / 2 + kappa * (p - chi)**2 / 2`` plus ``line_tension``
    times the summed length of all free edges. The period is held fixed.
    Return the energy and its partial derivatives with respect to the
    stored coordinates of every vertex (a vertex that appears through
    several periodic images collects the derivative of each image).

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
    tuple
        ``(energy, gradient)``: a float and a float array shaped like
        ``vertices``.

    Raises
    ------
    ValueError
        If ``kappa`` or ``chi`` is not a finite positive real number,
        ``line_tension`` is not a finite non-negative real number (booleans
        are rejected), any edge has zero length, or
        ``compute_cell_geometry`` rejects the mesh.
    """
    return energy, gradient

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_stripe_energy(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
    kappa: float,
    chi: float,
    line_tension: float,
) -> tuple:
    """Reference implementation (analytic polygon derivatives gathered onto vertices)."""
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
    if np.any(lengths <= 0.0):
        raise ValueError("every edge must have a positive length")
    v = np.asarray(vertices, dtype=float)
    c = np.asarray(cells, dtype=int)
    polygon = v[c]
    polygon[..., 0] += np.asarray(shifts, dtype=int) * float(period)
    following = np.roll(polygon, -1, axis=1)
    preceding = np.roll(polygon, 1, axis=1)

    pressure = areas - 1.0
    energy = (0.5 * np.sum(pressure ** 2) + 0.5 * float(kappa) * np.sum((perimeters - float(chi)) ** 2)
              + float(line_tension) * np.sum(lengths * free))
    area_gradient = 0.5 * np.stack([following[..., 1] - preceding[..., 1],
                                    preceding[..., 0] - following[..., 0]], axis=2)
    # Each edge carries its cell's perimeter tension plus the line tension if free;
    # it pulls its start forward along the edge and its end backward.
    tension = float(kappa) * (perimeters - float(chi))[:, None] + float(line_tension) * free
    pull = (tension / lengths)[..., None] * (following - polygon)
    per_vertex = pressure[:, None, None] * area_gradient - pull + np.roll(pull, 1, axis=1)
    gradient = np.zeros_like(v)
    np.add.at(gradient, c.ravel(), per_vertex.reshape(-1, 2))
    return float(energy), gradient

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
        "    energy, gradient = out\n"
        "    g = np.asarray(gradient, dtype=float)\n"
        "    if g.ndim != 2 or g.shape[1] != 2:\n"
        "        return -1.0\n"
        "    w = np.cos(0.9 * np.arange(g.shape[0]))\n"
        "    return float(energy + w @ g[:, 0] + 0.7 * (w @ g[:, 1]))\n"
    )
    status = mesh + (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "v, c, s, L = _mesh(2, 2, 0.6)\n"
        "collapsed = v.copy()\n"
        "collapsed[c[0, 1]] = collapsed[c[0, 0]]\n"
    )
    return [
        {
            "setup": mesh,
            "call": "_signature(compute_stripe_energy(*_mesh(3, 2, 0.6, 1.3, 0.02, 1), 0.16, 3.5, 0.03))",
            "gold_call": "_signature(_oracle_compute_stripe_energy(*_mesh(3, 2, 0.6, 1.3, 0.02, 1), 0.16, 3.5, 0.03))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_stripe_energy(*_mesh(2, 3, 0.58, 1.0, 0.04, 2), 0.35, 2.45, 0.0))",
            "gold_call": "_signature(_oracle_compute_stripe_energy(*_mesh(2, 3, 0.58, 1.0, 0.04, 2), 0.35, 2.45, 0.0))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_stripe_energy(*_mesh(1, 2, 0.62, 1.1, 0.03, 5), 0.2, 3.0, 0.1))",
            "gold_call": "_signature(_oracle_compute_stripe_energy(*_mesh(1, 2, 0.62, 1.1, 0.03, 5), 0.2, 3.0, 0.1))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_stripe_energy(*_mesh(5, 2, 0.6069024253629453), 0.16, 3.5, 0.0))",
            "gold_call": "_signature(_oracle_compute_stripe_energy(*_mesh(5, 2, 0.6069024253629453), 0.16, 3.5, 0.0))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_stripe_energy(*_mesh(4, 2, 0.6, 1.5), 0.25, 3.3, 0.05))",
            "gold_call": "_signature(_oracle_compute_stripe_energy(*_mesh(4, 2, 0.6, 1.5), 0.25, 3.3, 0.05))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_stripe_energy(v, c, s, L, 0.16, 3.5, -0.01))",
            "gold_call": "_status(lambda: _oracle_compute_stripe_energy(v, c, s, L, 0.16, 3.5, -0.01))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_stripe_energy(collapsed, c, s, L, 0.16, 3.5, 0.03))",
            "gold_call": "_status(lambda: _oracle_compute_stripe_energy(collapsed, c, s, L, 0.16, 3.5, 0.03))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_stripe_energy(v, c, s, L, True, 3.5, 0.03))",
            "gold_call": "_status(lambda: _oracle_compute_stripe_energy(v, c, s, L, True, 3.5, 0.03))",
        },
    ]
