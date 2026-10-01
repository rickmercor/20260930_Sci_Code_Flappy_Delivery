"""
Resolve the tensile load that the periodic stripe transmits along the load into the contributions of its cell rows, using the cells' Virial stresses.

For a tissue that is periodic along the load, the area-weighted Virial stress of its cells divided by the period gives the force it transmits, so each row of cells carries an identifiable share.

Returns
-------
np.ndarray: (n_rows,) axial load carried by each row; their sum is the stripe's tensile load.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_row_loads(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
    kappa: float,
    chi: float,
    line_tension: float,
    n_rows: int,
) -> "np.ndarray":
    """Return the axial load carried by each row of cells.

    The cells are listed row by row as in ``build_stress_free_stripe``, with
    the same number of cells in each of the ``n_rows`` rows. The load of row
    ``r`` is the sum over its cells of ``a_i * sigma_xx_i`` divided by
    ``period``, where ``a_i`` is the cell area of ``compute_cell_geometry``
    and ``sigma_xx_i`` the first component returned by
    ``compute_virial_stresses``. The row loads add up to the derivative of
    the total energy of ``compute_stripe_energy`` with respect to the period
    when every vertex ``x`` coordinate is scaled in proportion to it.

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
    n_rows : int
        Number of cell rows, at least 1, dividing the number of cells.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_rows,)``, row 0 first.

    Raises
    ------
    ValueError
        If ``n_rows`` is not an integer of at least 1 that divides the number
        of cells (booleans are rejected), or if ``compute_virial_stresses``
        rejects its input.
    """
    return row_loads

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_row_loads(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
    kappa: float,
    chi: float,
    line_tension: float,
    n_rows: int,
) -> "np.ndarray":
    """Reference implementation (row sums of area-weighted Virial stresses)."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    stresses = _oracle_compute_virial_stresses(vertices, cells, shifts, period, kappa, chi, line_tension)
    areas = _oracle_compute_cell_geometry(vertices, cells, shifts, period)[0]
    count = stresses.shape[0]
    if not (_is_integer(n_rows) and n_rows >= 1 and count % n_rows == 0):
        raise ValueError("n_rows must be a positive integer dividing the number of cells")
    # With periodic images unwrapped, the virial of all cells equals the period times
    # the transmitted force, so each row's share is its virial over the period.
    moments = areas * stresses[:, 0]
    return moments.reshape(int(n_rows), count // int(n_rows)).sum(axis=1) / float(period)

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
        "    if a.ndim != 1:\n"
        "        return -1.0\n"
        "    w = np.arange(1, a.size + 1) / a.size\n"
        "    return float(w @ a + np.sum(a))\n"
    )
    status = mesh + (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "v, c, s, L = _mesh(3, 2, 0.6, 1.2)\n"
    )
    return [
        {
            "setup": mesh,
            "call": "_signature(compute_row_loads(*_mesh(3, 2, 0.6, 1.3, 0.02, 1), 0.16, 3.5, 0.03, 3))",
            "gold_call": "_signature(_oracle_compute_row_loads(*_mesh(3, 2, 0.6, 1.3, 0.02, 1), 0.16, 3.5, 0.03, 3))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_row_loads(*_mesh(2, 3, 0.58, 1.0, 0.04, 2), 0.35, 2.45, 0.0, 2))",
            "gold_call": "_signature(_oracle_compute_row_loads(*_mesh(2, 3, 0.58, 1.0, 0.04, 2), 0.35, 2.45, 0.0, 2))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_row_loads(*_mesh(4, 2, 0.6069024253629453), 0.16, 3.5, 0.0, 4))",
            "gold_call": "_signature(_oracle_compute_row_loads(*_mesh(4, 2, 0.6069024253629453), 0.16, 3.5, 0.0, 4))",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_row_loads(*_mesh(6, 2, 0.6, 1.5), 0.25, 3.3, 0.05, 6))",
            "gold_call": "_signature(_oracle_compute_row_loads(*_mesh(6, 2, 0.6, 1.5), 0.25, 3.3, 0.05, 6))",
        },
        {
            "setup": mesh,
            "call": "float(compute_row_loads(*_mesh(1, 2, 0.62, 1.4, 0.03, 9), 0.2, 3.0, 0.1, 1)[0])",
            "gold_call": "float(_oracle_compute_row_loads(*_mesh(1, 2, 0.62, 1.4, 0.03, 9), 0.2, 3.0, 0.1, 1)[0])",
        },
        {
            "setup": mesh,
            "call": "_signature(compute_row_loads(*_mesh(2, 4, 0.6, 1.2, 0.03, 6), 0.16, 3.5, 0.03, 4))",
            "gold_call": "_signature(_oracle_compute_row_loads(*_mesh(2, 4, 0.6, 1.2, 0.03, 6), 0.16, 3.5, 0.03, 4))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_row_loads(v, c, s, L, 0.16, 3.5, 0.03, 4))",
            "gold_call": "_status(lambda: _oracle_compute_row_loads(v, c, s, L, 0.16, 3.5, 0.03, 4))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_row_loads(v, c, s, L, 0.16, 3.5, 0.03, 0))",
            "gold_call": "_status(lambda: _oracle_compute_row_loads(v, c, s, L, 0.16, 3.5, 0.03, 0))",
        },
    ]
