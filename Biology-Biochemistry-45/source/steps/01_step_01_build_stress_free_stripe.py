"""
Build the vertex mesh of an ordered epithelial stripe, periodic along the load, made of regular hexagonal cells at the stress-free size of the area-perimeter cell energy.

An ordered vertex-model sheet relaxes to regular hexagons whose size balances area elasticity against perimeter tension, and a stripe cut from that honeycomb along the load keeps zigzag free edges on both sides.

Returns
-------
tuple: (vertices (V, 2) float, cells (C, 6) int, shifts (C, 6) int, period float) of the stress-free stripe.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_stress_free_stripe(
    n_rows: int,
    n_columns: int,
    kappa: float,
    chi: float,
) -> tuple:
    """Return the vertex mesh of the stress-free stripe.

    A cell of dimensionless area ``a`` and perimeter ``p`` carries the energy
    ``e = (a - 1)**2 / 2 + kappa * (p - chi)**2 / 2``. The stripe is built
    from regular hexagons whose edge length ``d`` minimises ``e`` among
    regular hexagons (relative accuracy ``1e-13``). It has ``n_rows`` rows
    running along the load direction ``x`` and is periodic along ``x`` with
    ``n_columns`` cells per row and period ``n_columns * sqrt(3) * d``.

    The cell in row ``r`` (``r = 0`` lowest) and column ``c`` is centred at
    ``((c + (r % 2) / 2) * sqrt(3) * d, 1.5 * d * r)`` and has two edges
    perpendicular to ``x``. Its six vertices, counter-clockwise from the
    lower end of its right perpendicular edge, lie at the centre plus ``d``
    times ``(sqrt(3)/2, -1/2)``, ``(sqrt(3)/2, 1/2)``, ``(0, 1)``,
    ``(-sqrt(3)/2, 1/2)``, ``(-sqrt(3)/2, -1/2)`` and ``(0, -1)``.

    Cells are listed row by row and, within a row, by column. A vertex at
    ``(x, y)`` is stored at ``(x - s * period, y)`` with the integer
    ``s = floor(x / period + 1e-9)``, and the cell records ``s`` so that its
    unwrapped vertex is ``vertices[v] + (s * period, 0)``. Stored positions
    that agree within ``1e-9 * d`` are one vertex; vertices are numbered in
    order of first appearance in the cell listing and keep the stored
    position of that first appearance.

    Parameters
    ----------
    n_rows : int
        Number of cell rows, at least 1.
    n_columns : int
        Number of cells per row within one period, at least 2.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.

    Returns
    -------
    tuple
        ``(vertices, cells, shifts, period)``: float array ``(V, 2)``, integer
        arrays ``(n_rows * n_columns, 6)`` of vertex indices and periodic
        shifts, and the float period.

    Raises
    ------
    ValueError
        If ``n_rows`` is not an integer of at least 1, ``n_columns`` is not
        an integer of at least 2 (booleans are rejected), or ``kappa`` or
        ``chi`` is not a finite positive real number.
    """
    return vertices, cells, shifts, period

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_stress_free_stripe(
    n_rows: int,
    n_columns: int,
    kappa: float,
    chi: float,
) -> tuple:
    """Reference implementation (bisection for the stress-free edge, one ordered traversal)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    if not (_is_integer(n_rows) and n_rows >= 1):
        raise ValueError("n_rows must be an integer of at least 1")
    if not (_is_integer(n_columns) and n_columns >= 2):
        raise ValueError("n_columns must be an integer of at least 2")
    if not (_is_number(kappa) and kappa > 0.0):
        raise ValueError("kappa must be a finite positive number")
    if not (_is_number(chi) and chi > 0.0):
        raise ValueError("chi must be a finite positive number")
    kappa, chi = float(kappa), float(chi)
    root3 = np.sqrt(3.0)
    # A regular hexagon of size scale k has area k**2 and perimeter k * chi_star.
    chi_star = np.sqrt(8.0 * root3)

    def _slope(k):
        return 2.0 * k * (k * k - 1.0) + kappa * chi_star * (k * chi_star - chi)

    low, high = 0.0, max(1.0, chi / chi_star)
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _slope(middle) < 0.0:
            low = middle
        else:
            high = middle
    edge = 0.5 * (low + high) * chi_star / 6.0

    period = n_columns * root3 * edge
    offsets = ((0.5 * root3, -0.5), (0.5 * root3, 0.5), (0.0, 1.0),
               (-0.5 * root3, 0.5), (-0.5 * root3, -0.5), (0.0, -1.0))
    index, vertices, cells, shifts = {}, [], [], []
    for row in range(n_rows):
        for column in range(n_columns):
            cell, shift = [], []
            for dx, dy in offsets:
                x = (column + 0.5 * (row % 2)) * root3 * edge + dx * edge
                y = 1.5 * edge * row + dy * edge
                s = int(np.floor(x / period + 1e-9))
                # Honeycomb vertices sit on a lattice of half cell widths and
                # half edges, so the rounded lattice indices identify them.
                key = (int(round((x - s * period) / (0.5 * root3 * edge))) % (2 * n_columns),
                       int(round(y / (0.5 * edge))))
                if key not in index:
                    index[key] = len(vertices)
                    vertices.append((x - s * period, y))
                cell.append(index[key])
                shift.append(s)
            cells.append(cell)
            shifts.append(shift)
    return (np.array(vertices, dtype=float), np.array(cells, dtype=int),
            np.array(shifts, dtype=int), float(period))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    signature = (
        "import numpy as np\n"
        "def _signature(out):\n"
        "    vertices, cells, shifts, period = out\n"
        "    v = np.asarray(vertices, dtype=float)\n"
        "    c = np.asarray(cells)\n"
        "    s = np.asarray(shifts)\n"
        "    if v.ndim != 2 or v.shape[1] != 2 or c.ndim != 2 or c.shape[1] != 6 or s.shape != c.shape:\n"
        "        return -1.0\n"
        "    w = np.cos(0.7 * np.arange(v.shape[0]))\n"
        "    q = np.cos(0.3 * np.arange(c.size)).reshape(c.shape)\n"
        "    return float(period + w @ v[:, 0] + 0.5 * (w @ v[:, 1]) + 1e-3 * np.sum(q * c)\n"
        "                 + 0.1 * np.sum(q * s) + 0.01 * v.shape[0])\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": signature,
            "call": "_signature(build_stress_free_stripe(3, 2, 0.16, 3.5))",
            "gold_call": "_signature(_oracle_build_stress_free_stripe(3, 2, 0.16, 3.5))",
        },
        {
            "setup": signature,
            "call": "_signature(build_stress_free_stripe(1, 2, 0.35, 2.45))",
            "gold_call": "_signature(_oracle_build_stress_free_stripe(1, 2, 0.35, 2.45))",
        },
        {
            "setup": signature,
            "call": "_signature(build_stress_free_stripe(4, 3, 0.6, 1.8))",
            "gold_call": "_signature(_oracle_build_stress_free_stripe(4, 3, 0.6, 1.8))",
        },
        {
            "setup": signature,
            "call": "_signature(build_stress_free_stripe(2, 4, 0.05, 4.6))",
            "gold_call": "_signature(_oracle_build_stress_free_stripe(2, 4, 0.05, 4.6))",
        },
        {
            "setup": signature,
            "call": "float(build_stress_free_stripe(10, 2, 0.16, 3.5)[3])",
            "gold_call": "float(_oracle_build_stress_free_stripe(10, 2, 0.16, 3.5)[3])",
        },
        {
            "setup": signature,
            "call": "_signature(build_stress_free_stripe(2, 2, 40.0, 1.5))",
            "gold_call": "_signature(_oracle_build_stress_free_stripe(2, 2, 40.0, 1.5))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_stress_free_stripe(3, 1, 0.16, 3.5))",
            "gold_call": "_status(lambda: _oracle_build_stress_free_stripe(3, 1, 0.16, 3.5))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_stress_free_stripe(3, 2, 0.0, 3.5))",
            "gold_call": "_status(lambda: _oracle_build_stress_free_stripe(3, 2, 0.0, 3.5))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_stress_free_stripe(True, 2, 0.16, 3.5))",
            "gold_call": "_status(lambda: _oracle_build_stress_free_stripe(True, 2, 0.16, 3.5))",
        },
    ]
