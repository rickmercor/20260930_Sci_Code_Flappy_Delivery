"""
Relax every vertex of the periodic stripe at fixed period to the mechanical equilibrium reached from the supplied configuration.

In quasi-static loading of a vertex model the vertices follow overdamped force balance until the energy stops decreasing, so each loading state is a local energy minimum.

Returns
-------
np.ndarray: (V, 2) equilibrium vertex positions at fixed period with the input centroid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relax_stripe(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
    kappa: float,
    chi: float,
    line_tension: float,
    tolerance: float,
) -> "np.ndarray":
    """Return the equilibrium vertex positions reached from ``vertices``.

    Holding ``period`` fixed, minimise the energy of
    ``compute_stripe_energy`` starting from ``vertices`` and return the
    local minimum in whose basin of energy descent the starting
    configuration lies. Every component of the energy gradient at the
    returned positions must be at most ``tolerance`` in magnitude, and the
    mean of the returned vertex positions must equal that of ``vertices``
    (the energy is unchanged by a rigid translation).

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
    tolerance : float
        Positive gradient tolerance, at most ``1e-6``.

    Returns
    -------
    np.ndarray
        Float array shaped like ``vertices``.

    Raises
    ------
    ValueError
        If ``tolerance`` is not a finite positive real number at most
        ``1e-6`` (booleans are rejected), if the gradient tolerance is not
        reached, or if ``compute_stripe_energy`` rejects its input.
    """
    return relaxed

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_relax_stripe(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
    kappa: float,
    chi: float,
    line_tension: float,
    tolerance: float,
) -> "np.ndarray":
    """Reference implementation (L-BFGS descent, then Newton polish on a difference Hessian)."""
    import numpy as np
    from scipy.optimize import minimize

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not (_is_number(tolerance) and 0.0 < tolerance <= 1e-6):
        raise ValueError("tolerance must be a finite number in (0, 1e-6]")
    start = np.asarray(vertices, dtype=float)
    _oracle_compute_stripe_energy(start, cells, shifts, period, kappa, chi, line_tension)
    shape = start.shape

    def _objective(z):
        energy, gradient = _oracle_compute_stripe_energy(z.reshape(shape), cells, shifts, period,
                                                         kappa, chi, line_tension)
        return energy, gradient.ravel()

    result = minimize(_objective, start.ravel(), jac=True, method="L-BFGS-B",
                      options={"gtol": 1e-10, "ftol": 1e-16, "maxiter": 50000, "maxcor": 30})
    z = result.x
    size = z.size
    for _ in range(25):
        gradient = _objective(z)[1]
        if np.max(np.abs(gradient)) <= tolerance:
            break
        hessian = np.empty((size, size))
        for j in range(size):
            step = np.zeros(size)
            step[j] = 1e-6
            hessian[:, j] = (_objective(z + step)[1] - _objective(z - step)[1]) / 2e-6
        hessian = 0.5 * (hessian + hessian.T)
        # Rigid translations span the null space; the minimum-norm step ignores them.
        z = z + np.linalg.lstsq(hessian, -gradient, rcond=1e-10)[0]
    relaxed = z.reshape(shape)
    relaxed = relaxed - relaxed.mean(axis=0) + start.mean(axis=0)
    if np.max(np.abs(_objective(relaxed.ravel())[1])) > tolerance:
        raise ValueError("the equilibrium gradient tolerance was not reached")
    return relaxed

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
        "def _signature(start, out, cells, shifts, period):\n"
        "    v = np.asarray(out, dtype=float)\n"
        "    if v.shape != start.shape:\n"
        "        return -1.0\n"
        "    p = v[cells].copy()\n"
        "    p[..., 0] += shifts * period\n"
        "    q = np.roll(p, -1, axis=1)\n"
        "    lengths = np.linalg.norm(q - p, axis=2)\n"
        "    areas = 0.5 * np.sum(p[..., 0] * q[..., 1] - q[..., 0] * p[..., 1], axis=1)\n"
        "    w = np.cos(0.6 * np.arange(lengths.shape[0]))\n"
        "    drift = np.sum(np.abs(v.mean(axis=0) - start.mean(axis=0)))\n"
        "    return float(np.mean(lengths) + np.mean(areas) + (w @ lengths[:, 0]) / lengths.shape[0] + drift)\n"
    )

    status = mesh + (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "v, c, s, L = _mesh(2, 2, 0.6, 1.2)\n"
    )
    # (stripe geometry for _mesh, energy parameters and gradient tolerance)
    specs = [
        ("3, 2, 0.6069024253629453, 1.3, 0.01, 4", "0.16, 3.5, 0.03, 1e-13"),
        ("2, 2, 0.5862449350825937, 1.0", "0.25, 3.3, 0.05, 1e-13"),
        ("4, 3, 0.554002421990666, 1.45, 0.02, 8", "0.3, 3.0, 0.0, 1e-13"),
        ("5, 2, 0.6069024253629453, 1.52", "0.16, 3.5, 0.03, 1e-13"),
        ("3, 2, 0.6069024253629453", "0.16, 3.5, 0.0, 1e-13"),
    ]
    relaxed = [
        {
            "setup": mesh + f"v, c, s, L = _mesh({geometry})\n",
            "call": f"_signature(v, relax_stripe(v.copy(), c.copy(), s.copy(), L, {parameters}), c, s, L)",
            "gold_call": (f"_signature(v, _oracle_relax_stripe(v.copy(), c.copy(), s.copy(), L, {parameters}),"
                          " c, s, L)"),
        }
        for geometry, parameters in specs
    ]
    return relaxed + [
        {
            "setup": status,
            "call": "_status(lambda: relax_stripe(v, c, s, L, 0.16, 3.5, 0.03, 1e-3))",
            "gold_call": "_status(lambda: _oracle_relax_stripe(v, c, s, L, 0.16, 3.5, 0.03, 1e-3))",
        },
        {
            "setup": status,
            "call": "_status(lambda: relax_stripe(v, c, s, L, 0.16, 3.5, 0.03, 0.0))",
            "gold_call": "_status(lambda: _oracle_relax_stripe(v, c, s, L, 0.16, 3.5, 0.03, 0.0))",
        },
    ]
