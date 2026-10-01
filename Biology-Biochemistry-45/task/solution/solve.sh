#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
def build_stress_free_stripe(
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

import numpy as np
def compute_cell_geometry(
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

import numpy as np
def compute_stripe_energy(
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

    areas, perimeters, lengths, free = compute_cell_geometry(vertices, cells, shifts, period)
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

import numpy as np
def compute_virial_stresses(
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

    areas, perimeters, lengths, free = compute_cell_geometry(vertices, cells, shifts, period)
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

import numpy as np
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
    compute_stripe_energy(start, cells, shifts, period, kappa, chi, line_tension)
    shape = start.shape

    def _objective(z):
        energy, gradient = compute_stripe_energy(z.reshape(shape), cells, shifts, period,
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

import numpy as np
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
    """Reference implementation (row sums of area-weighted Virial stresses)."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    stresses = compute_virial_stresses(vertices, cells, shifts, period, kappa, chi, line_tension)
    areas = compute_cell_geometry(vertices, cells, shifts, period)[0]
    count = stresses.shape[0]
    if not (_is_integer(n_rows) and n_rows >= 1 and count % n_rows == 0):
        raise ValueError("n_rows must be a positive integer dividing the number of cells")
    # With periodic images unwrapped, the virial of all cells equals the period times
    # the transmitted force, so each row's share is its virial over the period.
    moments = areas * stresses[:, 0]
    return moments.reshape(int(n_rows), count // int(n_rows)).sum(axis=1) / float(period)

import numpy as np
def locate_bifurcation_stretch(
    n_rows: int,
    n_columns: int,
    kappa: float,
    chi: float,
    line_tension: float,
    threshold: float,
    tolerance: float,
) -> float:
    """Reference implementation (march in 0.05 increments, then bisection on relaxed states)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not (_is_number(line_tension) and line_tension >= 0.0):
        raise ValueError("line_tension must be a finite non-negative number")
    if not (_is_number(threshold) and threshold > 0.0):
        raise ValueError("threshold must be a finite positive number")
    if not (_is_number(tolerance) and 0.0 < tolerance <= 1e-6):
        raise ValueError("tolerance must be a finite number in (0, 1e-6]")
    vertices, cells, shifts, period = build_stress_free_stripe(n_rows, n_columns, kappa, chi)

    def _gap(stretch):
        start = vertices.copy()
        start[:, 0] *= stretch
        relaxed = relax_stripe(start, cells, shifts, stretch * period, kappa, chi,
                                       line_tension, 1e-12)
        lengths = compute_cell_geometry(relaxed, cells, shifts, stretch * period)[2]
        return float(np.min(lengths[:, 0])) - float(threshold)

    if not _gap(1.0) > 0.0:
        raise ValueError("the load-perpendicular junctions are not longer than the threshold at stretch 1")
    low, high = 1.0, None
    for step in range(1, 61):
        stretch = 1.0 + 0.05 * step
        if _gap(stretch) <= 0.0:
            high = stretch
            break
        low = stretch
    if high is None:
        raise ValueError("no junction reaches the threshold up to stretch 4")
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _gap(middle) > 0.0:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))

import numpy as np
def estimate_tension_load_ratio(
    n_rows: int = 10,
    kappa: float = 0.16,
    chi: float = 3.5,
    line_tension: float = 0.03,
    threshold: float = 0.1,
    n_columns: int = 2,
    tolerance: float = 1e-12,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not (_is_number(line_tension) and line_tension > 0.0):
        raise ValueError("line_tension must be a finite positive number")
    vertices, cells, shifts, period = build_stress_free_stripe(n_rows, n_columns, kappa, chi)

    def _state(stretch, tension):
        start = vertices.copy()
        start[:, 0] *= stretch
        relaxed = relax_stripe(start, cells, shifts, stretch * period, kappa, chi, tension, 1e-12)
        rows = compute_row_loads(relaxed, cells, shifts, stretch * period, kappa, chi, tension, n_rows)
        lengths = compute_cell_geometry(relaxed, cells, shifts, stretch * period)[2]
        return float(np.sum(rows)), float(np.min(lengths))

    loads = []
    for tension in (float(line_tension), 0.0):
        stretch = locate_bifurcation_stretch(n_rows, n_columns, kappa, chi, tension,
                                                     threshold, tolerance)
        load, shortest = _state(stretch, tension)
        below, _ = _state(stretch * (1.0 - 1e-6), tension)
        # Necking bifurcates at the load maximum: the load must still be rising into
        # the first exchange, and no other junction may have exchanged earlier.
        if not below < load:
            raise ValueError("the load is not rising into the first neighbour exchange")
        if shortest < float(threshold) - 1e-6:
            raise ValueError("another junction reaches the threshold before the first exchange")
        loads.append(load)
    return loads[0] / loads[1]
SCICODE_GOLD_EOF
