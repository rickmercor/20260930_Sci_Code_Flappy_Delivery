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
def assemble_upwind_convection_diffusion(velocity_x: "np.ndarray", velocity_y: "np.ndarray", *, fitted: dict = None) -> "np.ndarray":
    """Reference implementation (node-by-node stencil assembly)."""
    import numpy as np

    def _fitted_operator():
        if not isinstance(fitted, dict):
            raise ValueError("fitted must be a dictionary")
        names = ("x_edges", "y_edges", "diffusion_x", "diffusion_y",
                 "reaction", "robin_left", "robin_right")
        if any(name not in fitted for name in names):
            raise ValueError("missing fitted-grid data")
        xe, ye, dx_face, dy_face, reaction, left, right = [
            np.asarray(fitted[name], dtype=float) for name in names]
        vx, vy = np.asarray(velocity_x, dtype=float), np.asarray(velocity_y, dtype=float)
        if (xe.ndim != 1 or ye.ndim != 1 or xe.size < 2 or ye.size < 2
                or not np.all(np.isfinite(xe)) or not np.all(np.isfinite(ye))
                or np.any(np.diff(xe) <= 0) or np.any(np.diff(ye) <= 0)):
            raise ValueError("edges must be finite strictly increasing vectors")
        nx, ny = xe.size - 1, ye.size - 1
        if (vx.shape != (ny, nx) or vy.shape != (ny, nx)
                or dx_face.shape != (ny, nx + 1) or dy_face.shape != (ny, nx)
                or reaction.shape != (ny, nx) or left.shape != (ny,) or right.shape != (ny,)):
            raise ValueError("inconsistent fitted-grid shapes")
        if not all(np.all(np.isfinite(a)) for a in (vx, vy, dx_face, dy_face, reaction)):
            raise ValueError("cell and diffusion data must be finite")
        if np.any(dx_face <= 0) or np.any(dy_face <= 0) or np.any(reaction < 0):
            raise ValueError("diffusion must be positive and reaction nonnegative")
        if any(np.any(np.isnan(a)) or np.any(a < 0) for a in (left, right)):
            raise ValueError("Robin coefficients must be nonnegative, allowing positive infinity")
        hx, hy = np.diff(xe), np.diff(ye)
        volume = hy[:, None] * hx[None, :]
        L = np.diag((reaction * volume).ravel())

        def _rates(diffusion, distance, velocity):
            conductance = diffusion / distance
            pe = velocity / conductance
            # c_plus*u_left - c_minus*u_right solves the constant-flux ODE.
            z = abs(pe)
            if z < 1e-4:
                even = 1.0 + z*z/12.0 - z**4/720.0 + z**6/30240.0
                low = conductance * (even - z/2.0)
            elif z > 50:
                low = np.exp(np.log(abs(velocity)) - z - np.log(-np.expm1(-z)))
            else:
                decay = np.exp(-z)
                low = abs(velocity) * decay / (-np.expm1(-z))
            high = low + abs(velocity)
            return (high, low) if velocity >= 0 else (low, high)

        def _face(a, b, diffusion, distance, velocity, area):
            if a == b:
                return  # Both contributions of a periodic self-face cancel exactly.
            outgoing, incoming = _rates(diffusion, distance, velocity)
            L[a, a] += area * outgoing
            L[a, b] -= area * incoming
            L[b, a] -= area * outgoing
            L[b, b] += area * incoming

        def _boundary(cell, diffusion, distance, outward_speed, area, rho):
            outgoing, incoming = _rates(diffusion, distance, outward_speed)
            if rho == 0:
                effective = 0.0
            elif np.isposinf(rho):
                effective = outgoing
            elif rho >= incoming:
                effective = outgoing / (1.0 + incoming/rho)
            else:
                effective = outgoing * (rho/incoming) / (1.0 + rho/incoming)
            L[cell, cell] += area * effective

        for j in range(ny):
            for i in range(nx - 1):
                distance = (hx[i] + hx[i + 1]) / 2.0
                speed = (hx[i + 1]*vx[j, i] + hx[i]*vx[j, i + 1]) / (hx[i] + hx[i + 1])
                _face(j*nx+i, j*nx+i+1, dx_face[j, i+1], distance, speed, hy[j])
            _boundary(j*nx, dx_face[j, 0], hx[0]/2.0, -vx[j, 0], hy[j], left[j])
            _boundary(j*nx+nx-1, dx_face[j, -1], hx[-1]/2.0, vx[j, -1], hy[j], right[j])
        for j in range(ny):
            above = (j + 1) % ny
            distance = (hy[j] + hy[above]) / 2.0
            for i in range(nx):
                speed = (hy[above]*vy[j, i] + hy[j]*vy[above, i]) / (hy[j] + hy[above])
                _face(j*nx+i, above*nx+i, dy_face[j, i], distance, speed, hx[i])
        root_volume = np.sqrt(volume.ravel())
        return L / root_volume[:, None] / root_volume[None, :]

    return _fitted_operator()

import numpy as np
def seed_power_pattern(matrix: "np.ndarray", column: int) -> "np.ndarray":
    """Reference implementation (union of the column supports of I, A and A^2)."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
        raise ValueError("matrix must be a nonempty square 2-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must have finite entries")
    size = array.shape[0]
    if not (_is_integer(column) and 0 <= column < size):
        raise ValueError("column must be an integer index of the matrix")
    k = int(column)
    first = array[:, k]
    # Column k of A @ A is A applied to column k of A.
    second = array @ first
    mask = first != 0.0
    mask |= second != 0.0
    mask[k] = True
    return np.flatnonzero(mask).astype(int)

import numpy as np
def _solve_reduced_column(array, k, pattern):
    """Least-squares column on a pattern, solved on the rows the pattern reaches."""
    import numpy as np

    positions = np.asarray(pattern, dtype=int)
    # Rows outside those reached by matrix[:, pattern] cannot be changed, so the
    # minimizer is the solution of the reduced problem on the reached rows.
    rows = np.flatnonzero(np.any(array[:, positions] != 0.0, axis=1))
    target = (rows == k).astype(float)
    vector = np.zeros(array.shape[0])
    reduced = array[np.ix_(rows, positions)]
    scales = np.max(np.abs(reduced), axis=0)
    q, r = np.linalg.qr(reduced / scales, mode="reduced")
    vector[positions] = np.linalg.solve(r, q.T @ target) / scales
    return vector


def initialize_adaptive_column(matrix: "np.ndarray", column: int) -> tuple:
    """Reference implementation (a-priori seed, then the reduced least-squares solve)."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
        raise ValueError("matrix must be a nonempty square 2-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must have finite entries")
    if not (_is_integer(column) and 0 <= column < array.shape[0]):
        raise ValueError("column must be an integer index of the matrix")
    k = int(column)
    pattern = seed_power_pattern(array, k)
    vector = _solve_reduced_column(array, k, pattern)
    return pattern, vector

import numpy as np
def select_residual_rows(residual: "np.ndarray", rows: "np.ndarray", used: "np.ndarray", threshold: float, cap: int) -> "np.ndarray":
    """Reference implementation (norm-relative threshold with a per-pass cap)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    def _index_array(values, size, name):
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array")
        if values.size == 0:
            return np.zeros(0, dtype=int)
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must hold integer indices")
        values = values.astype(int)
        if values.min() < 0 or values.max() >= size:
            raise ValueError(f"{name} indices must lie in [0, n)")
        return values

    vector = np.asarray(residual, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("residual must be a nonempty 1-D array of finite numbers")
    size = vector.size
    candidates = _index_array(rows, size, "rows")
    if np.unique(candidates).size != candidates.size:
        raise ValueError("rows must not repeat an index")
    previous = _index_array(used, size, "used")
    if not (_is_number(threshold) and threshold >= 0.0):
        raise ValueError("threshold must be a finite nonnegative number")
    if not (_is_integer(cap) and cap >= 1):
        raise ValueError("cap must be an integer of at least 1")
    candidates = candidates[~np.isin(candidates, previous)]
    if candidates.size == 0:
        return np.zeros(0, dtype=int)
    level = float(threshold) * float(np.linalg.norm(vector))
    magnitudes = np.abs(vector[candidates])
    keep = magnitudes >= level
    candidates, magnitudes = candidates[keep], magnitudes[keep]
    # Primary key: decreasing magnitude; secondary key: increasing index.
    order = np.lexsort((candidates, -magnitudes))
    return candidates[order][: int(cap)].astype(int)

import numpy as np
def enlarge_column_pattern(matrix: "np.ndarray", pattern: "np.ndarray", selected: "np.ndarray") -> "np.ndarray":
    """Reference implementation (row patterns of the selected residual rows)."""
    import numpy as np

    def _index_array(values, size, name):
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array")
        if values.size == 0:
            return np.zeros(0, dtype=int)
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must hold integer indices")
        values = values.astype(int)
        if values.min() < 0 or values.max() >= size:
            raise ValueError(f"{name} indices must lie in [0, n)")
        return values

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
        raise ValueError("matrix must be a nonempty square 2-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must have finite entries")
    size = array.shape[0]
    current = _index_array(pattern, size, "pattern")
    if np.unique(current).size != current.size:
        raise ValueError("pattern must not repeat a position")
    chosen = _index_array(selected, size, "selected")
    if chosen.size == 0:
        return np.sort(current).astype(int)
    # Residual entry i equals sum_j matrix[i, j] m[j] - delta_ik, so m[j]
    # reaches entry i exactly when matrix[i, j] is nonzero: row patterns.
    reach = np.flatnonzero(np.any(array[chosen, :] != 0.0, axis=0))
    return np.union1d(current, reach).astype(int)

import numpy as np
def advance_column_pattern(matrix: "np.ndarray", pattern: "np.ndarray", residual: "np.ndarray", used: "np.ndarray", threshold: float, cap: int) -> tuple:
    """Reference implementation (select on the reduced rows, enlarge, record)."""
    import numpy as np

    def _index_array(values, size, name):
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array")
        if values.size == 0:
            return np.zeros(0, dtype=int)
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must hold integer indices")
        values = values.astype(int)
        if values.min() < 0 or values.max() >= size or np.unique(values).size != values.size:
            raise ValueError(f"{name} must hold distinct indices in [0, n)")
        return values

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
        raise ValueError("matrix must be a nonempty square 2-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must have finite entries")
    size = array.shape[0]
    current = _index_array(pattern, size, "pattern")
    if current.size == 0:
        raise ValueError("pattern must not be empty")
    previous = _index_array(used, size, "used")
    vector = np.asarray(residual, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError("residual must be a finite vector of length n")
    # Candidate rows are the rows of the current reduced problem (Eq. 11).
    rows = np.flatnonzero(np.any(array[:, current] != 0.0, axis=1))
    chosen = select_residual_rows(vector, rows, previous, threshold, cap)
    if chosen.size == 0:
        return np.sort(current).astype(int), np.sort(previous).astype(int)
    enlarged = enlarge_column_pattern(array, current, chosen)
    admitted = np.union1d(previous, chosen).astype(int)
    return enlarged, admitted

import numpy as np


def extend_reduced_qr(state: tuple, added_rows: "np.ndarray", added_columns: "np.ndarray", coupling: "np.ndarray", column: int, size: int) -> tuple:
    """Scale-aware orthogonal extension followed by a small column reorder."""

    def _qr_column_norms(block):
        """Column norms without squaring the physical column scales."""
        largest = np.max(np.abs(block), axis=0)
        return largest * np.linalg.norm(block / largest, axis=0)

    if not (isinstance(size, (int, np.integer)) and not isinstance(size, bool) and size > 0):
        raise ValueError("size must be a positive integer")
    if not (isinstance(column, (int, np.integer)) and not isinstance(column, bool) and 0 <= column < size):
        raise ValueError("column must be a full-space integer index")
    if not isinstance(state, tuple) or len(state) != 4:
        raise ValueError("state must contain I, J, Q and R")

    def _indices(values):
        values = np.asarray(values)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("indices must be one-dimensional integer arrays")
        if np.unique(values).size != values.size or np.any(values < 0) or np.any(values >= size):
            raise ValueError("indices must be distinct and in range")
        return values.astype(int)

    old_rows, old_columns = _indices(state[0]), _indices(state[1])
    new_rows, new_columns = _indices(added_rows), _indices(added_columns)
    if np.intersect1d(old_rows, new_rows).size or np.intersect1d(old_columns, new_columns).size:
        raise ValueError("old and added index sets must be disjoint")
    old_q, old_r = np.asarray(state[2], dtype=float), np.asarray(state[3], dtype=float)
    new_block = np.asarray(coupling, dtype=float)
    a, p, d, t = old_rows.size, old_columns.size, new_rows.size, new_columns.size
    if (old_q.shape != (a, p) or old_r.shape != (p, p)
            or new_block.shape != (a + d, t) or p > a
            or p + t == 0 or p + t > a + d):
        raise ValueError("inconsistent reduced dimensions")
    if not all(np.all(np.isfinite(x)) for x in (old_q, old_r, new_block)):
        raise ValueError("factors and couplings must be finite")

    embedded_q = np.zeros((a + d, p))
    embedded_q[:a] = old_q
    old_scales = _qr_column_norms(old_r) if p else np.empty(0)
    old_normalized_r = old_r / old_scales if p else old_r.copy()
    if t:
        new_scales = _qr_column_norms(new_block)
        normalized_block = new_block / new_scales
        cross = embedded_q.T @ normalized_block
        orthogonal_block = normalized_block - embedded_q @ cross
        # Reprojection preserves weak new directions in almost-aligned blocks.
        correction = embedded_q.T @ orthogonal_block
        cross += correction
        orthogonal_block -= embedded_q @ correction
        added_q, added_r = np.linalg.qr(orthogonal_block, mode="reduced")
        working_q = np.column_stack((embedded_q, added_q))
        working_r = np.zeros((p + t, p + t))
        working_r[:p, :p] = old_normalized_r
        working_r[:p, p:] = cross
        working_r[p:, p:] = added_r
    else:
        new_scales = np.empty(0)
        working_q, working_r = embedded_q, old_normalized_r

    row_ids = np.concatenate((old_rows, new_rows))
    column_ids = np.concatenate((old_columns, new_columns))
    row_order, column_order = np.argsort(row_ids), np.argsort(column_ids)
    # The factor describes sorted global columns, not append order.
    rotation, sorted_r = np.linalg.qr(working_r[:, column_order], mode="reduced")
    sorted_q = (working_q @ rotation)[row_order]
    signs = np.where(np.diag(sorted_r) < 0.0, -1.0, 1.0)
    sorted_q *= signs
    sorted_r *= signs[:, None]
    scales = np.concatenate((old_scales, new_scales))[column_order]
    rows, columns = row_ids[row_order], column_ids[column_order]
    target = (rows == column).astype(float)
    projected = sorted_q.T @ target
    scaled_solution = np.linalg.solve(sorted_r, projected)
    vector = np.zeros(size)
    vector[columns] = scaled_solution / scales
    reduced_residual = sorted_q @ projected - target
    norm = np.sqrt(reduced_residual @ reduced_residual + float(not np.any(rows == column)))
    return (rows, columns, sorted_q, sorted_r * scales), vector, float(norm)

import numpy as np
def compute_adaptive_column(matrix: "np.ndarray", column: int, tolerance: float, threshold: float, cap: int, max_passes: int, *, seed_mode: str = "power") -> tuple:
    """Reference implementation (initial column, then threshold-driven passes)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
        raise ValueError("matrix must be a nonempty square 2-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must have finite entries")
    size = array.shape[0]
    if not (_is_integer(column) and 0 <= column < size):
        raise ValueError("column must be an integer index of the matrix")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    if not (_is_number(threshold) and threshold >= 0.0):
        raise ValueError("threshold must be a finite nonnegative number")
    if not (_is_integer(cap) and cap >= 1):
        raise ValueError("cap must be an integer of at least 1")
    if not (_is_integer(max_passes) and max_passes >= 0):
        raise ValueError("max_passes must be an integer of at least 0")
    if seed_mode not in ("power", "diagonal"):
        raise ValueError("seed_mode must be 'power' or 'diagonal'")
    k = int(column)

    def _residual(vector):
        # Full residual: row k keeps its -1 whenever the pattern misses it.
        out = array @ vector
        out[k] -= 1.0
        return out

    def _fresh_state(support):
        # Empty factorization extended by the whole reduced block of `support`.
        rows = np.flatnonzero(np.any(array[:, support] != 0.0, axis=1))
        empty = np.empty(0, dtype=int)
        initial_state = (empty, empty.copy(), np.empty((0, 0)), np.empty((0, 0)))
        return extend_reduced_qr(
            initial_state, rows, support, array[np.ix_(rows, support)], k, size
        )

    if seed_mode == "diagonal":
        pattern = np.array([k], dtype=int)
        state, vector, _ = _fresh_state(pattern)
    else:
        pattern, vector = initialize_adaptive_column(array, k)
        state = None
    residual = _residual(vector)
    admitted = np.zeros(0, dtype=int)
    passes = 0
    while np.linalg.norm(residual) > tolerance and passes < max_passes:
        grown, recorded = advance_column_pattern(
            array, pattern, residual, admitted, threshold, cap
        )
        if recorded.size == admitted.size:
            break  # nothing qualified for admission
        if state is None:
            state, _, _ = _fresh_state(pattern)
        added_columns = np.setdiff1d(grown, state[1])
        reached = np.flatnonzero(np.any(array[:, grown] != 0.0, axis=1))
        added_rows = np.setdiff1d(reached, state[0])
        row_order = np.concatenate((state[0], added_rows))
        state, vector, _ = extend_reduced_qr(
            state, added_rows, added_columns,
            array[np.ix_(row_order, added_columns)], k, size
        )
        pattern, admitted = grown, recorded
        residual = _residual(vector)
        passes += 1
    return vector, float(np.linalg.norm(residual)), int(passes)

import numpy as np
def evaluate_preconditioner_benchmark(
    n_side: int = 18,
    flow_x: float = 30.0,
    flow_y: float = 60.0,
    flow_center: float = 0.37,
    tolerance: float = 0.1,
    threshold: float = 0.25,
    cap: int = 3,
    max_passes: int = 3,
    *, seed_mode: str = "power",
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    if not (_is_integer(n_side) and n_side >= 1):
        raise ValueError("n_side must be an integer of at least 1")
    for value in (flow_x, flow_y, flow_center):
        if not _is_number(value):
            raise ValueError("flow parameters must be finite numbers")
    if seed_mode not in ("power", "diagonal"):
        raise ValueError("seed_mode must be 'power' or 'diagonal'")
    size = int(n_side)
    edges = np.arange(size + 1, dtype=float) / size
    xe, ye = edges**1.7, edges**1.3
    xc, yc = (xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2
    x_grid, y_grid = np.meshgrid(xc, yc, indexing="xy")
    velocity_x = float(flow_x)*(1+x_grid)*np.cos(2*np.pi*y_grid)
    velocity_y = float(flow_y)*np.sin(2*np.pi*(x_grid-float(flow_center)))
    xf, yf = np.meshgrid(xe, yc, indexing="xy")
    dx_face = 0.08*(1+0.6*np.sin(np.pi*xf)**2*np.cos(2*np.pi*yf)**2)
    xf, yf = np.meshgrid(xc, ye[1:], indexing="xy")
    dy_face = 0.12*(1+0.5*np.cos(np.pi*xf)**2*np.sin(2*np.pi*yf)**2)
    geometry = dict(x_edges=xe, y_edges=ye, diffusion_x=dx_face, diffusion_y=dy_face,
                    reaction=0.1+0.05*np.cos(np.pi*x_grid)**2*np.sin(2*np.pi*y_grid)**2,
                    robin_left=0.2+2*np.sin(2*np.pi*yc)**2,
                    robin_right=0.5+3*np.cos(2*np.pi*yc)**2)
    matrix = assemble_upwind_convection_diffusion(velocity_x, velocity_y, fitted=geometry)
    unknowns = matrix.shape[0]
    preconditioner = np.zeros((unknowns, unknowns))
    # Frobenius-norm minimization decouples M into independent columns.
    for k in range(unknowns):
        vector, _, _ = compute_adaptive_column(
            matrix, k, tolerance, threshold, cap, max_passes, seed_mode=seed_mode
        )
        preconditioner[:, k] = vector
    defect = matrix @ preconditioner - np.eye(unknowns)
    value = float(np.linalg.norm(defect, "fro"))
    if not np.isfinite(value):
        raise ValueError("the Frobenius norm must be finite")
    return value
SCICODE_GOLD_EOF
