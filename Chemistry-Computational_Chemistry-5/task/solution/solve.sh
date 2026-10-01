#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def _surface_constants() -> tuple:
    """Return the parameters of the reduced free-energy surface."""
    return (8.10,        # lattice period, angstrom
            0.20, 0.55,  # short-range repulsion strength (eV) and range (angstrom)
            0.12, 2.60,  # separated-basin stiffness (eV/angstrom^2) and centre
            0.25, 0.80, 0.25,   # molecular well depth (eV), centre, width
            0.040, 1.22, 0.30,  # saddle height (eV), centre, width
            0.020, 0.045,       # terrace corrugation and edge depth, eV
            3.0,                # edge enhancement of the saddle
            0.06)               # double-bond exclusion penalty, eV


def _is_integer(value) -> bool:
    """Return True for a genuine integer, rejecting booleans."""
    import numpy as np
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


def _is_positive_float(value) -> bool:
    """Return True for a finite positive real scalar."""
    import numpy as np
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return bool(np.isfinite(float(value))) and float(value) > 0.0


def _surface_gradient(r1: float, r2: float, s: float) -> tuple:
    """Return the three partial derivatives of the free-energy surface."""
    import numpy as np

    (period, c_rep, r_core, k_sep, r_sep, d_mol, r_mol, w_mol,
     b_sad, r_sad, w_sad, a_cor, a_edge, c_edge, g_excl) = _surface_constants()

    omega = 2.0 * np.pi / period
    edge = 0.5 * (1.0 + np.cos(omega * s))
    d_edge = -0.5 * omega * np.sin(omega * s)
    scale = 1.0 + c_edge * edge

    grads = []
    saddle_s = 0.0
    for r in (r1, r2):
        saddle = b_sad * np.exp(-((r - r_sad) ** 2) / (2.0 * w_sad ** 2))
        well = d_mol * np.exp(-((r - r_mol) ** 2) / (2.0 * w_mol ** 2))
        grads.append(-6.0 * c_rep * r_core ** 6 / r ** 7
                     + k_sep * (r - r_sep)
                     + well * (r - r_mol) / w_mol ** 2
                     - saddle * (r - r_sad) / w_sad ** 2 * scale)
        saddle_s += saddle * c_edge * d_edge

    exclusion = g_excl * np.exp(-(((r1 - r_mol) ** 2) + ((r2 - r_mol) ** 2))
                                / (2.0 * w_mol ** 2))
    grads[0] -= exclusion * (r1 - r_mol) / w_mol ** 2
    grads[1] -= exclusion * (r2 - r_mol) / w_mol ** 2

    grad_s = (a_cor * 3.0 * omega * np.sin(3.0 * omega * s)
              + a_edge * omega * np.sin(omega * s)
              + saddle_s)
    return grads[0], grads[1], grad_s


import numpy as np
def simulate_surface_trajectory(n_frames: int, seed: int, dt_ps: float = 0.01,
                                        d_bond: float = 0.5, d_lateral: float = 2.0,
                                        kt_ev: float = 0.0387780) -> np.ndarray:
    """Reference implementation (Euler-Maruyama on the reduced surface)."""
    import numpy as np

    if not (_is_integer(n_frames) and int(n_frames) >= 2):
        raise ValueError("n_frames must be an integer of at least 2")
    if not _is_integer(seed):
        raise ValueError("seed must be an integer")
    for name, value in (("dt_ps", dt_ps), ("d_bond", d_bond),
                        ("d_lateral", d_lateral), ("kt_ev", kt_ev)):
        if not _is_positive_float(value):
            raise ValueError(f"{name} must be a positive finite float")

    period = _surface_constants()[0]
    start = _surface_constants()[4]
    n_frames = int(n_frames)
    dt_ps, kt_ev = float(dt_ps), float(kt_ev)
    d_bond, d_lateral = float(d_bond), float(d_lateral)

    noise = np.random.default_rng(int(seed)).standard_normal((n_frames, 3))
    mob_bond = d_bond * dt_ps / kt_ev
    mob_lateral = d_lateral * dt_ps / kt_ev
    amp_bond = np.sqrt(2.0 * d_bond * dt_ps)
    amp_lateral = np.sqrt(2.0 * d_lateral * dt_ps)

    trajectory = np.empty((n_frames, 3), dtype=float)
    r1 = r2 = start
    s = 0.0
    for t in range(n_frames):
        trajectory[t, 0] = r1
        trajectory[t, 1] = r2
        trajectory[t, 2] = s
        if not (r1 > 0.0 and r2 > 0.0):
            raise ValueError("integration produced a non-positive hydrogen separation")
        g1, g2, gs = _surface_gradient(r1, r2, s)
        r1 = r1 - mob_bond * g1 + amp_bond * noise[t, 0]
        r2 = r2 - mob_bond * g2 + amp_bond * noise[t, 1]
        s = (s - mob_lateral * gs + amp_lateral * noise[t, 2]) % period
    return trajectory

def _descriptor_geometry() -> tuple:
    """Return the Rh row geometry used by the adsorbate-centred descriptors."""
    return 8.10, 0.30, 1.00   # lattice period, edge-atom rise, adsorbate height


def _cosine_cutoff(distance, r_cut: float):
    """Return the cosine cutoff, zero at and beyond ``r_cut``."""
    import numpy as np
    inside = distance < r_cut
    return np.where(inside, 0.5 * (1.0 + np.cos(np.pi * np.minimum(distance, r_cut) / r_cut)), 0.0)


def _rh_image_positions(n_images: int) -> tuple:
    """Return the lateral positions and heights of the Rh images."""
    import numpy as np
    period, rise, _ = _descriptor_geometry()
    spacing = period / 3.0
    cells = np.arange(-int(n_images), int(n_images) + 1, dtype=float)
    offsets = np.array([0.0, spacing, 2.0 * spacing])
    lateral = (cells[:, None] * period + offsets[None, :]).ravel()
    heights = np.tile(np.array([rise, 0.0, 0.0]), cells.size)
    return lateral, heights


import numpy as np
def build_local_descriptors(trajectory: np.ndarray, n_radial: int = 7,
                                    r_cut: float = 7.0, n_images: int = 2) -> np.ndarray:
    """Reference implementation (three smoothly truncated radial channels)."""
    import numpy as np

    if not (_is_integer(n_radial) and int(n_radial) >= 1):
        raise ValueError("n_radial must be an integer of at least 1")
    if not (_is_integer(n_images) and int(n_images) >= 0):
        raise ValueError("n_images must be an integer of at least 0")
    if not _is_positive_float(r_cut):
        raise ValueError("r_cut must be a positive finite float")
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 3 or traj.shape[0] < 1:
        raise ValueError("trajectory must be a non-empty array of shape (T, 3)")
    if not np.all(np.isfinite(traj)):
        raise ValueError("trajectory must be finite")
    if np.any(traj[:, :2] <= 0.0):
        raise ValueError("hydrogen-hydrogen distances must be positive")

    n_radial, r_cut = int(n_radial), float(r_cut)
    _, _, height = _descriptor_geometry()
    centres = (np.arange(n_radial, dtype=float) + 0.5) * r_cut / n_radial
    sigma = r_cut / (2.0 * n_radial)
    two_sigma_sq = 2.0 * sigma ** 2

    r1, r2, s = traj[:, 0], traj[:, 1], traj[:, 2]
    lateral, heights = _rh_image_positions(int(n_images))
    d_rh = np.sqrt((height - heights)[None, :] ** 2 + (s[:, None] - lateral[None, :]) ** 2)
    cut_rh = _cosine_cutoff(d_rh, r_cut)
    cut_1, cut_2 = _cosine_cutoff(r1, r_cut), _cosine_cutoff(r2, r_cut)
    mean_hh = 0.5 * (r1 + r2)

    descriptors = np.empty((traj.shape[0], 3 * n_radial), dtype=float)
    for k, mu in enumerate(centres):
        descriptors[:, k] = np.sum(cut_rh * np.exp(-((d_rh - mu) ** 2) / two_sigma_sq), axis=1)
        descriptors[:, n_radial + k] = (cut_1 * np.exp(-((r1 - mu) ** 2) / two_sigma_sq)
                                        + cut_2 * np.exp(-((r2 - mu) ** 2) / two_sigma_sq))
        descriptors[:, 2 * n_radial + k] = cut_1 * cut_2 * np.exp(-((mean_hh - mu) ** 2) / two_sigma_sq)
    return descriptors

def _centred_moving_average(values, window_frames: int):
    """Return the centred moving average of each column, truncated at the ends."""
    import numpy as np
    n_rows = values.shape[0]
    half = int(window_frames) // 2
    cumulative = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    index = np.arange(n_rows)
    low = np.maximum(index - half, 0)
    high = np.minimum(index + half + 1, n_rows)
    return (cumulative[high] - cumulative[low]) / (high - low)[:, None]


import numpy as np
def label_reaction_endpoints(trajectory: np.ndarray, window_frames: int = 5,
                                     r_atomic: float = 1.5, r_molecular: float = 1.0) -> np.ndarray:
    """Reference implementation (smoothed thresholds with the crowding exclusion)."""
    import numpy as np

    if not (_is_integer(window_frames) and int(window_frames) >= 1 and int(window_frames) % 2 == 1):
        raise ValueError("window_frames must be an odd integer of at least 1")
    if not (_is_positive_float(r_atomic) and _is_positive_float(r_molecular)):
        raise ValueError("r_atomic and r_molecular must be positive finite floats")
    if not float(r_molecular) < float(r_atomic):
        raise ValueError("r_molecular must be strictly smaller than r_atomic")
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 3 or traj.shape[0] < 1:
        raise ValueError("trajectory must be a non-empty array of shape (T, 3)")
    if not np.all(np.isfinite(traj)):
        raise ValueError("trajectory must be finite")

    smoothed = _centred_moving_average(traj[:, :2], int(window_frames))
    nearest = smoothed.min(axis=1)
    crowded = np.all(smoothed < float(r_atomic), axis=1)

    labels = np.full(traj.shape[0], 2, dtype=np.int64)
    labels[(~crowded) & (nearest < float(r_molecular))] = 1
    labels[(~crowded) & (nearest > float(r_atomic))] = 0
    return labels

def _fix_loading_signs(loadings):
    """Return loadings with the largest-magnitude entry of each column positive."""
    import numpy as np
    signed = loadings.copy()
    for column in range(signed.shape[1]):
        pivot = int(np.argmax(np.abs(signed[:, column])))
        if signed[pivot, column] < 0.0:
            signed[:, column] = -signed[:, column]
    return signed


import numpy as np
def project_dynamical_components(descriptors: np.ndarray, lag_frames: int = 20,
                                         n_components: int = 5,
                                         rank_cutoff: float = 1e-6) -> np.ndarray:
    """Reference implementation (whitened symmetric lagged eigenproblem)."""
    import numpy as np

    features = np.asarray(descriptors, dtype=float)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("descriptors must be a two-dimensional array with at least two rows")
    if not np.all(np.isfinite(features)):
        raise ValueError("descriptors must be finite")
    n_rows = features.shape[0]
    if not (_is_integer(lag_frames) and 1 <= int(lag_frames) < n_rows):
        raise ValueError("lag_frames must be an integer in 1 .. T - 1")
    if not (_is_integer(n_components) and int(n_components) >= 1):
        raise ValueError("n_components must be an integer of at least 1")
    if isinstance(rank_cutoff, bool) or not isinstance(rank_cutoff, (int, float, np.floating, np.integer)):
        raise ValueError("rank_cutoff must be a float strictly between 0 and 1")
    if not 0.0 < float(rank_cutoff) < 1.0:
        raise ValueError("rank_cutoff must be a float strictly between 0 and 1")

    lag = int(lag_frames)
    centred = features - features.mean(axis=0)
    instant = centred.T @ centred / n_rows
    early, late = centred[:n_rows - lag], centred[lag:]
    lagged = (early.T @ late + late.T @ early) / (2.0 * (n_rows - lag))

    values, vectors = np.linalg.eigh(instant)
    if values.max() <= 0.0:
        raise ValueError("descriptors have no variance")
    keep = values > float(rank_cutoff) * values.max()
    whitener = vectors[:, keep] / np.sqrt(values[keep])
    if int(n_components) > int(keep.sum()):
        raise ValueError("n_components exceeds the retained rank of the covariance")

    reduced = whitener.T @ lagged @ whitener
    reduced = 0.5 * (reduced + reduced.T)
    slow_values, slow_vectors = np.linalg.eigh(reduced)
    order = np.argsort(slow_values)[::-1][:int(n_components)]
    loadings = _fix_loading_signs(whitener @ slow_vectors[:, order])
    return centred @ loadings

def _seed_centres(points, n_centres: int, rng):
    """Return the initial centres chosen by the squared-distance seeding rule."""
    import numpy as np
    n_points = points.shape[0]
    centres = np.empty((n_centres, points.shape[1]), dtype=float)
    centres[0] = points[int(rng.integers(n_points))]
    nearest = np.sum((points - centres[0]) ** 2, axis=1)
    for index in range(1, n_centres):
        total = nearest.sum()
        weights = nearest / total if total > 0.0 else np.full(n_points, 1.0 / n_points)
        centres[index] = points[int(rng.choice(n_points, p=weights))]
        nearest = np.minimum(nearest, np.sum((points - centres[index]) ** 2, axis=1))
    return centres


def _lloyd_refine(points, centres, max_iter: int):
    """Return the assignment produced by Lloyd iterations from given centres."""
    import numpy as np
    assignment = np.full(points.shape[0], -1, dtype=np.int64)
    for _ in range(int(max_iter)):
        distances = np.sum((points[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        proposal = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(proposal, assignment):
            break
        assignment = proposal
        for index in range(centres.shape[0]):
            member = assignment == index
            if member.any():
                centres[index] = points[member].mean(axis=0)
    return assignment


import numpy as np
def assign_intermediate_microstates(components: np.ndarray, labels: np.ndarray,
                                            n_microstates: int = 64, seed: int = 0,
                                            max_iter: int = 100) -> np.ndarray:
    """Reference implementation (squared-distance seeding plus Lloyd refinement)."""
    import numpy as np

    coords = np.asarray(components, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 1 or coords.shape[1] < 1:
        raise ValueError("components must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(coords)):
        raise ValueError("components must be finite")
    tags = np.asarray(labels)
    if tags.ndim != 1 or tags.shape[0] != coords.shape[0] or not np.issubdtype(tags.dtype, np.integer):
        raise ValueError("labels must be an integer array matching the number of rows")
    if not np.all(np.isin(tags, (0, 1, 2))):
        raise ValueError("labels must contain only 0, 1 and 2")
    if not _is_integer(seed):
        raise ValueError("seed must be an integer")
    if not (_is_integer(max_iter) and int(max_iter) >= 1):
        raise ValueError("max_iter must be an integer of at least 1")
    interior = np.flatnonzero(tags == 2)
    if not (_is_integer(n_microstates) and 1 <= int(n_microstates) <= interior.size):
        raise ValueError("n_microstates must be an integer between 1 and the number of intermediate frames")

    points = coords[interior]
    rng = np.random.default_rng(int(seed))
    centres = _seed_centres(points, int(n_microstates), rng)
    assignment = _lloyd_refine(points, centres, int(max_iter))

    microstates = np.full(coords.shape[0], -1, dtype=np.int64)
    microstates[interior] = assignment
    return microstates

def _validated_labels(labels):
    """Return the labels as a one-dimensional integer array of the right shape."""
    import numpy as np
    tags = np.asarray(labels)
    if tags.ndim != 1 or tags.shape[0] < 1 or not np.issubdtype(tags.dtype, np.integer):
        raise ValueError("labels must be a non-empty one-dimensional integer array")
    return tags.astype(np.int64)


import numpy as np
def compute_stopping_times(labels: np.ndarray) -> np.ndarray:
    """Reference implementation (running minima and maxima of endpoint indices)."""
    import numpy as np

    tags = _validated_labels(labels)
    if not np.all(np.isin(tags, (0, 1, 2))):
        raise ValueError("labels must contain only 0, 1 and 2")
    n_frames = tags.shape[0]
    index = np.arange(n_frames, dtype=np.int64)
    endpoint = tags < 2

    forward = np.minimum.accumulate(np.where(endpoint, index, n_frames)[::-1])[::-1]
    backward = np.maximum.accumulate(np.where(endpoint, index, -1))
    return np.column_stack([forward, backward]).astype(np.int64)

def _check_state_arrays(labels, microstates, stopping, n_microstates):
    """Return the validated label, microstate and stopping arrays."""
    import numpy as np
    tags = _validated_labels(labels)
    if not np.all(np.isin(tags, (0, 1, 2))):
        raise ValueError("labels must contain only 0, 1 and 2")
    cells = np.asarray(microstates)
    stops = np.asarray(stopping)
    if cells.ndim != 1 or cells.shape[0] != tags.shape[0] or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("microstates must be an integer array matching labels")
    if stops.ndim != 2 or stops.shape != (tags.shape[0], 2) or not np.issubdtype(stops.dtype, np.integer):
        raise ValueError("stopping must be an integer array of shape (T, 2)")
    if not (_is_integer(n_microstates) and int(n_microstates) >= 1):
        raise ValueError("n_microstates must be an integer of at least 1")
    interior = tags == 2
    if np.any(cells[~interior] != -1):
        raise ValueError("only intermediate frames may carry a microstate index")
    if interior.any():
        inside = cells[interior]
        if inside.min() < 0 or inside.max() >= int(n_microstates):
            raise ValueError("microstate indices must lie in 0 .. n_microstates - 1")
    return tags, cells.astype(np.int64), stops.astype(np.int64)


def _accumulate_half(system, rows, cols, tags_at_cols, n_microstates):
    """Add one directional half of the Galerkin estimate in place."""
    import numpy as np
    stays = tags_at_cols == 2
    np.add.at(system, (rows[stays], cols[stays]), 1.0)
    reaches = tags_at_cols == 1
    np.add.at(system[:, n_microstates], rows[reaches], 1.0)
    np.add.at(system, (rows, rows), -1.0)


import numpy as np
def build_galerkin_system(labels: np.ndarray, microstates: np.ndarray,
                                  stopping: np.ndarray, lag_frames: int = 20,
                                  n_microstates: int = 64) -> np.ndarray:
    """Reference implementation (forward and time-reversed halted windows)."""
    import numpy as np

    tags, cells, stops = _check_state_arrays(labels, microstates, stopping, n_microstates)
    n_frames = tags.shape[0]
    if not (_is_integer(lag_frames) and 1 <= int(lag_frames) < n_frames):
        raise ValueError("lag_frames must be an integer in 1 .. T - 1")

    lag = int(lag_frames)
    size = int(n_microstates)
    start = np.arange(n_frames - lag, dtype=np.int64)
    system = np.zeros((size, size + 1), dtype=float)

    forward = start[tags[start] == 2]
    reached = np.minimum(forward + lag, stops[forward, 0])
    _accumulate_half(system, cells[forward], cells[reached], tags[reached], size)

    backward = (start + lag)[tags[start + lag] == 2]
    left = np.maximum(backward - lag, stops[backward, 1])
    _accumulate_half(system, cells[backward], cells[left], tags[left], size)

    return system / (2.0 * (n_frames - lag))

def _solve_basis_coefficients(system):
    """Return the coefficient vector of the projected boundary value problem."""
    import numpy as np
    matrix = np.asarray(system, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != matrix.shape[0] + 1:
        raise ValueError("system must have shape (n, n + 1) with n at least 1")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("system must be finite")
    try:
        return np.linalg.solve(matrix[:, :-1], -matrix[:, -1])
    except np.linalg.LinAlgError as error:
        raise ValueError("the Galerkin matrix is singular") from error


import numpy as np
def solve_committor(labels: np.ndarray, microstates: np.ndarray,
                            system: np.ndarray) -> np.ndarray:
    """Reference implementation (linear solve plus indicator expansion)."""
    import numpy as np

    tags = _validated_labels(labels)
    if not np.all(np.isin(tags, (0, 1, 2))):
        raise ValueError("labels must contain only 0, 1 and 2")
    cells = np.asarray(microstates)
    if cells.ndim != 1 or cells.shape[0] != tags.shape[0] or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("microstates must be an integer array matching labels")
    coefficients = _solve_basis_coefficients(system)

    interior = tags == 2
    if np.any(cells[~interior] != -1):
        raise ValueError("only intermediate frames may carry a microstate index")
    if interior.any():
        inside = cells[interior]
        if inside.min() < 0 or inside.max() >= coefficients.size:
            raise ValueError("microstate indices must lie in 0 .. n - 1")

    committor = np.where(tags == 1, 1.0, 0.0)
    committor[interior] = coefficients[cells[interior]]
    return committor

def _window_flux(forward, backward, stops, starts, lag):
    """Return the summed flux of one block of windows in both time directions."""
    import numpy as np
    offsets = np.arange(lag, dtype=np.int64)[None, :]
    origin = starts[:, None]
    steps = origin + offsets
    left = np.maximum(origin, stops[steps, 1])
    right = np.minimum(origin + lag, stops[steps + 1, 0])
    total = float(np.sum(backward[left] * (forward[steps + 1] - forward[steps]) * forward[right]))
    total += float(np.sum(forward[left] * (backward[steps + 1] - backward[steps]) * backward[right]))
    return total


import numpy as np
def estimate_reactive_flux(labels: np.ndarray, committor: np.ndarray,
                                   stopping: np.ndarray, lag_frames: int = 20,
                                   block_frames: int = 20000) -> float:
    """Reference implementation (halted rolling-window reactive current)."""
    import numpy as np

    tags = _validated_labels(labels)
    if not np.all(np.isin(tags, (0, 1, 2))):
        raise ValueError("labels must contain only 0, 1 and 2")
    n_frames = tags.shape[0]
    forward = np.asarray(committor, dtype=float)
    if forward.ndim != 1 or forward.shape[0] != n_frames or not np.all(np.isfinite(forward)):
        raise ValueError("committor must be a finite one-dimensional array matching labels")
    stops = np.asarray(stopping)
    if stops.ndim != 2 or stops.shape != (n_frames, 2) or not np.issubdtype(stops.dtype, np.integer):
        raise ValueError("stopping must be an integer array of shape (T, 2)")
    if not (_is_integer(lag_frames) and 1 <= int(lag_frames) < n_frames):
        raise ValueError("lag_frames must be an integer in 1 .. T - 1")
    if not (_is_integer(block_frames) and int(block_frames) >= 1):
        raise ValueError("block_frames must be an integer of at least 1")

    lag = int(lag_frames)
    block = int(block_frames)
    stops = stops.astype(np.int64)
    backward = 1.0 - forward
    n_windows = n_frames - lag

    total = 0.0
    for low in range(0, n_windows, block):
        starts = np.arange(low, min(low + block, n_windows), dtype=np.int64)
        total += _window_flux(forward, backward, stops, starts, lag)
    return float(total / (2.0 * n_windows * lag))

import numpy as np
def compute_association_rate(n_frames: int = 200000, seed: int = 20260212,
                                     dt_ps: float = 0.01, lag_frames: int = 20,
                                     n_components: int = 5, n_microstates: int = 64,
                                     cluster_seed: int = 0) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    if not _is_positive_float(dt_ps):
        raise ValueError("dt_ps must be a positive finite float")

    trajectory = simulate_surface_trajectory(n_frames, seed, dt_ps)
    descriptors = build_local_descriptors(trajectory)
    labels = label_reaction_endpoints(trajectory)
    components = project_dynamical_components(descriptors, lag_frames, n_components)
    microstates = assign_intermediate_microstates(components, labels,
                                                          n_microstates, cluster_seed)
    stopping = compute_stopping_times(labels)
    system = build_galerkin_system(labels, microstates, stopping,
                                           lag_frames, n_microstates)
    committor = solve_committor(labels, microstates, system)
    flux = estimate_reactive_flux(labels, committor, stopping, lag_frames)

    backward_mean = float(np.mean(1.0 - committor))
    if not backward_mean > 0.0:
        raise ValueError("the mean backward committor must be positive")
    return float(flux / (backward_mean * float(dt_ps)))
SCICODE_GOLD_EOF
