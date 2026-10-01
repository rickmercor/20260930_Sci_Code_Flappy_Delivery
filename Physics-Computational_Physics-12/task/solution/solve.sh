#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def _design(times, degree, drop_constant):
    import numpy as np
    powers = np.arange(1, degree + 1) if drop_constant else np.arange(0, degree + 1)
    return np.power.outer(np.asarray(times, dtype=float), powers)


def _fit_leading(times, values, drop_constant):
    import numpy as np
    matrix = _design(times[:6], 3, drop_constant)
    solution, _, _, _ = np.linalg.lstsq(matrix, np.asarray(values, dtype=float)[:6], rcond=None)
    return float(solution[0])


def short_time_coefficients(node_times, recorded_density):
    import numpy as np
    times = np.asarray(node_times, dtype=float)
    density = np.asarray(recorded_density, dtype=float)
    if times.ndim != 1 or times.size < 6:
        raise ValueError("node_times must be a one-dimensional array with at least six entries")
    if density.shape != (4, 4, times.size):
        raise ValueError("recorded_density must have shape (4, 4, len(node_times))")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(density)):
        raise ValueError("node_times and recorded_density must be finite")
    if np.any(np.diff(times) <= 0.0) or times[0] <= 0.0:
        raise ValueError("node_times must be positive and strictly increasing")
    if np.any(density < 0.0):
        raise ValueError("recorded_density must be non-negative")
    out = np.zeros((2, 4))
    for link in range(2):
        plus, minus = 2 * link, 2 * link + 1
        out[link, 0] = _fit_leading(times, density[minus, plus], False)
        out[link, 1] = _fit_leading(times, density[plus, minus], False)
        out[link, 2] = _fit_leading(times, density[plus, plus], True)
        out[link, 3] = _fit_leading(times, density[minus, minus], True)
    return out

def detection_and_rate_constants(short_time):
    import numpy as np
    values = np.asarray(short_time, dtype=float)
    if values.shape != (2, 4) or not np.all(np.isfinite(values)):
        raise ValueError("short_time must be a finite array of shape (2, 4)")
    out = np.zeros((2, 4))
    for link in range(2):
        c_plus, c_minus, a_pp, a_mm = values[link]
        if c_plus <= 0.0 or c_minus <= 0.0:
            raise ValueError("both zero-time densities of each junction must be positive")
        if a_pp < 0.0 or a_mm < 0.0:
            raise ValueError("both zero-time slopes of each junction must be non-negative")
        product = c_plus * c_minus
        eta_plus = product / (product + a_mm)
        eta_minus = product / (product + a_pp)
        out[link] = [eta_plus, eta_minus, c_plus / eta_plus, c_minus / eta_minus]
    return out

def kernel_moments(node_times, bin_widths, recorded_density):
    import numpy as np
    times = np.asarray(node_times, dtype=float)
    widths = np.asarray(bin_widths, dtype=float)
    density = np.asarray(recorded_density, dtype=float)
    if times.ndim != 1 or widths.shape != times.shape:
        raise ValueError("node_times and bin_widths must be one-dimensional arrays of equal length")
    if density.shape != (4, 4, times.size):
        raise ValueError("recorded_density must have shape (4, 4, len(node_times))")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(widths)) and np.all(np.isfinite(density))):
        raise ValueError("all inputs must be finite")
    if np.any(widths <= 0.0) or np.any(density < 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("bin widths must be positive, densities non-negative, node times increasing")
    weight = density * widths
    mass = weight.sum(axis=2)
    first = (weight * times).sum(axis=2)
    second = (weight * times ** 2).sum(axis=2)
    row_total = mass.sum(axis=1)
    if np.any(row_total <= 0.0):
        raise ValueError("each row of the recorded kernel must carry positive weight")
    scale = row_total.reshape(4, 1)
    return np.stack([mass / scale, first / scale, second / scale])

def _stationary(matrix):
    import numpy as np
    system = np.vstack([matrix.T - np.eye(4), np.ones(4)])
    target = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    solution, _, _, _ = np.linalg.lstsq(system, target, rcond=None)
    return solution


def junction_dissipation(moments, detection_and_rates):
    import numpy as np
    stack = np.asarray(moments, dtype=float)
    values = np.asarray(detection_and_rates, dtype=float)
    if stack.shape != (3, 4, 4) or not np.all(np.isfinite(stack)):
        raise ValueError("moments must be a finite array of shape (3, 4, 4)")
    if values.shape != (2, 4) or not np.all(np.isfinite(values)):
        raise ValueError("detection_and_rates must be a finite array of shape (2, 4)")
    if np.any(values[:, :2] <= 0.0) or np.any(values[:, :2] > 1.0):
        raise ValueError("registration probabilities must lie in (0, 1]")
    if np.any(values[:, 2:] <= 0.0):
        raise ValueError("every junction rate constant must be positive")
    transition, first = stack[0], stack[1]
    weights = _stationary(transition)
    mean_time = float(weights @ first.sum(axis=1))
    if mean_time <= 0.0:
        raise ValueError("mean registered waiting time must be positive")
    frequencies = weights / mean_time
    dissipation = np.zeros(2)
    for link in range(2):
        eta_plus, eta_minus, k_plus, k_minus = values[link]
        restored = frequencies[2 * link] / eta_plus - frequencies[2 * link + 1] / eta_minus
        dissipation[link] = restored * np.log(k_plus / k_minus)
    return np.concatenate([frequencies, dissipation])

def discarded_record_moments(moments, keep_probabilities):
    import numpy as np
    stack = np.asarray(moments, dtype=float)
    keep = np.asarray(keep_probabilities, dtype=float).reshape(-1)
    if stack.shape != (3, 4, 4) or not np.all(np.isfinite(stack)):
        raise ValueError("moments must be a finite array of shape (3, 4, 4)")
    if keep.size != 4 or not np.all(np.isfinite(keep)):
        raise ValueError("keep_probabilities must hold four finite numbers")
    if np.any(keep <= 0.0) or np.any(keep > 1.0):
        raise ValueError("keep probabilities must lie in (0, 1]")
    transition, first, second = stack
    kept = np.diag(keep)
    dropped = np.diag(1.0 - keep)
    a0, a1, a2 = transition, -first, 0.5 * second
    b0, b1, b2 = a0 @ dropped, a1 @ dropped, a2 @ dropped
    n0, n1, n2 = a0 @ kept, a1 @ kept, a2 @ kept
    core = np.eye(4) - b0
    if abs(np.linalg.det(core)) < 1e-14:
        raise ValueError("surviving kernel resummation is singular")
    y0 = np.linalg.inv(core)
    y1 = y0 @ b1 @ y0
    y2 = y0 @ b2 @ y0 + y0 @ b1 @ y0 @ b1 @ y0
    c0 = y0 @ n0
    c1 = y0 @ n1 + y1 @ n0
    c2 = y0 @ n2 + y1 @ n1 + y2 @ n0
    return np.stack([c0, -c1, 2.0 * c2])

def _swap_columns(base, replacements):
    import numpy as np
    matrix = base.copy()
    for column, values in replacements.items():
        matrix[:, column] = values
    return matrix


def _first_variation(base, derivative):
    import numpy as np
    return sum(np.linalg.det(_swap_columns(base, {k: derivative[:, k]})) for k in range(base.shape[0]))


def _second_variation(base, first, other, mixed):
    import numpy as np
    size = base.shape[0]
    total = sum(np.linalg.det(_swap_columns(base, {k: mixed[:, k]})) for k in range(size))
    for k in range(size):
        for m in range(size):
            if k == m:
                continue
            total += np.linalg.det(_swap_columns(base, {k: first[:, k], m: other[:, m]}))
    return total


def counting_cumulants(moments):
    import numpy as np
    stack = np.asarray(moments, dtype=float)
    if stack.shape != (3, 4, 4) or not np.all(np.isfinite(stack)):
        raise ValueError("moments must be a finite array of shape (3, 4, 4)")
    transition, first, second = stack
    base = transition - np.eye(4)
    tilt = [transition * np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])[i] for i in range(2)]
    tilt_two = [[transition * np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])[i] * np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])[j] for j in range(2)] for i in range(2)]
    lap = -first
    lap_two = second
    cross = [-first * np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])[i] for i in range(2)]
    grad_lap = _first_variation(base, lap)
    if abs(grad_lap) < 1e-300:
        raise ValueError("degenerate kernel: the characteristic determinant is stationary")
    current = np.array([-_first_variation(base, tilt[i]) / grad_lap for i in range(2)])
    diffusion = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            curvature = (_second_variation(base, tilt[i], tilt[j], tilt_two[i][j])
                         + _second_variation(base, tilt[i], lap, cross[i]) * current[j]
                         + _second_variation(base, lap, tilt[j], cross[j]) * current[i]
                         + _second_variation(base, lap, lap, lap_two) * current[i] * current[j])
            diffusion[i, j] = -0.5 * curvature / grad_lap
    if diffusion[0, 0] <= 0.0 or diffusion[1, 1] <= 0.0:
        raise ValueError("both diffusion coefficients must be positive")
    determinant = diffusion[0, 0] * diffusion[1, 1] - diffusion[0, 1] * diffusion[1, 0]
    if determinant <= 0.0:
        raise ValueError("the diffusion matrix must be positive definite")
    ratio = float(current @ np.linalg.solve(diffusion, current))
    return np.array([current[0], current[1], diffusion[0, 0], diffusion[1, 1],
                     0.5 * (diffusion[0, 1] + diffusion[1, 0]), ratio], dtype=float)

def _working_kernel(times, density, short_time):
    import numpy as np
    augmented_times = np.concatenate([[0.0], times])
    augmented = np.zeros((4, 4, augmented_times.size))
    augmented[:, :, 1:] = density
    for link in range(2):
        plus, minus = 2 * link, 2 * link + 1
        augmented[minus, plus, 0] = short_time[link, 0]
        augmented[plus, minus, 0] = short_time[link, 1]
    grid = np.linspace(0.0, augmented_times[-1], 401)
    resampled = np.zeros((4, 4, 401))
    for previous in range(4):
        for following in range(4):
            resampled[previous, following] = np.interp(
                grid, augmented_times, augmented[previous, following]
            )
    return grid, resampled


def _survivor_kernel(grid, kernel, keep):
    import numpy as np
    step = grid[1] - grid[0]
    kept = np.diag(keep)
    dropped = np.diag(1.0 - keep)
    source = np.einsum('rsn,st->nrt', kernel, kept)
    feedback = np.einsum('rsn,st->nrt', kernel, dropped)
    surviving = np.zeros_like(source)
    surviving[0] = source[0]
    core = np.eye(4) - 0.5 * step * feedback[0]
    inverse = np.linalg.inv(core)
    for index in range(1, grid.size):
        weights = np.full(index, step)
        weights[-1] = 0.5 * step
        accumulated = np.einsum(
            'k,kij,kjl->il', weights, feedback[1:index + 1], surviving[index - 1::-1]
        )
        surviving[index] = inverse @ (source[index] + accumulated)
    return np.moveaxis(surviving, 0, 2)


def _integrate(values, grid):
    import numpy as np
    spacing = np.diff(grid)
    return np.sum(0.5 * spacing * (values[..., 1:] + values[..., :-1]), axis=-1)


def _stationary(matrix):
    import numpy as np
    system = np.vstack([matrix.T - np.eye(4), np.ones(4)])
    solution, _, _, _ = np.linalg.lstsq(system, np.array([0.0, 0.0, 0.0, 0.0, 1.0]), rcond=None)
    return solution


def waiting_time_irreversibility(node_times, recorded_density, short_time, keep_probabilities):
    import numpy as np
    times = np.asarray(node_times, dtype=float)
    density = np.asarray(recorded_density, dtype=float)
    coefficients = np.asarray(short_time, dtype=float)
    keep = np.asarray(keep_probabilities, dtype=float).reshape(-1)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("node_times must be a one-dimensional array with at least two entries")
    if density.shape != (4, 4, times.size):
        raise ValueError("recorded_density must have shape (4, 4, len(node_times))")
    if coefficients.shape != (2, 4) or keep.size != 4:
        raise ValueError("short_time must have shape (2, 4) and keep_probabilities four entries")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(density))
            and np.all(np.isfinite(coefficients)) and np.all(np.isfinite(keep))):
        raise ValueError("all inputs must be finite")
    if np.any(np.diff(times) <= 0.0) or times[0] <= 0.0:
        raise ValueError("node_times must be positive and strictly increasing")
    if np.any(keep <= 0.0) or np.any(keep > 1.0):
        raise ValueError("keep probabilities must lie in (0, 1]")
    grid, kernel = _working_kernel(times, density, coefficients)
    surviving = _survivor_kernel(grid, kernel, keep)
    mass = _integrate(surviving, grid)
    first = _integrate(surviving * grid, grid)
    row_total = mass.sum(axis=1)
    if np.any(row_total <= 0.0):
        raise ValueError("each row of the surviving kernel must carry positive weight")
    scale = row_total.reshape(4, 1)
    weights = _stationary(mass / scale)
    mean_time = float(weights @ (first / scale).sum(axis=1))
    if mean_time <= 0.0:
        raise ValueError("mean surviving waiting time must be positive")
    frequencies = weights / mean_time
    values = np.clip(surviving, 0.0, None)
    floor = 1e-300
    total = 0.0
    for previous in range(4):
        for following in range(4):
            back_previous = (1, 0, 3, 2)[following]
            back_following = (1, 0, 3, 2)[previous]
            if back_previous == previous and back_following == following:
                continue
            forward = values[previous, following]
            reverse = values[back_previous, back_following]
            total += float(_integrate(
                frequencies[previous] * forward * np.log((forward + floor) / (reverse + floor)), grid
            ))
    return np.array([frequencies[0], frequencies[1], frequencies[2], frequencies[3], total], dtype=float)

def _common_levels(detection):
    import numpy as np
    return np.array([min(detection[0, 0], detection[0, 1]),
                     min(detection[1, 0], detection[1, 1])])


def _keep_vector(detection, level):
    import numpy as np
    return np.array([level[0] / detection[0, 0], level[0] / detection[0, 1],
                     level[1] / detection[1, 0], level[1] / detection[1, 1]])


def _registration_constants(moments, detection):
    import numpy as np
    """Seven constants of the registration dependence of the joint cumulants."""
    reference = _common_levels(detection)
    points = [(reference[0], reference[1]),
              (0.5 * reference[0], reference[1]),
              (reference[0], 0.5 * reference[1])]
    cumulants = []
    for level in points:
        surviving = discarded_record_moments(moments, _keep_vector(detection, level))
        cumulants.append(counting_cumulants(surviving))
    alpha_a = cumulants[0][0] / points[0][0]
    alpha_b = cumulants[0][1] / points[0][1]
    system_a = np.array([[points[0][0], points[0][0] ** 2], [points[1][0], points[1][0] ** 2]])
    system_b = np.array([[points[0][1], points[0][1] ** 2], [points[2][1], points[2][1] ** 2]])
    if abs(np.linalg.det(system_a)) < 1e-300 or abs(np.linalg.det(system_b)) < 1e-300:
        raise ValueError("degenerate registration dependence")
    beta_a, gamma_a = np.linalg.solve(system_a, np.array([cumulants[0][2], cumulants[1][2]]))
    beta_b, gamma_b = np.linalg.solve(system_b, np.array([cumulants[0][3], cumulants[2][3]]))
    delta = cumulants[0][4] / (points[0][0] * points[0][1])
    return np.array([alpha_a, alpha_b, beta_a, gamma_a, beta_b, gamma_b, delta], dtype=float)


def _complete_registration_value(moments, detection):
    import numpy as np
    alpha_a, alpha_b, beta_a, gamma_a, beta_b, gamma_b, delta = _registration_constants(moments, detection)
    current = np.array([alpha_a, alpha_b])
    diffusion = np.array([[beta_a + gamma_a, delta], [delta, beta_b + gamma_b]])
    if diffusion[0, 0] <= 0.0 or diffusion[1, 1] <= 0.0:
        raise ValueError("the extended diffusion coefficients must be positive")
    if np.linalg.det(diffusion) <= 0.0:
        raise ValueError("the extended diffusion matrix must be positive definite")
    return float(current @ np.linalg.solve(diffusion, current))


def certified_dissipation_ratio(node_times, bin_widths, recorded_density):
    import numpy as np
    times = np.asarray(node_times, dtype=float)
    widths = np.asarray(bin_widths, dtype=float)
    density = np.asarray(recorded_density, dtype=float)
    if times.ndim != 1 or widths.shape != times.shape:
        raise ValueError("node_times and bin_widths must be one-dimensional arrays of equal length")
    if density.shape != (4, 4, times.size):
        raise ValueError("recorded_density must have shape (4, 4, len(node_times))")
    short_time = short_time_coefficients(times, density)
    detection = detection_and_rate_constants(short_time)
    moments = kernel_moments(times, widths, density)
    junction = junction_dissipation(moments, detection)
    uncertainty = _complete_registration_value(moments, detection)
    keep = _keep_vector(detection, _common_levels(detection))
    waiting = waiting_time_irreversibility(times, density, short_time, keep)
    resolved = junction[4] + junction[5]
    if resolved <= 0.0:
        raise ValueError("the resolved junctions must carry positive dissipation together")
    certified = max(uncertainty, waiting[4])
    return float(certified / resolved)
SCICODE_GOLD_EOF
