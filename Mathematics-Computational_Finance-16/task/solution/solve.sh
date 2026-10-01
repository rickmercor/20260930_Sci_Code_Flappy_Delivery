#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np  # noqa: E402, F811


def advance_controlled_diffusion(
    states: np.ndarray,
    noises: np.ndarray,
    drift_matrix: np.ndarray,
    drift_bias: np.ndarray,
    sigma_diag: np.ndarray,
    policy_matrix: np.ndarray,
    policy_bias: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Reference implementation."""
    x = np.asarray(states, dtype=float)
    xi = np.asarray(noises, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("states must be a non-empty two-dimensional array")
    if xi.shape != x.shape:
        raise ValueError("noises must have the same shape as states")
    sample_count, dimension = x.shape
    if np.asarray(drift_matrix).shape != (dimension, dimension):
        raise ValueError("drift_matrix must have shape (d, d)")
    if np.asarray(drift_bias).shape != (dimension,):
        raise ValueError("drift_bias must have shape (d,)")
    if np.asarray(sigma_diag).shape != (dimension,):
        raise ValueError("sigma_diag must have shape (d,)")
    if np.asarray(policy_matrix).shape != (dimension, dimension):
        raise ValueError("policy_matrix must have shape (d, d)")
    if np.asarray(policy_bias).shape != (dimension,):
        raise ValueError("policy_bias must have shape (d,)")
    arrays = (
        x,
        xi,
        np.asarray(drift_matrix, dtype=float),
        np.asarray(drift_bias, dtype=float),
        np.asarray(sigma_diag, dtype=float),
        np.asarray(policy_matrix, dtype=float),
        np.asarray(policy_bias, dtype=float),
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all array inputs must be finite")
    sigma = arrays[4]
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_diag must be positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    drift = x @ arrays[2].T + arrays[3]
    policy = x @ arrays[5].T + arrays[6]
    diffusion = np.broadcast_to(sigma, (sample_count, dimension))
    return x + (drift + diffusion * policy) * dt + diffusion * xi * np.sqrt(dt)

import numpy as np  # noqa: E402, F811


def evaluate_tt_continuation(
    points: np.ndarray, cores: list[np.ndarray]
) -> np.ndarray:
    """Reference implementation."""
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("points must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(x)):
        raise ValueError("points must be finite")
    if not isinstance(cores, (list, tuple)) or len(cores) != x.shape[1]:
        raise ValueError("cores must contain one tensor for each coordinate")
    parsed = [np.asarray(core, dtype=float) for core in cores]
    if any(core.ndim != 3 or core.shape[1] < 2 for core in parsed):
        raise ValueError("every core must have shape (r_left, m_i, r_right) with m_i >= 2")
    if parsed[0].shape[0] != 1 or parsed[-1].shape[2] != 1:
        raise ValueError("the exterior TT ranks must equal one")
    if any(parsed[i].shape[2] != parsed[i + 1].shape[0] for i in range(len(parsed) - 1)):
        raise ValueError("adjacent TT ranks must agree")
    if any(not np.all(np.isfinite(core)) for core in parsed):
        raise ValueError("cores must be finite")

    sample_count, dimension = x.shape
    result = np.empty((sample_count, dimension + 1), dtype=float)
    for k in range(sample_count):
        basis = []
        derivative = []
        for z, core in zip(x[k], parsed):
            degree = np.arange(core.shape[1], dtype=int)
            basis.append(np.power(z, degree))
            differentiated = np.zeros(core.shape[1], dtype=float)
            differentiated[1:] = degree[1:] * np.power(z, degree[:-1])
            derivative.append(differentiated)
        value_product = np.ones((1, 1), dtype=float)
        for coordinate, core in enumerate(parsed):
            value_product = value_product @ np.tensordot(
                core, basis[coordinate], axes=(1, 0)
            )
        result[k, 0] = value_product.item()
        for differentiated in range(dimension):
            gradient_product = np.ones((1, 1), dtype=float)
            for coordinate, core in enumerate(parsed):
                vector = derivative[coordinate] if coordinate == differentiated else basis[coordinate]
                gradient_product = gradient_product @ np.tensordot(core, vector, axes=(1, 0))
            result[k, differentiated + 1] = gradient_product.item()
    return result

import numpy as np  # noqa: E402, F811


def form_bsde_targets(
    continuation: np.ndarray,
    next_states: np.ndarray,
    policy_matrix: np.ndarray,
    policy_bias: np.ndarray,
    drift_trace: float,
    sigma_diag: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Reference implementation."""
    data = np.asarray(continuation, dtype=float)
    points = np.asarray(next_states, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("next_states must be a non-empty two-dimensional array")
    sample_count, dimension = points.shape
    if data.shape != (sample_count, dimension + 1):
        raise ValueError("continuation must have shape (K, d + 1)")
    if np.asarray(policy_matrix).shape != (dimension, dimension):
        raise ValueError("policy_matrix must have shape (d, d)")
    if np.asarray(policy_bias).shape != (dimension,):
        raise ValueError("policy_bias must have shape (d,)")
    if np.asarray(sigma_diag).shape != (dimension,):
        raise ValueError("sigma_diag must have shape (d,)")
    arrays = (
        data,
        points,
        np.asarray(policy_matrix, dtype=float),
        np.asarray(policy_bias, dtype=float),
        np.asarray(sigma_diag, dtype=float),
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all array inputs must be finite")
    if not np.isfinite(drift_trace):
        raise ValueError("drift_trace must be finite")
    if np.any(arrays[4] <= 0.0):
        raise ValueError("sigma_diag must be positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    value = data[:, 0]
    gradient = data[:, 1:]
    scaled_gradient = gradient * arrays[4]
    policy = points @ arrays[2].T + arrays[3]
    nonlinear_term = (
        float(drift_trace)
        + 0.5 * np.sum(scaled_gradient * scaled_gradient, axis=1)
        + np.sum(policy * scaled_gradient, axis=1)
    )
    return value - nonlinear_term * dt

import numpy as np  # noqa: E402, F811


def build_weighted_features(
    points: np.ndarray,
    noises: np.ndarray,
    sigma_diag: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Reference implementation."""
    x = np.asarray(points, dtype=float)
    xi = np.asarray(noises, dtype=float)
    sigma = np.asarray(sigma_diag, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("points must be a non-empty two-dimensional array")
    if xi.shape != x.shape:
        raise ValueError("noises must have the same shape as points")
    if sigma.shape != (x.shape[1],):
        raise ValueError("sigma_diag must have shape (d,)")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(xi)) or not np.all(np.isfinite(sigma)):
        raise ValueError("array inputs must be finite")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_diag must be positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    sample_count, dimension = x.shape
    basis = np.stack((np.ones_like(x), x, x * x), axis=2)
    derivative = np.stack((np.zeros_like(x), np.ones_like(x), 2.0 * x), axis=2)
    weighted = np.broadcast_to(basis, (dimension + 1, sample_count, dimension, 3)).copy()
    sigma_noise = xi * sigma[None, :] * np.sqrt(dt)
    for coordinate in range(dimension):
        weighted[coordinate + 1, :, coordinate, :] = (
            sigma_noise[:, coordinate, None] * derivative[:, coordinate, :]
        )
    return weighted

import numpy as np  # noqa: E402, F811


def assemble_local_system(
    weighted_features: np.ndarray,
    cores: list[np.ndarray],
    core_index: int,
) -> np.ndarray:
    """Reference implementation."""
    features = np.asarray(weighted_features, dtype=float)
    if features.ndim != 4:
        raise ValueError("weighted_features must have four dimensions")
    channel_count, sample_count, dimension, basis_count = features.shape
    if (
        dimension == 0
        or sample_count == 0
        or basis_count == 0
        or channel_count <= 1
        or (channel_count - 1) % dimension != 0
    ):
        raise ValueError("weighted_features must have shape (1 + q*d, K, d, m) for q >= 1")
    if not np.all(np.isfinite(features)):
        raise ValueError("weighted_features must be finite")
    if not isinstance(core_index, (int, np.integer)) or not 0 <= int(core_index) < dimension:
        raise ValueError("core_index must identify one TT core")
    if not isinstance(cores, (list, tuple)) or len(cores) != dimension:
        raise ValueError("cores must contain one tensor per coordinate")
    parsed = [np.asarray(core, dtype=float) for core in cores]
    if any(core.ndim != 3 or core.shape[1] != basis_count for core in parsed):
        raise ValueError("every core must have the feature basis mode")
    if parsed[0].shape[0] != 1 or parsed[-1].shape[2] != 1:
        raise ValueError("the exterior TT ranks must equal one")
    if any(parsed[i].shape[2] != parsed[i + 1].shape[0] for i in range(dimension - 1)):
        raise ValueError("adjacent TT ranks must agree")
    if any(not np.all(np.isfinite(core)) for core in parsed):
        raise ValueError("cores must be finite")

    position = int(core_index)
    local_shape = parsed[position].shape
    matrix = np.zeros((sample_count, int(np.prod(local_shape))), dtype=float)
    for k in range(sample_count):
        local_row = np.zeros(local_shape, dtype=float)
        for channel in range(channel_count):
            left = np.ones(1, dtype=float)
            for coordinate in range(position):
                evaluated = np.tensordot(
                    parsed[coordinate], features[channel, k, coordinate], axes=(1, 0)
                )
                left = left @ evaluated
            right = np.ones(1, dtype=float)
            for coordinate in range(dimension - 1, position, -1):
                evaluated = np.tensordot(
                    parsed[coordinate], features[channel, k, coordinate], axes=(1, 0)
                )
                right = evaluated @ right
            local_row += np.einsum(
                "a,q,b->aqb", left, features[channel, k, position], right
            )
        matrix[k] = local_row.reshape(-1)
    return matrix

import numpy as np  # noqa: E402, F811


def solve_ridge_core(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    tau: float,
    core_shape: tuple[int, int, int],
) -> np.ndarray:
    """Reference implementation."""
    matrix = np.asarray(design_matrix, dtype=float)
    y = np.asarray(targets, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("design_matrix must be non-empty and two dimensional")
    if y.shape != (matrix.shape[0],):
        raise ValueError("targets must have one entry per matrix row")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(y)):
        raise ValueError("design_matrix and targets must be finite")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    if not isinstance(core_shape, (tuple, list)) or len(core_shape) != 3:
        raise ValueError("core_shape must contain three dimensions")
    if any(not isinstance(size, (int, np.integer)) or int(size) <= 0 for size in core_shape):
        raise ValueError("core_shape dimensions must be positive integers")
    shape = tuple(int(size) for size in core_shape)
    if int(np.prod(shape)) != matrix.shape[1]:
        raise ValueError("core_shape product must match the matrix column count")

    normal_matrix = matrix.T @ matrix + float(tau) * np.eye(matrix.shape[1])
    coefficients = np.linalg.solve(normal_matrix, matrix.T @ y)
    return coefficients.reshape(shape)

import numpy as np  # noqa: E402, F811


def _assemble_last_system(features, cores):
    """Assemble the core-index-two local system for a three-core TT."""
    sample_count = features.shape[1]
    local_shape = cores[2].shape
    matrix = np.zeros((sample_count, int(np.prod(local_shape))), dtype=float)
    for k in range(sample_count):
        row = np.zeros(local_shape, dtype=float)
        for channel in range(features.shape[0]):
            left = np.ones(1, dtype=float)
            for coordinate in range(2):
                evaluated = np.tensordot(
                    cores[coordinate], features[channel, k, coordinate], axes=(1, 0)
                )
                left = left @ evaluated
            row += np.einsum("a,q,b->aqb", left, features[channel, k, 2], np.ones(1))
        matrix[k] = row.reshape(-1)
    return matrix


def advance_adaptive_sweep(
    weighted_features: np.ndarray,
    middle_design: np.ndarray,
    targets: np.ndarray,
    left_core: np.ndarray,
    updated_middle_core: np.ndarray,
    right_core: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Reference implementation."""
    features = np.asarray(weighted_features, dtype=float)
    matrix = np.asarray(middle_design, dtype=float)
    y = np.asarray(targets, dtype=float)
    first = np.asarray(left_core, dtype=float)
    middle = np.asarray(updated_middle_core, dtype=float)
    last = np.asarray(right_core, dtype=float)
    if (
        features.ndim != 4
        or features.shape[2] != 3
        or features.shape[0] <= 1
        or (features.shape[0] - 1) % 3 != 0
    ):
        raise ValueError("weighted_features must have shape (1 + 3*q, K, 3, m) for q >= 1")
    sample_count, basis_count = features.shape[1], features.shape[3]
    if sample_count == 0 or basis_count == 0:
        raise ValueError("weighted_features must be non-empty")
    if matrix.shape != (sample_count, middle.size):
        raise ValueError("middle_design must match the samples and middle core size")
    if y.shape != (sample_count,):
        raise ValueError("targets must have one value per sample")
    if first.ndim != 3 or first.shape[0] != 1 or first.shape[1] != basis_count:
        raise ValueError("left_core must have shape (1, m, r_1)")
    if middle.ndim != 3 or middle.shape[:2] != (first.shape[2], basis_count):
        raise ValueError("updated_middle_core must have shape (r_1, m, r_2)")
    if last.shape != (middle.shape[2], basis_count, 1):
        raise ValueError("right_core must have shape (r_2, m, 1)")
    arrays = (features, matrix, y, first, middle, last)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all array inputs must be finite")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and positive")
    if middle.shape[0] * basis_count < middle.shape[2]:
        raise ValueError("the SVD shift cannot preserve the right rank")

    residual_sq = float(np.dot(matrix @ middle.reshape(-1) - y, matrix @ middle.reshape(-1) - y))
    core_norm_sq = float(np.dot(middle.reshape(-1), middle.reshape(-1)))
    if core_norm_sq == 0.0:
        raise ValueError("the updated middle core must have positive norm")
    tau_next = float(gamma) * residual_sq / core_norm_sq
    if not np.isfinite(tau_next) or tau_next <= 0.0:
        raise ValueError("the adaptive regularization magnitude must be positive")

    left_vectors, singular_values, right_vectors = np.linalg.svd(
        middle.reshape(middle.shape[0] * basis_count, middle.shape[2]),
        full_matrices=False,
    )
    transfer = singular_values[:, None] * right_vectors
    absorbed_matrix = transfer @ last.reshape(last.shape[0], basis_count)
    row_energy = np.sum(absorbed_matrix * absorbed_matrix, axis=1)
    bond_order = np.lexsort((np.arange(row_energy.size), -row_energy))
    left_vectors = left_vectors[:, bond_order]
    absorbed_matrix = absorbed_matrix[bond_order]
    for bond in range(absorbed_matrix.shape[0]):
        right_pivot = int(np.argmax(np.abs(absorbed_matrix[bond])))
        sign = float(np.sign(absorbed_matrix[bond, right_pivot]))
        if sign == 0.0:
            left_pivot = int(np.argmax(np.abs(left_vectors[:, bond])))
            sign = float(np.sign(left_vectors[left_pivot, bond]))
            if sign == 0.0:
                sign = 1.0
        left_vectors[:, bond] *= sign
        absorbed_matrix[bond] *= sign
    shifted_middle = left_vectors.reshape(middle.shape)
    absorbed_last = absorbed_matrix.reshape(last.shape)
    last_design = _assemble_last_system(features, [first, shifted_middle, absorbed_last])
    normal_matrix = last_design.T @ last_design + tau_next * np.eye(last_design.shape[1])
    updated_last = np.linalg.solve(normal_matrix, last_design.T @ y).reshape(last.shape)
    return np.concatenate(([tau_next], shifted_middle.reshape(-1), updated_last.reshape(-1)))

import numpy as np  # noqa: E402, F811


def recover_feedback_control(
    query: np.ndarray,
    left_core: np.ndarray,
    adaptive_state: np.ndarray,
    sigma_diag: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    point = np.asarray(query, dtype=float)
    first = np.asarray(left_core, dtype=float)
    packed = np.asarray(adaptive_state, dtype=float)
    sigma = np.asarray(sigma_diag, dtype=float)
    if point.shape != (3,):
        raise ValueError("query must have shape (3,)")
    if first.ndim != 3 or first.shape[0] != 1 or first.shape[1] < 2 or first.shape[2] == 0:
        raise ValueError("left_core must have shape (1, m, r_1) with m >= 2")
    if packed.ndim != 1 or packed.size < 2:
        raise ValueError("adaptive_state must be a non-empty vector")
    if sigma.shape != (3,):
        raise ValueError("sigma_diag must have shape (3,)")
    if not all(np.all(np.isfinite(array)) for array in (point, first, packed, sigma)):
        raise ValueError("all inputs must be finite")
    if packed[0] <= 0.0:
        raise ValueError("the stored adaptive regularization must be positive")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_diag must be positive")
    basis_count = first.shape[1]
    rank_left = first.shape[2]
    denominator = basis_count * (rank_left + 1)
    if (packed.size - 1) % denominator != 0:
        raise ValueError("adaptive_state length is incompatible with left_core")
    rank_right = (packed.size - 1) // denominator
    if rank_right <= 0:
        raise ValueError("the packed right rank must be positive")

    middle_size = rank_left * basis_count * rank_right
    middle = packed[1 : 1 + middle_size].reshape(rank_left, basis_count, rank_right)
    last = packed[1 + middle_size :].reshape(rank_right, basis_count, 1)
    cores = [first, middle, last]
    degree = np.arange(basis_count, dtype=int)
    basis = [np.power(z, degree) for z in point]
    derivative = []
    for z in point:
        differentiated = np.zeros(basis_count, dtype=float)
        differentiated[1:] = degree[1:] * np.power(z, degree[:-1])
        derivative.append(differentiated)
    gradient = np.empty(3, dtype=float)
    for differentiated in range(3):
        product = np.ones((1, 1), dtype=float)
        for coordinate, core in enumerate(cores):
            vector = derivative[coordinate] if coordinate == differentiated else basis[coordinate]
            product = product @ np.tensordot(core, vector, axes=(1, 0))
        gradient[differentiated] = product.item()
    return -sigma * gradient

import numpy as np  # noqa: E402, F811


def run_full_pipeline(
    states: np.ndarray,
    noises: np.ndarray,
    drift_matrix: np.ndarray,
    drift_bias: np.ndarray,
    sigma_diag: np.ndarray,
    policy_matrix: np.ndarray,
    policy_bias: np.ndarray,
    next_cores: list[np.ndarray],
    initial_cores: list[np.ndarray],
    dt: float,
    tau_initial: float,
    gamma: float,
    query: np.ndarray,
) -> float:
    """Reference implementation."""
    x = np.asarray(states, dtype=float)
    xi = np.asarray(noises, dtype=float)
    matrix = np.asarray(drift_matrix, dtype=float)
    drift_offset = np.asarray(drift_bias, dtype=float)
    sigma = np.asarray(sigma_diag, dtype=float)
    policy = np.asarray(policy_matrix, dtype=float)
    policy_offset = np.asarray(policy_bias, dtype=float)
    evaluation_point = np.asarray(query, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 3:
        raise ValueError("states must have non-empty shape (K, 3)")
    if xi.shape != x.shape:
        raise ValueError("noises must have the same shape as states")
    if matrix.shape != (3, 3) or policy.shape != (3, 3):
        raise ValueError("drift_matrix and policy_matrix must have shape (3, 3)")
    if drift_offset.shape != (3,) or policy_offset.shape != (3,) or sigma.shape != (3,):
        raise ValueError("biases and sigma_diag must have shape (3,)")
    if evaluation_point.shape != (3,):
        raise ValueError("query must have shape (3,)")
    arrays = (x, xi, matrix, drift_offset, sigma, policy, policy_offset, evaluation_point)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all direct array inputs must be finite")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_diag must be positive")
    scalars = (dt, tau_initial, gamma)
    if any(not np.isfinite(value) or value <= 0.0 for value in scalars):
        raise ValueError("dt, tau_initial, and gamma must be finite and positive")
    if not isinstance(next_cores, (list, tuple)) or len(next_cores) != 3:
        raise ValueError("next_cores must contain three TT cores")
    if not isinstance(initial_cores, (list, tuple)) or len(initial_cores) != 3:
        raise ValueError("initial_cores must contain three TT cores")
    continuation_cores = [np.asarray(core, dtype=float) for core in next_cores]
    current_cores = [np.asarray(core, dtype=float) for core in initial_cores]
    for core_set in (continuation_cores, current_cores):
        if any(core.ndim != 3 or core.shape[1] != 3 for core in core_set):
            raise ValueError("every TT core must have basis mode three")
        if core_set[0].shape[0] != 1 or core_set[-1].shape[2] != 1:
            raise ValueError("exterior TT ranks must equal one")
        if core_set[0].shape[2] != core_set[1].shape[0] or core_set[1].shape[2] != core_set[2].shape[0]:
            raise ValueError("adjacent TT ranks must agree")
        if any(not np.all(np.isfinite(core)) for core in core_set):
            raise ValueError("TT cores must be finite")
    first_matrix = current_cores[0].reshape(3, current_cores[0].shape[2])
    last_matrix = current_cores[2].reshape(current_cores[2].shape[0], 3)
    if not np.allclose(first_matrix.T @ first_matrix, np.eye(first_matrix.shape[1]), atol=1e-12):
        raise ValueError("the first current-time core must be left orthonormal")
    if not np.allclose(last_matrix @ last_matrix.T, np.eye(last_matrix.shape[0]), atol=1e-12):
        raise ValueError("the last current-time core must be right orthonormal")

    next_states = advance_controlled_diffusion(  # noqa: F821
        x, xi, matrix, drift_offset, sigma, policy, policy_offset, dt
    )
    continuation = evaluate_tt_continuation(  # noqa: F821
        next_states, continuation_cores
    )
    targets = form_bsde_targets(  # noqa: F821
        continuation,
        next_states,
        policy,
        policy_offset,
        float(np.trace(matrix)),
        sigma,
        dt,
    )
    weighted_features = build_weighted_features(x, xi, sigma, dt)  # noqa: F821
    middle_design = assemble_local_system(  # noqa: F821
        weighted_features, current_cores, 1
    )
    updated_middle = solve_ridge_core(  # noqa: F821
        middle_design, targets, tau_initial, current_cores[1].shape
    )
    adaptive_state = advance_adaptive_sweep(  # noqa: F821
        weighted_features,
        middle_design,
        targets,
        current_cores[0],
        updated_middle,
        current_cores[2],
        gamma,
    )
    feedback = recover_feedback_control(  # noqa: F821
        evaluation_point, current_cores[0], adaptive_state, sigma
    )
    return float(feedback[0])
SCICODE_GOLD_EOF
