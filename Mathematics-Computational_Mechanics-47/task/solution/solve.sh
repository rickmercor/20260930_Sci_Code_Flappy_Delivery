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


def cubic_heat_rhs(
    state: np.ndarray,
    time: float,
    amplitudes: np.ndarray,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _finite_vector(name: str, values, size: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.shape[0] != size:
            raise ValueError(f"{name} must be a 1-D array of length {size}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} entries must be finite")
        return array

    field = np.asarray(state, dtype=float)
    if field.ndim != 1 or field.shape[0] < 1:
        raise ValueError("state must be a 1-D array with at least one entry")
    if not np.all(np.isfinite(field)):
        raise ValueError("state entries must be finite")
    n_nodes = field.shape[0]

    amps = _finite_vector("amplitudes", amplitudes, 2)
    bounds = _finite_vector("boundary_values", boundary_values, 2)
    if not _is_number(time) or not math.isfinite(float(time)):
        raise ValueError("time must be a finite real scalar")
    if not _is_number(domain_length) or not math.isfinite(float(domain_length)):
        raise ValueError("domain_length must be a finite real scalar")
    if not _is_number(diffusivity) or not math.isfinite(float(diffusivity)):
        raise ValueError("diffusivity must be a finite real scalar")
    length = float(domain_length)
    kappa = float(diffusivity)
    if length <= 0.0:
        raise ValueError("domain_length must be positive")
    if kappa <= 0.0:
        raise ValueError("diffusivity must be positive")

    spacing = length / (n_nodes + 1.0)
    positions = spacing * np.arange(1, n_nodes + 1, dtype=float)
    scaled = positions / length

    curvature = np.empty(n_nodes, dtype=float)
    if n_nodes == 1:
        curvature[0] = bounds[0] - 2.0 * field[0] + bounds[1]
    else:
        curvature[0] = bounds[0] - 2.0 * field[0] + field[1]
        curvature[1:-1] = field[:-2] - 2.0 * field[1:-1] + field[2:]
        curvature[-1] = field[-2] - 2.0 * field[-1] + bounds[1]

    moment = float(time)
    lobe_left = 1.0 / (1.0 + 100.0 * (scaled - 0.25) ** 2)
    lobe_right = 1.0 / (1.0 + 100.0 * (scaled - 0.75) ** 2)
    source = (amps[0] * math.sin(2.0 * math.pi * moment) * lobe_left
              + amps[1] * math.sin(4.0 * math.pi * moment) * lobe_right)

    derivative = kappa / spacing ** 2 * curvature - field ** 3 + source
    return np.asarray(derivative, dtype=float)

import numpy as np


def march_cubic_fom(
    initial_state: np.ndarray,
    time_step: float,
    n_steps: int,
    amplitudes: np.ndarray,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
    residual_tolerance: float,
    max_newton_iterations: int,
    rhs_function=None,
) -> np.ndarray:
    """Reference implementation."""
    from numbers import Integral, Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _is_integer(value) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    field = np.asarray(initial_state, dtype=float)
    if field.ndim != 1 or field.shape[0] < 1:
        raise ValueError("initial_state must be a 1-D array with at least one entry")
    if not np.all(np.isfinite(field)):
        raise ValueError("initial_state entries must be finite")
    n_nodes = field.shape[0]

    bounds = np.asarray(boundary_values, dtype=float)
    if bounds.ndim != 1 or bounds.shape[0] != 2 or not np.all(np.isfinite(bounds)):
        raise ValueError("boundary_values must be a finite 1-D array of length 2")
    if not _is_number(time_step) or not math.isfinite(float(time_step)) or float(time_step) <= 0.0:
        raise ValueError("time_step must be a finite positive real scalar")
    if not _is_integer(n_steps) or int(n_steps) < 0:
        raise ValueError("n_steps must be a non-negative integer")
    if not _is_number(domain_length) or not math.isfinite(float(domain_length)) or float(domain_length) <= 0.0:
        raise ValueError("domain_length must be a finite positive real scalar")
    if not _is_number(diffusivity) or not math.isfinite(float(diffusivity)) or float(diffusivity) <= 0.0:
        raise ValueError("diffusivity must be a finite positive real scalar")
    if (not _is_number(residual_tolerance) or not math.isfinite(float(residual_tolerance))
            or float(residual_tolerance) <= 0.0):
        raise ValueError("residual_tolerance must be a finite positive real scalar")
    if not _is_integer(max_newton_iterations) or int(max_newton_iterations) < 1:
        raise ValueError("max_newton_iterations must be a positive integer")

    step = float(time_step)
    total = int(n_steps)
    tolerance = float(residual_tolerance)
    cap = int(max_newton_iterations)
    rhs = cubic_heat_rhs if rhs_function is None else rhs_function
    spacing = float(domain_length) / (n_nodes + 1.0)
    curvature_scale = float(diffusivity) / spacing ** 2

    stencil = np.diag(-2.0 * np.ones(n_nodes, dtype=float))
    if n_nodes > 1:
        off = np.ones(n_nodes - 1, dtype=float)
        stencil += np.diag(off, 1) + np.diag(off, -1)
    diffusion = curvature_scale * stencil
    identity = np.eye(n_nodes, dtype=float)

    trajectory = np.empty((n_nodes, total + 1), dtype=float)
    trajectory[:, 0] = field
    for index in range(1, total + 1):
        moment = index * step
        previous = trajectory[:, index - 1]
        current = previous.copy()
        for _ in range(cap):
            rate = rhs(current, moment, amplitudes, domain_length,
                       diffusivity, boundary_values)
            residual = current - previous - step * np.asarray(rate, dtype=float)
            if float(np.linalg.norm(residual)) < tolerance:
                break
            jacobian = identity - step * (diffusion - np.diag(3.0 * current ** 2))
            current = current - np.linalg.solve(jacobian, residual)
        trajectory[:, index] = current
    return trajectory

import numpy as np


def lifted_polynomial_operators(
    n_nodes: int,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    from numbers import Integral, Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _is_integer(value) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    if not _is_integer(n_nodes) or int(n_nodes) < 1:
        raise ValueError("n_nodes must be a positive integer")
    bounds = np.asarray(boundary_values, dtype=float)
    if bounds.ndim != 1 or bounds.shape[0] != 2 or not np.all(np.isfinite(bounds)):
        raise ValueError("boundary_values must be a finite 1-D array of length 2")
    if not _is_number(domain_length) or not math.isfinite(float(domain_length)) or float(domain_length) <= 0.0:
        raise ValueError("domain_length must be a finite positive real scalar")
    if not _is_number(diffusivity) or not math.isfinite(float(diffusivity)) or float(diffusivity) <= 0.0:
        raise ValueError("diffusivity must be a finite positive real scalar")

    count = int(n_nodes)
    width = 2 * count
    length = float(domain_length)
    spacing = length / (count + 1.0)
    scale = float(diffusivity) / spacing ** 2
    scaled = spacing * np.arange(1, count + 1, dtype=float) / length
    profile = np.stack([1.0 / (1.0 + 100.0 * (scaled - 0.25) ** 2),
                        1.0 / (1.0 + 100.0 * (scaled - 0.75) ** 2)], axis=1)

    constant = np.zeros(width, dtype=float)
    linear = np.zeros((width, width), dtype=float)
    quadratic = np.zeros((width, width * width), dtype=float)
    inputs = np.zeros((width, 2), dtype=float)
    bilinear = np.zeros((width, 2 * width), dtype=float)

    def _deposit(row: int, first: int, second: int, value: float) -> None:
        low, high = (first, second) if first <= second else (second, first)
        quadratic[row, low * width + high] += value

    for node in range(count):
        upper = count + node
        # Temperature block: diffusion, ghost values, sink written as -q*w, source.
        linear[node, node] += -2.0 * scale
        if node - 1 >= 0:
            linear[node, node - 1] += scale
        else:
            constant[node] += scale * bounds[0]
        if node + 1 <= count - 1:
            linear[node, node + 1] += scale
        else:
            constant[node] += scale * bounds[1]
        _deposit(node, node, upper, -1.0)
        inputs[node, :] = profile[node, :]
        # Auxiliary block: twice the temperature times the temperature equation.
        _deposit(upper, node, node, -4.0 * scale)
        if node - 1 >= 0:
            _deposit(upper, node, node - 1, 2.0 * scale)
        else:
            linear[upper, node] += 2.0 * scale * bounds[0]
        if node + 1 <= count - 1:
            _deposit(upper, node, node + 1, 2.0 * scale)
        else:
            linear[upper, node] += 2.0 * scale * bounds[1]
        _deposit(upper, upper, upper, -2.0)
        for channel in range(2):
            bilinear[upper, channel * width + node] += 2.0 * profile[node, channel]

    return constant, linear, quadratic, inputs, bilinear

import numpy as np


def block_pod_basis(
    state_snapshots: np.ndarray,
    energy_tolerance: float,
) -> tuple[np.ndarray, int, int]:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    snapshots = np.asarray(state_snapshots, dtype=float)
    if snapshots.ndim != 2 or snapshots.shape[0] < 1 or snapshots.shape[1] < 1:
        raise ValueError("state_snapshots must be a 2-D array with positive extents")
    if not np.all(np.isfinite(snapshots)):
        raise ValueError("state_snapshots entries must be finite")
    if not _is_number(energy_tolerance) or not math.isfinite(float(energy_tolerance)):
        raise ValueError("energy_tolerance must be a finite real scalar")
    tolerance = float(energy_tolerance)
    if not 0.0 < tolerance < 1.0:
        raise ValueError("energy_tolerance must lie strictly between 0 and 1")

    n_nodes = snapshots.shape[0]
    auxiliary = snapshots * snapshots

    modes = []
    counts = []
    for matrix in (snapshots, auxiliary):
        left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        energy = singular ** 2
        total = float(np.sum(energy))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("a snapshot block carries no energy")
        neglected = 1.0 - np.cumsum(energy) / total
        admissible = np.nonzero(neglected < tolerance)[0]
        retained = int(admissible[0]) + 1 if admissible.size else int(singular.size)
        block = np.array(left[:, :retained], dtype=float)
        for column in range(retained):
            pivot = int(np.argmax(np.abs(block[:, column])))
            if block[pivot, column] < 0.0:
                block[:, column] = -block[:, column]
        modes.append(block)
        counts.append(retained)

    first, second = counts
    basis = np.zeros((2 * n_nodes, first + second), dtype=float)
    basis[:n_nodes, :first] = modes[0]
    basis[n_nodes:, first:] = modes[1]
    return basis, first, second

import numpy as np


def hrf_operator_gram(
    trial_basis: np.ndarray,
    constant_operator: np.ndarray,
    linear_operator: np.ndarray,
    quadratic_operator: np.ndarray,
    input_operator: np.ndarray,
    bilinear_operator: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    def _finite(name: str, values, ndim: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != ndim:
            raise ValueError(f"{name} must have ndim={ndim}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} entries must be finite")
        return array

    basis = _finite("trial_basis", trial_basis, 2)
    constant = _finite("constant_operator", constant_operator, 1)
    linear = _finite("linear_operator", linear_operator, 2)
    quadratic = _finite("quadratic_operator", quadratic_operator, 2)
    inputs = _finite("input_operator", input_operator, 2)
    bilinear = _finite("bilinear_operator", bilinear_operator, 2)

    full, reduced = basis.shape
    if full < 1 or reduced < 1:
        raise ValueError("trial_basis must have positive extents")
    if constant.shape[0] != full:
        raise ValueError("constant_operator must have one entry per full-order row")
    if linear.shape != (full, full):
        raise ValueError("linear_operator must be square with the full-order size")
    if quadratic.shape != (full, full * full):
        raise ValueError("quadratic_operator must have full**2 columns")
    if inputs.shape[0] != full or inputs.shape[1] < 1:
        raise ValueError("input_operator must have one row per full-order row")
    n_inputs = inputs.shape[1]
    if bilinear.shape != (full, n_inputs * full):
        raise ValueError("bilinear_operator must have n_inputs * full columns")

    quadratic_columns = quadratic @ np.kron(basis, basis)
    bilinear_columns = bilinear @ np.kron(np.eye(n_inputs, dtype=float), basis)
    blocks = np.concatenate(
        [
            basis,
            linear @ basis,
            quadratic_columns,
            bilinear_columns,
            constant.reshape(full, 1),
            inputs,
        ],
        axis=1,
    )
    gram = blocks.T @ blocks
    return np.asarray(gram, dtype=float)

import numpy as np


def kronecker_square_jacobian(reduced_state: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    import numpy as np

    vector = np.asarray(reduced_state, dtype=float)
    if vector.ndim != 1 or vector.shape[0] < 1:
        raise ValueError("reduced_state must be a 1-D array with at least one entry")
    if not np.all(np.isfinite(vector)):
        raise ValueError("reduced_state entries must be finite")

    size = vector.shape[0]
    identity = np.eye(size, dtype=float)
    column = vector.reshape(size, 1)
    # The first term differentiates the left factor of the Kronecker square and
    # the second term differentiates the right factor; both are needed because
    # the two factors occupy different index positions.
    derivative = np.kron(column, identity) + np.kron(identity, column)
    return np.asarray(derivative, dtype=float)

import numpy as np


def residual_coefficient_vector(
    reduced_history: np.ndarray,
    input_history: np.ndarray,
    time_step: float,
    alphas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _finite(name: str, values, ndim: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != ndim:
            raise ValueError(f"{name} must have ndim={ndim}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} entries must be finite")
        return array

    states = _finite("reduced_history", reduced_history, 2)
    controls = _finite("input_history", input_history, 2)
    state_weights = _finite("alphas", alphas, 1)
    rate_weights = _finite("betas", betas, 1)
    if states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("reduced_history must have positive extents")
    if controls.shape[0] != states.shape[0] or controls.shape[1] < 1:
        raise ValueError("input_history must have one row per stored step")
    if state_weights.shape[0] != states.shape[0] or rate_weights.shape[0] != states.shape[0]:
        raise ValueError("alphas and betas must have one entry per stored step")
    if not _is_number(time_step) or not math.isfinite(float(time_step)) or float(time_step) <= 0.0:
        raise ValueError("time_step must be a finite positive real scalar")

    step = float(time_step)
    n_reduced = states.shape[1]
    n_inputs = controls.shape[1]
    quadratic = n_reduced * n_reduced
    bilinear = n_inputs * n_reduced
    total = 2 * n_reduced + quadratic + bilinear + 1 + n_inputs

    first = n_reduced
    second = 2 * n_reduced
    third = second + quadratic
    fourth = third + bilinear

    coefficients = np.zeros(total, dtype=float)
    for index in range(states.shape[0]):
        state = states[index]
        control = controls[index]
        weight = float(state_weights[index])
        rate = -step * float(rate_weights[index])
        coefficients[:first] += weight * state
        coefficients[first:second] += rate * state
        coefficients[second:third] += rate * np.kron(state, state)
        coefficients[third:fourth] += rate * np.kron(control, state)
        coefficients[fourth] += rate
        coefficients[fourth + 1:] += rate * control
    return coefficients

import numpy as np


def reduced_newton_direction(
    gram: np.ndarray,
    coefficients: np.ndarray,
    reduced_state: np.ndarray,
    current_input: np.ndarray,
    time_step: float,
    alpha_zero: float,
    beta_zero: float,
    scheme: str,
    kronecker_function=None,
) -> tuple[np.ndarray, float]:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _finite(name: str, values, ndim: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != ndim:
            raise ValueError(f"{name} must have ndim={ndim}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} entries must be finite")
        return array

    matrix = _finite("gram", gram, 2)
    coeff = _finite("coefficients", coefficients, 1)
    state = _finite("reduced_state", reduced_state, 1)
    control = _finite("current_input", current_input, 1)
    if not isinstance(scheme, str) or scheme not in ("galerkin", "lspg"):
        raise ValueError("scheme must be 'galerkin' or 'lspg'")
    if not _is_number(time_step) or not math.isfinite(float(time_step)) or float(time_step) <= 0.0:
        raise ValueError("time_step must be a finite positive real scalar")
    for name, value in (("alpha_zero", alpha_zero), ("beta_zero", beta_zero)):
        if not _is_number(value) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite real scalar")

    n_reduced = state.shape[0]
    n_inputs = control.shape[0]
    if n_reduced < 1 or n_inputs < 1:
        raise ValueError("reduced_state and current_input must be non-empty")
    total = 2 * n_reduced + n_reduced * n_reduced + n_inputs * n_reduced + 1 + n_inputs
    if matrix.shape != (total, total):
        raise ValueError("gram must be square with the column-basis dimension")
    if coeff.shape[0] != total:
        raise ValueError("coefficients must match the column-basis dimension")

    step = float(time_step)
    lead_state = float(alpha_zero)
    lead_rate = float(beta_zero)
    identity = np.eye(n_reduced, dtype=float)
    square_jacobian = (kronecker_square_jacobian
                       if kronecker_function is None else kronecker_function)

    first = n_reduced
    second = 2 * n_reduced
    third = second + n_reduced * n_reduced
    fourth = third + n_inputs * n_reduced

    factor = np.zeros((total, n_reduced), dtype=float)
    factor[:first, :] = lead_state * identity
    factor[first:second, :] = -step * lead_rate * identity
    factor[second:third, :] = -step * lead_rate * square_jacobian(state)
    factor[third:fourth, :] = -step * lead_rate * np.kron(control.reshape(n_inputs, 1), identity)

    if scheme == "lspg":
        # The test basis is the full-order Jacobian itself, so both sides are
        # contracted through the Gram matrix rather than through the trial block.
        weighted = matrix @ factor
        system = factor.T @ weighted
        gradient = factor.T @ (matrix @ coeff)
    else:
        trial_rows = matrix[:first, :]
        system = trial_rows @ factor
        gradient = trial_rows @ coeff

    direction = np.linalg.solve(system, -gradient)
    return np.asarray(direction, dtype=float), float(np.linalg.norm(gradient))

import numpy as np


def march_reduced_rom(
    gram: np.ndarray,
    initial_reduced_state: np.ndarray,
    time_step: float,
    n_steps: int,
    amplitudes: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    scheme: str,
    residual_tolerance: float,
    max_newton_iterations: int,
    coefficient_function=None,
    direction_function=None,
    kronecker_function=None,
) -> np.ndarray:
    """Reference implementation."""
    from numbers import Integral, Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _is_integer(value) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    def _finite(name: str, values, ndim: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != ndim:
            raise ValueError(f"{name} must have ndim={ndim}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} entries must be finite")
        return array

    matrix = _finite("gram", gram, 2)
    start = _finite("initial_reduced_state", initial_reduced_state, 1)
    amps = _finite("amplitudes", amplitudes, 1)
    state_weights = _finite("alphas", alphas, 1)
    rate_weights = _finite("betas", betas, 1)
    if not isinstance(scheme, str) or scheme not in ("galerkin", "lspg"):
        raise ValueError("scheme must be 'galerkin' or 'lspg'")
    if start.shape[0] < 1:
        raise ValueError("initial_reduced_state must be non-empty")
    if amps.shape[0] != 2:
        raise ValueError("amplitudes must have exactly two entries")
    if state_weights.shape[0] < 1 or state_weights.shape[0] != rate_weights.shape[0]:
        raise ValueError("alphas and betas must be non-empty and equally long")
    if not _is_number(time_step) or not math.isfinite(float(time_step)) or float(time_step) <= 0.0:
        raise ValueError("time_step must be a finite positive real scalar")
    if not _is_integer(n_steps) or int(n_steps) < 0:
        raise ValueError("n_steps must be a non-negative integer")
    if (not _is_number(residual_tolerance) or not math.isfinite(float(residual_tolerance))
            or float(residual_tolerance) <= 0.0):
        raise ValueError("residual_tolerance must be a finite positive real scalar")
    if not _is_integer(max_newton_iterations) or int(max_newton_iterations) < 1:
        raise ValueError("max_newton_iterations must be a positive integer")

    n_reduced = start.shape[0]
    levels = state_weights.shape[0]
    total = 2 * n_reduced + n_reduced * n_reduced + 2 * n_reduced + 1 + 2
    if matrix.shape != (total, total):
        raise ValueError("gram must be square with the column-basis dimension")

    step = float(time_step)
    horizon = int(n_steps)
    tolerance = float(residual_tolerance)
    cap = int(max_newton_iterations)
    make_coefficients = (residual_coefficient_vector
                         if coefficient_function is None else coefficient_function)
    make_direction = (reduced_newton_direction
                      if direction_function is None else direction_function)

    def _input_at(level: int) -> np.ndarray:
        moment = level * step
        return np.array([amps[0] * math.sin(2.0 * math.pi * moment),
                         amps[1] * math.sin(4.0 * math.pi * moment)], dtype=float)

    trajectory = np.empty((n_reduced, horizon + 1), dtype=float)
    trajectory[:, 0] = start
    for index in range(1, horizon + 1):
        current = trajectory[:, index - 1].copy()
        past = [trajectory[:, max(index - j, 0)] for j in range(1, levels)]
        controls = np.array([_input_at(max(index - j, 0)) for j in range(levels)], dtype=float)
        for _ in range(cap):
            history = np.array([current] + past, dtype=float)
            coefficients = make_coefficients(
                history, controls, step, state_weights, rate_weights)
            direction, measure = make_direction(
                matrix, coefficients, current, controls[0], step,
                float(state_weights[0]), float(rate_weights[0]), scheme,
                kronecker_function=kronecker_function)
            if float(measure) < tolerance:
                break
            current = current + np.asarray(direction, dtype=float)
        trajectory[:, index] = current
    return trajectory

import numpy as np


def run_hrf_rom_pipeline(
    quantity: str,
    n_nodes: int,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
    time_step: float,
    n_training_steps: int,
    n_online_steps: int,
    training_amplitudes: np.ndarray,
    test_amplitudes: np.ndarray,
    energy_tolerance: float,
    alphas: np.ndarray,
    betas: np.ndarray,
    residual_tolerance: float,
    max_newton_iterations: int,
) -> float:
    """Reference implementation."""
    from numbers import Integral, Real

    import math
    import numpy as np

    def _is_integer(value) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    recognised = ("lspg_error_ratio", "galerkin_error_ratio", "lspg_state_error",
                  "galerkin_state_error", "projection_error", "temperature_mode_count",
                  "auxiliary_mode_count", "worst_excess_error_ratio")
    if not isinstance(quantity, str) or quantity not in recognised:
        raise ValueError("quantity is not a recognised reporting name")
    if not _is_integer(n_nodes) or int(n_nodes) < 1:
        raise ValueError("n_nodes must be a positive integer")

    training = np.asarray(training_amplitudes, dtype=float)
    if training.ndim != 2 or training.shape[0] < 1 or training.shape[1] != 2:
        raise ValueError("training_amplitudes must have shape (n_train, 2)")
    if not np.all(np.isfinite(training)):
        raise ValueError("training_amplitudes entries must be finite")
    testing = np.asarray(test_amplitudes, dtype=float)
    if quantity == "worst_excess_error_ratio":
        if (testing.ndim != 2 or testing.shape[0] < 1 or testing.shape[1] != 2
                or not np.all(np.isfinite(testing))):
            raise ValueError("test_amplitudes must have finite shape (n_test, 2)")
    elif testing.ndim != 1 or testing.shape[0] != 2 or not np.all(np.isfinite(testing)):
        raise ValueError("test_amplitudes must be a finite 1-D array of length 2")

    count = int(n_nodes)
    spacing = float(domain_length) / (count + 1.0)
    scaled = spacing * np.arange(1, count + 1, dtype=float) / float(domain_length)
    initial = (scaled * (1.0 - scaled)
               * (6.0 * (1.0 - scaled) ** 2 * np.exp(-scaled)
                  - 10.0 * np.exp(scaled) * np.sin(scaled / 6.0))
               + scaled)

    snapshots = np.concatenate(
        [np.asarray(march_cubic_fom(initial, time_step, n_training_steps, row,
                                            domain_length, diffusivity, boundary_values,
                                            residual_tolerance, max_newton_iterations,
                                            rhs_function=cubic_heat_rhs), dtype=float)
         for row in training],
        axis=1,
    )
    basis, n_temperature, n_auxiliary = block_pod_basis(
        snapshots, energy_tolerance)
    if quantity == "temperature_mode_count":
        return float(n_temperature)
    if quantity == "auxiliary_mode_count":
        return float(n_auxiliary)

    basis = np.asarray(basis, dtype=float)
    modes = basis[:count, :int(n_temperature)]

    if quantity == "worst_excess_error_ratio":
        constant, linear, quadratic, inputs, bilinear = lifted_polynomial_operators(
            n_nodes, domain_length, diffusivity, boundary_values)
        gram = hrf_operator_gram(
            basis, constant, linear, quadratic, inputs, bilinear)
        lifted_initial = np.concatenate([initial, initial * initial])
        reduced_initial = basis.T @ lifted_initial
        excess_ratios = []

        for amplitudes in testing:
            reference = np.asarray(march_cubic_fom(
                initial, time_step, n_online_steps, amplitudes, domain_length,
                diffusivity, boundary_values, residual_tolerance,
                max_newton_iterations, rhs_function=cubic_heat_rhs),
                                   dtype=float)
            residual_field = reference - modes @ (modes.T @ reference)
            projection_energy = float(np.sum(residual_field ** 2))
            if not math.isfinite(projection_energy) or projection_energy <= 0.0:
                raise ValueError("the projection energy must be finite and positive")

            prediction_energies = {}
            for scheme in ("lspg", "galerkin"):
                reduced = np.asarray(march_reduced_rom(
                    gram, reduced_initial, time_step, n_online_steps, amplitudes,
                    alphas, betas, scheme, residual_tolerance,
                    max_newton_iterations,
                    coefficient_function=residual_coefficient_vector,
                    direction_function=reduced_newton_direction,
                    kronecker_function=kronecker_square_jacobian),
                                     dtype=float)
                reconstructed = modes @ reduced[:int(n_temperature), :]
                prediction_energies[scheme] = float(
                    np.sum((reference - reconstructed) ** 2))

            galerkin_excess = prediction_energies["galerkin"] - projection_energy
            lspg_excess = prediction_energies["lspg"] - projection_energy
            if (not math.isfinite(galerkin_excess) or galerkin_excess <= 0.0
                    or not math.isfinite(lspg_excess) or lspg_excess < 0.0):
                raise ValueError("the excess errors must be finite with a positive denominator")
            excess_ratios.append(lspg_excess / galerkin_excess)

        value = float(max(excess_ratios))
        if not math.isfinite(value):
            raise ValueError("the reported scalar must be finite")
        return value

    reference = np.asarray(march_cubic_fom(
        initial, time_step, n_online_steps, testing, domain_length, diffusivity,
        boundary_values, residual_tolerance, max_newton_iterations,
        rhs_function=cubic_heat_rhs),
                           dtype=float)
    residual_field = reference - modes @ (modes.T @ reference)
    projection_energy = float(np.sum(residual_field ** 2))
    reference_energy = float(np.sum(reference ** 2))
    if quantity == "projection_error":
        return float(projection_energy / reference_energy)

    constant, linear, quadratic, inputs, bilinear = lifted_polynomial_operators(
        n_nodes, domain_length, diffusivity, boundary_values)
    gram = hrf_operator_gram(
        basis, constant, linear, quadratic, inputs, bilinear)
    lifted_initial = np.concatenate([initial, initial * initial])
    reduced_initial = basis.T @ lifted_initial

    scheme = "lspg" if quantity.startswith("lspg") else "galerkin"
    reduced = np.asarray(march_reduced_rom(
        gram, reduced_initial, time_step, n_online_steps, testing, alphas, betas,
        scheme, residual_tolerance, max_newton_iterations,
        coefficient_function=residual_coefficient_vector,
        direction_function=reduced_newton_direction,
        kronecker_function=kronecker_square_jacobian),
                         dtype=float)
    reconstructed = modes @ reduced[:int(n_temperature), :]
    prediction_energy = float(np.sum((reference - reconstructed) ** 2))

    if quantity.endswith("state_error"):
        value = prediction_energy / reference_energy
    else:
        if not math.isfinite(projection_energy) or projection_energy <= 0.0:
            raise ValueError("the projection energy must be finite and positive")
        value = prediction_energy / projection_energy
    if not math.isfinite(value):
        raise ValueError("the reported scalar must be finite")
    return float(value)
SCICODE_GOLD_EOF
